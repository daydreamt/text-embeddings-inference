use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DeBertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: Option<usize>,
    pub position_embedding_type: Option<String>,
    pub use_cache: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, u32>>,

    // DeBERTa specific configurations
    pub relative_attention: bool,
    pub pos_att_type: Option<Vec<String>>,
    pub max_relative_positions: Option<i64>,
    pub position_buckets: Option<i64>,
    pub share_att_key: Option<bool>,
    pub norm_rel_ebd: Option<String>,
    pub attention_head_size: Option<usize>,
    pub position_biased_input: Option<bool>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<HiddenAct>,
    pub pooler_hidden_size: Option<usize>,
}

impl Default for DeBertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu, // DeBERTa uses GELU activation
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: Some(0),
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
            id2label: None,
            label2id: None,
            relative_attention: true,
            pos_att_type: Some(vec!["p2c".to_string(), "c2p".to_string()]),
            max_relative_positions: Some(256),
            position_buckets: Some(256),
            share_att_key: Some(true),
            norm_rel_ebd: Some("none".to_string()),
            attention_head_size: None,
            position_biased_input: Some(true),
            pooler_dropout: None,
            pooler_hidden_act: None,
            pooler_hidden_size: None,
        }
    }
}

#[derive(Debug)]
pub struct DeBertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    position_biased_input: bool,
    span: tracing::Span,
}

impl DeBertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let position_biased_input = config.position_biased_input.unwrap_or(true);

        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let position_embeddings = if position_biased_input {
            Some(Embedding::new(
                vb.pp("position_embeddings").get(
                    (config.max_position_embeddings, config.hidden_size),
                    "weight",
                )?,
                config.hidden_size,
            ))
        } else {
            None
        };

        // Only create token_type_embeddings if type_vocab_size > 0
        let token_type_embeddings = if config.type_vocab_size > 0 {
            match vb
                .pp("token_type_embeddings")
                .get((config.type_vocab_size, config.hidden_size), "weight")
            {
                Ok(w) => Some(Embedding::new(w, config.hidden_size)),
                Err(_) => None,
            }
        } else {
            None
        };

        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            position_biased_input,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut embeddings = self.word_embeddings.forward(input_ids)?;

        if self.position_biased_input {
            if let Some(ref position_embeddings) = self.position_embeddings {
                embeddings = embeddings.add(&position_embeddings.forward(position_ids)?)?;
            }
        }

        if let Some(ref token_type_embeddings) = self.token_type_embeddings {
            embeddings = embeddings.add(&token_type_embeddings.forward(token_type_ids)?)?;
        }

        let embeddings = self.layer_norm.forward(&embeddings, None)?;

        Ok(embeddings)
    }
}

struct DeBertaDisentangledSelfAttention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,

    pos_att_type: Vec<String>,
    max_relative_positions: i64,
    position_buckets: i64,
    share_att_key: bool,
    pos_key_proj: Option<Linear>,
    pos_query_proj: Option<Linear>,

    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,

    pos_dropout: f64,
    softmax_scale: f64,

    span: tracing::Span,
}

impl DeBertaDisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query_proj = Linear::new(
            vb.pp("query_proj")
                .get((all_head_size, config.hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(vb.pp("query_proj").get(all_head_size, "bias")?),
            None,
        );

        let key_proj = Linear::new(
            vb.pp("key_proj")
                .get((all_head_size, config.hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(vb.pp("key_proj").get(all_head_size, "bias")?),
            None,
        );

        let value_proj = Linear::new(
            vb.pp("value_proj")
                .get((all_head_size, config.hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(vb.pp("value_proj").get(all_head_size, "bias")?),
            None,
        );

        let share_att_key = config.share_att_key.unwrap_or(true);
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();

        let (pos_key_proj, pos_query_proj) = if config.relative_attention && !share_att_key {
            let pos_key_proj = if pos_att_type.iter().any(|s| s == "c2p") {
                Some(Linear::new(
                    vb.pp("pos_key_proj")
                        .get((all_head_size, config.hidden_size), "weight")?
                        .transpose(0, 1)?,
                    Some(vb.pp("pos_key_proj").get(all_head_size, "bias")?),
                    None,
                ))
            } else {
                None
            };

            let pos_query_proj = if pos_att_type.iter().any(|s| s == "p2c") {
                Some(Linear::new(
                    vb.pp("pos_query_proj")
                        .get((all_head_size, config.hidden_size), "weight")?
                        .transpose(0, 1)?,
                    None, // No bias for pos_query_proj
                    None,
                ))
            } else {
                None
            };

            (pos_key_proj, pos_query_proj)
        } else {
            (None, None)
        };

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            pos_att_type,
            max_relative_positions: config.max_relative_positions.unwrap_or(256),
            position_buckets: config.position_buckets.unwrap_or(256),
            share_att_key,
            pos_key_proj,
            pos_query_proj,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            pos_dropout: config.hidden_dropout_prob,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "disentangled_self_attention"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Project to query, key, value
        let query_layer = self.query_proj.forward(hidden_states)?;
        let key_layer = self.key_proj.forward(hidden_states)?;
        let value_layer = self.value_proj.forward(hidden_states)?;

        // Reshape and transpose for attention computation
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let query_layer = self.transpose_for_scores(&query_layer, batch_size, seq_len)?;
        let key_layer = self.transpose_for_scores(&key_layer, batch_size, seq_len)?;
        let value_layer = self.transpose_for_scores(&value_layer, batch_size, seq_len)?;

        // Compute attention scores
        let key_layer_t = key_layer.transpose(2, 3)?;
        let attention_scores = query_layer.matmul(&key_layer_t)?;
        let mut attention_scores = (attention_scores * self.softmax_scale)?;

        // Add relative attention bias if available
        if self.pos_att_type.len() > 0 && relative_embeddings.is_some() && relative_pos.is_some() {
            let rel_att = self.disentangled_attention_bias(
                &query_layer,
                &key_layer,
                relative_pos.unwrap(),
                relative_embeddings.unwrap(),
            )?;
            attention_scores = attention_scores.broadcast_add(&rel_att)?;
        }

        let attention_scores = if let Some(attention_mask) = attention_mask {
            attention_scores.broadcast_add(attention_mask)?
        } else {
            attention_scores
        };

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Transpose back and reshape
        let context_layer = context_layer.transpose(1, 2)?;
        let context_layer = context_layer.contiguous()?;
        let context_layer = context_layer.reshape((batch_size, seq_len, self.all_head_size))?;

        Ok(context_layer)
    }

    fn transpose_for_scores(
        &self,
        x: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let x = x.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        x.transpose(1, 2)?.contiguous()
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: &Tensor,
        rel_embeddings: &Tensor,
    ) -> Result<Tensor> {
        // FIXME: Implement
        let device = query_layer.device();
        let dtype = query_layer.dtype();

        // FIXME: Implement
        Tensor::zeros(
            (
                query_layer.dim(0)?,
                query_layer.dim(1)?,
                query_layer.dim(2)?,
                key_layer.dim(2)?,
            ),
            dtype,
            device,
        )
    }
}

struct DeBertaAttention {
    self_attention: DeBertaDisentangledSelfAttention,
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DeBertaAttention {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let self_attention = DeBertaDisentangledSelfAttention::load(vb.pp("self"), config)?;

        let dense = Linear::new(
            vb.pp("output")
                .pp("dense")
                .get((config.hidden_size, config.hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(
                vb.pp("output")
                    .pp("dense")
                    .get(config.hidden_size, "bias")?,
            ),
            None,
        );

        let layer_norm = LayerNorm::load(
            vb.pp("output").pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            self_attention,
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let self_output = self.self_attention.forward(
            hidden_states,
            attention_mask,
            relative_embeddings,
            relative_pos,
        )?;
        let attention_output = self.dense.forward(&self_output)?;
        let attention_output = self
            .layer_norm
            .forward(&attention_output, Some(hidden_states))?;

        Ok(attention_output)
    }
}

struct DeBertaLayer {
    attention: DeBertaAttention,
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DeBertaLayer {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let attention = DeBertaAttention::load(vb.pp("attention"), config)?;

        let intermediate = Linear::new(
            vb.pp("intermediate")
                .pp("dense")
                .get((config.intermediate_size, config.hidden_size), "weight")?,
            Some(
                vb.pp("intermediate")
                    .pp("dense")
                    .get(config.intermediate_size, "bias")?,
            ),
            Some(config.hidden_act.clone()),
        );

        let output = Linear::new(
            vb.pp("output")
                .pp("dense")
                .get((config.hidden_size, config.intermediate_size), "weight")?,
            Some(
                vb.pp("output")
                    .pp("dense")
                    .get(config.hidden_size, "bias")?,
            ),
            None,
        );

        let layer_norm = LayerNorm::load(
            vb.pp("output").pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            attention,
            intermediate,
            output,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            relative_embeddings,
            relative_pos,
        )?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self
            .layer_norm
            .forward(&layer_output, Some(&attention_output))?;

        Ok(layer_output)
    }
}

struct DeBertaRelativeEmbeddings {
    embeddings: Embedding,
    dropout: f64,
    max_relative_positions: i64,
    position_buckets: i64,
    span: tracing::Span,
}

impl DeBertaRelativeEmbeddings {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let max_relative_positions = config.max_relative_positions.unwrap_or(256);
        let position_buckets = config.position_buckets.unwrap_or(256);

        // When max_relative_positions is -1, use position_buckets
        let embedding_size = if max_relative_positions < 1 {
            position_buckets as usize
        } else {
            max_relative_positions as usize
        };

        let hidden_size = config.hidden_size;

        // DeBERTa uses 2x the size for positive and negative positions
        let embeddings = Embedding::new(
            vb.get((embedding_size * 2, hidden_size), "weight")?,
            hidden_size,
        );

        Ok(Self {
            embeddings,
            dropout: config.hidden_dropout_prob,
            max_relative_positions: embedding_size as i64,
            position_buckets,
            span: tracing::span!(tracing::Level::TRACE, "relative_embeddings"),
        })
    }

    fn get_rel_embedding(&self) -> Result<Tensor> {
        Ok(self.embeddings.embeddings().clone())
    }
}

struct DeBertaEncoder {
    layers: Vec<DeBertaLayer>,
    relative_attention: Option<DeBertaRelativeEmbeddings>,
    layer_norm: Option<LayerNorm>,
    relative_attention_enabled: bool,
    span: tracing::Span,
}

impl DeBertaEncoder {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DeBertaLayer::load(vb.pp(&format!("layer.{}", i)), config)?);
        }

        let relative_attention = if config.relative_attention {
            Some(DeBertaRelativeEmbeddings::load(
                vb.pp("rel_embeddings"),
                config,
            )?)
        } else {
            None
        };

        let norm_rel_ebd = config
            .norm_rel_ebd
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("none");
        let layer_norm = if norm_rel_ebd.contains("layer_norm") {
            match LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            ) {
                Ok(ln) => Some(ln),
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(Self {
            layers,
            relative_attention,
            layer_norm,
            relative_attention_enabled: config.relative_attention,
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        let (relative_embeddings, relative_pos) = if self.relative_attention_enabled {
            if let Some(ref relative_attention) = self.relative_attention {
                let rel_emb = relative_attention.get_rel_embedding()?;
                let rel_emb = if let Some(ref layer_norm) = self.layer_norm {
                    layer_norm.forward(&rel_emb, None)?
                } else {
                    rel_emb
                };
                (Some(rel_emb), None) // You'll need to implement relative position calculation
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                attention_mask,
                relative_embeddings.as_ref(),
                relative_pos.as_ref(),
            )?;
        }

        Ok(hidden_states)
    }
}

struct DeBertaPooler {
    dense: Linear,
    span: tracing::Span,
}

impl DeBertaPooler {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let dense = Linear::new(
            vb.pp("dense")
                .get((config.hidden_size, config.hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(vb.pp("dense").get(config.hidden_size, "bias")?),
            None,
        );

        Ok(Self {
            dense,
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Pool the first token (CLS token) representation
        let first_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&first_token)?;
        // Apply tanh activation
        pooled.tanh()
    }
}

pub struct DeBertaContextPooler {
    dense: Linear,
    dropout: f64,
    activation: HiddenAct,
    span: tracing::Span,
}

impl DeBertaContextPooler {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let pooler_hidden_size = config.pooler_hidden_size.ok_or_else(|| {
            candle::Error::Msg("pooler_hidden_size is required for context pooler".to_string())
        })?;

        let dense = Linear::new(
            vb.pp("dense")
                .get((pooler_hidden_size, pooler_hidden_size), "weight")?
                .transpose(0, 1)?,
            Some(vb.pp("dense").get(pooler_hidden_size, "bias")?),
            None,
        );

        let dropout = config.pooler_dropout.unwrap_or(config.hidden_dropout_prob);
        let activation = config.pooler_hidden_act.clone().unwrap_or(HiddenAct::Gelu);

        Ok(Self {
            dense,
            dropout,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "context_pooler"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Take the first token
        let context_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&context_token)?;

        // Apply activation based on the configured type
        match self.activation {
            HiddenAct::Gelu => pooled.gelu(),
            HiddenAct::Relu => pooled.relu(),
            HiddenAct::Silu => pooled.silu(),
            HiddenAct::Swiglu => {
                // Swiglu is not typically used for pooler activation in DeBERTa
                candle::bail!("Swiglu activation is not supported for DeBERTa context pooler")
            }
        }
    }
}

pub struct DeBertaClassificationHead {
    dropout: f64,
    classifier: Linear,
    span: tracing::Span,
}

impl DeBertaClassificationHead {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let classifier = Linear::new(
            vb.pp("classifier")
                .get((n_classes, config.hidden_size), "weight")?,
            Some(vb.pp("classifier").get(n_classes, "bias")?),
            None,
        );

        Ok(Self {
            dropout: config
                .classifier_dropout
                .unwrap_or(config.hidden_dropout_prob),
            classifier,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.classifier.forward(hidden_states)
    }
}

pub struct DeBertaModel {
    embeddings: DeBertaEmbeddings,
    encoder: DeBertaEncoder,
    pooler: Option<DeBertaPooler>,
    pool: Pool,
    classifier: Option<DeBertaClassificationHead>,

    num_attention_heads: usize,
    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl DeBertaModel {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig, model_type: ModelType) -> Result<Self> {
        let (pool, classifier) = match model_type {
            ModelType::Classifier => {
                let classifier = DeBertaClassificationHead::load(vb.clone(), config)?;
                (Pool::Cls, Some(classifier))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("DeBERTa does not support Splade pooling");
                }
                (pool, None)
            }
        };

        let embeddings = DeBertaEmbeddings::load(vb.pp("deberta.embeddings"), config)?;
        let encoder = DeBertaEncoder::load(vb.pp("deberta.encoder"), config)?;

        // Pooler might be at root level (not under deberta)
        let pooler = if classifier.is_none() {
            match DeBertaPooler::load(vb.pp("pooler"), config) {
                Ok(pooler) => Some(pooler),
                Err(_) => {
                    // Try under deberta
                    match DeBertaPooler::load(vb.pp("deberta.pooler"), config) {
                        Ok(pooler) => Some(pooler),
                        Err(_) => None,
                    }
                }
            }
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pool,
            classifier,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    // Rest of the implementation remains the same...
    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let (input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask) =
            self.prepare_batch(&batch)?;

        let shape = (batch_size, max_length);
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;

        let encoder_output = self
            .encoder
            .forward(&embedding_output, attention_bias.as_ref())?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            self.pool_embeddings(
                encoder_output.clone(),
                &batch,
                attention_mask.as_ref(),
                input_lengths,
                has_raw_requests,
            )?
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            self.get_raw_embeddings(
                encoder_output,
                &batch,
                batch_size,
                max_length,
                has_pooling_requests,
            )?
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }

    fn prepare_batch(
        &self,
        batch: &Batch,
    ) -> Result<(
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Tensor,
        Option<Tensor>,
        Option<Tensor>,
    )> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        if batch_size > 1 {
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut type_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
            let mut attention_mask = Vec::with_capacity(elems);
            let mut attention_bias = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);
            let mut masking = false;

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = (end - start) as u32;
                input_lengths.push(seq_length as f32);

                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    type_ids.push(batch.token_type_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                    attention_mask.push(1.0_f32);
                    attention_bias.push(0.0);
                }

                let padding = batch.max_length - seq_length;
                if padding > 0 {
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(0);
                        type_ids.push(0);
                        position_ids.push(0);
                        attention_mask.push(0.0_f32);
                        attention_bias.push(f32::NEG_INFINITY);
                    }
                }
            }

            let input_lengths = Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?
                .to_dtype(self.dtype)?;

            let (attention_bias, attention_mask) = if masking {
                let attention_mask = if self.pool == Pool::Mean {
                    Some(
                        Tensor::from_vec(
                            attention_mask,
                            (batch_size, max_length, 1),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?,
                    )
                } else {
                    None
                };

                let attention_bias =
                    Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                        .to_dtype(self.dtype)?;

                let attention_bias = attention_bias
                    .broadcast_as((batch_size, self.num_attention_heads, max_length, max_length))?
                    .contiguous()?;

                (Some(attention_bias), attention_mask)
            } else {
                (None, None)
            };

            Ok((
                input_ids,
                type_ids,
                position_ids,
                input_lengths,
                attention_bias,
                attention_mask,
            ))
        } else {
            let input_lengths =
                Tensor::from_vec(vec![batch.max_length as f32], (1, 1), &self.device)?
                    .to_dtype(self.dtype)?;

            Ok((
                batch.input_ids.clone(),
                batch.token_type_ids.clone(),
                batch.position_ids.clone(),
                input_lengths,
                None,
                None,
            ))
        }
    }

    fn pool_embeddings(
        &self,
        mut outputs: Tensor,
        batch: &Batch,
        attention_mask: Option<&Tensor>,
        mut input_lengths: Tensor,
        has_raw_requests: bool,
    ) -> Result<Option<Tensor>> {
        let pooled_indices_length = batch.pooled_indices.len();

        let pooled_indices = if has_raw_requests {
            let pooled_indices = Tensor::from_vec(
                batch.pooled_indices.clone(),
                pooled_indices_length,
                &self.device,
            )?;
            outputs = outputs.index_select(&pooled_indices, 0)?;
            Some(pooled_indices)
        } else {
            None
        };

        let pooled_embeddings = match self.pool {
            Pool::Cls => {
                if let Some(ref pooler) = self.pooler {
                    pooler.forward(&outputs)?
                } else {
                    outputs.i((.., 0))?
                }
            }
            Pool::LastToken => {
                candle::bail!("LastToken pooling is not supported for DeBERTa")
            }
            Pool::Mean => {
                if let Some(attention_mask) = attention_mask {
                    let attention_mask = if let Some(pooled_indices) = pooled_indices {
                        input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        attention_mask.index_select(&pooled_indices, 0)?
                    } else {
                        attention_mask.clone()
                    };

                    outputs = outputs.broadcast_mul(&attention_mask)?;
                }

                (outputs.sum(1)?.broadcast_div(&input_lengths))?
            }
            Pool::Splade => unreachable!(),
        };

        Ok(Some(pooled_embeddings))
    }

    fn get_raw_embeddings(
        &self,
        outputs: Tensor,
        batch: &Batch,
        batch_size: usize,
        max_length: usize,
        has_pooling_requests: bool,
    ) -> Result<Option<Tensor>> {
        let (b, l, h) = outputs.shape().dims3()?;
        let outputs = outputs.reshape((b * l, h))?;

        if batch_size > 1 && has_pooling_requests {
            let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);

            for i in batch.raw_indices.iter() {
                let start = i * batch.max_length;
                let i = *i as usize;
                let length = batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                for j in start..start + length {
                    final_indices.push(j);
                }
            }

            let final_indices_length = final_indices.len();
            let final_indices =
                Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

            Ok(Some(outputs.index_select(&final_indices, 0)?))
        } else {
            Ok(Some(outputs))
        }
    }
}

impl Model for DeBertaModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for embedding models"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}
