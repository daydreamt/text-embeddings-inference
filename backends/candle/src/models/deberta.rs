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
    pub pad_token_id: usize,
    pub position_embedding_type: Option<String>,
    pub use_cache: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub id2label: Option<HashMap<String, String>>,

    // DeBERTa specific configurations
    pub relative_attention: bool,
    pub pos_att_type: Option<Vec<String>>,
    pub max_relative_positions: Option<i64>,
    pub position_buckets: Option<i64>,
    pub share_att_key: Option<bool>,
    pub norm_rel_ebd: Option<String>,
}

#[derive(Debug)]
pub struct DeBertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DeBertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        // DeBERTa v2 with relative attention typically doesn't use position embeddings
        let position_embeddings = if !config.relative_attention {
            match vb.pp("position_embeddings").get(
                (config.max_position_embeddings, config.hidden_size),
                "weight",
            ) {
                Ok(w) => Some(Embedding::new(w, config.hidden_size)),
                Err(_) => None,
            }
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

        if let Some(ref token_type_embeddings) = self.token_type_embeddings {
            embeddings = embeddings.add(&token_type_embeddings.forward(token_type_ids)?)?;
        }

        if let Some(ref position_embeddings) = self.position_embeddings {
            embeddings = embeddings.add(&position_embeddings.forward(position_ids)?)?;
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

    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,

    pos_dropout: f64,

    span: tracing::Span,
}

impl DeBertaDisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query_proj = Linear::new(
            vb.pp("query_proj")
                .get((config.hidden_size, all_head_size), "weight")?,
            Some(vb.pp("query_proj").get(all_head_size, "bias")?),
            None,
        );

        let key_proj = Linear::new(
            vb.pp("key_proj")
                .get((config.hidden_size, all_head_size), "weight")?,
            Some(vb.pp("key_proj").get(all_head_size, "bias")?),
            None,
        );

        let value_proj = Linear::new(
            vb.pp("value_proj")
                .get((config.hidden_size, all_head_size), "weight")?,
            Some(vb.pp("value_proj").get(all_head_size, "bias")?),
            None,
        );

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            pos_att_type: config.pos_att_type.clone().unwrap_or_default(),
            max_relative_positions: config.max_relative_positions.unwrap_or(256),
            position_buckets: config.position_buckets.unwrap_or(256),
            share_att_key: config.share_att_key.unwrap_or(true),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            pos_dropout: config.hidden_dropout_prob,
            span: tracing::span!(tracing::Level::TRACE, "disentangled_self_attention"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        _relative_embeddings: &Tensor,
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

        // TODO: Implement disentangled attention mechanism
        // This is a simplified version - actual implementation would compute:
        // 1. Content-to-content attention
        // 2. Content-to-position attention
        // 3. Position-to-content attention

        // For now, just do standard attention as placeholder
        let attention_scores = query_layer.matmul(&key_layer.transpose(D::Minus2, D::Minus1)?)?;
        let attention_scores =
            (attention_scores * (1.0 / (self.attention_head_size as f64).sqrt()))?;

        let attention_scores = if let Some(attention_mask) = attention_mask {
            attention_scores.broadcast_add(attention_mask)?
        } else {
            attention_scores
        };

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Transpose back and reshape
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
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
        x.transpose(1, 2)
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
                .get((config.hidden_size, config.hidden_size), "weight")?,
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
        relative_embeddings: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let self_output =
            self.self_attention
                .forward(hidden_states, attention_mask, relative_embeddings)?;
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
        relative_embeddings: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attention_output =
            self.attention
                .forward(hidden_states, attention_mask, relative_embeddings)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self
            .layer_norm
            .forward(&layer_output, Some(&attention_output))?;

        Ok(layer_output)
    }
}

struct DeBertaEncoder {
    layers: Vec<DeBertaLayer>,
    relative_attention: Option<DeBertaRelativeEmbeddings>,
    layer_norm: Option<LayerNorm>, // Add this field
    span: tracing::Span,
}

impl DeBertaEncoder {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DeBertaLayer::load(vb.pp(&format!("layer.{}", i)), config)?);
        }

        // rel_embeddings is directly under encoder, not nested deeper
        let relative_attention = if config.relative_attention {
            Some(DeBertaRelativeEmbeddings::load(
                vb.pp("rel_embeddings"),
                config,
            )?)
        } else {
            None
        };

        let layer_norm = match LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        ) {
            Ok(ln) => Some(ln),
            Err(_) => None,
        };

        Ok(Self {
            layers,
            relative_attention,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        let relative_embeddings = if let Some(ref relative_attention) = self.relative_attention {
            relative_attention.get_rel_embedding()?
        } else {
            Tensor::zeros((1, 1, 1), hidden_states.dtype(), hidden_states.device())?
        };

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, &relative_embeddings)?;
        }

        if let Some(ref layer_norm) = self.layer_norm {
            hidden_states = layer_norm.forward(&hidden_states, None)?;
        }

        Ok(hidden_states)
    }
}

struct DeBertaPooler {
    dense: Linear,
    activation: candle_nn::Activation,
    span: tracing::Span,
}

impl DeBertaPooler {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let dense = Linear::new(
            vb.pp("dense")
                .get((config.hidden_size, config.hidden_size), "weight")?,
            Some(vb.pp("dense").get(config.hidden_size, "bias")?),
            None,
        );

        Ok(Self {
            dense,
            activation: candle_nn::Activation::Sigmoid, // FIXME: //candle_nn::Activation::Tanh,
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // We pool the first token (CLS token) representation
        let first_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&first_token)?;
        self.activation.forward(&pooled)
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
        // When max_relative_positions is -1, use position_buckets
        let embedding_size = match config.max_relative_positions {
            Some(-1) | None => config.position_buckets.unwrap_or(256) as usize,
            Some(n) => n as usize,
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
            position_buckets: config.position_buckets.unwrap_or(256),
            span: tracing::span!(tracing::Level::TRACE, "relative_embeddings"),
        })
    }

    fn get_rel_embedding(&self) -> Result<Tensor> {
        // TODO: Implement proper relative position embedding computation
        // For now, return the embeddings tensor
        Ok(self.embeddings.embeddings().clone())
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

        // Classifier is at root level, not under deberta
        let output_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let output_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(output_weight, Some(output_bias), None);

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

        // The input here should already be pooled embeddings of shape [batch_size, hidden_size]
        // Following BERT's pattern, we need to add and remove a dimension for the linear layer
        let mut hidden_states = hidden_states.unsqueeze(1)?; // [batch_size, 1, hidden_size]

        // Note: DeBERTa v2 typically doesn't have a separate pooler layer in the classifier
        // The pooling happens in the main model's forward method
        let logits = self.classifier.forward(&hidden_states)?; // [batch_size, 1, n_classes]
        let logits = logits.squeeze(1)?; // [batch_size, n_classes]

        Ok(logits)
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
                Err(_) => None,
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

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        // Prepare inputs similar to BERT
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
                // Use the forward method which handles pooling
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                // pooled_embeddings shape: [batch_size, hidden_size]
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}
