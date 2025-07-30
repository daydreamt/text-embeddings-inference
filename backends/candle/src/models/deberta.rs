use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{bail, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// Helper function to build relative position ids, adapted from Hugging Face's candle implementation.
fn build_relative_position(query_size: usize, key_size: usize, device: &Device) -> Result<Tensor> {
    let q_ids = Tensor::arange(0, query_size as u32, device)?.to_dtype(DType::I64)?;
    let k_ids = Tensor::arange(0, key_size as u32, device)?.to_dtype(DType::I64)?;
    let rel_pos_ids = q_ids.unsqueeze(1)?.broadcast_sub(&k_ids.unsqueeze(0)?)?;
    Ok(rel_pos_ids)
}

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
        self.layer_norm.forward(&embeddings, None)
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
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("query_proj").get(all_head_size, "bias")?),
            None,
        );
        let key_proj = Linear::new(
            vb.pp("key_proj")
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("key_proj").get(all_head_size, "bias")?),
            None,
        );
        let value_proj = Linear::new(
            vb.pp("value_proj")
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("value_proj").get(all_head_size, "bias")?),
            None,
        );

        let share_att_key = config.share_att_key.unwrap_or(false);
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();
        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }

        let (pos_key_proj, pos_query_proj) = if relative_attention && !share_att_key {
            let pos_key_proj = if pos_att_type.iter().any(|t| t == "c2p" || t == "p2p") {
                Some(Linear::new(
                    vb.pp("pos_key_proj")
                        .get((all_head_size, config.hidden_size), "weight")?,
                    Some(vb.pp("pos_key_proj").get(all_head_size, "bias")?),
                    None,
                ))
            } else {
                None
            };
            let pos_query_proj = if pos_att_type.iter().any(|t| t == "p2c" || t == "p2p") {
                Some(Linear::new(
                    vb.pp("pos_query_proj")
                        .get((all_head_size, config.hidden_size), "weight")?,
                    Some(vb.pp("pos_query_proj").get(all_head_size, "bias")?),
                    None,
                ))
            } else {
                None
            };
            (pos_key_proj, pos_query_proj)
        } else {
            (None, None)
        };

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            pos_att_type,
            max_relative_positions,
            position_buckets: config.position_buckets.unwrap_or(-1),
            share_att_key,
            pos_key_proj,
            pos_query_proj,
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
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let query_layer = self.query_proj.forward(hidden_states)?;
        let key_layer = self.key_proj.forward(hidden_states)?;
        let value_layer = self.value_proj.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer, batch_size, seq_len)?;
        let key_layer = self.transpose_for_scores(&key_layer, batch_size, seq_len)?;
        let value_layer = self.transpose_for_scores(&value_layer, batch_size, seq_len)?;

        let mut scale_factor = 1.0;
        if self.pos_att_type.contains(&"c2p".to_string()) {
            scale_factor += 1.0;
        }
        if self.pos_att_type.contains(&"p2c".to_string()) {
            scale_factor += 1.0;
        }
        let scale = (self.attention_head_size as f64 * scale_factor).sqrt();

        let mut attention_scores = query_layer.matmul(&key_layer.t()?)?;

        if let (Some(rel_embeddings), Some(relative_pos)) = (relative_embeddings, relative_pos) {
            let rel_att = self.disentangled_attention_bias(
                &query_layer,
                &key_layer,
                relative_pos,
                rel_embeddings,
            )?;
            attention_scores = attention_scores.add(&rel_att)?;
        }

        attention_scores = (attention_scores / scale)?;

        if let Some(attention_mask) = attention_mask {
            let (b, _, _, k) = attention_mask.dims4()?;
            let attention_mask = attention_mask.reshape((b, k))?.unsqueeze(1)?.repeat((
                self.num_attention_heads,
                1,
                1,
            ))?;
            attention_scores = attention_scores.broadcast_add(&attention_mask)?
        }

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;

        context_layer
            .reshape((
                batch_size,
                self.num_attention_heads,
                seq_len,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.all_head_size))
    }

    fn transpose_for_scores(
        &self,
        x: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        println!("DEBUG transpose_for_scores:");
        println!("  input shape: {:?}", x.shape());
        println!("  batch_size: {}, seq_len: {}", batch_size, seq_len);
        println!(
            "  num_attention_heads: {}, attention_head_size: {}",
            self.num_attention_heads, self.attention_head_size
        );

        let reshaped = x.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        println!("  after first reshape: {:?}", reshaped.shape());

        let transposed = reshaped.transpose(1, 2)?;
        println!("  after transpose: {:?}", transposed.shape());

        let contiguous = transposed.contiguous()?;

        let final_shape = contiguous.reshape((
            batch_size * self.num_attention_heads,
            seq_len,
            self.attention_head_size,
        ))?;
        println!("  final shape: {:?}", final_shape.shape());

        Ok(final_shape)
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: &Tensor,
        rel_embeddings: &Tensor,
    ) -> Result<Tensor> {
        let (total_bs_heads, q_len, _d_head) = query_layer.dims3()?;
        let k_len = key_layer.dim(1)?;
        let n_head = self.num_attention_heads;
        let bs = total_bs_heads / n_head;

        let att_span = if self.position_buckets > 0 {
            self.position_buckets
        } else {
            self.max_relative_positions
        };
        if att_span <= 0 {
            bail!("att_span must be a positive integer, but got {}", att_span);
        }

        let relative_pos = relative_pos
            .unsqueeze(1)?
            .repeat((1, n_head, 1, 1))?
            .reshape((total_bs_heads, q_len, k_len))?;

        let pos_idx = (relative_pos
            .broadcast_add(&Tensor::new(att_span, relative_pos.device())?)?)
        .clamp(0i64, 2 * att_span - 1)?
        .to_dtype(DType::U32)?;

        let rel_embeddings = rel_embeddings.to_dtype(query_layer.dtype())?;

        // Create positional query/key layers
        // When share_att_key is true, we use the main query/key projections
        // When false, we use the separate positional projections
        let (pos_query_layer, pos_key_layer) = if self.share_att_key {
            // Use the main query and key projections
            let pos_query = {
                let p_query = self.query_proj.forward(&rel_embeddings)?;
                let p_query = p_query.unsqueeze(0)?;
                let transposed = self.transpose_for_scores(&p_query, 1, 2 * att_span as usize)?;
                transposed.repeat((bs, 1, 1))?
            };

            let pos_key = {
                let p_key = self.key_proj.forward(&rel_embeddings)?;
                let p_key = p_key.unsqueeze(0)?;
                let transposed = self.transpose_for_scores(&p_key, 1, 2 * att_span as usize)?;
                transposed.repeat((bs, 1, 1))?
            };

            (Some(pos_query), Some(pos_key))
        } else {
            // Use separate positional projections if they exist
            let pos_key = if self.pos_att_type.iter().any(|t| t == "c2p" || t == "p2p") {
                if let Some(ref k_proj) = self.pos_key_proj {
                    let p_key = k_proj.forward(&rel_embeddings)?;
                    let p_key = p_key.unsqueeze(0)?;
                    let transposed = self.transpose_for_scores(&p_key, 1, 2 * att_span as usize)?;
                    Some(transposed.repeat((bs, 1, 1))?)
                } else {
                    None
                }
            } else {
                None
            };

            let pos_query = if self.pos_att_type.iter().any(|t| t == "p2c" || t == "p2p") {
                if let Some(ref q_proj) = self.pos_query_proj {
                    let p_query = q_proj.forward(&rel_embeddings)?;
                    let p_query = p_query.unsqueeze(0)?;
                    let transposed =
                        self.transpose_for_scores(&p_query, 1, 2 * att_span as usize)?;
                    Some(transposed.repeat((bs, 1, 1))?)
                } else {
                    None
                }
            } else {
                None
            };

            (pos_query, pos_key)
        };

        let mut score = Tensor::zeros(
            (total_bs_heads, q_len, k_len),
            query_layer.dtype(),
            query_layer.device(),
        )?;

        if self.pos_att_type.contains(&"c2p".to_string()) {
            let pos_key_layer = pos_key_layer
                .as_ref()
                .ok_or_else(|| candle::Error::Msg("pos_key_layer not found for c2p".to_string()))?;

            let c2p_att = query_layer.matmul(&pos_key_layer.transpose(1, 2)?)?;
            let c2p_att = c2p_att.gather(&pos_idx, 2)?;
            score = score.add(&c2p_att)?;
        }

        if self.pos_att_type.contains(&"p2c".to_string()) {
            let pos_query_layer = pos_query_layer.as_ref().ok_or_else(|| {
                candle::Error::Msg("pos_query_proj not found for p2c".to_string())
            })?;

            // Key insight: The Python code shows that p2c_att is gathered differently
            // We need to transpose the attention scores before gathering
            let p2c_att = pos_query_layer.matmul(&key_layer.transpose(1, 2)?)?;

            // Gather along dimension 1 (the sequence dimension) after getting the scores
            let c2p_pos = pos_idx.clone(); // Use the same indices as c2p
            let p2c_att = p2c_att.gather(&c2p_pos, 1)?;

            score = score.add(&p2c_att)?;
        }

        Ok(score)
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
        self.layer_norm
            .forward(&attention_output, Some(hidden_states))
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
        self.layer_norm
            .forward(&layer_output, Some(&attention_output))
    }
}

struct DeBertaRelativeEmbeddings {
    embeddings: Embedding,
    span: tracing::Span,
}

impl DeBertaRelativeEmbeddings {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let pos_ebd_size = if position_buckets > 0 {
            position_buckets * 2
        } else {
            max_relative_positions * 2
        };
        let embeddings = Embedding::new(
            vb.get((pos_ebd_size as usize, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        Ok(Self {
            embeddings,
            span: tracing::span!(tracing::Level::TRACE, "relative_embeddings"),
        })
    }

    fn get_rel_embedding(&self) -> Result<Tensor> {
        Ok(self.embeddings.embeddings().clone())
    }
}

struct DeBertaEncoder {
    layers: Vec<DeBertaLayer>,
    relative_attention_layer: Option<DeBertaRelativeEmbeddings>,
    layer_norm: Option<LayerNorm>,
    relative_attention: bool,
    span: tracing::Span,
}

impl DeBertaEncoder {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| DeBertaLayer::load(vb.pp(&format!("layer.{}", i)), config))
            .collect::<Result<Vec<_>>>()?;
        let relative_attention_layer = if config.relative_attention {
            Some(DeBertaRelativeEmbeddings::load(
                vb.pp("rel_embeddings"),
                config,
            )?)
        } else {
            None
        };
        let norm_rel_ebd = config.norm_rel_ebd.as_deref().unwrap_or("none");
        let layer_norm = if norm_rel_ebd.contains("layer_norm") {
            LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )
            .ok()
        } else {
            None
        };
        Ok(Self {
            layers,
            relative_attention_layer,
            layer_norm,
            relative_attention: config.relative_attention,
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut current_hidden_states = hidden_states.clone();
        let (q_len, k_len) = (hidden_states.dim(1)?, hidden_states.dim(1)?);

        let relative_pos = if self.relative_attention {
            Some(build_relative_position(
                q_len,
                k_len,
                hidden_states.device(),
            )?)
        } else {
            None
        };

        let relative_embeddings = if let Some(rel_attn_layer) = &self.relative_attention_layer {
            let mut embeddings = rel_attn_layer.get_rel_embedding()?;
            if let Some(ln) = &self.layer_norm {
                embeddings = ln.forward(&embeddings, None)?;
            }
            Some(embeddings)
        } else {
            None
        };

        for layer in &self.layers {
            current_hidden_states = layer.forward(
                &current_hidden_states,
                attention_mask,
                relative_embeddings.as_ref(),
                relative_pos.as_ref(),
            )?;
        }
        Ok(current_hidden_states)
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
                .get((config.hidden_size, config.hidden_size), "weight")?,
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
        let first_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&first_token)?;
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
                .get((pooler_hidden_size, pooler_hidden_size), "weight")?,
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
        let context_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&context_token)?;
        match self.activation {
            HiddenAct::Gelu => pooled.gelu(),
            HiddenAct::Relu => pooled.relu(),
            HiddenAct::Silu => pooled.silu(),
            HiddenAct::Swiglu => {
                candle::bail!("Swiglu activation is not supported for DeBERTa context pooler")
            }
        }
    }
}

pub struct DeBertaClassificationHead {
    classifier: Linear,
    span: tracing::Span,
}

impl DeBertaClassificationHead {
    pub fn load(vb: VarBuilder, config: &DeBertaConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };
        let classifier = Linear::new(
            vb.pp("classifier")
                .get((n_classes, config.hidden_size), "weight")?,
            Some(vb.pp("classifier").get(n_classes, "bias")?),
            None,
        );
        Ok(Self {
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
        let pooler = if classifier.is_none() {
            DeBertaPooler::load(vb.pp("pooler"), config)
                .or_else(|_| DeBertaPooler::load(vb.pp("deberta.pooler"), config))
                .ok()
        } else {
            None
        };
        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pool,
            classifier,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();
        let batch_len = batch.len();
        let max_length = batch.max_length as usize;
        let mut input_lengths = Vec::with_capacity(batch_len);
        for i in 0..batch_len {
            let length = batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];
            input_lengths.push(length as f32);
        }
        let (_, _, _, _, attention_bias, attention_mask) =
            self.prepare_batch(&batch, &input_lengths)?;
        let shape = (batch_len, max_length);
        let input_ids = Tensor::from_vec(batch.input_ids.clone(), shape, &self.device)?;
        let type_ids = Tensor::from_vec(batch.token_type_ids.clone(), shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids.clone(), shape, &self.device)?;
        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;
        let encoder_output = self
            .encoder
            .forward(&embedding_output, attention_bias.as_ref())?;
        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();
        let input_lengths_tensor =
            Tensor::from_vec(input_lengths.clone(), (batch_len, 1), &self.device)?
                .to_dtype(self.dtype)?;
        let pooled_embeddings = if has_pooling_requests {
            self.pool_embeddings(
                encoder_output.clone(),
                &batch,
                attention_mask.as_ref(),
                input_lengths_tensor,
                has_raw_requests,
            )?
        } else {
            None
        };
        let raw_embeddings = if has_raw_requests {
            self.get_raw_embeddings(
                encoder_output,
                &batch,
                &input_lengths,
                batch_len,
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
        input_lengths: &[f32],
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
            let mut attention_mask = Vec::with_capacity(batch_size * max_length);
            let mut attention_bias = Vec::with_capacity(batch_size * max_length);
            let mut masking = false;
            for &length in input_lengths.iter() {
                let seq_length = length as usize;
                for _ in 0..seq_length {
                    attention_mask.push(1.0_f32);
                    attention_bias.push(0.0_f32);
                }
                let padding = max_length - seq_length;
                if padding > 0 {
                    masking = true;
                    for _ in 0..padding {
                        attention_mask.push(0.0_f32);
                        attention_bias.push(f32::NEG_INFINITY);
                    }
                }
            }
            let input_lengths_tensor =
                Tensor::from_vec(input_lengths.to_vec(), (batch_size, 1), &self.device)?
                    .to_dtype(self.dtype)?;
            let (attention_bias, attention_mask_tensor) = if masking {
                let attention_mask_tensor = if self.pool == Pool::Mean {
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
                let attention_bias_tensor =
                    Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                        .to_dtype(self.dtype)?;
                (Some(attention_bias_tensor), attention_mask_tensor)
            } else {
                (None, None)
            };
            Ok((
                batch.input_ids.clone(),
                batch.token_type_ids.clone(),
                batch.position_ids.clone(),
                input_lengths_tensor,
                attention_bias,
                attention_mask_tensor,
            ))
        } else {
            let input_lengths_tensor =
                Tensor::from_vec(input_lengths.to_vec(), (1, 1), &self.device)?
                    .to_dtype(self.dtype)?;
            Ok((
                batch.input_ids.clone(),
                batch.token_type_ids.clone(),
                batch.position_ids.clone(),
                input_lengths_tensor,
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
        let pooled_indices = if has_raw_requests && pooled_indices_length > 0 {
            let pooled_indices_tensor = Tensor::from_vec(
                batch.pooled_indices.clone(),
                pooled_indices_length,
                &self.device,
            )?;
            outputs = outputs.index_select(&pooled_indices_tensor, 0)?;
            Some(pooled_indices_tensor)
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
                bail!("LastToken pooling is not supported for DeBERTa")
            }
            Pool::Mean => {
                let mut outputs_for_pooling = outputs;
                if let Some(attention_mask) = attention_mask {
                    let attention_mask = if let Some(pooled_indices) = pooled_indices {
                        input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        attention_mask.index_select(&pooled_indices, 0)?
                    } else {
                        attention_mask.clone()
                    };
                    outputs_for_pooling = outputs_for_pooling.broadcast_mul(&attention_mask)?;
                }
                (outputs_for_pooling.sum(1)?.broadcast_div(&input_lengths))?
            }
            Pool::Splade => unreachable!(),
        };
        Ok(Some(pooled_embeddings))
    }

    fn get_raw_embeddings(
        &self,
        outputs: Tensor,
        batch: &Batch,
        input_lengths: &[f32],
        batch_size: usize,
        max_length: usize,
        has_pooling_requests: bool,
    ) -> Result<Option<Tensor>> {
        let (_b, _l, h) = outputs.shape().dims3()?;
        let outputs = outputs.reshape((batch_size * max_length, h))?;
        if batch_size > 1 && has_pooling_requests {
            let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);
            for &i_u32 in batch.raw_indices.iter() {
                let i = i_u32 as usize;
                let start = (i * max_length) as u32;
                let length = input_lengths[i];
                for j in start..(start + length as u32) {
                    final_indices.push(j);
                }
            }
            let final_indices_length = final_indices.len();
            let final_indices_tensor =
                Tensor::from_vec(final_indices, final_indices_length, &self.device)?;
            Ok(Some(outputs.index_select(&final_indices_tensor, 0)?))
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
