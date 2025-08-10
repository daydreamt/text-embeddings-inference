use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{bail, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DebertaV2Config {
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
    // DebertaV2-v2 ConvLayer, which is not supported.
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub conv_groups: Option<usize>,
    #[serde(default)]
    pub conv_act: Option<String>,
    #[serde(default)]
    pub embedding_size: Option<usize>,
}

#[derive(Debug)]
pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    embed_proj: Option<Linear>,
    embedding_size: usize,
    hidden_size: usize,
    layer_norm: LayerNorm,
    position_biased_input: bool,
    span: tracing::Span,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let position_biased_input = config.position_biased_input.unwrap_or(true);
        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);
        let hidden_size = config.hidden_size;

        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get((config.vocab_size, embedding_size), "weight")?,
            embedding_size,
        );

        let position_embeddings = if position_biased_input {
            Some(Embedding::new(
                vb.pp("position_embeddings")
                    .get((config.max_position_embeddings, embedding_size), "weight")?,
                embedding_size,
            ))
        } else {
            None
        };

        // token_type_embeddings are projected from hidden_size
        let token_type_embeddings = if config.type_vocab_size > 0 {
            match vb
                .pp("token_type_embeddings")
                .get((config.type_vocab_size, hidden_size), "weight")
            {
                Ok(w) => Some(Embedding::new(w, hidden_size)),
                Err(_) => candle::bail!(
                    "configuration type_vocab_size > 0 but no token_type_embeddings in the model"
                ),
            }
        } else {
            None
        };

        let embed_proj = if embedding_size != hidden_size {
            let proj = Linear::new(
                vb.pp("embed_proj")
                    .get((embedding_size, hidden_size), "weight")?,
                None,
                None,
            );
            Some(proj)
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
            embed_proj,
            embedding_size,
            hidden_size,
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

        // Token + (optional) position + (optional) proj + (optional) token-type
        let mut embeddings = self.word_embeddings.forward(input_ids)?;

        if let Some(ref position_embeddings) = self.position_embeddings {
            embeddings = embeddings.add(&position_embeddings.forward(position_ids)?)?;
        }
        if let Some(proj) = &self.embed_proj {
            embeddings = proj.forward(&embeddings)?;
        }
        if let Some(ref token_type_embeddings) = self.token_type_embeddings {
            embeddings = embeddings.add(&token_type_embeddings.forward(token_type_ids)?)?;
        }

        // LN only (we do not zero pad here; masking is handled in attention/pooling).
        self.layer_norm.forward(&embeddings, None)
    }
}

struct DebertaV2DisentangledSelfAttention {
    qkv_linear: Linear, // TODO: also pos_key_proj, pos_query_proj?
    // used only when share_att_key == true (for rel-embeddings path)
    content_q_proj_for_rel: Linear,
    content_k_proj_for_rel: Linear,
    use_c2p_att: bool,
    use_p2c_att: bool,
    max_relative_positions: i64,
    position_buckets: i64,
    share_att_key: bool,
    pos_key_proj: Option<Linear>,
    pos_query_proj: Option<Linear>,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    span: tracing::Span,
    device: Device,
    cached_pos_key_proj: Mutex<Option<Tensor>>, // (2*att_span, all_head_size)
    cached_pos_query_proj: Mutex<Option<Tensor>>, // (2*att_span, all_head_size)
    cached_att_span_i64: Tensor,                // scalar [att_span] as i64
}

impl DebertaV2DisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden = config.hidden_size;

        // Fused QKV for the content path.
        let qw = vb.pp("query_proj").get((all_head_size, hidden), "weight")?;
        let qb = vb.pp("query_proj").get(all_head_size, "bias")?;
        let kw = vb.pp("key_proj").get((all_head_size, hidden), "weight")?;
        let kb = vb.pp("key_proj").get(all_head_size, "bias")?;
        let vw = vb.pp("value_proj").get((all_head_size, hidden), "weight")?;
        let vbias = vb.pp("value_proj").get(all_head_size, "bias")?;

        let qkv_w = Tensor::cat(&[&qw, &kw, &vw], 0)?;
        let qkv_b = Tensor::cat(&[&qb, &kb, &vbias], 0)?;
        let qkv_linear = Linear::new(qkv_w, Some(qkv_b), None);

        // Keep slim Q/K linears for the share_att_key branch of rel-emb.
        let content_q_proj_for_rel = Linear::new(qw.clone(), Some(qb.clone()), None);
        let content_k_proj_for_rel = Linear::new(kw.clone(), Some(kb.clone()), None);

        let share_att_key = config.share_att_key.unwrap_or(false);
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();
        let use_c2p_att = pos_att_type.iter().any(|t| t == "c2p");
        let use_p2c_att = pos_att_type.iter().any(|t| t == "p2c");

        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }

        let (pos_key_proj, pos_query_proj) = if relative_attention && !share_att_key {
            let pos_key_proj = if use_c2p_att {
                Some(Linear::new(
                    vb.pp("pos_key_proj")
                        .get((all_head_size, config.hidden_size), "weight")?,
                    Some(vb.pp("pos_key_proj").get(all_head_size, "bias")?),
                    None,
                ))
            } else {
                None
            };
            let pos_query_proj = if use_p2c_att {
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

        let device = vb.device().clone();

        let att_span = if config.position_buckets.unwrap_or(-1) > 0 {
            config.position_buckets.unwrap()
        } else {
            config
                .max_relative_positions
                .unwrap_or(config.max_position_embeddings as i64)
        };

        Ok(Self {
            qkv_linear,
            content_q_proj_for_rel,
            content_k_proj_for_rel,
            use_c2p_att,
            use_p2c_att,
            max_relative_positions,
            position_buckets: config.position_buckets.unwrap_or(-1),
            share_att_key,
            pos_key_proj,
            pos_query_proj,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            span: tracing::span!(tracing::Level::TRACE, "disentangled_self_attention"),
            device: device.clone(),
            cached_pos_key_proj: Mutex::new(None),
            cached_pos_query_proj: Mutex::new(None),
            cached_att_span_i64: Tensor::new(&[att_span as i64], &device)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attn_bias_bhl: Option<&Tensor>, // (BH,L,L) or None
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, _) = hidden_states.dims3()?;

        // Fused QKV -> split to (B*H, L, d)
        let qkv = self
            .qkv_linear
            .forward(hidden_states)?
            .reshape((b, l, self.num_attention_heads * 3, self.attention_head_size))?
            .transpose(1, 2)?; // (B, 3H, L, d)
        let chunks = qkv.chunk(3, 1)?;
        let q = chunks[0].reshape((b * self.num_attention_heads, l, self.attention_head_size))?;
        let k = chunks[1].reshape((b * self.num_attention_heads, l, self.attention_head_size))?;
        let v = chunks[2].reshape((b * self.num_attention_heads, l, self.attention_head_size))?;

        // Scale like the fast reference: divide K by sqrt(d * scale_factor).
        let mut scale_factor: usize = 1;
        if self.use_c2p_att {
            scale_factor += 1;
        }
        if self.use_p2c_att {
            scale_factor += 1;
        }

        let inv_scale: f64 = ((self.attention_head_size * scale_factor) as f64)
            .sqrt()
            .recip();

        // Scale Q once, then Q @ K^T (drop extra clone).
        let q_scaled = (&q * inv_scale)?; // f32/f16-safe scalar mul
        let mut attn = q_scaled.matmul(&k.transpose(1, 2)?)?; // (B*H, L, L)

        // Relative bias (uses the same scalar inv_scale)
        if let (Some(rel_e), Some(rel_p)) = (relative_embeddings, relative_pos) {
            // Use UNscaled q for the disentangled paths to avoid double scaling.
            attn = attn.add(&self.disentangled_attention_bias(&q, &k, rel_p, rel_e, inv_scale)?)?;
        }

        if let Some(bias_bhl) = attn_bias_bhl {
            // Mask already provided as (BH,L,L); avoid per-layer reshape.
            attn = attn.add(bias_bhl)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&attn)?;

        // Context -> (B,L,all_head)
        probs
            .matmul(&v)?
            .reshape((b, self.num_attention_heads, l, self.attention_head_size))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.all_head_size))
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,    // (B*H, Lq, d)
        key_layer: &Tensor,      // (B*H, Lk, d)
        relative_pos: &Tensor,   // 2D/3D/4D; normalized below
        rel_embeddings: &Tensor, // (2*att_span, hidden)
        inv_scale: f64,          // 1 / sqrt(d * scale_factor)
    ) -> Result<Tensor> {
        let (total_bs_heads, _q_len, d_head) = query_layer.dims3()?;
        let n_head = self.num_attention_heads;
        let bs = total_bs_heads / n_head;

        let att_span = if self.position_buckets > 0 {
            self.position_buckets
        } else {
            self.max_relative_positions
        };

        // Normalize relative_pos to 4D: (1,1,Lq,Lk)
        let relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            4 => relative_pos.clone(),
            other => bail!(
                "Relative position ids must be of dim 2, 3, or 4. Got {}",
                other
            ),
        };

        // Match dtypes once.
        let rel_embeddings = rel_embeddings.to_dtype(query_layer.dtype())?;

        // Helper: cache projector outputs on rel_embeddings.
        let get_cached_proj = |cache: &Mutex<Option<Tensor>>,
                               direct: &dyn Fn(&Tensor) -> Result<Tensor>|
         -> Result<Tensor> {
            let mut guard = cache.lock().unwrap();
            if let Some(t) = guard.as_ref() {
                return Ok(t.clone());
            }
            let t = direct(&rel_embeddings)?;
            *guard = Some(t.clone());
            Ok(t)
        };

        // Reshape projector output to (H, 2*span, d)
        let reshape_base = |base: Tensor| -> Result<Tensor> {
            base.reshape((2 * att_span as usize, n_head, d_head))?
                .transpose(0, 1)? // (H, 2*span, d)
                .contiguous()
        };

        let mut score: Option<Tensor> = None;

        // ---- c2p ----
        if self.use_c2p_att {
            let base = if self.share_att_key {
                get_cached_proj(&self.cached_pos_key_proj, &|e| {
                    self.content_k_proj_for_rel.forward(e)
                })?
            } else {
                let proj = self.pos_key_proj.as_ref().expect("pos_key_proj not found");
                get_cached_proj(&self.cached_pos_key_proj, &|e| proj.forward(e))?
            };

            let h_2span_d = reshape_base(base)?; // (H,2*span,d)
                                                 // Materialize with repeat for better matmul perf.
            let pos_key_layer = if bs > 1 {
                h_2span_d.repeat(bs)?.contiguous()?.reshape((
                    total_bs_heads,
                    2 * att_span as usize,
                    d_head,
                ))?
            } else {
                h_2span_d.reshape((total_bs_heads, 2 * att_span as usize, d_head))?
            };

            let c2p_att = query_layer.matmul(&pos_key_layer.transpose(1, 2)?)?;

            // integer-only indices
            let c2p_pos = relative_pos
                .broadcast_add(&self.cached_att_span_i64)?
                .clamp(0i64, (2 * att_span - 1) as i64)?
                .squeeze(0)?
                .squeeze(0)?; // (Lq,Lk)

            let c2p_indices = c2p_pos
                .unsqueeze(0)?
                .broadcast_as((total_bs_heads, c2p_pos.dim(0)?, c2p_pos.dim(1)?))?
                .to_dtype(DType::U32)?;

            let c2p_att_gathered = c2p_att.gather(&c2p_indices, D::Minus1)?;
            let c2p_scaled = (c2p_att_gathered * inv_scale)?;
            score = Some(match score {
                Some(s) => s.add(&c2p_scaled)?,
                None => c2p_scaled,
            });
        }

        // ---- p2c ----
        if self.use_p2c_att {
            let base = if self.share_att_key {
                get_cached_proj(&self.cached_pos_query_proj, &|e| {
                    self.content_q_proj_for_rel.forward(e)
                })?
            } else {
                let proj = self
                    .pos_query_proj
                    .as_ref()
                    .expect("pos_query_proj not found");
                get_cached_proj(&self.cached_pos_query_proj, &|e| proj.forward(e))?
            };

            let h_2span_d = reshape_base(base)?; // (H,2*span,d)
                                                 // Materialize with repeat for better matmul perf.
            let pos_query_layer = if bs > 1 {
                h_2span_d.repeat(bs)?.contiguous()?.reshape((
                    total_bs_heads,
                    2 * att_span as usize,
                    d_head,
                ))?
            } else {
                h_2span_d.reshape((total_bs_heads, 2 * att_span as usize, d_head))?
            };

            let p2c_att = key_layer.matmul(&pos_query_layer.transpose(1, 2)?)?;

            // integer-only: att_span - rel_pos
            let p2c_pos = self
                .cached_att_span_i64
                .broadcast_sub(&relative_pos)?
                .clamp(0i64, (2 * att_span - 1) as i64)?
                .squeeze(0)?
                .squeeze(0)?; // (Lq,Lk)

            let p2c_indices = p2c_pos
                .unsqueeze(0)?
                .broadcast_as((total_bs_heads, p2c_pos.dim(0)?, p2c_pos.dim(1)?))?
                .to_dtype(DType::U32)?;

            let p2c_att_gathered = p2c_att.gather(&p2c_indices, D::Minus1)?.transpose(1, 2)?;
            let p2c_scaled = (p2c_att_gathered * inv_scale)?;
            score = Some(match score {
                Some(s) => s.add(&p2c_scaled)?,
                None => p2c_scaled,
            });
        }

        Ok(match score {
            Some(s) => s,
            None => query_layer.zeros_like()?,
        })
    }
}

struct DebertaV2Attention {
    self_attention: DebertaV2DisentangledSelfAttention,
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DebertaV2Attention {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let self_attention = DebertaV2DisentangledSelfAttention::load(vb.pp("self"), config)?;
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
        attn_bias_bhl: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_out = self.self_attention.forward(
            hidden_states,
            attn_bias_bhl,
            relative_embeddings,
            relative_pos,
        )?;
        let y = self.dense.forward(&self_out)?;
        self.layer_norm.forward(&y, Some(hidden_states))
    }
}

struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DebertaV2Layer {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.pp("attention"), config)?;
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
        attn_bias_bhl: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = self.attention.forward(
            hidden_states,
            attn_bias_bhl,
            relative_embeddings,
            relative_pos,
        )?;
        let h = self.intermediate.forward(&x)?;
        self.layer_norm.forward(&self.output.forward(&h)?, Some(&x))
    }
}

struct DebertaV2RelativeEmbeddings {
    embeddings: Embedding,
    span: tracing::Span,
}

impl DebertaV2RelativeEmbeddings {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }
        let position_buckets = config.position_buckets.unwrap_or(-1);
        // Buckets take precedence over max_position_embeddings.
        let pos_ebd_size = if position_buckets > 0 {
            position_buckets
        } else {
            max_relative_positions
        };
        // The embedding table is 2x the size (for negative and positive positions)
        let embeddings = Embedding::new(
            vb.get((pos_ebd_size as usize * 2, config.hidden_size), "weight")?,
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

struct DebertaV2Encoder {
    layers: Vec<DebertaV2Layer>,
    relative_attention_layer: Option<DebertaV2RelativeEmbeddings>,
    layer_norm: Option<LayerNorm>,
    relative_attention: bool,
    position_buckets: i64,
    max_relative_positions: i64,
    rel_pos_full: Option<Tensor>,          // (1, Lmax, Lmax), I64
    rel_embeddings_cached: Option<Tensor>, // (2*att_span, hidden), dtype of params
    span: tracing::Span,
}

impl DebertaV2Encoder {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| DebertaV2Layer::load(vb.pp(&format!("layer.{}", i)), config))
            .collect::<Result<Vec<_>>>()?;
        let relative_attention_layer = if config.relative_attention {
            Some(DebertaV2RelativeEmbeddings::load(
                vb.pp("rel_embeddings"),
                config,
            )?)
        } else {
            None
        };
        let norm_rel_ebd = config.norm_rel_ebd.as_deref().unwrap_or("none");
        let layer_norm = if norm_rel_ebd.contains("layer_norm") {
            Some(LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?)
        } else {
            None
        };
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }

        let rel_pos_full = if config.relative_attention {
            let lmax = config.max_position_embeddings; // sequence length cap
            Some(build_relative_position(
                lmax,
                lmax,
                vb.device(),
                Some(position_buckets as isize),
                Some(max_relative_positions as isize),
            )?)
        } else {
            None
        };

        // Cache normalized rel-embeddings once
        let rel_embeddings_cached = if let Some(rel_attn_layer) = &relative_attention_layer {
            let mut e = rel_attn_layer.get_rel_embedding()?;
            if let Some(ln) = &layer_norm {
                e = ln.forward(&e, None)?;
            }
            Some(e)
        } else {
            None
        };

        Ok(Self {
            layers,
            relative_attention_layer,
            layer_norm,
            relative_attention: config.relative_attention,
            position_buckets,
            max_relative_positions,
            rel_pos_full,
            rel_embeddings_cached,
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    #[inline]
    fn get_rel_pos(&self, hidden_states: &Tensor) -> Result<Option<Tensor>> {
        if !self.relative_attention {
            return Ok(None);
        }
        let l = hidden_states.dim(1)?;
        let rp = self
            .rel_pos_full
            .as_ref()
            .expect("rel_pos_full missing")
            .narrow(1, 0, l)?
            .narrow(2, 0, l)?;
        Ok(Some(rp))
    }

    #[inline]
    fn get_rel_embedding(&self) -> Result<Option<Tensor>> {
        Ok(self.rel_embeddings_cached.clone())
    }

    fn forward(&self, hidden_states: &Tensor, attn_mask_b11l: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let relative_pos = self.get_rel_pos(hidden_states)?;
        let relative_embeddings = self.get_rel_embedding()?;

        let (b, l, _) = hidden_states.dims3()?;
        let h = self.layers[0].attention.self_attention.num_attention_heads;
        // Build (BH,L,L) once; pass through layers without reshaping again.
        let attn_bias_bhl: Option<Tensor> = match attn_mask_b11l {
            Some(m) => {
                // (B,1,1,L) -> (B,H,L,L) -> materialize -> (B*H,L,L)
                let bias = m
                    .broadcast_as((b, h, l, l))?
                    .contiguous()?
                    .reshape((b * h, l, l))?;
                Some(bias)
            }
            None => None,
        };

        let mut current_ref: &Tensor = hidden_states;
        let mut current_owned: Option<Tensor> = None;

        for layer in &self.layers {
            let next = layer.forward(
                current_ref,
                attn_bias_bhl.as_ref(),
                relative_embeddings.as_ref(),
                relative_pos.as_ref(),
            )?;
            current_owned = Some(next);
            current_ref = current_owned.as_ref().unwrap();
        }

        match current_owned {
            Some(t) => Ok(t),
            None => bail!("DebertaV2Encoder has no layers"),
        }
    }
}

pub struct DebertaV2Pooler {
    dense: Linear,
    activation: HiddenAct,
    classifier: Option<Linear>,
    span: tracing::Span,
}

impl DebertaV2Pooler {
    fn load_base(
        vb: VarBuilder,
        config: &DebertaV2Config,
        pooler_hidden_size: usize,
    ) -> Result<(Linear, HiddenAct)> {
        let dense = Linear::new(
            vb.pp("dense")
                .get((pooler_hidden_size, config.hidden_size), "weight")?,
            Some(vb.pp("dense").get(pooler_hidden_size, "bias")?),
            None,
        );

        let activation = match &config.pooler_hidden_act {
            Some(act @ (HiddenAct::Gelu | HiddenAct::Relu | HiddenAct::Silu)) => act.clone(),
            Some(other) => bail!("Unsupported activation function: {:?}", other),
            None => bail!("pooler_hidden_act must be specified"),
        };

        Ok((dense, activation))
    }

    pub fn load_embedding(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let pooler_hidden_size = config.pooler_hidden_size.unwrap_or(config.hidden_size);
        let (dense, activation) = Self::load_base(vb, config, pooler_hidden_size)?;

        Ok(Self {
            dense,
            activation,
            classifier: None,
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    pub fn load_classification(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let n_classes = config
            .id2label
            .as_ref()
            .ok_or_else(|| {
                candle::Error::Msg("`id2label` must be set for classifier models".to_string())
            })?
            .len();

        let pooler_hidden_size = config.pooler_hidden_size.ok_or_else(|| {
            candle::Error::Msg("`pooler_hidden_size` must be set for classifier models".to_string())
        })?;

        let pooler_vb = vb.pp("pooler");
        let (dense, activation) = Self::load_base(pooler_vb, config, pooler_hidden_size)?;

        let classifier = Linear::new(
            vb.pp("classifier")
                .get((n_classes, pooler_hidden_size), "weight")?,
            Some(vb.pp("classifier").get(n_classes, "bias")?),
            None,
        );

        Ok(Self {
            dense,
            activation,
            classifier: Some(classifier),
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let first_token = hidden_states.i((.., 0))?;
        let pooled = self.dense.forward(&first_token)?;
        let activated = match self.activation {
            HiddenAct::Gelu => pooled.gelu(),
            HiddenAct::Relu => pooled.relu(),
            HiddenAct::Silu => pooled.silu(),
            _ => unreachable!("Invalid activation should have been caught during loading"),
        }?;
        let result = match &self.classifier {
            Some(classifier) => classifier.forward(&activated),
            None => Ok(activated),
        };
        result
    }

    pub fn is_classifier(&self) -> bool {
        self.classifier.is_some()
    }
}

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    pooler: Option<DebertaV2Pooler>,
    pool: Pool,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    neg_mask_value: Tensor,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config, model_type: ModelType) -> Result<Self> {
        if config.conv_kernel_size.is_some()
            || config.conv_groups.is_some()
            || config.conv_act.is_some()
        {
            candle::bail!(
                "Unsupported DebertaV2 configuration: `conv_kernel_size`, `conv_groups`, or `conv_act` was found - not implemented!"
            )
        }

        let (pool, pooler) = match model_type {
            ModelType::Classifier => {
                let pooler = DebertaV2Pooler::load_classification(vb.clone(), config)?;
                (Pool::Cls, Some(pooler))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("DebertaV2 does not support Splade pooling");
                }
                let pooler = DebertaV2Pooler::load_embedding(vb.pp("pooler"), config)
                    .or_else(|_| DebertaV2Pooler::load_embedding(vb.pp("deberta.pooler"), config))
                    .ok();
                (pool, pooler)
            }
        };

        let embeddings = DebertaV2Embeddings::load(vb.pp("deberta.embeddings"), config)?;
        let encoder = DebertaV2Encoder::load(vb.pp("deberta.encoder"), config)?;

        // Use a large finite negative to avoid NaNs when an attention row is fully masked.
        let neg_val = match vb.dtype() {
            DType::F32 => -1.0e9f32,
            _ => -65504.0f32,
        };
        let neg_mask_value =
            Tensor::from_slice(&[neg_val], (1,), vb.device())?.to_dtype(vb.dtype());

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pool,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            neg_mask_value: neg_mask_value?,
        })
    }

    #[inline]
    fn make_additive_key_padding_mask(&self, mask_2d: &Tensor) -> Result<Tensor> {
        // mask_2d: (B,L) in compute dtype, 1.0=real, 0.0=pad
        let key = mask_2d.unsqueeze(1)?.unsqueeze(1)?; // (B,1,1,L)

        // Threshold in the same dtype to avoid fp32 upcasts on MPS/CUDA.
        let thr = Tensor::new(0.5f32, key.device())?
            .to_dtype(key.dtype())?
            .broadcast_as(key.shape())?;

        let inv = key.le(&thr)?.to_dtype(self.dtype)?; // (B,1,1,L) in {0,1}
        inv.broadcast_mul(&self.neg_mask_value) // (B,1,1,L) in {0,-big}
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let elems = batch_size * max_length;

        let mut input_ids = Vec::with_capacity(elems);
        let mut type_ids = Vec::with_capacity(elems);
        let mut position_ids = Vec::with_capacity(elems);
        let mut attention_mask = Vec::with_capacity(elems);
        let mut input_lengths = Vec::with_capacity(batch_size);
        let mut masking = false;

        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            let seq_len = end - start;
            input_lengths.push(seq_len as f32);

            input_ids.extend_from_slice(&batch.input_ids[start..end]);
            type_ids.extend_from_slice(&batch.token_type_ids[start..end]);
            position_ids.extend_from_slice(&batch.position_ids[start..end]);
            attention_mask.extend(std::iter::repeat(1.0f32).take(seq_len));

            let pad = max_length - seq_len;
            if pad > 0 {
                masking = true;
                input_ids.extend(std::iter::repeat(0).take(pad));
                type_ids.extend(std::iter::repeat(0).take(pad));
                position_ids.extend(std::iter::repeat(0).take(pad));
                attention_mask.extend(std::iter::repeat(0.0f32).take(pad));
            }
        }

        let shape = (batch_size, max_length);
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let attention_mask_2d: Option<Tensor> = if masking {
            Some(
                Tensor::from_vec(attention_mask, (batch_size, max_length), &self.device)?
                    .to_dtype(self.dtype)?,
            )
        } else {
            None
        };

        // Embeddings (no mask multiply here)
        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;

        // Build key-only additive mask (B,1,1,L) once
        let attention_add_mask: Option<Tensor> = match &attention_mask_2d {
            Some(m) => Some(self.make_additive_key_padding_mask(m)?),
            None => None,
        };

        let encoder_output = self
            .encoder
            .forward(&embedding_output, attention_add_mask.as_ref())?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            self.pool_embeddings(
                encoder_output.clone(),
                attention_mask_2d.as_ref(),
                &batch,
                &input_lengths,
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
                batch_size,
                max_length,
                has_pooling_requests,
            )?
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }

    fn pool_embeddings(
        &self,
        mut outputs: Tensor,      // (B,L,H)
        mask_2d: Option<&Tensor>, // (B,L) or None
        batch: &Batch,
        input_lengths: &[f32],
        has_raw_requests: bool,
    ) -> Result<Option<Tensor>> {
        let pooled_indices_length = batch.pooled_indices.len();

        // Build once and reuse for outputs and masks.
        let pooled_indices_tensor = if has_raw_requests && pooled_indices_length > 0 {
            Some(Tensor::from_vec(
                batch.pooled_indices.clone(),
                pooled_indices_length,
                &self.device,
            )?)
        } else {
            None
        };
        if let Some(ref idx) = pooled_indices_tensor {
            outputs = outputs.index_select(idx, 0)?;
        }

        let pooled_embeddings = match self.pool {
            Pool::Cls => {
                if let Some(ref pooler) = self.pooler {
                    pooler.forward(&outputs)?
                } else {
                    outputs.i((.., 0))?
                }
            }
            Pool::Mean => {
                // If a mask exists, compute denom from it on-device; otherwise fall back to input_lengths.
                if let Some(m2d_in) = mask_2d {
                    // Select mask rows if needed and zero out padded tokens.
                    let m2d = if let Some(ref idx) = pooled_indices_tensor {
                        m2d_in.index_select(idx, 0)?
                    } else {
                        m2d_in.clone()
                    };
                    outputs = outputs.broadcast_mul(&m2d.unsqueeze(2)?)?;

                    // denom = sum over tokens of the (selected) mask
                    let mut denom = m2d.sum(1)?.unsqueeze(1)?.to_dtype(self.dtype)?;
                    let eps = Tensor::new(1e-6f32, &self.device)?.to_dtype(self.dtype)?;
                    let zero_mask =
                        denom.le(&Tensor::new(0.5f32, &self.device)?.to_dtype(self.dtype)?)?;
                    denom = denom.broadcast_add(&eps)?;
                    let mean = outputs.sum(1)?.broadcast_div(&denom)?;
                    let zeros = mean.zeros_like()?;
                    zero_mask.where_cond(&zeros, &mean)?
                } else {
                    // No mask: use provided input_lengths.
                    let mut denom = if let Some(_) = pooled_indices_tensor {
                        let selected: Vec<f32> = batch
                            .pooled_indices
                            .iter()
                            .map(|&idx| input_lengths[idx as usize])
                            .collect();
                        Tensor::from_vec(selected, (pooled_indices_length, 1), &self.device)?
                            .to_dtype(self.dtype)?
                    } else {
                        Tensor::from_vec(input_lengths.to_vec(), (batch.len(), 1), &self.device)?
                            .to_dtype(self.dtype)?
                    };
                    let eps = Tensor::new(1e-6f32, &self.device)?.to_dtype(self.dtype)?;
                    let zero_mask =
                        denom.le(&Tensor::new(0.5f32, &self.device)?.to_dtype(self.dtype)?)?;
                    denom = denom.broadcast_add(&eps)?;
                    let mean = outputs.sum(1)?.broadcast_div(&denom)?;
                    let zeros = mean.zeros_like()?;
                    zero_mask.where_cond(&zeros, &mean)?
                }
            }
            Pool::LastToken | Pool::Splade => unreachable!(),
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

impl Model for DebertaV2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        let pooler = self
            .pooler
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("No pooler available for prediction".to_string()))?;

        if !pooler.is_classifier() {
            candle::bail!("`predict` requires a classification pooler");
        }

        let (pooled_embeddings, _) = self.forward(batch)?;
        pooled_embeddings
            .ok_or_else(|| candle::Error::Msg("No pooled embeddings available".to_string()))
    }
}

pub(crate) fn build_relative_position(
    query_size: usize,
    key_size: usize,
    device: &Device,
    bucket_size: Option<isize>,
    max_position: Option<isize>,
) -> Result<Tensor> {
    let q_ids = Tensor::arange(0, query_size as i64, device)?.unsqueeze(0)?;
    let k_ids: Tensor = Tensor::arange(0, key_size as i64, device)?.unsqueeze(D::Minus1)?;
    let mut rel_pos_ids = k_ids.broadcast_sub(&q_ids)?;
    let bucket_size = bucket_size.unwrap_or(-1);
    let max_position = max_position.unwrap_or(-1);

    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position, device)?;
    }

    rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    rel_pos_ids.unsqueeze(0)
}

pub(crate) fn make_log_bucket_position(
    relative_pos: Tensor,
    bucket_size: isize,
    max_position: isize,
    device: &Device,
) -> Result<Tensor> {
    let sign = relative_pos.to_dtype(DType::F32)?.sign()?;

    let mid = bucket_size / 2;

    let lt_mid = relative_pos.lt(mid as i64)?;
    let gt_neg_mid = relative_pos.gt(-mid as i64)?;

    let condition = lt_mid
        .to_dtype(candle::DType::F32)?
        .mul(&gt_neg_mid.to_dtype(candle::DType::F32)?)?
        .to_dtype(DType::U8)?;

    let on_true = Tensor::new(&[(mid - 1) as u32], device)?
        .broadcast_as(relative_pos.shape())?
        .to_dtype(relative_pos.dtype())?;

    let on_false = relative_pos
        .to_dtype(DType::F32)?
        .abs()?
        .to_dtype(DType::I64)?;

    let abs_pos = condition.where_cond(&on_true, &on_false)?;

    let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), device)?;

    let log_pos = {
        let first_log = abs_pos
            .to_dtype(DType::F32)?
            .broadcast_div(&mid_as_tensor)?
            .log()?;

        let second_log =
            Tensor::from_slice(&[((max_position as f32 - 1.0) / mid as f32)], (1,), device)?
                .log()?;

        let first_div_second = first_log.broadcast_div(&second_log)?;

        let to_ceil = first_div_second
            .broadcast_mul(Tensor::from_slice(&[(mid - 1) as f32], (1,), device)?.as_ref())?;

        let ceil = to_ceil.ceil()?;

        ceil.broadcast_add(&mid_as_tensor)?
    };

    Ok({
        let abs_pos_lte_mid = abs_pos.to_dtype(DType::F32)?.broadcast_le(&mid_as_tensor)?;
        let relative_pos = relative_pos.to_dtype(relative_pos.dtype())?;
        let log_pos_mul_sign = log_pos.broadcast_mul(&sign.to_dtype(DType::F32)?)?;
        abs_pos_lte_mid.where_cond(&relative_pos.to_dtype(DType::F32)?, &log_pos_mul_sign)?
    })
}
