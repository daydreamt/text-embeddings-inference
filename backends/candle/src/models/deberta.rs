use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{bail, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
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
    pub label2id: Option<HashMap<String, u32>>,
    // DebertaV2 specific configurations
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
                    .get((embedding_size, hidden_size), "weight")?, // (embedding_size, hidden_size)
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
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        // println!(
        //    "[F1 DebertaV2Embeddings::forward] START. input_ids shape: {:?}",
        //    input_ids.shape()
        //);

        let mut embeddings = self.word_embeddings.forward(input_ids)?;
        // println!(
        //    "[F1 DebertaV2Embeddings::forward] After word_embeddings. Shape: {:?}, Sum: {:?}",
        //    embeddings.shape(),
        //    embeddings.sum_all()
        //);

        if let Some(ref position_embeddings) = self.position_embeddings {
            let pos_embeds = position_embeddings.forward(position_ids)?;
            // println!("[F1 DebertaV2Embeddings::forward] Applying position_embeddings. Shape: {:?}, Sum: {:?}", pos_embeds.shape(), pos_embeds.sum_all());
            embeddings = embeddings.add(&pos_embeds)?;
        }
        // println!(
        //    "[F1 DebertaV2Embeddings::forward] After position_embeddings. Shape: {:?}, Sum: {:?}",
        //    embeddings.shape(),
        //    embeddings.sum_all()
        //);

        if let Some(proj) = &self.embed_proj {
            embeddings = proj.forward(&embeddings)?;
            // println!(
            //    "[F1 DebertaV2Embeddings::forward] After embed_proj. Shape: {:?}, Sum: {:?}",
            //    embeddings.shape(),
            //    embeddings.sum_all()
            //);
        }

        if let Some(ref token_type_embeddings) = self.token_type_embeddings {
            let tte = token_type_embeddings.forward(token_type_ids)?;
            // println!("[F1 DebertaV2Embeddings::forward] Applying token_type_embeddings. Shape: {:?}, Sum: {:?}", tte.shape(), tte.sum_all());
            embeddings = embeddings.add(&tte)?;
        }
        // println!(     "[F1 DebertaV2Embeddings::forward] After token_type_embeddings. Shape: {:?}, Sum: {:?}", embeddings.shape(), embeddings.sum_all());

        let mut embeddings = self.layer_norm.forward(&embeddings, None)?;
        // println!( "[F1 DebertaV2Embeddings::forward] After layer_norm. Shape: {:?}, Sum: {:?}", embeddings.shape(), embeddings.sum_all() );

        if let Some(mask) = attention_mask {
            let mut mask = mask.clone();
            if mask.dims() != embeddings.dims() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }
            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&mask)?;
            /* println!(
                "[F1 DebertaV2Embeddings::forward] After masking. Shape: {:?}, Sum: {:?}",
                embeddings.shape(),
                embeddings.sum_all()
            );
            */
        }

        /* println!(
            "[F1 DebertaV2Embeddings::forward] END. Final shape: {:?}",
            embeddings.shape()
        );
        */
        Ok(embeddings)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/DebertaV2_v2/modeling_DebertaV2_v2.py#L72
struct XSoftmax {}

impl XSoftmax {
    fn apply(input: &Tensor, mask: &Tensor, dim: D, device: &Device) -> Result<Tensor> {
        /* println!(
            "[F1 XSoftmax::apply] START. input shape: {:?}, mask shape: {:?}",
            input.shape(),
            mask.shape()
        );
        */
        let mut rmask = mask.broadcast_as(input.shape())?.to_dtype(DType::F32)?;
        /* println!(
            "[F1 XSoftmax::apply] rmask pre-logic shape: {:?}",
            rmask.shape()
        );
        */

        rmask = rmask
            .broadcast_lt(&Tensor::new(&[1.0_f32], device)?)?
            .to_dtype(DType::U8)?;
        // println!(
        //    "[F1 XSoftmax::apply] rmask post-logic shape: {:?}",
        //    rmask.shape()
        //);

        let min_value_tensor = Tensor::new(&[f32::MIN], device)?
            .to_dtype(input.dtype())?
            .broadcast_as(input.shape())?;
        let mut output = rmask.where_cond(&min_value_tensor, input)?;
        // println!(
        //    "[F1 XSoftmax::apply] after where_cond with MIN. Shape: {:?}, Sum: {:?}",
        //    output.shape(),
        //    output.sum_all()
        //);

        output = candle_nn::ops::softmax(&output, dim)?;
        // println!(
        //    "[F1 XSoftmax::apply] after softmax. Shape: {:?}, Sum: {:?}",
        //    output.shape(),
        //    output.sum_all()
        //);

        let t_zeroes = Tensor::new(&[0f32], device)?
            .to_dtype(output.dtype())?
            .broadcast_as(output.shape())?;
        output = rmask.where_cond(&t_zeroes, &output)?;
        // println!(
        //    "[F1 XSoftmax::apply] after where_cond with ZEROS. Shape: {:?}, Sum: {:?}",
        //    output.shape(),
        //    output.sum_all()
        //);

        Ok(output)
    }
}

struct DebertaV2DisentangledSelfAttention {
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
    // ADDED: Device for XSoftmax
    device: Device,
}

impl DebertaV2DisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
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
            device: vb.device().clone(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] START. hidden_states shape: {:?}, attention_mask shape: {:?}", hidden_states.shape(), attention_mask.shape());
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let q_proj = self.query_proj.forward(hidden_states)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] q_proj shape: {:?}",
            q_proj.shape()
        );
        */
        let query_layer = self.transpose_for_scores(q_proj, batch_size, seq_len)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] query_layer shape: {:?}",
            query_layer.shape()
        );
        */

        let k_proj = self.key_proj.forward(hidden_states)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] k_proj shape: {:?}",
            k_proj.shape()
        );
        */

        let key_layer = self.transpose_for_scores(k_proj, batch_size, seq_len)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] key_layer shape: {:?}",
            key_layer.shape()
        );
        */

        let v_proj = self.value_proj.forward(hidden_states)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] v_proj shape: {:?}",
            v_proj.shape()
        );
        */
        let value_layer = self.transpose_for_scores(v_proj, batch_size, seq_len)?;
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] value_layer shape: {:?}",
            value_layer.shape()
        );
        */

        let mut scale_factor = 1.0;
        if self.pos_att_type.iter().any(|s| s == "c2p") {
            scale_factor += 1.0;
        }
        if self.pos_att_type.iter().any(|s| s == "p2c") {
            scale_factor += 1.0;
        }
        let scale = (self.attention_head_size as f64 * scale_factor).sqrt();
        /* println!(
            "[F1 DebertaV2DisentangledSelfAttention::forward] scale: {}",
            scale
        );
        */

        let scale_tensor =
            Tensor::new(&[scale as f32], query_layer.device())?.to_dtype(query_layer.dtype())?;
        // Scale key before matmul
        let key_layer_scaled = key_layer.transpose(1, 2)?.broadcast_div(&scale_tensor)?;
        let mut attention_scores = query_layer.matmul(&key_layer_scaled)?;

        // Compute raw attention scores
        // let mut attention_scores = query_layer.matmul(&key_layer.transpose(1, 2)?)?;
        // Scale after multiplication (more numerically stable)
        //let scale_tensor =
        //    Tensor::new(&[scale as f32], query_layer.device())?.to_dtype(query_layer.dtype())?;
        //attention_scores = attention_scores.broadcast_div(&scale_tensor)?;
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] attention_scores (content). Shape: {:?}, Sum: {:?}", attention_scores.shape(), attention_scores.sum_all());

        if let (Some(rel_embeddings), Some(relative_pos)) = (relative_embeddings, relative_pos) {
            // println!("[F1 DebertaV2DisentangledSelfAttention::forward] Applying disentangled_attention_bias.");
            let rel_att = self.disentangled_attention_bias(
                &query_layer,
                &key_layer,
                relative_pos,
                rel_embeddings,
                scale,
            )?;
            /* println!(
                "[F1 DebertaV2DisentangledSelfAttention::forward] rel_att shape: {:?}, Sum: {:?}",
                rel_att.shape(),
                rel_att.sum_all()
            );
            */
            attention_scores = attention_scores.add(&rel_att)?;
            // println!("[F1 DebertaV2DisentangledSelfAttention::forward] attention_scores (content+relative). Shape: {:?}, Sum: {:?}", attention_scores.shape(), attention_scores.sum_all());
        }

        let attention_scores =
            attention_scores.reshape((batch_size, self.num_attention_heads, seq_len, seq_len))?;
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] attention_scores reshaped (pre-softmax). Shape: {:?}", attention_scores.shape());

        let attention_probs =
            XSoftmax::apply(&attention_scores, attention_mask, D::Minus1, &self.device)?;
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] attention_probs from XSoftmax. Shape: {:?}, Sum: {:?}", attention_probs.shape(), attention_probs.sum_all());

        let attention_probs =
            attention_probs.reshape((batch_size * self.num_attention_heads, seq_len, seq_len))?;
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] attention_probs reshaped. Shape: {:?}", attention_probs.shape());

        let context_layer = attention_probs.matmul(&value_layer)?;
        // println!("[F1 DebertaV2DisentangledSelfAttention::forward] context_layer. Shape: {:?}, Sum: {:?}", context_layer.shape(), context_layer.sum_all());

        let final_context = context_layer
            .reshape((
                batch_size,
                self.num_attention_heads,
                seq_len,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.all_head_size));

        if let Ok(ref final_context_tensor) = final_context {
            /* println!(
                "[F1 DebertaV2DisentangledSelfAttention::forward] END. final_context shape: {:?}",
                final_context_tensor.shape()
            );
            */
        }

        final_context
    }

    fn transpose_for_scores(&self, x: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let reshaped = x.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;

        let transposed = reshaped.transpose(1, 2)?.contiguous()?;

        transposed.reshape((
            batch_size * self.num_attention_heads,
            transposed.dim(2)?, // This should be seq_len, but get it from tensor
            transposed.dim(3)?, // This should be attention_head_size
        ))
    }
    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: &Tensor,
        rel_embeddings: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        // println!("[F1 disentangled_attention_bias] START. query_layer shape: {:?}, rel_embeddings shape: {:?}", query_layer.shape(), rel_embeddings.shape());
        let (total_bs_heads, q_len, d_head) = query_layer.dims3()?;
        let k_len = key_layer.dim(1)?;
        let n_head = self.num_attention_heads;
        let bs = total_bs_heads / n_head;

        let scale_tensor =
            Tensor::new(scale as f32, query_layer.device())?.to_dtype(query_layer.dtype())?;
        // println!("[F1 disentangled_attention_bias] scale: {:?}", scale);

        let att_span = if self.position_buckets > 0 {
            self.position_buckets
        } else {
            self.max_relative_positions
        };

        // Normalize relative_pos dimensions once
        let relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            4 => relative_pos.clone(),
            other => bail!(
                "Relative position ids must be of dim 2, 3, or 4. Got {}",
                other
            ),
        };
        // println!("[F1 disentangled_attention_bias] normalized relative_pos shape: {:?}", relative_pos.shape());

        let rel_embeddings = rel_embeddings.to_dtype(query_layer.dtype())?;

        let mut score = Tensor::zeros(
            (total_bs_heads, q_len, k_len),
            query_layer.dtype(),
            query_layer.device(),
        )?;
        // println!("[F1 disentangled_attention_bias] Initial score shape: {:?}", score.shape());

        // Improved reshape function that minimizes operations
        let reshape_pos_embedding = |pos_embedding: Tensor| -> Result<Tensor> {
            // Reshape: [2*att_span, hidden] -> [n_head, 2*att_span, d_head]
            let reshaped = pos_embedding
                .reshape((2 * att_span as usize, n_head, d_head))?
                .transpose(0, 1)?;

            // Only expand if batch_size > 1
            if bs > 1 {
                reshaped
                    .unsqueeze(0)?
                    .expand(&[bs, n_head, 2 * att_span as usize, d_head])?
                    .reshape((total_bs_heads, 2 * att_span as usize, d_head))
            } else {
                Ok(reshaped.contiguous()?)
            }
        };

        if self.pos_att_type.iter().any(|s| s == "c2p") {
            // println!("[F1 disentangled_attention_bias] Calculating c2p attention.");
            let pos_key = if self.share_att_key {
                self.key_proj.forward(&rel_embeddings)?
            } else {
                self.pos_key_proj
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("pos_key_proj not found".to_string()))?
                    .forward(&rel_embeddings)?
            };
            // println!("[F1 disentangled_attention_bias] pos_key shape: {:?}, Sum: {:?}", pos_key.shape(), pos_key.sum_all());

            let pos_key_layer = reshape_pos_embedding(pos_key)?;
            // println!("[F1 disentangled_attention_bias] pos_key_layer (reshaped) shape: {:?}, Sum: {:?}", pos_key_layer.shape(), pos_key_layer.sum_all());

            /*
            // Scale pos_key_layer before matmul for better numerical stability
            let pos_key_scaled = pos_key_layer
                .transpose(1, 2)?
                .broadcast_div(&scale_tensor)?;
            let c2p_att = query_layer.matmul(&pos_key_scaled)?;
            */
            let c2p_att = query_layer.matmul(&pos_key_layer.transpose(1, 2)?)?;
            // println!("[F1 disentangled_attention_bias] c2p_att shape: {:?}, Sum: {:?}", c2p_att.shape(), c2p_att.sum_all());

            // Improved position index calculation
            let c2p_pos = relative_pos
                .broadcast_add(&Tensor::new(&[att_span as i64], relative_pos.device())?)?
                .clamp(0i64, (2 * att_span - 1) as i64)?
                .to_dtype(DType::I64)?; // Keep as I64 for gather
                                        // println!("[F1 disentangled_attention_bias] c2p_pos shape: {:?}", c2p_pos.shape());

            // Prepare indices for gather - ensure contiguous
            let c2p_indices = c2p_pos
                .squeeze(0)?
                .squeeze(0)?
                .expand(&[total_bs_heads, q_len, k_len])?
                .contiguous()?;
            // println!("[F1 disentangled_attention_bias] c2p_indices shape: {:?}", c2p_indices.shape());

            // let c2p_att_gathered = c2p_att.gather(&c2p_indices, 2)?; // Use dimension 2
            let c2p_att_gathered = c2p_att.gather(&c2p_indices, D::Minus1)?;
            // println!("[F1 disentangled_attention_bias] c2p_att_gathered shape: {:?}, Sum: {:?}", c2p_att_gathered.shape(), c2p_att_gathered.sum_all());
            //            score = score.add(&c2p_att_gathered)?;
            let c2p_att_scaled = c2p_att_gathered.broadcast_div(&scale_tensor)?;
            score = score.add(&c2p_att_scaled)?;
            // println!("[F1 disentangled_attention_bias] score after c2p. Sum: {:?}", score.sum_all());
        }

        if self.pos_att_type.iter().any(|s| s == "p2c") {
            // println!("[F1 disentangled_attention_bias] Calculating p2c attention.");
            let pos_query = if self.share_att_key {
                self.query_proj.forward(&rel_embeddings)?
            } else {
                self.pos_query_proj
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("pos_query_proj not found".to_string()))?
                    .forward(&rel_embeddings)?
            };
            // println!("[F1 disentangled_attention_bias] pos_query shape: {:?}, Sum: {:?}", pos_query.shape(), pos_query.sum_all());

            let r_pos = if key_layer.dim(1)? != query_layer.dim(1)? {
                // println!("[F1 disentangled_attention_bias] Rebuilding relative positions for p2c.");
                build_relative_position(
                    key_layer.dim(1)?,
                    key_layer.dim(1)?,
                    relative_pos.device(),
                    Some(self.position_buckets as isize),
                    Some(self.max_relative_positions as isize),
                )?
            } else {
                relative_pos.clone()
            };

            // Improved negation handling - avoid unnecessary conversions
            let neg_r_pos = {
                // Try integer negation first (more precise)
                let zero = Tensor::zeros_like(&r_pos)?;
                zero.broadcast_sub(&r_pos)?
            };

            let p2c_pos = neg_r_pos
                .broadcast_add(&Tensor::new(&[att_span as i64], relative_pos.device())?)?
                .clamp(0i64, (2 * att_span - 1) as i64)?
                .to_dtype(DType::I64)?; // Keep as I64 for gather
                                        // println!("[F1 disentangled_attention_bias] p2c_pos shape: {:?}", p2c_pos.shape());

            let pos_query_layer = reshape_pos_embedding(pos_query)?;
            // println!("[F1 disentangled_attention_bias] pos_query_layer (reshaped) shape: {:?}, Sum: {:?}", pos_query_layer.shape(), pos_query_layer.sum_all());

            /*
            // Scale before matmul
            let pos_query_scaled = pos_query_layer
                .transpose(1, 2)?
                .broadcast_div(&scale_tensor)?;
            let p2c_att = key_layer.matmul(&pos_query_scaled)?;
            */
            // For p2c attention - NO scaling before matmul
            let p2c_att = key_layer.matmul(&pos_query_layer.transpose(1, 2)?)?;
            // println!("[F1 disentangled_attention_bias] p2c_att shape: {:?}, Sum: {:?}", p2c_att.shape(), p2c_att.sum_all());

            // Prepare indices for gather
            let p2c_indices = p2c_pos
                .squeeze(0)?
                .squeeze(0)?
                .expand(&[total_bs_heads, k_len, k_len])?
                .contiguous()?;
            // println!("[F1 disentangled_attention_bias] p2c_indices shape: {:?}", p2c_indices.shape());

            /*
            let p2c_att_gathered = p2c_att
                .gather(&p2c_indices, 2)? // Use dimension 2
                .transpose(1, 2)?;
            */
            let p2c_att_gathered = p2c_att
                .gather(&p2c_indices, 2)? // or D::Minus1
                .transpose(1, 2)?; // Don't forget the transpose after gather!

            // println!("[F1 disentangled_attention_bias] p2c_att_gathered shape: {:?}, Sum: {:?}", p2c_att_gathered.shape(), p2c_att_gathered.sum_all());

            /*
            score = score.add(&p2c_att_gathered)?;
            */
            // Scale AFTER gather and transpose
            score = score.add(&p2c_att_gathered.broadcast_div(&scale_tensor)?)?;

            // println!("[F1 disentangled_attention_bias] score after p2c. Sum: {:?}", score.sum_all());
        }

        // println!("[F1 disentangled_attention_bias] END. Final score shape: {:?}", score.shape());
        Ok(score)
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
        attention_mask: &Tensor,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        /* println!(
            "[F1 DebertaV2Attention::forward] START. hidden_states shape: {:?}",
            hidden_states.shape()
        );
        */

        let self_output = self.self_attention.forward(
            hidden_states,
            attention_mask,
            relative_embeddings,
            relative_pos,
        )?;
        /* println!(
            "[F1 DebertaV2Attention::forward] After self_attention. Shape: {:?}, Sum: {:?}",
            self_output.shape(),
            self_output.sum_all()
        );
        */

        let attention_output = self.dense.forward(&self_output)?;
        /* println!(
            "[F1 DebertaV2Attention::forward] After dense. Shape: {:?}, Sum: {:?}",
            attention_output.shape(),
            attention_output.sum_all()
        );
        */

        let result = self
            .layer_norm
            .forward(&attention_output, Some(hidden_states));

        if let Ok(ref res) = result {
            /* println!(
                "[F1 DebertaV2Attention::forward] END. After layer_norm. Shape: {:?}, Sum: {:?}",
                res.shape(),
                res.sum_all()
            );
            */
        }
        result
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
        attention_mask: &Tensor,
        relative_embeddings: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        /* println!(
            "[F1 DebertaV2Layer::forward] START. hidden_states shape: {:?}",
            hidden_states.shape()
        );
        */

        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            relative_embeddings,
            relative_pos,
        )?;
        /* println!(
            "[F1 DebertaV2Layer::forward] After attention. Shape: {:?}, Sum: {:?}",
            attention_output.shape(),
            attention_output.sum_all()
        );
        */

        let intermediate_output = self.intermediate.forward(&attention_output)?;
        /* println!(
            "[F1 DebertaV2Layer::forward] After intermediate. Shape: {:?}, Sum: {:?}",
            intermediate_output.shape(),
            intermediate_output.sum_all()
        );
        */

        let layer_output = self.output.forward(&intermediate_output)?;
        /* println!(
            "[F1 DebertaV2Layer::forward] After output. Shape: {:?}, Sum: {:?}",
            layer_output.shape(),
            layer_output.sum_all()
        );
        */

        let result = self
            .layer_norm
            .forward(&layer_output, Some(&attention_output));

        if let Ok(ref res) = result {
            /* println!(
                "[F1 DebertaV2Layer::forward] END. After layer_norm. Shape: {:?}, Sum: {:?}",
                res.shape(),
                res.sum_all()
            );
            */
        }
        result
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
            let ln = LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?;
            Some(ln)
        } else {
            None
        };
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
        if max_relative_positions < 1 {
            max_relative_positions = config.max_position_embeddings as i64;
        }

        Ok(Self {
            layers,
            relative_attention_layer,
            layer_norm,
            relative_attention: config.relative_attention,
            position_buckets,
            max_relative_positions,
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    fn get_rel_pos(&self, hidden_states: &Tensor) -> Result<Option<Tensor>> {
        if !self.relative_attention {
            return Ok(None);
        }
        let (q_len, k_len) = (hidden_states.dim(1)?, hidden_states.dim(1)?);
        let rel_pos = build_relative_position(
            q_len,
            k_len,
            hidden_states.device(),
            Some(self.position_buckets as isize),
            Some(self.max_relative_positions as isize),
        )?;
        Ok(Some(rel_pos))
    }

    fn get_rel_embedding(&self) -> Result<Option<Tensor>> {
        if let Some(rel_attn_layer) = &self.relative_attention_layer {
            let mut embeddings = rel_attn_layer.get_rel_embedding()?;
            if let Some(ln) = &self.layer_norm {
                embeddings = ln.forward(&embeddings, None)?;
            }
            Ok(Some(embeddings))
        } else {
            Ok(None)
        }
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        /* println!(
            "[F1 DebertaV2Encoder::forward] START. hidden_states shape: {:?}",
            hidden_states.shape()
        );
        */

        let relative_pos = self.get_rel_pos(hidden_states)?;
        if let Some(ref rp) = relative_pos {
            /* println!(
                "[F1 DebertaV2Encoder::forward] Got relative_pos. Shape: {:?}",
                rp.shape()
            );
            */
        }

        let relative_embeddings = self.get_rel_embedding()?;
        if let Some(ref re) = relative_embeddings {
            /* println!(
                "[F1 DebertaV2Encoder::forward] Got relative_embeddings. Shape: {:?}",
                re.shape()
            );
            */
        }

        let mut current_hidden_states = hidden_states.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            /* println!(
                "[F1 DebertaV2Encoder::forward] >> Layer {i} START. Shape: {:?}",
                current_hidden_states.shape()
            );
            */
            current_hidden_states = layer.forward(
                &current_hidden_states,
                attention_mask,
                relative_embeddings.as_ref(),
                relative_pos.as_ref(),
            )?;
            /* println!(
                "[F1 DebertaV2Encoder::forward] >> Layer {i} END. Shape: {:?}, Sum: {:?}",
                current_hidden_states.shape(),
                current_hidden_states.sum_all()
            );
            */
        }
        /* println!(
            "[F1 DebertaV2Encoder::forward] END. Final hidden_states shape: {:?}",
            current_hidden_states.shape()
        );
        */
        Ok(current_hidden_states)
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
        /* println!(
            "[F1 DebertaV2Pooler::forward] START. hidden_states shape: {:?}",
            hidden_states.shape()
        );
        */
        let first_token = hidden_states.i((.., 0))?;
        /* println!(
            "[F1 DebertaV2Pooler::forward] first_token shape: {:?}",
            first_token.shape()
        );
        */

        let pooled = self.dense.forward(&first_token)?;
        /* println!(
            "[F1 DebertaV2Pooler::forward] after dense. Shape: {:?}, Sum: {:?}",
            pooled.shape(),
            pooled.sum_all()
        );
        */

        let activated = match self.activation {
            HiddenAct::Gelu => pooled.gelu(),
            HiddenAct::Relu => pooled.relu(),
            HiddenAct::Silu => pooled.silu(),
            _ => unreachable!("Invalid activation should have been caught during loading"),
        }?;
        /* println!(
            "[F1 DebertaV2Pooler::forward] after activation. Shape: {:?}, Sum: {:?}",
            activated.shape(),
            activated.sum_all()
        );
        */
        let result = match &self.classifier {
            Some(classifier) => classifier.forward(&activated),
            None => Ok(activated),
        };

        if let Ok(ref res) = result {
            /* println!(
                "[F1 DebertaV2Pooler::forward] END. Final output shape: {:?}",
                res.shape()
            );
            */
        }
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

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pool,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();
        /* println!(
            "[F1 DebertaV2Model::forward] START. batch len: {}, max_length: {}",
            batch.len(),
            batch.max_length
        );
        */
        let batch_len = batch.len();
        let max_length = batch.max_length as usize;

        let mut input_lengths = Vec::with_capacity(batch_len);
        for i in 0..batch_len {
            let length =
                (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize;
            input_lengths.push(length);
        }

        let mut attention_mask_vec = Vec::with_capacity(batch_len * max_length);
        for &seq_len in &input_lengths {
            // Use 1.0f32 for valid tokens
            attention_mask_vec.extend(std::iter::repeat(1.0f32).take(seq_len));
            // Use 0.0f32 for padding
            attention_mask_vec.extend(std::iter::repeat(0.0f32).take(max_length - seq_len));
        }
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_len, max_length), &self.device)?
                .to_dtype(self.dtype)?; // Ensure it's in the model's dtype
                                        /* println!(
                                            "[F1 DebertaV2Model::forward] attention_mask_2d shape: {:?}",
                                            attention_mask_2d.shape()
                                        );
                                        */

        let attention_mask_4d = self._get_attention_mask(attention_mask_2d.clone())?;
        /* println!(
            "[F1 DebertaV2Model::forward] attention_mask_4d shape: {:?}",
            attention_mask_4d.shape()
        );
        */

        // The pooling logic downstream requires f32 lengths.
        let input_lengths_f32: Vec<f32> = input_lengths.iter().map(|&l| l as f32).collect();

        let shape = (batch_len, max_length);
        let input_ids = Tensor::from_vec(batch.input_ids.clone(), shape, &self.device)?;
        let type_ids = Tensor::from_vec(batch.token_type_ids.clone(), shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids.clone(), shape, &self.device)?;

        // println!("[F1 DebertaV2Model::forward] Calling embeddings.forward...");
        let embedding_output = self.embeddings.forward(
            &input_ids,
            &type_ids,
            &position_ids,
            Some(&attention_mask_2d),
        )?;
        /* println!(
            "[F1 DebertaV2Model::forward] After embeddings.forward. Shape: {:?}, Sum: {:?}",
            embedding_output.shape(),
            embedding_output.sum_all()
        );
        */

        // println!("[F1 DebertaV2Model::forward] Calling encoder.forward...");
        let encoder_output = self
            .encoder
            .forward(&embedding_output, &attention_mask_4d)?;
        /* println!(
            "[F1 DebertaV2Model::forward] After encoder.forward. Shape: {:?}, Sum: {:?}",
            encoder_output.shape(),
            encoder_output.sum_all()
        );
        */

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            // println!("[F1 DebertaV2Model::forward] Pooling embeddings...");
            self.pool_embeddings(
                encoder_output.clone(),
                &batch,
                &input_lengths_f32,
                has_raw_requests,
            )?
        } else {
            None
        };
        if let Some(ref p) = pooled_embeddings {
            // println!(
            //    "[F1 DebertaV2Model::forward] Pooled embeddings shape: {:?}",
            //    p.shape()
            //);
        }

        let raw_embeddings = if has_raw_requests {
            // println!("[F1 DebertaV2Model::forward] Getting raw embeddings...");
            self.get_raw_embeddings(
                encoder_output,
                &batch,
                &input_lengths_f32,
                batch_len,
                max_length,
                has_pooling_requests,
            )?
        } else {
            None
        };
        if let Some(ref r) = raw_embeddings {
            /* println!(
                "[F1 DebertaV2Model::forward] Raw embeddings shape: {:?}",
                r.shape()
            );
            */
        }

        // println!("[F1 DebertaV2Model::forward] END.");
        Ok((pooled_embeddings, raw_embeddings))
    }

    fn _get_attention_mask(&self, mut attention_mask: Tensor) -> Result<Tensor> {
        match attention_mask.dims().len() {
            2 => {
                let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
                attention_mask = extended_attention_mask.broadcast_mul(
                    &extended_attention_mask
                        .squeeze(D::Minus2)?
                        .unsqueeze(D::Minus1)?,
                )?;
            }
            3 => attention_mask = attention_mask.unsqueeze(1)?,
            4 => {}
            len => bail!("Unsupported attention mask dimensions: {len}"),
        }
        attention_mask.to_dtype(self.dtype)
    }

    fn pool_embeddings(
        &self,
        mut outputs: Tensor,
        batch: &Batch,
        input_lengths: &[f32],
        has_raw_requests: bool,
    ) -> Result<Option<Tensor>> {
        let pooled_indices_length = batch.pooled_indices.len();

        if has_raw_requests && pooled_indices_length > 0 {
            let pooled_indices_tensor = Tensor::from_vec(
                batch.pooled_indices.clone(),
                pooled_indices_length,
                &self.device,
            )?;
            outputs = outputs.index_select(&pooled_indices_tensor, 0)?;
        }

        let pooled_embeddings = match self.pool {
            Pool::Cls => {
                if let Some(ref pooler) = self.pooler {
                    pooler.forward(&outputs)?
                } else {
                    outputs.i((.., 0))?
                }
            }
            Pool::LastToken => {
                bail!("LastToken pooling is not supported for DebertaV2")
            }
            Pool::Mean => {
                let input_lengths_tensor = if has_raw_requests && pooled_indices_length > 0 {
                    let selected_lengths: Vec<f32> = batch
                        .pooled_indices
                        .iter()
                        .map(|&idx| input_lengths[idx as usize])
                        .collect();
                    Tensor::from_vec(selected_lengths, (pooled_indices_length, 1), &self.device)?
                        .to_dtype(self.dtype)?
                } else {
                    Tensor::from_vec(input_lengths.to_vec(), (batch.len(), 1), &self.device)?
                        .to_dtype(self.dtype)?
                };

                outputs.sum(1)?.broadcast_div(&input_lengths_tensor)?
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
