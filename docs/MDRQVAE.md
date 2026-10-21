# MD-RQ-VAE: the multi-modal decomposed tokenizer (old implementation)

What the GRID code actually does, cited to `legacy/grid/` (paths below are relative to it). This
describes the reference only. For how to rebuild it in MACRec, see `PORTING.md`.

## 1. Summary

Each item has a text embedding `x_t` and an image embedding `x_i`. MD-RQ-VAE turns the pair into a
3-token semantic ID `[f, t, i]`:

- `f` is a **shared** code, quantized from a cross-attention fusion of both modalities;
- `t` is a **text-specific** code, quantized from `[shared residual ; text latent]`;
- `i` is an **image-specific** code, quantized from `[shared residual ; image latent]`.

Text is reconstructed from `e_f + e_t` and image from `e_f + e_i`. This forces the shared code to
carry what both modalities have in common, and each modality code to add what is specific to its
modality (the "additive" property). Unlike the chain RQ-VAE, `t` and `i` are siblings, not
parent and child: given `f`, they are produced by two separate branches. That is what makes the
factorized decoder (MFD, see `MFD.md`) natural.

## 2. Architecture (forward pass)

Source: `src/modules/clustering/md_rqvae.py`. Dimensions come from
`configs/experiment/md_rqvae_train_flat.yaml`.

| step | code | shape / config |
|---|---|---|
| inputs | SigLIP2-so400m-patch14-384 text and image embeddings (`yaml:31-34`) | 1152-d each (`embedding_dim: 1152`, `yaml:36`) |
| input norm | `x = LayerNorm(L2normalize(raw))`, per modality (`md_rqvae.py:147-148`, `215-216`) | 1152 |
| encoders | `z_t = E_t(x_t)`, `z_i = E_i(x_i)` (`md_rqvae.py:217-218`) | MLP 1152→768→256→512, ReLU (`yaml:195-215`); `latent_dim: 512` (`yaml:37`) |
| fusion | `z_f = F(z_t, z_i)` (`md_rqvae.py:219`) | `CrossAttentionFusion` (`md_rqvae.py:16-51`), 4 heads (`yaml:217-220`) |
| shared quant | `f, e_f = C_f(z_f)` (`md_rqvae.py:226-230`) | EMA VQ, `codebook_width_l1` entries |
| shared residual | `r_f = z_f − e_f.detach()` (`md_rqvae.py:236`) | residual against the **hard** code vector |
| text branch | `v_t = MLP_t([r_f ; z_t])`; `t, e_t = C_t(v_t)` (`md_rqvae.py:239-244`) | single Linear 1024→512, no activation (`yaml:223-230`; `src/models/components/network_blocks/mlp.py:33-45`) |
| image branch | `v_i = MLP_i([r_f ; z_i])`; `i, e_i = C_i(v_i)` (`md_rqvae.py:247-252`) | same as the text branch (`yaml:232-239`) |
| decoders | `x̂_t = D_t(e_f + e_t)`, `x̂_i = D_i(e_f + e_i)` (`md_rqvae.py:257-258`) | MLP 512→256→768→1152, ReLU (`yaml:300-320`) |
| output SID | `[f, t, i]` stacked in that order (`md_rqvae.py:442`) | |

**CrossAttentionFusion** (`md_rqvae.py:33-51`): each latent is treated as a length-1 sequence.
Text attends to image (`t2i_attn`) and image attends to text (`i2t_attn`). Each result goes through
a residual add and its own LayerNorm, the two are concatenated, then
`z_f = LayerNorm(Linear_{2d→d}(·))`.

**Gradient plumbing.** All three codebooks use straight-through estimation:
`e_ste = z + (e − z).detach()` (`src/components/quantization_strategies.py:116-137`). The
decoders consume the STE outputs, so gradients reach the encoders. `r_f` uses the detached hard
`e_f` (`md_rqvae.py:232-236`), but `z_f` still gets gradient through `r_f`, from the text and
image branches.

**Codebooks** (`yaml:242-297`; `src/modules/clustering/vector_quantization.py:135-287`):
`EMAVectorQuantization`, squared Euclidean distance, `ema_decay: 0.99`.
- Centroids are not trained by gradient (`vector_quantization.py:171-172`). They are EMA running
  means with Laplace smoothing (`vector_quantization.py:186-204`).
- Dead codes (EMA count < `dead_code_threshold: 1.0`) are re-seeded from random batch points at
  every training step (`vector_quantization.py:206-214`, `270-273`).
- Initialization is K-means++ on a buffer of `init_buffer_size: 8192` points (`yaml:192`;
  `vector_quantization.py:222-252`). The class default of 1000 is overridden. Until every
  codebook is initialized, the codebooks return zeros.

## 3. Loss

`md_rqvae.py:296-332`, weights from `yaml:190-191`:

```
L = L_rec + α·L_aux + β·L_vq,     α = 1.0, β = 1.0
L_rec = MSE(x̂_t, text_emb) + MSE(x̂_i, image_emb)         md_rqvae.py:323
L_aux = InfoNCE(z_f, z_t) + InfoNCE(z_f, z_i)              md_rqvae.py:328, 273-282
L_vq  = Σ_{f,t,i} MSE(v, e.detach())   (commitment only)    vector_quantization.py:280-281
```

- `L_rec` is **zero until all three codebooks have been K-means-initialized**
  (`md_rqvae.py:320-325`). `L_aux` is active from step 0 (`md_rqvae.py:327-328`).
- InfoNCE is in-batch with a learnable temperature: `log_temperature` is initialized at
  `log(0.07)` (`md_rqvae.py:170-171`) and the temperature is clamped at ≥ 1e-4 (`md_rqvae.py:279`).
  It is **one-directional**: cross-entropy over the rows of `a·pᵀ` only (`md_rqvae.py:280-282`),
  even though the docstring says "symmetric".
- The reconstruction target is the **raw** embedding (`md_rqvae.py:323`), not the
  L2-normalized + LayerNormed encoder input (`md_rqvae.py:215-216`).
- With EMA codebooks, `L_vq` is the commitment term only, with weight 1 (`β`).

## 4. Training hyperparameters (shipped `2048_1024` run)

| knob | value | source |
|---|---|---|
| codebook sizes shared / text / image | **2048 / 1024 / 1024** | `sbatchfiles/train_mdrqvae.sbatch:29-31` |
| optimizer | Adagrad, lr 1e-3, weight decay 0 | `yaml:412-417` |
| batch size | 2048 per device | `yaml:108` |
| steps | `max_steps: 10000` (`max_epochs: 10`); checkpoint `checkpoint_000_010000.ckpt` | `yaml:375-376`, `yaml:332-346` |
| precision | bf16-mixed | `yaml:380` |
| seed | 42 | `yaml:46` |
| hardware | 1×H100, 300 GB RAM | `sbatchfiles/train_mdrqvae.sbatch:3-9` |
| items | every item id in `items_dir=/scratch/yw8866/rec-tmall/image_sqf`; ids missing from an embedding file get zero vectors | `sbatchfiles/train_mdrqvae.sbatch:28`, `yaml:53-54` |
| train/val split | none: trains on the whole catalog (`val_check_interval: 1e8`, `yaml:382`) | |

SID extraction: `configs/experiment/md_rqvae_inference_flat.yaml` (`ckpt_path` required, `:23`)
runs `predict_step` (`md_rqvae.py:436-453`) and writes `item_id → [f,t,i]` to a pickle
(`md_rqvae_inference_flat.yaml:266-274`). `src/utils/verify_sids.py:234-275` (`convert_to_pt`)
converts it to `semantic_ids.pt` with shape `[3, max_item_id+1]`, column = item id, missing ids =
`[0,0,0]`. The measured file has shape `(3, 8133508)`, widths `[2048, 1024, 1024]`
(`rec-tmall/results/modality_decomp_report.txt:2`). 5,409,793 of those columns are real items; the
rest are placeholders (`rec-tmall/results/analyze_sid_factor.out:14`).

## 5. Collision handling

**Effectively none: collisions are kept.**
- The inference config does list `deduplicate_rows_in_tensor` as a post-processing step
  (`md_rqvae_inference_flat.yaml:275-279`). But that function returns immediately unless the
  file ends in `.pt` (`src/utils/tensor_utils.py:143-144`), and the writer produces a `.pkl`
  (`md_rqvae_inference_flat.yaml:266-274`). The shipped SID file has 3 rows, not 4, which
  confirms no dedup column was added (`modality_decomp_report.txt:2`).
- Measured collision rate: **36.2%** of 5.41M items share their SID with another item (4.16M
  unique tuples, largest group 152) (`AGENTS.md:418`).
- Downstream, the generator and evaluator work on **SID tuples, not item ids**: MFD builds its
  catalog as the unique tuple set (`src/models/modules/semantic_id/mfd_generation_model.py:217-221`).
  So a collided pair counts as a hit for either item. On a 5.4M-item catalog this was tolerable.
  On Amazon (6k–14k items) it must be resolved (see `PORTING.md`, open decisions).

## 6. Verified properties (with the scripts that measured them)

All the numbers below were measured on Tmall with the shipped checkpoint and SIDs.

**Additivity** (`rec-tmall/analyze_modality_decomposition.py` TEST 2, `:285-307`; run by
`sbatchfiles/modality_decomposition.sbatch:54`; output `rec-tmall/results/modality_decomp_report.txt:53-65`).
The sample is 200,000 items (`report.txt:1`). The metric is cosine to the true SigLIP vector after
decoding three code sums.

| branch | cos(D(e_f+e_mod), x) | cos(D(e_f), x) | cos(D(e_mod), x) | fraction where the sum beats both parts |
|---|---|---|---|---|
| text | 0.8816 | 0.8275 | 0.7375 | **0.9918** |
| image | 0.8096 | 0.7477 | 0.5509 | **0.9885** |

**Shared-code balance**: `cos(D_t(e_f),x_t) − cos(D_i(e_f),x_i) = +0.0798 [+0.0794, +0.0803]`
(`report.txt:64`). The shared code alone reconstructs text better than image, so it leans
slightly toward text.

**Modality separation (TEST 1b**, `analyze_modality_decomposition.py:234-282`; baseline
`:164-184`). Take pairs of items that agree on two codes and differ at one level, and measure
their SigLIP text and image distances. Each distance is divided by its random-pair baseline
(text 0.3734, image 0.5266, ratio 0.709, `report.txt:31-32`). The normalized separation ratio
below is (text/base) ÷ (image/base); values above 1 mean the level moves text more than image.

| level | MD-RQ-VAE text/base | image/base | **normalized ratio** | chain RQ-VAE level | normalized ratio |
|---|---|---|---|---|---|
| shared / L0 | 0.682 | 0.734 | 0.929 | level0 | 0.953 |
| text / L1 | 0.549 | 0.497 | **1.105** | level1 | 0.893 |
| image / L2 | 0.447 | 0.601 | **0.745** | level2 | 0.869 |

Source rows: `report.txt:41-43` (MD) and `report.txt:49-51` (chain). Reading: the text level is
the only level, in either tokenizer, that moves text more than image (1.105). The image level is
more image-selective than any chain level (0.745 vs ≥ 0.869). The asymmetry is real but moderate,
not a clean split.

**Decoder intervention (TEST 1a**, `analyze_modality_decomposition.py:196-231`;
`report.txt:20-29`). Swapping the text code drops text reconstruction cosine by 0.107. Swapping
the image code drops image cosine by 0.109. Swapping the shared code drops both (0.087 / 0.158).
Caveat: the cross-modality drop is 0 **by construction**, because `x̂_i` does not depend on `t`
(`report.txt:22-24`). Do not cite the "selectivity = inf" column as evidence.

**Conditional independence of t and i given f** (`rec-tmall/analyze_sid_behavior_coupling.py`
`--part factor`, `part_g` at `:170-235`; estimator `_cmi_from_triples` `:132-167`; shuffle bias
floor `:205-221`; run by `sbatchfiles/analyze_sid_factor.sbatch:30`; output
`rec-tmall/results/analyze_sid_factor.out:19-42`). Computed over the 5,409,793 catalog items, not
over interaction events.

| quantity | value |
|---|---|
| unconditional MI(t; i) | 1.594 bits = 16.7% of min(H_t, H_i) |
| H(t\|f), H(i\|f) | 7.11, 6.90 bits |
| CMI(t; i \| f): plug-in / shuffle null / **bias-corrected** | 2.841 / 2.721 / **0.120 bits** |
| corrected CMI / H(t\|f), / H(i\|f), / H(t,i\|f) | **1.69%**, **1.74%**, 1.07% |
| share of the t–i coupling absorbed by f | 92.5% |

Caveat: the raw plug-in estimate is about 96% estimator bias (2.72 of 2.84 bits). The 0.12-bit
result is a difference of two large numbers, and it depends on the shuffle floor. On a 6k–14k
item Amazon catalog the plug-in estimate will be almost all bias, so the test has to be
redesigned (see `PORTING.md` §5).

**Codebook health** (`AGENTS.md:417-418`): all three levels about 100% active
(2046/2048, 1024/1024, 1024/1024).

## 7. Known issues and Tmall-specific choices (do not copy blindly)

- **Stale comment.** `md_rqvae.py:254-256` describes "dropout on the modality-specific codes",
  but no dropout is implemented: the decoders always see `e_f + e_mod` (`:257-258`).
- **InfoNCE is one-directional** while its docstring says symmetric (`md_rqvae.py:276-282`).
- **Reconstruction target is the raw embedding**, not the normalized encoder input (§3).
- **Scale.** Codebook sizes 2048/1024/1024 were chosen for a 5.4M-item catalog. For 6k–14k
  items, a 2048-entry shared codebook gives each item a nearly unique `f`, and the factorization
  loses its meaning. Sizes must be re-chosen (PORTING.md, open decision).
- **Text is Chinese Tmall product names**, embedded with the SigLIP2 text tower; image is the
  product photo with the SigLIP2 image tower. Both towers share one vector space (`AGENTS.md`
  §2, steps c1/c2). MACRec instead uses LLaMA-7B mean-pooled title+description (4096-d) and
  CLIP ViT-L/14 image features. These live in different spaces, which matters for
  `CrossAttentionFusion`'s InfoNCE pull of `z_f` toward both modalities.
- **Collisions are left in** (§5), which is acceptable only because evaluation is tuple-level.
- **Codebook-init injection.** `src/utils/codebook_embedding_init.py:55-98` orthogonally
  up-projects the 512-d centroids into the T5 table (`W_up`, `:81-85`). The shipped MFD (Run-C)
  did **not** use it: it ran with `codebook_init_path=null`
  (`sbatchfiles/train_mfd_pmq_qf.sbatch:55`).
- **No validation split** during tokenizer training. Checkpoint selection is by `train/loss`
  (`yaml:336`).
