# PORTING: re-implementing MD-RQ-VAE + MFD inside MACRec

Goal: add MD-RQ-VAE (tokenizer) and MFD (factorized decoder) to `macrec/` as new modules next to
MACRec's own, and evaluate them with MACRec's own data splits and metrics, so the numbers are
directly comparable to TIGER, MQL4GRec and MACRec. **Nothing is ported yet.**

Citation conventions:
- `macrec/...` paths are the pristine upstream at commit `7a48e74`.
- `legacy:` paths are under `legacy/grid/`.
- Old-implementation details live in `MDRQVAE.md` and `MFD.md`.

---

## 1. MACRec component map (what exists today)

| stage | file:function | what it does | notes for the port |
|---|---|---|---|
| raw → processed data | `macrec/data_process/amazon18_data_process.py` (`parse_args` l.265) | Amazon-2018 reviews/meta → 5-core, leave-one-out `.inter.json`, `.item.json`, `.item2id` | processed outputs **ship** in `macrec/data/{Instruments,Arts,Games}`; the raw data is not shipped |
| text features | `macrec/data_process/amazon_text_emb.py`: `generate_text` l.25, `generate_item_embedding` l.50-104 | LLaMA-7B (`huggyllama/llama-7b`, l.114), mean-pooled last hidden state, title and description encoded separately then averaged → `<D>.emb-llama-td.npy`, row = item id | fp32, batch size 1. `local_files_only` is set whenever `--model_cache_dir` exists (l.131) |
| image features | `macrec/data_process/load_all_figures.py` (downloads `imageURLHighRes` from raw meta), `clip_feature.py` | CLIP ViT-L/14 image features → `<D>.emb-ViT-L-14.npy` | **needs the raw Amazon-2018 meta**, which is not in the repo |
| pseudo-classes | `macrec/data/kmeans.py` | KMeans(512) on each embedding → `<D>.index_{lemb,vitemb}_kmeans512.json` (positives for the cross-RQ contrast) | hard-codes `dataset="Games"` and relative paths (l.4-5) |
| tokenizer model | `macrec/cross_index/models/rqvae.py`: `CrossRQVAE` l.10-270 | per modality: align MLP (in→768) → MLP encoder → RQ (`ResidualVectorQuantizer`, `models/rq.py`) → mirrored decoder. From `begin_cross_layer` on, each level adds an in-batch contrast with cross-modal pseudo-class positives (`class_con_cross_rq` l.98-150). Loss = recon + `quant_loss_weight`·RQ + `recon_contrast_weight`·InfoNCE(text_x_q, image_x_q) (l.222-231) | **two independent RQ stacks** (text and image), each with its own full SID. There is no shared code |
| tokenizer training | `macrec/cross_index/main.py` (args l.14-59, build l.103-130), `trainer.py`: `CrossTrainer.fit` l.390-450 | full-batch-ish epochs (batch 2048), eval every `eval_step` on the training data; saves `best_text_collision_model.pth` / `best_image_collision_model.pth` (l.244-246) | `--use_cross_rq` is `type=bool` (l.41): **any non-empty value, including "False", enables it**. Omit the flag to disable. `main.py` always loads **both** modalities (l.106) |
| SID assignment + collisions | `macrec/cross_index/generate_indices_distance.py` | `get_indices(use_sk=False)` (l.114). Collisions are resolved by **re-assigning the last level, then the second-to-last**, to each item's next-nearest code, ordered by distance; the item closest to the shared code keeps it (l.158-237, at most 3 rounds, l.178). Writes `{item_idx: ["<a_12>","<b_3>",...]}` (l.247-263); prefixes `<a_>`…`<e_>` for text, `<A_>`…`<E_>` for image (l.66-69) | loads the checkpoint with a bare `torch.load` (l.71): needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` under torch ≥ 2.6 (see `envs/SETUP.md`). Rebuilds the model from the **saved** args (l.72-95) |
| sequence data | `macrec/data.py`: `SeqRecDataset` l.181-383 | `_remap_items` (l.219-224) turns each item into the concatenated token string of its SID. Train = all prefixes of `items[:-2]` (mode 0, l.231-243); valid target = `items[-2]` (l.270-285); test target = `items[-1]` (l.287-308); `max_his_len=20` | `BaseDataset._load_data` opens **both** index files (l.42-55), so a single-modality run must point `--image_index_file` at the text index |
| multi-task data | `macrec/data.py`: `ItemImageDataset` l.386, `FusionSeqRecDataset` l.437; `utils.py`: `load_datasets` l.160-198 | MACRec's cross-modal tasks (`item2image`, `seqitem2image`, …) | not needed for TIGER / MFD |
| tokenizer + vocab | `macrec/finetune_contrastive.py` l.32-43, 75-77, 90 | T5 tokenizer from `config/ckpt`; adds every SID token from the index files as a new token; resizes embeddings | T5 config `config/ckpt/config.json`: d_model 128, 4+4 layers, 6 heads, d_ff 1024, the same as the legacy backbone |
| model | `macrec/modeling.py`: `CrossModalContrastive(T5ForConditionalGeneration)` l.31 | T5 seq2seq LM loss, plus `0.01`× in-batch encoder contrast for `task_flag==1` rows (l.36-38, 65-82) | for `seqrec`, `task_flag=0` (`data.py:383`), so it is plain T5 |
| training loop | `macrec/finetune_contrastive.py`: `train` l.16-156 | HF `Trainer`, early stopping on eval loss (`--patient`), `save_total_limit=1`, `load_best_model_at_end` | `report_to=None` (l.141) means **all** installed integrations (wandb if present). `finetune_full.sh` sets `WANDB_MODE=disabled` |
| collate | `macrec/collator.py`: `Collator` l.13-50, `TestCollator(Save)` l.52-111 | tokenizes the input string (history) and the label string (target SID) | |
| constrained beam search | `macrec/generation_trie.py`: `Trie` l.7-89, `prefix_allowed_tokens_fn` l.91; built in `macrec/test_ddp_save.py:66-75` | trie over `[0] + tokenizer.encode(sid_string)` for every catalog item; HF `generate(num_beams=20, max_new_tokens=10, prefix_allowed_tokens_fn=…)` (l.116-126) | chain decoding: validity by construction |
| evaluation | `macrec/test_ddp_save.py:test_ddp` l.34-249; `macrec/evaluate.py`: `get_topk_results` l.34-63, `hit_k` l.94, `ndcg_k` l.80 | sorts the beams by `sequences_scores`; `--filter_items` gives invalid strings −1000; leave-one-out, so IDCG = 1. Metrics `hit@1,5,10, ndcg@5,10` (`utils.py:130`) | targets are SID strings. Since SIDs are de-collided, matching a string is matching an item. Uses `DistributedSampler(drop_last=True)` (l.62) |
| ensemble | `macrec/ensemble.py`: `get_topk_results_ensemble` l.37-101 | merges the text-SID run and the image-SID run, ranking by item | the "MACRec" headline number comes from this ensemble (`scripts/finetune_full.sh`, last step) |

### Baselines available in this code
- **MACRec**: `cross_index/scripts/run_cross_rqvae.sh` (4×256, cross-RQ, contrast weights), then
  `gen_code.sh`, then `scripts/finetune_full.sh` (6 tasks, seqrec + seqimage test, ensemble).
  Upstream ships no embeddings or index files, so everything must be regenerated.
  `cross_index/scripts/gen.sh` calls a script that does not exist; use `gen_code.sh`.
- **TIGER**: not a separate script. The closest faithful recipe within this code is a
  text-only RQ-VAE (cross-RQ off), the collision re-assignment, then `--tasks seqrec`, testing
  `seqrec`. This is exactly the smoke test (`sbatch/smoke_tiger_instruments.sh`). It differs from
  the TIGER paper in two ways: (a) TIGER uses Sentence-T5 text features, MACRec LLaMA-7B;
  (b) TIGER resolves collisions with an extra dedup token, MACRec re-assigns codes.
- **MQL4GRec**: not a separate script. MACRec is built on the MQL4GRec code, and the nearest
  setting is: cross-RQ off (omit the flag), MQL4GRec's task mix, no cross contrast. The exact
  MQL4GRec settings (tasks, losses) are not encoded here. To reproduce MQL4GRec faithfully, use
  its own repo or its paper's table. **Open item.**

---

## 2. What MD-RQ-VAE + MFD need, and where it goes

Principle: **new files next to MACRec's own, with MACRec's interfaces** (same `.npy` inputs, same
`{item_idx: [tokens]}` index JSON, same `SeqRecDataset` splits, same `evaluate.py` metrics). Every
number then lands in the same table as TIGER and MACRec. Upstream files stay unmodified; where a
behavior change is unavoidable, subclass.

### 2.1 Target configuration (to start from; see open decisions)

```
SID = [f, t1, t2, i1, i2]        5 tokens, 256 codes per level
       f        shared code (from the cross-attention fusion z_f)
       t1, t2   text-specific residual chain of length 2   (RQ on v_t = MLP([r_f ; z_t]))
       i1, i2   image-specific residual chain of length 2  (RQ on v_i = MLP([r_f ; z_i]))
reconstruction: x̂_t = D_t(e_f + e_t1 + e_t2),  x̂_i = D_i(e_f + e_i1 + e_i2)
```

This generalizes `legacy:src/modules/clustering/md_rqvae.py:225-258` from one code per branch to
a two-level residual stack per branch. Capacity: 256 · 256² · 256² ≈ 1.1e12 tuples for 6k–14k
items. Collisions will still occur, because codes concentrate.

### 2.2 MFD generalization (f, then two parallel chains of length 2)

```
chain (TIGER-style, 5 tokens)  P = P(f) P(t1|f) P(t2|f,t1) P(i1|f,t1,t2) P(i2|f,t1,t2,i1)
MFD (this port)                P = P(f) · [P(t1|f) P(t2|f,t1)] · [P(i1|f) P(i2|f,i1)]
```

The cross-branch edges `t* → i*` are dropped, while the within-branch chain edges stay.

**Decoder layout (no behavior token; see `MFD.md` §2c).** Run the encoder once and the decoder in
two passes per example:
- text pass: `[BOS, f, t1] → predicts f @BOS, t1 @f, t2 @t1`;
- image pass: `[BOS, f, i1] → predicts i1 @f, i2 @i1` (count the f loss once).

In HF terms, these are two `decoder_input_ids` sequences that share `encoder_outputs`. Standard
causal masking makes the two branches conditionally independent given `f` by construction. This
is the legacy PMQ "separate passes" design (`legacy:mfd.py:359-366`) without the query slots.
Query slots (`q_f`, `q_t`, `q_i`) are an optional second step (legacy Run-C, `MFD.md` §2b).

**Decoding.** Generalizes `legacy:mfd.py:546-610`:
1. one pass gives the top-`K_f` shared codes `f`;
2. per root, the **text** branch runs a small beam over `(t1, t2)`, keeping `K_t` pairs; the
   **image** branch independently keeps `K_i` pairs;
3. compose `K_t × K_i` per root, filter against the catalog (sorted packed keys plus
   `searchsorted`, `legacy:mfd.py:213-221`, `300-308`);
4. score `logP(f) + (1/τ)[logP(t1,t2|f) + logP(i1,i2|f)]`;
5. take the global top-`K_pool`.

With depth 2, `K_t` and `K_i` are **no longer free**: each branch needs one extra decoder step
per kept `t1` / `i1`. Decoder-state cost per item: `1 + K_f·(2 + K_t1 + K_i1)`, versus
`nb·(2+K_f)` in the legacy code (`legacy:mfd.py:1184-1188`).

**Exact wide-catalog alternative** (`legacy:mfd.py:958-1017`). The Amazon catalogs have
6k–14k tuples, so every catalog item whose `f` is among the kept roots can be scored exactly.
Group the catalog by `(f, t1)` and `(f, i1)` prefixes, run one step per distinct prefix, and
gather. This keeps "no t/i pruning" at a cost bounded by the number of distinct prefixes under
the kept roots. **Report the decoder-state count for every configuration**, so the chain
comparison can be matched on compute (`legacy:sbatchfiles/matched_compute.sbatch`).

**Output format.** Emit each candidate as its SID token string, in score order, and reuse
`macrec/evaluate.py:get_topk_results / get_metrics_results` unchanged. MFD's hit@k / ndcg@k are
then computed by exactly the code that computes TIGER's.

### 2.3 New files (proposed) and function signatures

```
macrec/cross_index/models/mdrqvae.py        # tokenizer model (port of legacy md_rqvae.py)
    class CrossAttentionFusion(nn.Module):
        def forward(self, z_t: Tensor, z_i: Tensor) -> Tensor                       # (B,d)
    class MDRQVAE(nn.Module):
        def __init__(self, text_in_dim, image_in_dim, latent_dim=512,
                     shared_codes=256, text_codes=(256,256), image_codes=(256,256),
                     enc_hidden=(768,256), fusion_heads=4, alpha_nce=1.0, beta_vq=1.0,
                     vq_type="ema", ema_decay=0.99, dead_code_threshold=1.0,
                     kmeans_init=True, sk_epsilons=None)
        def forward(self, x_t, x_i) -> dict  # ids (B,5), recon (x̂_t,x̂_i), latents (z_f,z_t,z_i), vq_loss
        def compute_loss(self, out, x_t, x_i) -> tuple[Tensor, dict]                # total, parts
        @torch.no_grad()
        def get_indices(self, x_t, x_i) -> tuple[Tensor, list[Tensor]]              # ids (B,5), per-level distances
macrec/cross_index/main_mdrqvae.py           # mirrors main.py: DualEmbDataset + MDRQVAE + trainer
macrec/cross_index/trainer_mdrqvae.py        # (or reuse CrossTrainer if forward/compute_loss match)
macrec/cross_index/generate_indices_mdrqvae.py
    def assign_sids(model, data, device) -> tuple[np.ndarray, list[np.ndarray]] # (N,5) ids, distances
    def resolve_collisions(ids, dists, policy="reassign"|"extra_code") -> np.ndarray
    def write_index(ids, out_path, prefixes=("<f_{}>","<t_{}>","<u_{}>","<i_{}>","<j_{}>")) -> None
macrec/data_mfd.py
    class MFDSeqRecDataset(SeqRecDataset):   # same splits; target split into the two branch strings
        def __getitem__(self, i) -> dict     # input_ids (history str), labels_text "<f><t><u>", labels_image "<f><i><j>"
macrec/collator_mfd.py                       # tokenizes both label strings
macrec/modeling_mfd.py
    class MFDT5(T5ForConditionalGeneration):
        def forward(self, input_ids, attention_mask, labels_text, labels_image, **kw) -> Seq2SeqLMOutput
            # encoder once; decoder twice (text pass, image pass); loss = CE_f + CE_t1 + CE_t2 + CE_i1 + CE_i2
        @torch.no_grad()
        def generate_mfd(self, input_ids, attention_mask, catalog: "Catalog", K_f, K_t, K_i, K_pool,
                         tau=1.0, wide_catalog=False) -> tuple[list[list[str]], Tensor, dict]
            # candidate SID strings per example, scores (B,K_pool), stats incl. decoder-state count
macrec/mfd_catalog.py
    class Catalog:  # sorted packed keys of every item's 5-tuple, plus prefix groups for wide scoring
        def __init__(self, index_json: dict, tokenizer, code_num=256)
        def valid(self, f, t1, t2, i1, i2) -> BoolTensor
macrec/finetune_mfd.py                       # mirrors finetune_contrastive.py (HF Trainer), tasks=seqrec only
macrec/test_mfd.py                           # mirrors test_ddp_save.py; calls generate_mfd; same results JSON
macrec/analysis/cmi_branches.py              # §5 test
macrec/analysis/decomposition_tests.py       # port of legacy rec-tmall/analyze_modality_decomposition.py
```

The token prefixes are a suggestion; the requirement is that they don't clash with MACRec's
`<a_>…<e_>`, `<A_>…<E_>`. The chain baseline on the same MD SID needs no new code: write the
5-token index JSON and run the stock `finetune_contrastive.py --tasks seqrec` + `test_ddp_save.py`
(trie over the 5-token strings). That isolates the decoder, as in the legacy ablation.

### 2.4 What changes vs the legacy implementation

| legacy (Tmall) | port (Amazon / MACRec) |
|---|---|
| SigLIP2 1152-d, both modalities in one space | MACRec: LLaMA-7B 4096-d text, CLIP ViT-L/14 768-d image (different spaces; **open decision**) |
| 3-token SID `[f,t,i]`, 2048/1024/1024 | 5-token `[f,t1,t2,i1,i2]`, 256 each |
| collisions kept (36.2%) | must resolve (MACRec's eval is per item) |
| custom T5 with `inputs_embeds`, stride-4 behavior-interleaved history | HF `T5ForConditionalGeneration` over tokenized SID strings |
| behavior token, t1/t2/t3 tracks, marginalization | none |
| catalog = unique tuples | catalog = item SIDs (unique after collision handling) |
| Lightning + Hydra | argparse + HF `Trainer` (MACRec's style) |

---

## 3. Validating the tokenizer before training any recommender

Port `legacy:rec-tmall/analyze_modality_decomposition.py` and re-run its tests on the Amazon
SIDs, both for MD-RQ-VAE and for the chain RQ-VAE (MACRec text RQ):

1. **Additivity** (`:285-307`): fraction of items where `D(e_f+e_mod)` beats both parts, and the
   shared-balance gap. Tmall: 0.9918 / 0.9885, +0.0798.
2. **Normalized separation ratio** (`:164-184`, `:234-282`). Tmall: text level 1.105, image level
   0.745, against chain 0.95/0.89/0.87.
3. **Codebook usage and collision rate before and after resolution**
   (`legacy:src/utils/verify_sids.py:84-119`).

## 4. Training and evaluation protocol

- Splits, history length (20) and metrics exactly as MACRec (§1). Report hit@{1,5,10} and
  ndcg@{5,10}, plus coverage@pool and `P(rank<10 | covered)` for MFD and chain
  (`legacy:rec-tmall/analyze_mfd_audit.py`).
- Compare MFD and chain **at matched decoder-state budget**, with **paired** event bootstrap CIs
  over the same test users, and over **≥3 seeds**. On Tmall, a single seed understated the
  uncertainty of the lead (`legacy:rec-tmall/GOAL.md:38-41`).

## 5. Test: are the two branches conditionally independent given f?

The licensing test for dropping the cross-branch edges, generalized from
`legacy:rec-tmall/analyze_sid_behavior_coupling.py` part G (`:170-235`).

- **Count level (catalog)**: `CMI((t1,t2); (i1,i2) | f)` and the level-wise `CMI(t1; i1 | f)`.
  With 6k–14k items over 256 f-groups, the plug-in estimator is dominated by bias. On Tmall, the
  bias was 2.72 of 2.84 bits even with 5.4M items (`MDRQVAE.md` §6). So the
  within-f-group shuffle floor (`:205-221`) is **mandatory**. Report raw, null and corrected
  values; consider also a Miller–Madow or NSB estimator for a sanity check.
- **Model level (events, recommended as the primary test)**: train the chain decoder and the MFD
  decoder on the same MD SID, and compare held-out `−log P(i1,i2 | x, f)` under MFD with
  `−log P(i1,i2 | x, f, t1, t2)` under the chain. The gap in nats per event is the predictive CMI
  that the factorization discards. Unlike the count test, it is measured on the interaction
  distribution that the metric uses.
- Decision rule (proposal): report CMI as a fraction of `H(i-branch | f)`. Tmall's licensing
  value was 1.7%.

---

## 6. OPEN DECISIONS (listed, not decided)

1. **Branch depth.** One code per branch (`[f,t,i]`, the legacy design, 3 tokens) or two
   (`[f,t1,t2,i1,i2]`, 5 tokens, the target above)?
   - Depth 1 keeps `K_t`/`K_i` free, but needs larger codebooks to stay collision-free, and the
     sequence is shorter than TIGER/MACRec's 4 tokens.
   - Depth 2 matches MACRec's token budget better, but adds per-branch beam cost.
2. **Codebook sizes.** Shared vs branch sizes (256 everywhere vs a larger `f`). On Tmall,
   2048/1024/1024 for 5.4M items; for 6k–14k items, 256 shared codes means about 25–55 items per
   `f`. How many items per `f` the factorization needs is an empirical question.
3. **Collision handling.** MACRec-style **re-assignment** of the last branch levels to
   next-nearest codes (`macrec/cross_index/generate_indices_distance.py:168-237`), or an
   **extra dedup code** (TIGER-style)?
   - With re-assignment, which branch is re-assigned: text, image, or alternating? That changes
     the semantics of the moved codes.
   - An extra code adds a 6th token and must live in one of the branches, or after both.
4. **Encoders.** SigLIP2 (one shared space for text and image; what MD-RQ-VAE was validated
   with) or MACRec's LLaMA-7B (text) + CLIP ViT-L/14 (image), which TIGER/MACRec numbers use?
   - Matching MACRec isolates tokenizer and decoder for a fair comparison.
   - SigLIP2 is what the decomposition properties were measured on.
   - Either way, image features need the raw Amazon-2018 metadata (image URLs), which is not in
     the repo.
5. **Reuse of MACRec's alignment losses.** Whether to add MACRec's cross-modal pieces to MD-RQ-VAE:
   - pseudo-class contrast `class_con_cross_rq` (`macrec/cross_index/models/rqvae.py:98-150`);
   - quantized-space InfoNCE (`rqvae.py:212-220`, weight `recon_contrast_weight`);
   - the multi-task T5 alignment tasks (`item2image`, …) and the encoder contrast
     (`macrec/modeling.py:49-82`).

   These can be added on top of the shared-code InfoNCE that MD-RQ-VAE already has
   (`legacy:md_rqvae.py:273-282`). They can also confound "MD vs MACRec" comparisons.
6. **VQ type.** Legacy EMA + dead-code revival + K-means++ (`legacy:vector_quantization.py:135-287`),
   or MACRec's gradient VQ with k-means init and optional Sinkhorn (`macrec/cross_index/models/vq.py`)?
7. **Query slots.** Plain two-pass MFD first, or go straight to PMQ / Query-f (legacy Run-C)?
8. **MQL4GRec baseline source** (§1): reimplement inside MACRec, or take it from its own repo
   or paper.
