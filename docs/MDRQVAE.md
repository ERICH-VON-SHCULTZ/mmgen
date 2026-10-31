# MD-RQ-VAE: the multi-modal decomposed tokenizer

This doc has two parts. **Part A is the current implementation on Amazon** (mmgen, MACRec code
base): what was built, the official SIDs, and where everything lives. **Part B is the old Tmall
implementation** in `legacy/grid/`, kept as the design reference (cited file:line). Part B's
section numbers (§1–§7) are stable; other docs link to them.

Reading order for a new member: Part A here (A3 = exact recipe to train the SIDs yourself) → `docs/MDRQVAE_REPORT.md` (results and decisions) →
`docs/MDRQVAE_LOG.md` (every attempt and control, in order) → `docs/BASELINE_REPORT.md` (the
baselines MD-RQ-VAE is compared against) → `docs/MFD.md` (the planned decoder; design not final).

# Part A. Current implementation (Amazon Instruments / Arts / Games)

## A1. What was built

SID = `[f ; t_1..t_L ; i_1..i_L]`: one shared code `f`, then an L-level residual text chain and an
L-level residual image chain. Two configurations: **1+2+2** (L = 2, SID length 5) and **1+3+3**
(L = 3, length 7); 256 codes per level, 32-d codes. The design principles (fixed by the task, not
tuned): the shared code is quantized first from a fused representation; each modality branch is
built from the shared residual plus the modality latent; text is reconstructed from shared + text
codes, image from shared + image codes; conditional independence of the two chains given `f` is
measured, not assumed.

Forward pass (`macrec/cross_index/models/mdrqvae.py`, class `MDRQVAE`):

| step | Amazon implementation |
|---|---|
| inputs | LLaMA-7B text features (4096-d, `<D>.emb-llama-td.npy`) and CLIP ViT-L/14 image features (768-d, `<D>.emb-ViT-L-14.npy`): the baselines' own files |
| input norm | per modality, L2-normalize × √dim (unit per-dimension RMS); this is also the reconstruction target |
| encoders | per modality, MLP [2048, 1024, 512, 256, 128, 64] → 32 (the A/B RQ-VAE MLP) |
| fusion | cross-attention of the two latents as 2 tokens, 4 heads (Tmall `CrossAttentionFusion`) → `z_f` |
| shared quant | 1 level × 256, `q_f` |
| branches | `r_f = z_f − q_f.detach()`; `v_t = Linear([r_f ; z_t])`, `v_i = Linear([r_f ; z_i])` |
| modality quant | residual VQ, L levels × 256 per chain |
| decoders | `x̂_t = D_t(q_f + q_t)`, `x̂_i = D_i(q_f + q_i)` (mirror MLPs) |
| codebooks | MACRec `VectorQuantizer`: gradient-trained, k-means init on the first batch, L2 lookup, β = 0.25; dead codes (unused by the whole catalog) re-seeded from encoder outputs every 10 epochs until epoch 400 |
| loss | MSE_t + MSE_i + VQ + 1.0 × (InfoNCE(z_f, z_t) + InfoNCE(z_f, z_i)), one-directional, learnable temperature |
| training | AdamW lr 1e-3, wd 1e-4, batch 2048, 500 epochs, seed 42, whole catalog (no interactions used) |
| checkpoint | **final epoch (500), no selection** (`final.pth`); see A2 |

Final settings = attempt **a05** (1+2+2) and **a06** (1+3+3), found on Instruments in 6 attempts
(one change at a time, `docs/MDRQVAE_LOG.md`), applied unchanged to Arts and Games. Differences from
Tmall are tabulated in `docs/MDRQVAE_REPORT.md` §2.

Code and jobs (all new files; MACRec's own files are untouched):

| file | role |
|---|---|
| `macrec/cross_index/models/mdrqvae.py` | the model |
| `macrec/cross_index/main_mdrqvae.py` | trainer: writes `run_config.json` first, then `train_log.jsonl` (every 10 epochs: utilization, full-SID collision, reconstruction cos), `best.pth` (lowest collision; superseded rule), `final.pth` (official), `train_summary.json` |
| `macrec/cross_index/generate_indices_md.py` | codes from a checkpoint (`codes.npy`, `sid_stats.json`); its index file adds the old `<x_k>` token (superseded) |
| `macrec/cross_index/reassign_md.py` | **official index**: distance-based reassignment, no extra token (rule in A2) |
| `macrec/md_mt_data.py` | control only: C-style multi-task datasets on one MD SID (`mdshared`, `mdtext`, `mdimage`, `mdtrans`); run name `mdmt122`; result: no gain (`docs/MDRQVAE_REPORT.md` §5.2) |
| `macrec/cross_index/md_diagnostics.py` | diagnostics 1–7 (reconstruction vs A/B, utilization, collision, CMI with permutation null, AMI with A/B, separation, additivity, shared balance) |
| `sbatch/md/tok_attempt.sbatch` | one tokenizer attempt: train → codes → diagnostics (on `best.pth`, as in the attempt log) |
| `sbatch/md/reassign.sbatch` | reassignment + diagnostics on the new index (`CKPT=final`, official) |
| `sbatch/tools/freeze_md_sids.py` | versioned, hashed SID sets; `--lock` makes them read-only |
| `sbatch/tools/cmi_shared_image.py` | first-level CMI without items whose image file is shared (acceptance check) |
| `sbatch/baselines/run_baseline.sh` case `md*` | chain decoder on an MD index: TIGER settings (plain T5, batch 256, lr 5e-4, seqrec only, one list) |

Run directories: tokenizers in `mmgen_runs/md/<D>/tok/<attempt>/` (`a05_122`, `a06_133` are the
final ones); chain-decoder runs in `mmgen_runs/baselines/<D>/<tag>/seed42/`, with
`<tag>` = `md122_a05_ep500_reassign` or `md133_a06_ep500_reassign`.

## A2. Official SID on Amazon (decision 2026-10-27; v1 locked 2026-10-28)

The official SID is the **reassigned, token-free index**: `[shared, text chain, image chain]`,
length 5 for 1+2+2 (a05 settings) and 7 for 1+3+3 (a06 settings), with no trailing disambiguation
token. Every later run, chain decoder and MFD alike, uses it; chain vs MFD comparisons must both
use the same set. Index files: `mmgen_runs/data/<D>/<D>.index_md122_a05_ep500_reassign.json` (1+2+2) and
`<D>.index_md133_a06_ep500_reassign.json` (1+3+3). Two older index files exist only as controls
for the Step-3 comparison (`docs/MDRQVAE_REPORT.md` §7.3): `<D>.index_md1X2_a0X.json` (best-collision
checkpoint + an extra `<x_k>` token) and `<D>.index_md1X2_a0X_reassign.json` (best-collision
checkpoint, reassigned). Do not use them for new work.

Reassignment rule (`macrec/cross_index/reassign_md.py`; the tokenizers are not retrained):
1. the shared code never changes;
2. only full-SID collision groups are touched; the item with the lowest total quantization
   distance to its own codes (ties: lower item id) keeps its SID;
3. every other item of the group changes exactly one code, on the LAST level of the text or image
   chain: the code with the smallest increase in that level's quantization distance (searched over
   both chains), among codes whose new SID is not already used;
4. only if no such code exists, the second-to-last level (MQL4GRec logic); never the shared level;
5. deterministic (no randomness): groups ordered by their first item id, items by id.

Frozen sets: `/scratch/yw8866/mmgen_runs/md/sids/<version>/<D>/<cfg>/` with `tokenizer.pth`,
`run_config.json`, `index.json`, `index.sha256`, `reassign.json`, `diagnostics.{json,md}`, `STATUS`
(`sbatch/tools/freeze_md_sids.py`). **v1 LOCKED 2026-10-28** (user confirmation after the
Instruments chain-decoder control): every file read-only. Any change becomes v2 with a note; v1 is
never overwritten. Jobs read `mmgen_runs/data/<D>/<D>.index_md1X2_a0X_ep500_reassign.json`; check it
against `index.sha256` before a run.

Checkpoint rule (decision 2026-10-27): the tokenizer at the **final epoch (500), no selection**
(`final.pth`). Reconstruction rises roughly monotonically in all six runs and is still rising at
epoch 500. The earlier rule (lowest full-SID collision, `best.pth`) is MQL4GRec's; A, B, E, C and D
keep it, except the final-epoch E-norm control (`tiger_fnormj_ep500`). Acceptance for a final-epoch
set: corrected first-level CMI after excluding items with byte-identical images ≤ ~2% of H(t₁|f).

- v1 Instruments 1+2+2: `mmgen_runs/md/sids/v1/Instruments/1+2+2/index.json` sha256 `512221a10e72602185bb0bae1127daef72f6f4f54e67ca0b1a704de331166564` (final epoch)
- v1 Instruments 1+3+3: `mmgen_runs/md/sids/v1/Instruments/1+3+3/index.json` sha256 `1fe4df38cd83c69ec0bb18c94293c441b9bb57e61a43c8bc1270f35b8955c3af` (final epoch; same as best-collision)
- v1 Arts 1+2+2: `mmgen_runs/md/sids/v1/Arts/1+2+2/index.json` sha256 `cbc70c4ed3da57fe3172bc280a2feccee5ab92bec1689691c9cb3eb88a7e0960` (final epoch; same as best-collision)
- v1 Arts 1+3+3: `mmgen_runs/md/sids/v1/Arts/1+3+3/index.json` sha256 `8612e3cbfd92fa1929df7134aa0898c842c2515b98a3889aa93ccb81f6a14961` (final epoch)
- v1 Games 1+2+2: `mmgen_runs/md/sids/v1/Games/1+2+2/index.json` sha256 `48851182c58e4cfd89106efa7bfbdbd080c9b36a7761ff8c062622ea44c5e144` (final epoch)
- v1 Games 1+3+3: `mmgen_runs/md/sids/v1/Games/1+3+3/index.json` sha256 `15c68e21bcec5ae48e6bfe9c37df84145209b2f5998140e5809d3d2798fb4121` (final epoch; excl. CMI 2.02%, accepted by the user as within ~2% so all six follow one rule)

Superseded (kept): `mmgen_runs/md/sids/superseded/v1_20261028/<D>/<cfg>/`, the best-collision sets
of Instruments 1+2+2 (sha256 `aa514208…`), Instruments 1+3+3, Arts 1+2+2, Arts 1+3+3
(`73d5b133…`), Games 1+2+2 (`f5a30990…`), Games 1+3+3 (`1c1f4220…`).

## A3. Appendix: how the v1 SIDs were produced, and how to train them yourself

This is the exact recipe behind the six v1 sets. Two stages per set: **(1) train the tokenizer**
(`main_mdrqvae.py`, 500 epochs, keep `final.pth`), **(2) reassign collisions** (`reassign_md.py`
on `final.pth`). Then `freeze_md_sids.py` copied the results into `sids/v1/` and `--lock` made them
read-only. No step uses interaction data; the tokenizer sees only the item features of the whole
catalog.

**What to expect when you train it yourself.** Following this recipe gives a tokenizer of the same
quality as v1, but the individual codes will not be identical: GPU training is not bit-for-bit
deterministic, so each run assigns somewhat different codes. **That is expected and acceptable.**
Check your set against the acceptance list in A3.7; if it passes, it is a valid MD-RQ-VAE SID set.
Two practical points:
- Give your run its own `NAME` / `--out_dir` and its own index `TAG` (e.g. `md122_a05_ep500_reassign_<you>`),
  so you never overwrite the official files in `mmgen_runs/data/<D>/`.
- To compare against the numbers in our reports exactly, use the locked v1 files
  (`sids/v1/<D>/<cfg>/index.json`), which every reported MD run used.

### A3.1 Environment

| item | value |
|---|---|
| container | Singularity image `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
| Python env | overlay `/scratch/yw8866/envs/macrec/overlay-25GB-500K.ext3` mounted `:ro`, `source /ext3/env.sh && conda activate py310` (Python 3.10.20, torch 2.6.0+cu124; full list `envs/requirements.lock`, rebuild `envs/SETUP.md`) |
| env vars | `USE_TF=0`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| hardware | one GPU; the original runs used A100 / H100 / H200 (per set below); training takes ~40–60 s |
| working dir | `macrec/cross_index/` (the trainer imports `models.mdrqvae`; `reassign_md.py` imports `generate_indices_md`) |

### A3.2 Inputs (frozen; produced by `sbatch/features.sbatch`)

Rows are in item-id order (`<D>.item2id`); 6,250 / 9,416 / 13,839 items.
- Text: `mmgen_runs/data/<D>/<D>.emb-llama-td.npy` (4096-d): LLaMA-7B (`huggyllama/llama-7b`),
  title + description, mean pooling (`macrec/data_process/amazon_text_emb.py`). On Instruments it
  equals MQL4GRec's shipped file (max abs diff 4e-5).
- Image: `mmgen_runs/data/<D>/<D>.emb-ViT-L-14.npy` (768-d): CLIP ViT-L/14 image branch on the main
  product image (`macrec/data_process/clip_feature_ext.py`, MACRec's `clip_feature.py` logic,
  tolerant to missing images).

| dataset | text sha256 | image sha256 |
|---|---|---|
| Instruments | `925f5151a7b276e4d44a85ab70f78e5e33566c1e33ca6f38b9cf43ace449d787` | `6e74c71e12d037687fc38f434a40716de3e727ddd65cbe97aa27c77439e41b64` |
| Arts | `f961fee57e7f403143707212b244f108ba66900f19c0469a352da808dead0369` | `32562b1defd6cafc075000340562f2a76a2f500ff8591e41399fe7f07a66ea99` |
| Games | `acdbd3aa224b1f498fa8134364bd606c25d5d2610f5c78244851739233728899` | `dddf6c2f9b3380478a0eef859440a419abe4ad77c67f9d4cc449599fce5c874b` |

### A3.3 Stage 1: tokenizer training

Full resolved settings (identical for all six sets except `n_modal`; also in each set's
`sids/v1/<D>/<cfg>/run_config.json`):

| setting | value | setting | value |
|---|---|---|---|
| `n_shared` | 1 | `n_modal` | **2** (1+2+2, attempt a05) or **3** (1+3+3, a06) |
| `codebook_size` | 256 per level | `e_dim` | 32 |
| `layers` (encoder MLP) | 2048 1024 512 256 128 64 | `input_mode` | `l2_sqrtd` (per-modality L2 norm × √dim) |
| `fusion` | `xattn` (cross-attention, 4 heads) | `info_nce_weight` | 1.0 |
| `quant_loss_weight` | 1.0 | `dead_code_reset` / `dead_code_until` | every 10 epochs / until epoch 400 |
| optimizer | AdamW, `lr` 1e-3, `weight_decay` 1e-4 | `batch_size` | 2048 |
| `epochs` | 500 | `eval_every` | 10 |
| `seed` | 42 (`random`, `numpy`, `torch`, `torch.cuda` seeded; cuDNN determinism not forced) | checkpoint used | **`final.pth`** (epoch 500, no selection) |

Non-default flags (everything else is the trainer's default): `--n_shared 1 --n_modal {2|3}
--dead_code_reset 10 --dead_code_until 400 --fusion xattn`.

Command, as run (job script `sbatch/md/tok_attempt.sbatch`; it also writes the superseded
best-collision index and diagnostics, which v1 does not use):

```bash
sbatch --export=ALL,D=<D>,NAME=a05_122,TAG=md122_a05,ARGS="--n_shared 1 --n_modal 2 --dead_code_reset 10 --dead_code_until 400 --fusion xattn" sbatch/md/tok_attempt.sbatch
sbatch --export=ALL,D=<D>,NAME=a06_133,TAG=md133_a06,ARGS="--n_shared 1 --n_modal 3 --dead_code_reset 10 --dead_code_until 400 --fusion xattn" sbatch/md/tok_attempt.sbatch
```

Equivalent direct call inside the container (what the job runs):

```bash
cd /scratch/yw8866/mmgen/macrec/cross_index
python -u main_mdrqvae.py --dataset <D> --out_dir /scratch/yw8866/mmgen_runs/md/<D>/tok/a05_122 \
    --n_shared 1 --n_modal 2 --dead_code_reset 10 --dead_code_until 400 --fusion xattn
# 1+3+3: --out_dir .../a06_133 and --n_modal 3
```

Outputs in `--out_dir`: `run_config.json`, `train_log.jsonl`, `best.pth` (superseded rule),
**`final.pth`**, `train_summary.json`.

### A3.4 Stage 2: reassignment (official index, no extra token)

```bash
sbatch --export=ALL,D=<D>,NAME=a05_122,TAG=md122_a05_ep500_reassign,CKPT=final sbatch/md/reassign.sbatch
sbatch --export=ALL,D=<D>,NAME=a06_133,TAG=md133_a06_ep500_reassign,CKPT=final sbatch/md/reassign.sbatch
# direct: cd macrec/cross_index && python -u reassign_md.py --ckpt <out_dir>/final.pth --tag md122_a05_ep500_reassign
```

Writes `mmgen_runs/data/<D>/<D>.index_<tag>.json` (the index the T5 jobs read) and, in the
checkpoint dir, `codes_<tag>.npy` and `reassign_<tag>.json` (every changed item). The rule is in
A2; it has no randomness. `reassign.sbatch` then runs `md_diagnostics.py` on the new codes
(`diagnostics_<tag>.json`).

### A3.5 Stage 3: freeze and lock

```bash
python3 sbatch/tools/freeze_md_sids.py --dataset <D> --cfg 122 --version v1   # and --cfg 133
python3 sbatch/tools/freeze_md_sids.py --dataset <D> --cfg 122 --version v1 --lock
```

Copies `final.pth` → `tokenizer.pth`, the index → `index.json` (+ `index.sha256`), the reassignment
summary and diagnostics into `sids/v1/<D>/<cfg>/`; `--lock` writes `STATUS = v1 locked` and makes
every file read-only. Locked on 2026-10-28.

### A3.6 Provenance of each locked set

| set | training job / commit / GPU | reassign job / commit | items changed | index sha256 |
|---|---|---|---|---|
| Instruments 1+2+2 | 21101478 / `796d07f` / A100 | 21164545 / `29a038b` | 24 | `512221a1…` |
| Instruments 1+3+3 | 21103580 / `796d07f` / H200 | 21164547 / `29a038b` | 16 | `1fe4df38…` |
| Arts 1+2+2 | 21149486 / `3928840` / H200 | 21164549 / `29a038b` | 54 | `cbc70c4e…` |
| Arts 1+3+3 | 21149487 / `3928840` / H100 | 21164553 / `fc5e4f7` | 44 | `8612e3cb…` |
| Games 1+2+2 | 21149488 / `3928840` / H200 | 21164554 / `fc5e4f7` | 141 | `48851182…` |
| Games 1+3+3 | 21149489 / `3e026d1` / H100 | 21164561 / `fc5e4f7` | 80 | `15c68e21…` |

Full hashes are in A2. Logs: `mmgen_runs/logs/mmgen_md_tok_<job>.out`,
`mmgen_runs/logs/mmgen_md_reassign_<job>.out`. Code changes since those commits do not affect
results: `main_mdrqvae.py` changed only in its docstring, and `reassign_md.py` gained a `.detach()`
on the codebook weights (`3e026d1`), which changes no value. `models/mdrqvae.py` and
`generate_indices_md.py` are unchanged since `9fe6fc3`; `models/{rq,vq,layers}.py` are MACRec's own, unmodified.

### A3.7 Checking your set (and what we measured)

**Acceptance list for a self-trained set** (tolerances from a full rerun of all six v1 recipes,
below). After stage 2, read `train_log.jsonl` (last line), `reassign_<tag>.json` and
`diagnostics_<tag>.json` (written by `sbatch/md/reassign.sbatch`):

| check | expected |
|---|---|
| reconstruction cosine text / image at epoch 500 | within ~0.005 of v1 (v1 values in the table below) |
| codebook use | ≥ ~250 of 256 codes on every level |
| full-SID collision before reassignment | ≤ ~1.2% (Games highest) |
| after reassignment | `all_unique: true`, `shared_unchanged: true` |
| corrected first-level CMI, items with shared images excluded | ≤ ~2% of H(t₁|f) (the v1 acceptance bar, A2) |

The code numbers themselves will differ from v1; do not compare SIDs item by item.

**What we measured** (2026-10-31, `sbatch/md/repro_v1.sbatch`, all on H200; outputs in
`mmgen_runs/md/repro/`):

| check | job | result |
|---|---|---|
| full retrain + reassignment (`MODE=full`) | 21507767 | same quality as v1 on every check above; different codes |
| reassignment only, on the locked `tokenizer.pth` (`MODE=reassign`) | 21508421 | identical to v1 for all six sets (same sha256) |

| set | rec cos text / image, v1 → rerun | collision before reassignment | items reassigned |
|---|---|---|---|
| Instruments 1+2+2 | 0.9600 → 0.9606 / 0.9150 → 0.9148 | 0.38% → 0.43% | 24 → 27 |
| Instruments 1+3+3 | 0.9659 → 0.9664 / 0.9265 → 0.9266 | 0.26% → 0.26% | 16 → 16 |
| Arts 1+2+2 | 0.9677 → 0.9675 / 0.9136 → 0.9131 | 0.57% → 0.61% | 54 → 57 |
| Arts 1+3+3 | 0.9676 → 0.9714 / 0.9160 → 0.9209 | 0.47% → 0.45% | 44 → 42 |
| Games 1+2+2 | 0.9560 → 0.9538 / 0.9176 → 0.9156 | 1.02% → 1.12% | 141 → 155 |
| Games 1+3+3 | 0.9601 → 0.9604 / 0.9223 → 0.9217 | 0.58% → 0.49% | 80 → 68 |

Codebook use was 253–256 per level in both. How much the codes differ: the item grouping on the
shared level agrees with v1 at AMI 0.20–0.38, deeper levels near 0
(`mmgen_runs/md/repro/repro_v1_compare.json`). The run-to-run differences come from GPU kernels
and the first-batch k-means codebook initialization, which are not forced to be deterministic.
Whether a retrained set changes the downstream T5 numbers by more than seed noise has not been
measured yet (`docs/MDRQVAE_REPORT.md` §6).

Rebuild the exact v1 index from the locked tokenizer (stage 2 only; writes under
`mmgen_runs/md/repro/reassign/`, never into v1 or the data dir):

```bash
sbatch --export=ALL,MODE=reassign sbatch/md/repro_v1.sbatch            # all six; or SETS="Arts:122"
```

# Part B. The old Tmall implementation (design reference, `legacy/grid/`)

What the GRID code actually does, cited to `legacy/grid/` (paths below are relative to it). For how
it maps onto MACRec, see `PORTING.md`.

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
