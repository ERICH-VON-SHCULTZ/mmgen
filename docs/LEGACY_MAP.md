# LEGACY_MAP: what is in `legacy/grid/` and what to do with it

`legacy/grid/` is a read-only snapshot of the old GRID fork (Tmall, multi-behavior) plus its
`rec-tmall/` analysis scripts. Nothing in this repo imports it. See `PROVENANCE.md` for the source
commit and the artifact locations.

Status legend:
- **KEEP-AS-REFERENCE**: read this when re-implementing MD-RQ-VAE, MFD or the evaluation in MACRec.
- **SUPPORT**: generic plumbing that the reference code depends on. Read it only to understand a call.
- **OBSOLETE**: Tmall- or behavior-specific. Do not port.

All paths below are relative to `legacy/grid/`.

## Tokenizer (MD-RQ-VAE)

| path | what | status |
|---|---|---|
| `src/modules/clustering/md_rqvae.py` | `CrossAttentionFusion` + `MDRQVAE` (shared/text/image codebooks, additive decoders, loss) | **KEEP-AS-REFERENCE** |
| `src/modules/clustering/vector_quantization.py` | `VectorQuantization`, `EMAVectorQuantization` (EMA updates, dead-code revival, K-means++ init buffer) | **KEEP-AS-REFERENCE** |
| `src/components/quantization_strategies.py` | `STEQuantization` (straight-through), Gumbel and rotation-trick variants | KEEP-AS-REFERENCE (STE only) |
| `src/components/clustering_initializers.py` | `KMeansPlusPlusInitInitializer` | SUPPORT |
| `src/components/distance_functions.py` | `SquaredEuclideanDistance` | SUPPORT |
| `src/components/loss_functions.py` | `WeightedSquaredError`, `BetaQuantizationLoss`, `FocalLoss` | SUPPORT (focal is OBSOLETE) |
| `src/components/training_loop_functions.py` | world-size loss scaling during codebook init | SUPPORT |
| `src/modules/clustering/residual_quantization.py` | single-modality RQ-VAE / RVQ (the chain/TIGER tokenizer) | KEEP-AS-REFERENCE (chain baseline) |
| `src/models/modules/clustering/*` | base clustering module, mini-batch k-means (rkmeans SID) | SUPPORT |
| `configs/experiment/md_rqvae_train_flat.yaml` | MD-RQ-VAE training config (all hyperparameters) | **KEEP-AS-REFERENCE** |
| `configs/experiment/md_rqvae_inference_flat.yaml` | SID extraction config | KEEP-AS-REFERENCE |
| `configs/experiment/rqvae_*`, `rvq_*`, `rkmeans_*`, `sem_embeds_*` | single-modality SID pipelines | KEEP-AS-REFERENCE (chain) |
| `sbatchfiles/train_mdrqvae.sbatch` | launcher for the shipped 2048/1024/1024 run | KEEP-AS-REFERENCE |
| `src/utils/verify_sids.py` | codebook usage, collision rate, pkl → `semantic_ids.pt` | KEEP-AS-REFERENCE |
| `src/utils/tensor_utils.py` | `deduplicate_rows_in_tensor` (a no-op on the MD path, see MDRQVAE.md §6), `build_sid_tensor_from_pkl` | SUPPORT |
| `src/utils/codebook_embedding_init.py` | orthogonal up-projection of codebook centroids into the T5 embedding table | KEEP-AS-REFERENCE (optional) |
| `src/inference.py`, `src/utils/inference_utils.py` | Hydra inference entry, pickle writer | SUPPORT |

## Generative model (chain T5 and MFD)

| path | what | status |
|---|---|---|
| `src/models/modules/semantic_id/mfd_generation_model.py` | `MFDMultiBehaviorEncoderDecoder`: factorized decoder, PMQ/Query-f slots, compositional beam, catalog constraint, wide-catalog pool, audit dump | **KEEP-AS-REFERENCE** (behavior parts OBSOLETE, see MFD.md §7) |
| `src/models/modules/semantic_id/tiger_generation_model.py` | chain T5 models: `SemanticIDEncoderDecoder` (single-behavior TIGER), `SemanticIDMultiBehaviorEncoderDecoder` (MB chain, parent of MFD), beam search | KEEP-AS-REFERENCE (chain decoding and table layout); BM-TV, item-PE, two-view align/rerank are OBSOLETE |
| `src/models/modules/semantic_id/pba_generation_model.py` | MBGen PBA (position-routed experts and behavior injection) | OBSOLETE |
| `src/models/modules/huggingface/transformer_base_module.py`, `base_module.py` | Lightning wrappers | SUPPORT |
| `src/models/components/network_blocks/*` | MLP and embedding helpers | SUPPORT |
| `configs/experiment/tiger_mb_mfd_flat.yaml` | MFD config, with every `mfd_*` knob documented inline | **KEEP-AS-REFERENCE** |
| `configs/experiment/tiger_mb_train_flat.yaml`, `tiger_train_flat.yaml` | chain MB / single-behavior TIGER configs | KEEP-AS-REFERENCE |
| `configs/experiment/tiger_mb_pba_flat.yaml`, `tiger_mb_cmir.yaml`, `mbgen_sid_gen.yaml` | PBA / CMIR / MBGen tokenizer | OBSOLETE |
| `sbatchfiles/train_mfd_pmq_qf.sbatch` | launcher for Run-C, the shipped MFD (PMQ + Query-f) | KEEP-AS-REFERENCE |
| `sbatchfiles/train_tiger_mb_mfd.sbatch`, `train_mfd_pmq.sbatch`, `mfd_kf_sweep.sbatch` | earlier MFD variants and the K_f sweep | KEEP-AS-REFERENCE |
| `sbatchfiles/matched_compute.sbatch` + `rec-tmall/analyze_matched_compute.py` | chain vs MFD at equal decoder-state budget | KEEP-AS-REFERENCE (method) |

## Evaluation

| path | what | status |
|---|---|---|
| `src/components/eval_metrics.py` | `NDCG`, `Recall` (l.87–138), `SIDRetrievalEvaluator` (l.262) | KEEP-AS-REFERENCE (metric definitions) |
| `src/components/eval_metrics.py` `MultiBehaviorSIDRetrievalEvaluator` (l.392) | t1/t2/t3 behavior tracks | OBSOLETE |
| `rec-tmall/analyze_mfd_audit.py` | coverage@pool / P(rank<10\|cov) / f-pruned decomposition from an audit dump | KEEP-AS-REFERENCE (diagnostic method) |
| `rec-tmall/align_and_breakdown.py`, `predictable_breakdown.py`, `seed_variance.py` | paired-bootstrap and multi-seed protocol | KEEP-AS-REFERENCE (protocol only) |

## Measurement scripts behind the MD-RQ-VAE claims

| path | what | status |
|---|---|---|
| `rec-tmall/analyze_modality_decomposition.py` (run by `sbatchfiles/modality_decomposition.sbatch:54`) | additivity, normalized separation ratios, shared balance, codebook purity | **KEEP-AS-REFERENCE** |
| `rec-tmall/results/modality_decomp_report.txt` | its output (copied from `/scratch/yw8866/rec-tmall/modality_decomp/report.txt`) | KEEP-AS-REFERENCE |
| `rec-tmall/analyze_sid_behavior_coupling.py` `--part factor` (run by `sbatchfiles/analyze_sid_factor.sbatch:30`) | CMI(text;image \| shared) with shuffle bias floor | **KEEP-AS-REFERENCE** (part G only; the other parts are behavior probes, OBSOLETE) |
| `rec-tmall/results/analyze_sid_factor.out` | its output | KEEP-AS-REFERENCE |

## Data pipeline (Tmall)

| path | what | status |
|---|---|---|
| `src/data/build_tmall_tfrecords.py`, `build_tmall_tfrecords_mb.py` | Tmall logs → per-user TFRecords, pv rebalancing (user_drop / event_downsample) | OBSOLETE |
| `src/data/generate_mbgen_sid.py` | MBGen tokenizer re-implementation | OBSOLETE |
| `src/data/loading/components/collate_functions.py` | `collate_with_behavior_sid_causal_duplicate` (train), `collate_with_behavior_interleave` (eval): stride-4 `[b,f,t,i]` sequences | OBSOLETE (MACRec has its own text-token pipeline); read only to understand MFD's input layout |
| `src/data/loading/components/pre_processing.py:125` `map_sparse_id_to_semantic_id` | item id → SID tuple lookup | SUPPORT |
| `src/data/loading/components/label_function.py:103` `NextKTokenMasking` | next-item label slicing | SUPPORT |
| `src/data/loading/*` (iterators, datamodule, interfaces) | TFRecord/text iterators, Lightning DataModule | OBSOLETE |
| `rec-tmall/download_image.py` | Tmall image download | OBSOLETE |

## Behavior modeling, rerankers, history forcing, Tmall analyses: all OBSOLETE

| path | what |
|---|---|
| `mfd_generation_model.py`: `mfd_behavior_marginalized`, `mfd_behavior_weight`, behavior slot / `behavior_mlp`, `generate_conditioned` (t2) | behavior marginalization / conditioning |
| `mfd_generation_model.py`: `mfd_force_history` (l.95, 165–171, 939–947, 975–985) | history forcing |
| `mfd_generation_model.py`: `mfd_dump_memory`, `mfd_audit_*`, `mfd_learned_fusion`, `mfd_branch_contrastive`, `use_cmir` | reranker dumps, learned fusion, side experiments |
| `rec-tmall/train_mfd_reranker.py`, `train_madr.py`, `chain_rescore.py`, `cf_rerank.py`, `bpf_rerank.py`, `nip_*.py`, `inject_rerank.py`, `sweep*_rerank.py`, `matched_rerank_compare.py`, `promote_fusion.py`, `tau_select.py` | second-stage rerankers and their sweeps |
| `rec-tmall/branch_calibrate.py`, `branch_realloc.py`, `coverage_expansion.py`, `convert_coverage.py`, `kpool_sweep.py`, `cap_eval*.py`, `headroom.py`, `oracle_gate.py`, `topk_diag.py` | pool / coverage experiments (the methods are reusable, the code is Tmall-bound) |
| `rec-tmall/revisit.py`, `time_heuristics.py`, `ts_sidecar.py`, `content_heuristics.py`, `bias_eval.py`, `gate_audit.py`, `head_sharpening.py`, `component_audit.py` | repeat-item and behavior analyses |
| `rec-tmall/analyze*.py` (except the two above), `debug_behavior_strings.py`, `analyze_tfrecords_mb.py` | Tmall data analyses |
| `rec-tmall/GOAL.md` | full research log for the "MFD beats chain by +0.01" goal (concluded: not reached; best lead +0.0024–0.0025) |
| `src/utils/restart_job*.py` | Slurm requeue callbacks |
| `sbatchfiles/*` other than those listed above | one launcher per experiment above |
| `AGENTS.md` | the old project handbook (§15 MD-RQ-VAE, §16 MFD, §17 reranker); a useful narrative, but every claim in `docs/` is re-cited to code |
