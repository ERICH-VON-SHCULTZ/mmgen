"""Independent re-implementation of the baselines' ranking metrics, written from scratch so that
our own pipeline (MD-RQ-VAE chain decoder, later MFD) can be checked against the MACRec/MQL4GRec
evaluation (test_ddp_save.py + evaluate.py, ensemble.py) on saved predictions.

Protocol reproduced (leave-one-out, one target per user):
- single list: the 20 beam outputs of a user, SID strings with spaces removed, sorted by score
  descending (stable, so ties keep beam order); HR@k = target among the first k; NDCG@k = 1/log2(r+1).
- merged list (MQL4GRec / MACRec merge): map each SID to its item; walk text_1, image_1, text_2,
  image_2, ...; an item seen again gets (old + new) / 2 + 1, otherwise it keeps its score; an
  unmapped SID gets score -1000; rank items by score descending (stable, first-seen order).

Usage:
  python3 rank_metrics.py --save save_seqrec_20.json [--results results_seqrec_20.json]
  python3 rank_metrics.py --save save_seqrec_20.json --save_image save_seqimage_20.json \
      --index <D>.index_lemb_x.json --image_index <D>.index_vitemb_x.json [--results results_ensemble_20.json]
With --results, exits non-zero unless every metric matches exactly (|diff| < 1e-12)."""
import argparse
import json
import math
import sys

KS_HIT, KS_NDCG = (1, 5, 10), (5, 10)


def metrics_from_ranks(ranks, n):
    """ranks: 1-based rank of the target per user, or None if absent."""
    out = {}
    for k in KS_HIT:
        out[f"hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    for k in KS_NDCG:
        out[f"ndcg@{k}"] = sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / n
    return out


def _clean(s):
    return s.strip().replace(" ", "")


def single_list(save, beams=20):
    outs, scores, targets = save["all_outputs"], save["all_scores"], save["all_targets"]
    ranks = []
    for u, tgt in enumerate(targets):
        cand = [(_clean(o), sc) for o, sc in zip(outs[u * beams:(u + 1) * beams], scores[u * beams:(u + 1) * beams])]
        order = sorted(range(len(cand)), key=lambda j: -cand[j][1])          # stable
        ranked = [cand[j][0] for j in order]
        ranks.append(ranked.index(tgt) + 1 if tgt in ranked else None)
    return metrics_from_ranks(ranks, len(targets))


def merged_list(save_t, save_i, idx_t, idx_i, beams=20):
    sid2item_t = {"".join(v): int(k) for k, v in idx_t.items()}
    sid2item_i = {"".join(v): int(k) for k, v in idx_i.items()}
    by_user_t = {u: b for b, u in enumerate(save_t["all_users"])}
    by_user_i = {u: b for b, u in enumerate(save_i["all_users"])}
    assert set(by_user_t) == set(by_user_i)
    ranks = []
    for u in sorted(by_user_t):
        bt, bi = by_user_t[u], by_user_i[u]
        target = sid2item_t[save_t["all_targets"][bt]]
        assert target == sid2item_i[save_i["all_targets"][bi]]
        score, first = {}, []
        for j in range(beams):
            for save, b, m in ((save_t, bt, sid2item_t), (save_i, bi, sid2item_i)):
                item = m.get(_clean(save["all_outputs"][b * beams + j]), -1)
                sc = save["all_scores"][b * beams + j] if item != -1 else -1000
                if item in score and item != -1:
                    score[item] = (sc + score[item]) / 2 + 1
                else:
                    if item not in score:
                        first.append(item)
                    score[item] = sc
        ranked = sorted(first, key=lambda it: -score[it])                    # stable on first-seen order
        ranks.append(ranked.index(target) + 1 if target in ranked else None)
    return metrics_from_ranks(ranks, len(ranks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", required=True)
    p.add_argument("--save_image")
    p.add_argument("--index")
    p.add_argument("--image_index")
    p.add_argument("--results", help="reference metrics JSON to compare against")
    a = p.parse_args()
    if a.save_image:
        m = merged_list(json.load(open(a.save)), json.load(open(a.save_image)),
                        json.load(open(a.index)), json.load(open(a.image_index)))
    else:
        m = single_list(json.load(open(a.save)))
    print(json.dumps(m))
    if a.results:
        ref = json.load(open(a.results))
        ref = ref.get("mean_results", ref)
        diffs = {k: m[k] - ref[k] for k in m}
        ok = all(abs(d) < 1e-12 for d in diffs.values())
        print("reference", json.dumps({k: ref[k] for k in m}))
        print("MATCH" if ok else f"MISMATCH {diffs}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
