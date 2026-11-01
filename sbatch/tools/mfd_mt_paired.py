"""Paired bootstrap for MFD with C's task mix (docs/MFD_REPORT.md §13), from saved per-user ranks only (CPU).

Lists (joint MFD list, rank of the target, clamped to 10):
  (i)   K_f=20, w=1          <run>/eval_kf20/ranked_<split>.pt
  (ii)  K_f=100, w=1/.5/.5   <run>/eval_ii/ranked_<split>.pt
  (iii) valid-chosen grid    <run>/sweep/apply_test_ranks.pt (test) / sweep_valid_ranks.pt (valid)
mfd122_mv at (i) and (ii); its (ii) is the current best MFD and the reference for (iii).
Plug-in 1+2+2 merged list: rebuilt from saved beams with MACRec's own code (plugin_paired.per_user, asserted against
its results files); users aligned by dataset index and checked item by item against the MFD targets.
2,000 resamples of the same users, seed 0, 95% percentile CI. Usage (inside the env, MACREC=<frozen macrec copy>,
plugin_paired.py next to this file): python mfd_mt_paired.py [D] -> mmgen_runs/mfd/<D>/mt_paired.json
"""
import json
import os
import sys

import numpy as np
import torch

import plugin_paired

RUNS = "/scratch/yw8866/mmgen_runs"
NB = 2000


def ranked(D, run, sub, split):
    R = torch.load(f"{RUNS}/mfd/{D}/{run}/seed42/{sub}/ranked_{split}.pt")
    ids, tg = R["joint"][0][:, :10], R["targets"]
    m = ids == tg[:, None]
    return torch.where(m.any(1), m.int().argmax(1), torch.full_like(tg, 10)), tg


def swept(D, run, split):
    base = f"{RUNS}/mfd/{D}/{run}/seed42/sweep"
    ch = json.load(open(f"{base}/chosen.json"))
    if split == "test":
        return torch.load(f"{base}/apply_test_ranks.pt")[ch["name"]].clamp(max=10), ch["name"]
    s = torch.load(f"{base}/sweep_valid_ranks.pt")
    ci = next(i for i, c in enumerate(s["configs"]) if "mode" not in c and all(c[k] == ch[k] for k in ("k_f", "w_f", "w_t", "w_i")))
    return s["ranks"][ci].clamp(max=10), ch["name"]


def plugin_merged(D, split, tg):
    ranks, _ = plugin_paired.per_user(D, "plug122", split)
    sv = json.load(open(f"{RUNS}/baselines/{D}/plug122/seed42/eval/{split}/save_seqrec_20.json"))
    idx = json.load(open(f"{RUNS}/data/{D}/{D}.index_plug122_text.json"))
    s2i = {"".join(v): int(k) for k, v in idx.items()}
    for k, u in enumerate(sv["all_users"]):
        assert s2i[sv["all_targets"][k].strip().replace(" ", "")] == int(tg[u]), ("user order", D, split, u)
    return torch.tensor(np.minimum(ranks["merged"], 10))


def ndcg(r):
    return torch.where(r < 10, 1.0 / torch.log2(r.float() + 2), torch.zeros(r.shape))


def paired(a, b, boot):
    out = {}
    for name, f in (("hr10", lambda r: (r < 10).float()), ("ndcg10", ndcg)):
        d = (f(a) - f(b)).numpy()
        bs = d[boot].mean(1)
        out[name] = dict(a=float(f(a).mean()), b=float(f(b).mean()), diff=float(d.mean()),
                         ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out["wins"], out["losses"] = int(((a < 10) & (b >= 10)).sum()), int(((a >= 10) & (b < 10)).sum())
    return out


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "Instruments"
    res = {"dataset": D, "comparisons": {}}
    for split in ("test", "valid"):
        L = {}
        L["mv (i)"], tg = ranked(D, "mfd122_mv", "eval_kf20", split)
        L["mv (ii)"], tg2 = ranked(D, "mfd122_mv", "eval_ii", split)
        assert torch.equal(tg, tg2)
        VS = [X for X in "ABC" if os.path.exists(f"{RUNS}/mfd/{D}/mfd122_mt{X}/seed42/sweep/apply_test_ranks.pt")]
        for X in VS:
            r = f"mfd122_mt{X}"
            L[f"{X} (i)"], t = ranked(D, r, "eval_kf20", split); assert torch.equal(t, tg)
            L[f"{X} (ii)"], t = ranked(D, r, "eval_ii", split); assert torch.equal(t, tg)
            L[f"{X} (iii)"], res[f"{X}_chosen"] = swept(D, r, split)
        L["plug-in 1+2+2 merged"] = plugin_merged(D, split, tg)
        boot = np.random.default_rng(0).integers(0, len(tg), size=(NB, len(tg)))
        pairs = [(f"{X} ({s})", f"mv ({s})") for X in VS for s in ("i", "ii")] + \
                [(f"{X} (iii)", "mv (ii)") for X in VS] + \
                [(f"{X} ({s})", "plug-in 1+2+2 merged") for X in VS for s in ("i", "ii", "iii")] + \
                [(f"{X} (ii)", "A (ii)") for X in VS if X != "A"] + [("mv (ii)", "plug-in 1+2+2 merged")]
        for a, b in pairs:
            p = paired(L[a], L[b], boot)
            res["comparisons"].setdefault(f"{a} vs {b}", {})[split] = p
            print(f"{split} {a:>8s} vs {b:<22s}: HR@10 {p['hr10']['a']:.4f} vs {p['hr10']['b']:.4f} "
                  f"diff {p['hr10']['diff']:+.4f} [{p['hr10']['ci95'][0]:+.4f}, {p['hr10']['ci95'][1]:+.4f}]  "
                  f"NDCG@10 diff {p['ndcg10']['diff']:+.4f} [{p['ndcg10']['ci95'][0]:+.4f}, {p['ndcg10']['ci95'][1]:+.4f}]  "
                  f"won/lost {p['wins']}/{p['losses']}", flush=True)
    json.dump(res, open(f"{RUNS}/mfd/{D}/mt_paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
