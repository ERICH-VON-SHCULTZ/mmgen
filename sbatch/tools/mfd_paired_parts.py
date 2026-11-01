"""Paired bootstrap for MFD Parts A-C (docs/MFD_REPORT.md), from saved per-user ranks only (no model, CPU).
Same users on both sides (dataset order; the chain's sampler order is aligned by all_users), 2,000 resamples,
seed 0, HR@10 / NDCG@10 differences with 95% percentile CIs.
Sources: MFD eval_kf20/ranked_<split>.pt (joint list), MFD Part B apply_<split>_ranks.pt (rank of the target),
chain eval/<split>/save_seqrec_20.json (beams sorted by score).
Usage (inside the env): python mfd_paired_parts.py -> mmgen_runs/mfd/paired_parts.json
"""
import json

import numpy as np
import torch

RUNS = "/scratch/yw8866/mmgen_runs"
IDX = {"122": ".index_md122_a05_ep500_reassign.json", "133": ".index_md133_a06_ep500_reassign.json"}
CHAIN = {"122": "md122_a05_ep500_reassign", "133": "md133_a06_ep500_reassign"}
NB = 2000


def targets_of(D, run, split):
    return torch.load(f"{RUNS}/mfd/{D}/{run}/seed42/eval_kf20/ranked_{split}.pt")["targets"]


def mfd_k20(D, run, split):
    R = torch.load(f"{RUNS}/mfd/{D}/{run}/seed42/eval_kf20/ranked_{split}.pt")
    ids, tg = R["joint"][0][:, :10], R["targets"]
    m = ids == tg[:, None]
    return torch.where(m.any(1), m.int().argmax(1), torch.full_like(tg, 10))


def mfd_partb(D, run, split):
    for sub in ("sweep", "sweep_applied"):
        try:
            d = torch.load(f"{RUNS}/mfd/{D}/{run}/seed42/{sub}/apply_{split}_ranks.pt")
        except FileNotFoundError:
            continue
        name = next(k for k in d if k not in ("stage1_reference", "target_f_rank"))
        return d[name].clamp(max=10)
    if split == "valid":                                                  # swept runs: valid ranks from the sweep
        ch = json.load(open(f"{RUNS}/mfd/{D}/{run}/seed42/sweep/chosen.json"))
        s = torch.load(f"{RUNS}/mfd/{D}/{run}/seed42/sweep/sweep_valid_ranks.pt")
        ci = next(i for i, c in enumerate(s["configs"]) if "mode" not in c and
                  all(c[k] == ch[k] for k in ("k_f", "w_f", "w_t", "w_i")))
        return s["ranks"][ci].clamp(max=10)
    raise FileNotFoundError(f"Part B ranks for {D}/{run}/{split}")


def chain(D, b, cfg, split, tg):
    idx = json.load(open(f"{RUNS}/data/{D}/{D}{IDX[cfg]}"))
    sid2item = {"".join(v): int(k) for k, v in idx.items()}
    sv = json.load(open(f"{RUNS}/baselines/{D}/{b}/seed42/eval/{split}/save_seqrec_20.json"))
    U = len(tg)
    assert sorted(sv["all_users"]) == list(range(U))
    r = torch.full((U,), 10, dtype=torch.long)
    for k, u in enumerate(sv["all_users"]):
        assert sid2item[sv["all_targets"][k].strip().replace(" ", "")] == int(tg[u]), "user order"
        beams = sv["all_outputs"][k * 20:(k + 1) * 20]
        sc = sv["all_scores"][k * 20:(k + 1) * 20]
        top = [sid2item.get(beams[j].strip().replace(" ", ""), -1) for j in sorted(range(20), key=lambda j: -sc[j])[:10]]
        if int(tg[u]) in top:
            r[u] = top.index(int(tg[u]))
    return r


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
    res = {}
    for D, cfgs in (("Instruments", ("122", "133")), ("Arts", ("122",))):
        for split in ("test", "valid"):
            tg = targets_of(D, "mfd122_stage1", split)
            rng = np.random.default_rng(0)
            boot = rng.integers(0, len(tg), size=(NB, len(tg)))
            R = {"chain": chain(D, CHAIN["122"], "122", split, tg), "chain_aug": chain(D, "mdmv122", "122", split, tg),
                 "mfd_k20": mfd_k20(D, "mfd122_stage1", split), "mfd_partb": mfd_partb(D, "mfd122_stage1", split),
                 "mfd_aug_k20": mfd_k20(D, "mfd122_mv", split), "mfd_aug_partb": mfd_partb(D, "mfd122_mv", split)}
            for run in ("mfd122_mv",):
                assert torch.equal(targets_of(D, run, split), tg), "user order differs between MFD runs"
            pairs = [("mfd_k20", "chain"), ("mfd_partb", "chain"), ("mfd_partb", "mfd_k20"),
                     ("mfd_aug_k20", "chain"), ("mfd_aug_partb", "chain"), ("mfd_aug_partb", "chain_aug"),
                     ("mfd_aug_k20", "mfd_k20"), ("mfd_aug_partb", "mfd_partb"), ("chain_aug", "chain")]
            if "133" in cfgs:
                tg3 = targets_of(D, "mfd133_stage1", split)
                assert torch.equal(tg3, tg)
                R["chain133"] = chain(D, CHAIN["133"], "133", split, tg)
                R["mfd133_k20"] = mfd_k20(D, "mfd133_stage1", split)
                R["mfd133_partb"] = mfd_partb(D, "mfd133_stage1", split)
                pairs += [("mfd133_k20", "chain133"), ("mfd133_partb", "chain133"), ("mfd133_partb", "mfd133_k20")]
            for a, b in pairs:
                p = paired(R[a], R[b], boot)
                res[f"{D}/{split}/{a}-{b}"] = p
                print(f"{D} {split} {a} - {b}: HR@10 {p['hr10']['a']:.4f} vs {p['hr10']['b']:.4f} "
                      f"diff {p['hr10']['diff']:+.4f} [{p['hr10']['ci95'][0]:+.4f}, {p['hr10']['ci95'][1]:+.4f}]  "
                      f"NDCG@10 diff {p['ndcg10']['diff']:+.4f} [{p['ndcg10']['ci95'][0]:+.4f}, {p['ndcg10']['ci95'][1]:+.4f}]  "
                      f"won/lost {p['wins']}/{p['losses']}", flush=True)
    json.dump(res, open(f"{RUNS}/mfd/paired_parts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
