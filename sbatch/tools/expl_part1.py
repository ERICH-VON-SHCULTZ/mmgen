"""Exploration Part 1 (docs/EXPLORATION.md): where does C (MQL4GRec seed 42, merged list) win? CPU, inside the env.

Per TEST user (dataset order), hit@10 of C merged vs each of our runs: mfd122_mv at setting (ii) (eval_ii joint), the
plug-in 1+2+2 merged list, the MD chain decoder (20 beams). Lists are rebuilt as reported (plugin_paired.per_user for
C / plug-in, asserted against their results files; geo_chain_paired.ranks for the chain; ranked_test.pt for MFD).
Groups: both hit, only C, only ours, neither. Per group:
  history length (items before the target, uncapped and capped at 20); target popularity = training interactions of
  the target item (items[:-2] of every user), bucketed; target f in the history; target item itself in the history;
  our first-step rank of the target's first token (f; plug-in: min over its text / image views) and coverage
  (mfd122_mv: f in the kept 100; chain: f among the 20 final beams' f; plug-in: first token in the top 20) with
  hit@10 given coverage; C's first-step rank of the target's first-level code (min over text / image lists);
  distinctiveness = cosine distance to the nearest other item in the LLaMA text and CLIP image feature spaces.
Usage: MACREC=<macrec dir> python expl_part1.py <D> -> mmgen_runs/expl/<D>/part1.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plugin_paired  # noqa: E402
from geo_chain_paired import ranks as chain_ranks  # noqa: E402

RUNS = "/scratch/yw8866/mmgen_runs"
POP_BUCKETS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 10 ** 9)]


def nn_dist(path):
    x = torch.from_numpy(np.load(path)).float()
    x = torch.nn.functional.normalize(x, dim=1)
    best = torch.empty(len(x))
    for s in range(0, len(x), 2048):
        c = x[s:s + 2048] @ x.t()
        c[torch.arange(c.shape[0]), torch.arange(s, s + c.shape[0])] = -2
        best[s:s + 2048] = c.max(1).values
    return (1 - best).numpy()


def describe(mask, F):
    n = int(mask.sum())
    if n == 0:
        return {"users": 0}
    d = {"users": n}
    for k, v in F.items():
        x = v[mask]
        if v.dtype == bool:
            d[k] = float(x.mean())
        else:
            d[k] = {"mean": float(np.mean(x)), "median": float(np.median(x))}
    pop = F["target_train_interactions"][mask]
    d["popularity_buckets"] = {f"{a}-{b - 1}" if b < 10 ** 9 else f"{a}+": float(((pop >= a) & (pop < b)).mean())
                               for a, b in POP_BUCKETS}
    return d


def main():
    D = sys.argv[1]
    inters = json.load(open(f"{RUNS}/data/{D}/{D}.inter.json"))
    seqs = list(inters.values())
    U = len(seqs)
    idx = json.load(open(f"{RUNS}/data/{D}/{D}.index_md122_a05_ep500_reassign.json"))
    f_of = {int(k): v[0] for k, v in idx.items()}
    train_cnt = np.zeros(len(idx), dtype=int)
    for s in seqs:
        for i in s[:-2]:
            train_cnt[i] += 1
    tgt = np.array([s[-1] for s in seqs])
    hist = [s[:-1] for s in seqs]
    E = f"{RUNS}/expl/{D}"
    first = {n: torch.load(f"{E}/first_{n}.pt") for n in ("C", "plug122", "chain", "mv")}
    nn_t = nn_dist(f"{RUNS}/data/{D}/{D}.emb-llama-td.npy")
    nn_i = nn_dist(f"{RUNS}/data/{D}/{D}.emb-ViT-L-14.npy")
    # ---- hit@10 per user ----
    hitC = plugin_paired.per_user(D, "mql", "test")[0]["merged"] < 10
    hitP = plugin_paired.per_user(D, "plug122", "test")[0]["merged"] < 10
    hitCh = chain_ranks(f"{RUNS}/baselines/{D}/md122_a05_ep500_reassign/seed42", "test") < 10
    R = torch.load(f"{RUNS}/mfd/{D}/mfd122_mv/seed42/eval_ii/ranked_test.pt")
    assert np.array_equal(R["targets"].numpy(), tgt), "MFD ranked targets != dataset order"
    hitM = (R["joint"][0][:, :10] == R["targets"][:, None]).any(1).numpy()
    # ---- per-user features ----
    fr_mv = first["mv"]["seqrec"]["rank"].numpy()
    fr_ch = first["chain"]["seqrec"]["rank"].numpy()
    fr_pl = np.minimum(first["plug122"]["seqrec"]["rank"].numpy(), first["plug122"]["seqimage"]["rank"].numpy())
    c_t, c_i = first["C"]["seqrec"]["rank"].numpy(), first["C"]["seqimage"]["rank"].numpy()
    sv = json.load(open(f"{RUNS}/baselines/{D}/md122_a05_ep500_reassign/seed42/eval/test/save_seqrec_20.json"))
    B = len(sv["all_outputs"]) // len(sv["all_users"])
    ch_cov = np.zeros(U, dtype=bool)
    fx = lambda x: x.strip().replace(" ", "").split(">")[0] + ">"
    for k, u in enumerate(sv["all_users"]):
        ch_cov[u] = fx(sv["all_targets"][k]) in {fx(b) for b in sv["all_outputs"][k * B:(k + 1) * B]}
    F = {"history_len": np.array([len(h) for h in hist]),
         "history_len_capped20": np.array([min(len(h), 20) for h in hist]),
         "target_train_interactions": train_cnt[tgt],
         "target_f_in_history": np.array([f_of[t] in {f_of[h] for h in hh[-20:]} for t, hh in zip(tgt, hist)]),
         "target_item_in_history": np.array([t in set(hh) for t, hh in zip(tgt, hist)]),
         "C_first_rank_min": np.minimum(c_t, c_i), "C_first_top1": np.minimum(c_t, c_i) == 0,
         "C_first_top20": np.minimum(c_t, c_i) < 20,
         "nn_text_cos_dist": nn_t[tgt], "nn_image_cos_dist": nn_i[tgt]}
    ours = {"mfd122_mv_ii": (hitM, fr_mv, fr_mv < 100), "plug122_merged": (hitP, fr_pl, fr_pl < 20),
            "md_chain": (hitCh, fr_ch, ch_cov)}
    out = {"dataset": D, "users": U,
           "hr10": {"C_merged": float(hitC.mean())} | {k: float(v[0].mean()) for k, v in ours.items()},
           "first_tokens": {n: {t: v["first_tokens"] for t, v in first[n].items()} for n in first},
           "catalog": {"nn_text_cos_dist_median": float(np.median(nn_t)), "nn_image_cos_dist_median": float(np.median(nn_i)),
                       "train_interactions_median": float(np.median(train_cnt))},
           "comparisons": {}}
    for name, (hitX, frX, covX) in ours.items():
        G = {"both": hitC & hitX, "only_C": hitC & ~hitX, "only_ours": ~hitC & hitX, "neither": ~hitC & ~hitX}
        FX = F | {"our_first_rank": frX, "our_first_top1": frX == 0, "our_first_top20": frX < 20, "our_covered": covX}
        res = {}
        for g, m in G.items():
            d = describe(m, FX)
            if m.sum():
                d["our_hit10_given_covered"] = float(hitX[m & covX].mean()) if (m & covX).any() else None
            res[g] = d
        out["comparisons"][name] = res
        print(f"== {D}: C merged vs {name}: both {G['both'].sum()}, only C {G['only_C'].sum()}, only ours {G['only_ours'].sum()}, neither {G['neither'].sum()}")
        for g in ("only_C", "only_ours", "both"):
            d = res[g]
            print(f"   {g:9s} n={d['users']:5d} hist {d['history_len']['median']:.0f} pop med {d['target_train_interactions']['median']:.0f} "
                  f"f_in_hist {d['target_f_in_history']:.2f} our f top1 {d['our_first_top1']:.2f} top20 {d['our_first_top20']:.2f} "
                  f"covered {d['our_covered']:.2f} | C first top1 {d['C_first_top1']:.2f} top20 {d['C_first_top20']:.2f} | "
                  f"nn text {d['nn_text_cos_dist']['median']:.3f} image {d['nn_image_cos_dist']['median']:.3f}")
    os.makedirs(E, exist_ok=True)
    json.dump(out, open(f"{E}/part1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
