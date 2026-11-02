"""Data check for the next f code, no training (docs/MFD_REPORT.md §14). CPU, inside the env.

Is content geometry (MD-RQ-VAE f codebook distances) or behavioural co-occurrence the better signal for the target
item's f, given the f codes of the user's history? Splits as MACRec (data.py): history = the last 20 items before the
target; test target items[-1], valid target items[-2]. Co-occurrence counts use TRAINING sequences only (items[:-2] of
every user: consecutive transitions f(i_t) -> f(i_t+1)), so valid / test targets never enter them.

1. Distances: for each user, squared codebook distance from the target's f to the history's f codes, nearest and mean,
   vs the same for a random f (uniform over the f codes, and drawn from the training-target f distribution). Also the
   share of users whose target f already occurs in the history, and the same distances on the other users.
2. Next-f predictors, each ranking all f codes per user (ties broken at random, seed 0): top-1/5/10/20 hit of the target f.
     geo_nearest  -min_h d2(f', h)          geo_mean  -mean_h d2(f', h)
     cooc_all     sum_h P(f' | h)           cooc_last P(f' | last history f)    (P = row-normalised training transitions)
     repeat       count of f' in the history (recency breaks ties)             pop  training next-f frequency
   Each also on the users whose target f is NOT in the history (where only a real signal can help).
3. Soft-label temperature: for the f codebook, the eps mass on the 5 and 10 nearest neighbours of q(f') =
   softmax_{f' != f*}(-d2/T), mean over f*, at the T used (median 10th-NN distance) and at the T that put 50%, 70%, 90%
   of the eps mass on the 10 nearest neighbours (bisection).
Usage: python geo_data_check.py <D> -> mmgen_runs/geo/<D>/data_check.json
"""
import json
import sys

import numpy as np
import torch

RUNS = "/scratch/yw8866/mmgen_runs"
KS = (1, 5, 10, 20)


def load(D):
    idx = json.load(open(f"{RUNS}/data/{D}/{D}.index_md122_a05_ep500_reassign.json"))
    f_of = {int(k): int(v[0].split("_")[1][:-1]) for k, v in idx.items()}      # <s0_k>
    inters = json.load(open(f"{RUNS}/data/{D}/{D}.inter.json"))
    sd = torch.load(f"{RUNS}/md/sids/v1/{D}/1+2+2/tokenizer.pth", map_location="cpu", weights_only=False)["state_dict"]
    C = sd["shared_rq.vq_layers.0.embedding.weight"].float()
    return f_of, inters, C


def topk_hits(scores, target, rng):
    """scores (U, K), target (U,): rank of the target with random tie-breaking; returns ranks (0 = top-1)."""
    noise = rng.random(scores.shape) * 1e-9
    s = scores + noise
    return (s > s[np.arange(len(target)), target][:, None]).sum(1)


def summ(r, mask=None):
    r = r if mask is None else r[mask]
    return {f"top{k}": float((r < k).mean()) for k in KS} | {"mean_rank": float(r.mean() + 1), "users": int(len(r))}


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "Instruments"
    f_of, inters, C = load(D)
    K = C.shape[0]
    d2 = torch.cdist(C, C).pow(2).numpy()
    off = ~np.eye(K, dtype=bool)
    trans = np.zeros((K, K))
    nxt = np.zeros(K)
    for items in inters.values():
        fs = [f_of[i] for i in items[:-2]]
        for a, b in zip(fs[:-1], fs[1:]):
            trans[a, b] += 1
            nxt[b] += 1
    P = trans / np.maximum(trans.sum(1, keepdims=True), 1)
    pop = nxt / nxt.sum()
    out = {"dataset": D, "f_codes": K, "train_transitions": int(trans.sum()),
           "transition_self_share": float(np.trace(trans) / trans.sum()),
           "mean_sq_dist_all_f_pairs": float(d2[off].mean()), "splits": {}}
    rng = np.random.default_rng(0)
    for split, t_off in (("valid", 2), ("test", 1)):
        hist, tgt = [], []
        for items in inters.values():
            h = items[:-t_off][-20:] if t_off == 1 else items[:-2][-20:]
            hist.append([f_of[i] for i in h])
            tgt.append(f_of[items[-t_off]])
        tgt = np.array(tgt)
        U = len(tgt)
        in_hist = np.array([t in h for t, h in zip(tgt, hist)])
        near = np.array([d2[t, h].min() for t, h in zip(tgt, hist)])
        mean = np.array([d2[t, h].mean() for t, h in zip(tgt, hist)])
        ru = rng.integers(0, K, U)
        rp = rng.choice(K, U, p=pop)
        near_u = np.array([d2[t, h].min() for t, h in zip(ru, hist)])
        mean_u = np.array([d2[t, h].mean() for t, h in zip(ru, hist)])
        near_p = np.array([d2[t, h].min() for t, h in zip(rp, hist)])
        mean_p = np.array([d2[t, h].mean() for t, h in zip(rp, hist)])
        nh = ~in_hist
        dist = {"target_f_in_history": float(in_hist.mean()),
                "random_uniform_f_in_history": float(np.mean([t in h for t, h in zip(ru, hist)])),
                "random_pop_f_in_history": float(np.mean([t in h for t, h in zip(rp, hist)])),
                "all_users": {"target": {"nearest": float(near.mean()), "mean": float(mean.mean())},
                              "random_uniform": {"nearest": float(near_u.mean()), "mean": float(mean_u.mean())},
                              "random_pop": {"nearest": float(near_p.mean()), "mean": float(mean_p.mean())}},
                "users_target_f_not_in_history": {
                    "users": int(nh.sum()),
                    "target": {"nearest": float(near[nh].mean()), "mean": float(mean[nh].mean())},
                    "random_uniform_not_in_history": {"nearest": float(near_u[[t not in h for t, h in zip(ru, hist)]].mean())},
                    "random_pop_not_in_history": {"nearest": float(near_p[[t not in h for t, h in zip(rp, hist)]].mean())}}}
        S = {n: np.zeros((U, K)) for n in ("geo_nearest", "geo_mean", "cooc_all", "cooc_last", "repeat", "pop")}
        for u, h in enumerate(hist):
            S["geo_nearest"][u] = -d2[:, h].min(1)
            S["geo_mean"][u] = -d2[:, h].mean(1)
            S["cooc_all"][u] = P[h].sum(0)
            S["cooc_last"][u] = P[h[-1]]
            for j, x in enumerate(h):
                S["repeat"][u, x] += 1 + 1e-3 * j      # count, recency breaks ties
            S["pop"][u] = pop
        S["cooc_all_plus_repeat"] = S["cooc_all"] + 1e3 * (S["repeat"] > 0)
        pred = {}
        for n, s in S.items():
            r = topk_hits(s, tgt, rng)
            pred[n] = {"all_users": summ(r), "target_f_not_in_history": summ(r, nh)}
        out["splits"][split] = {"users": U, "distances": dist, "predictors": pred}
        print(f"== {D} {split}: {U} users, target f in history {in_hist.mean():.3f}")
        print(f"   nearest / mean sq dist  target {near.mean():.2f} / {mean.mean():.2f}   random uniform {near_u.mean():.2f} / "
              f"{mean_u.mean():.2f}   random pop {near_p.mean():.2f} / {mean_p.mean():.2f}   (mean f pair {d2[off].mean():.2f})")
        print(f"   target f not in history ({nh.sum()} users): nearest {near[nh].mean():.2f} vs random uniform "
              f"{dist['users_target_f_not_in_history']['random_uniform_not_in_history']['nearest']:.2f}")
        for n, v in pred.items():
            a, b = v["all_users"], v["target_f_not_in_history"]
            print(f"   {n:20s} top1 {a['top1']:.4f} top5 {a['top5']:.4f} top10 {a['top10']:.4f} top20 {a['top20']:.4f}"
                  f" | not-in-history top1 {b['top1']:.4f} top10 {b['top10']:.4f} top20 {b['top20']:.4f}")
    # ---- 3. soft-label temperature ----
    order = np.argsort(np.where(off, d2, np.inf), axis=1)

    def mass(T):
        lg = np.where(off, -d2 / T, -np.inf)
        q = np.exp(lg - lg.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        qs = np.take_along_axis(q, order, 1)
        return float(qs[:, :5].sum(1).mean()), float(qs[:, :10].sum(1).mean())

    T0 = float(np.median(np.sort(np.where(off, d2, np.inf), 1)[:, 9]))   # 10th nearest neighbour (self excluded)
    temps = {"used (median 10th-NN sq dist)": T0}
    for target in (0.5, 0.7, 0.9):
        lo, hi = 1e-3, T0
        for _ in range(100):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if mass(mid)[1] > target else (lo, mid)
        temps[f"10NN mass {target}"] = (lo * hi) ** 0.5
    out["soft_label_temperature"] = {}
    print(f"== soft-label temperature ({D}, f codebook)")
    for name, T in temps.items():
        m5, m10 = mass(T)
        out["soft_label_temperature"][name] = {"T": T, "T_over_used": T / T0, "eps_mass_5nn": m5, "eps_mass_10nn": m10}
        print(f"   {name:32s} T {T:7.3f} ({T / T0:.3f} x used)  eps mass on 5-NN {m5:.3f}, on 10-NN {m10:.3f}")
    json.dump(out, open(f"{RUNS}/geo/{D}/data_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
