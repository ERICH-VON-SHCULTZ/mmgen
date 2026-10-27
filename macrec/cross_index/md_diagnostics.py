"""mmgen addition: diagnostics for an MD-RQ-VAE tokenizer (the Part-2 bar in docs/MDRQVAE_LOG.md).

  1 reconstruction per modality: mean cos(x_hat, x) for MD (shared + modality codes) and for the
    single-modality tokenizers of A (text) and B (image) on the same items (their best_collision ckpt)
  2 codebook utilisation per level (+ the curve from train_log.jsonl)
  3 full-SID collision rate (before any disambiguation code)
  4 CMI(t-chain ; i-chain | f): plug-in, as a fraction of H(t|f), against a permutation null that
    shuffles the image codes among items sharing f (100 reps); first level of each chain, then the
    full chains
  5 AMI(shared code, first level of A) and AMI(shared code, first level of B)
  6 separation (Tmall TEST 1b, adapted): pairs of items that agree on two of {f, t_1, i_1} and differ
    on the third; mean raw-feature cosine distance in text and in image, each divided by its
    random-pair baseline; ratio (text/base)/(image/base) > 1 means the code moves text more.
    The same test on the levels of the early-fusion chain tokenizer E (when its index exists) is the
    reference, as the chain RQ-VAE was on Tmall.
  7 additivity: cos of D(q_f + q_m), D(q_f), D(q_m); share of items where the sum beats both parts
Usage: python md_diagnostics.py --ckpt <dir>/best.pth  -> <dir>/diagnostics.json
"""
import argparse
import collections
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_mutual_info_score

from generate_indices_md import compute_codes, load
from models.rqvae import RQVAE

TOK = "/scratch/yw8866/mmgen_runs/baselines/{D}/tok/{name}/best_collision_model.pth"


def entropy(counter, n):
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def cond_entropy(x, given):
    """H(X | G) in bits, plug-in."""
    n = len(x)
    joint = collections.Counter(zip(given, x))
    marg = collections.Counter(given)
    return entropy(joint, n) - entropy(marg, n)


def cmi(t, i, f):
    """I(T; I | F) = H(T|F) + H(I|F) - H(T,I|F), plug-in, bits."""
    return cond_entropy(t, f) + cond_entropy(i, f) - cond_entropy(list(zip(t, i)), f)


def cmi_test(t, i, f, reps=100, seed=0):
    rng = np.random.default_rng(seed)
    real = cmi(t, i, f)
    h_t = cond_entropy(t, f)
    groups = collections.defaultdict(list)
    for k, g in enumerate(f):
        groups[g].append(k)
    arr = np.empty(len(i), dtype=object)
    arr[:] = list(i)
    i = arr
    null = []
    for _ in range(reps):
        ip = i.copy()
        for idx in groups.values():
            if len(idx) > 1:
                ip[idx] = i[rng.permutation(idx)]
        null.append(cmi(t, list(ip), f))
    null = np.array(null)
    return {"cmi_plugin_bits": real, "H_t_given_f": h_t, "H_i_given_f": cond_entropy(list(i), f),
            "frac_plugin": real / h_t if h_t else None,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_p95": float(np.quantile(null, 0.95)), "p_value": float((null >= real).mean()),
            "z": float((real - null.mean()) / (null.std() + 1e-12)),
            "cmi_corrected_bits": real - float(null.mean()),
            "frac_corrected": (real - float(null.mean())) / h_t if h_t else None}


@torch.no_grad()
def single_recon(D, name, x_raw, device):
    path = TOK.format(D=D, name=name)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = RQVAE(in_dim=x_raw.shape[1], num_emb_list=a.num_emb_list, e_dim=a.e_dim, layers=a.layers,
              dropout_prob=a.dropout_prob, bn=a.bn, loss_type=a.loss_type, quant_loss_weight=a.quant_loss_weight,
              kmeans_init=a.kmeans_init, kmeans_iters=a.kmeans_iters, sk_epsilons=a.sk_epsilons, sk_iters=a.sk_iters)
    m.load_state_dict(ck["state_dict"])
    m = m.to(device).eval()
    x = torch.from_numpy(x_raw.astype(np.float32)).to(device)
    out = torch.cat([m(x[s:s + 4096], use_sk=False)[0] for s in range(0, len(x), 4096)])
    return float(F.cosine_similarity(out, x, dim=-1).mean())


def cosdist_pairs(X, pairs):
    a, b = X[pairs[:, 0]], X[pairs[:, 1]]
    return float((1 - (a * b).sum(1)).mean())


def separation(keys_other, key_target, Xt, Xi, rng, max_pairs=200000):
    """Pairs agreeing on keys_other and differing on key_target."""
    groups = collections.defaultdict(list)
    for k, ko in enumerate(keys_other):
        groups[ko].append(k)
    pairs = []
    for idx in groups.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                if key_target[idx[x]] != key_target[idx[y]]:
                    pairs.append((idx[x], idx[y]))
    if not pairs:
        return {"pairs": 0}
    pairs = np.array(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return {"pairs": int(len(pairs)), "text_dist": cosdist_pairs(Xt, pairs), "image_dist": cosdist_pairs(Xi, pairs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--reps", type=int, default=100)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, c, epoch = load(a.ckpt)
    D, S, L = c["dataset"], c["n_shared"], c["n_modal"]
    codes, xt, xi = compute_codes(model, c, dev)
    N = len(codes)
    out = {"ckpt": a.ckpt, "epoch": epoch, "config": c}
    dd = f"{c['data_root']}/{D}/{D}"
    raw_t, raw_i = np.load(f"{dd}.emb-llama-td.npy"), np.load(f"{dd}.emb-ViT-L-14.npy")

    # 1 + 7 reconstruction and additivity
    q_f, q_t, q_i = model.code_vectors(torch.from_numpy(codes).to(dev))
    parts = model.decode_parts(q_f, q_t, q_i)
    cs = {k: F.cosine_similarity(v, xt if k[0] == "t" else xi, dim=-1) for k, v in parts.items()}
    out["recon"] = {"md_text": float(cs["t_sum"].mean()), "md_image": float(cs["i_sum"].mean()),
                    "A_text": single_recon(D, "tiger_text", raw_t, dev),
                    "B_image": single_recon(D, "tiger_image", raw_i, dev)}
    out["additivity"] = {}
    for m in ("t", "i"):
        s, f_, mo = cs[f"{m}_sum"], cs[f"{m}_f"], cs[f"{m}_m"]
        out["additivity"]["text" if m == "t" else "image"] = {
            "cos_sum": float(s.mean()), "cos_shared_only": float(f_.mean()), "cos_modal_only": float(mo.mean()),
            "frac_sum_beats_both": float(((s > f_) & (s > mo)).float().mean())}
    out["shared_balance_text_minus_image"] = float((cs["t_f"] - cs["i_f"]).mean())

    # 2 + 3 utilisation, collisions
    names = [f"s{j}" for j in range(S)] + [f"t{j}" for j in range(L)] + [f"v{j}" for j in range(L)]
    out["utilization"] = dict(zip(names, [int(len(np.unique(codes[:, j]))) for j in range(codes.shape[1])]))
    tup = [tuple(r) for r in codes.tolist()]
    out["full_sid_collision"] = (N - len(set(tup))) / N
    out["max_group"] = max(collections.Counter(tup).values())
    log = f"{os.path.dirname(a.ckpt)}/train_log.jsonl"
    if os.path.exists(log):
        out["util_curve"] = [(r["epoch"], r["util"]) for r in map(json.loads, open(log))][:6]

    # 4 conditional independence
    f = [tuple(r[:S]) for r in codes.tolist()]
    t1, i1 = codes[:, S].tolist(), codes[:, S + L].tolist()
    tch = [tuple(r[S:S + L]) for r in codes.tolist()]
    ich = [tuple(r[S + L:]) for r in codes.tolist()]
    out["cmi_first_level"] = cmi_test(t1, i1, f, a.reps)
    out["cmi_full_chain"] = cmi_test(tch, ich, f, a.reps)
    out["mi_unconditional_first_level"] = cond_entropy(t1, [0] * N) + cond_entropy(i1, [0] * N) - cond_entropy(list(zip(t1, i1)), [0] * N)
    out["shared_groups"] = len(set(f))

    # 5 modality balance of the shared code
    lvl0 = lambda tag: [json.load(open(f"{dd}.index_{tag}.json"))[str(k)][0] for k in range(N)]
    A0, B0 = lvl0("lemb_tiger"), lvl0("vitemb_tiger")
    f_lab = ["-".join(map(str, x)) for x in f]
    out["ami_shared_vs_A0"] = adjusted_mutual_info_score(f_lab, A0)
    out["ami_shared_vs_B0"] = adjusted_mutual_info_score(f_lab, B0)
    out["ami_A0_vs_B0"] = adjusted_mutual_info_score(A0, B0)

    # 6 separation
    rng = np.random.default_rng(0)
    Xt = raw_t / np.linalg.norm(raw_t, axis=1, keepdims=True)
    Xi = raw_i / np.linalg.norm(raw_i, axis=1, keepdims=True)
    rp = rng.integers(0, N, (200000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
    base_t, base_i = cosdist_pairs(Xt, rp), cosdist_pairs(Xi, rp)
    def norm(r):
        if r.get("pairs"):
            r["text_over_base"], r["image_over_base"] = r["text_dist"] / base_t, r["image_dist"] / base_i
            r["normalized_ratio"] = r["text_over_base"] / r["image_over_base"]
        return r
    sep = {"base_text": base_t, "base_image": base_i,
           "shared": norm(separation(list(zip(t1, i1)), f, Xt, Xi, rng)),
           "text": norm(separation(list(zip(f, i1)), t1, Xt, Xi, rng)),
           "image": norm(separation(list(zip(f, t1)), i1, Xt, Xi, rng))}
    efile = f"{dd}.index_fnormj_tiger.json"
    if os.path.exists(efile):
        E = json.load(open(efile)); E = [E[str(k)] for k in range(N)]
        for lv in range(3):
            other = [tuple(e[j] for j in range(3) if j != lv) for e in E]
            sep[f"E_level{lv}"] = norm(separation(other, [e[lv] for e in E], Xt, Xi, rng))
    out["separation"] = sep

    json.dump(out, open(f"{os.path.dirname(a.ckpt)}/diagnostics.json", "w"), indent=1, default=str)
    brief = {k: out[k] for k in ("recon", "utilization", "full_sid_collision", "ami_shared_vs_A0", "ami_shared_vs_B0")}
    brief["cmi1"] = {k: out["cmi_first_level"][k] for k in ("cmi_plugin_bits", "null_mean", "null_std", "p_value", "frac_plugin", "frac_corrected")}
    brief["sep_ratio"] = {k: v.get("normalized_ratio") for k, v in sep.items() if isinstance(v, dict)}
    brief["additivity"] = {k: v["frac_sum_beats_both"] for k, v in out["additivity"].items()}
    print(json.dumps(brief, default=str))


if __name__ == "__main__":
    main()
