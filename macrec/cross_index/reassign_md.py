"""mmgen addition: make every MD-RQ-VAE SID unique by distance-based reassignment (no extra token).

Rule (decision 2026-10-27; docs/MDRQVAE.md "official SID"):
 1. The shared code(s) of an item never change.
 2. Only items in a full-SID collision group are touched. In each group the item with the lowest
    total quantization distance to its own codes (sum over all levels of ||residual - code||^2;
    ties: lower item id) keeps its SID.
 3. Every other item of the group (processed in increasing item id) changes exactly one code, on the
    LAST level of the text chain or of the image chain: the replacement code with the smallest
    increase in that level's quantization distance (||r - e_new||^2 - ||r - e_own||^2, r = the
    item's residual entering that level), over both chains, among codes whose new full SID is not
    already used (by any item, including items reassigned earlier).
 4. Only if no such code exists on either last level: the second-to-last level of a chain, same
    criterion (the later levels keep their codes, as in MQL4GRec's distance reassignment).
 5. Deterministic: no randomness; order = groups by their first item id, items by id.
Writes <D>.index_<tag>.json (S + 2L tokens, no extra token), <out>/codes_<tag>.npy and
<out>/reassign_<tag>.json (every change: item, chain, level, old/new code, distance increase).
"""
import argparse
import collections
import json
import os

import numpy as np
import torch

from generate_indices_md import compute_codes, load


@torch.no_grad()
def level_residuals(model, xt, xi):
    """Per chain ('t', 'v'): list over levels of (residual entering the level, codebook)."""
    out = {"t": [], "v": [], "s": []}
    bs = 4096
    enc = [model.encode(xt[s:s + bs], xi[s:s + bs]) for s in range(0, xt.shape[0], bs)]
    for key, rq, inp in (("s", model.shared_rq, "z_f"), ("t", model.text_rq, "v_t"), ("v", model.image_rq, "v_i")):
        res = torch.cat([e[inp] for e in enc])
        for vq in rq.vq_layers:
            w = vq.embedding.weight
            idx = torch.cdist(res, w).argmin(-1)
            out[key].append((res.clone(), w))
            res = res - w[idx]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True, help="new index tag, e.g. md122_a05_reassign")
    a = p.parse_args()
    model, c, epoch = load(a.ckpt)
    model = model.cuda()
    codes, xt, xi = compute_codes(model, c)
    S, L, N = c["n_shared"], c["n_modal"], len(codes)
    R = level_residuals(model, xt, xi)
    # distances of every item to every code at every level: dist[chain][level] (N, K)
    dist = {k: [torch.cdist(r, w.detach()).pow(2).cpu().numpy() for r, w in v] for k, v in R.items()}
    col = {"s": list(range(S)), "t": list(range(S, S + L)), "v": list(range(S + L, S + 2 * L))}
    own = np.zeros(N)
    for k in ("s", "t", "v"):
        for lv, j in enumerate(col[k]):
            own += dist[k][lv][np.arange(N), codes[:, j]]
    assert np.array_equal(np.stack([dist[k][lv].argmin(1) for k in ("s", "t", "v") for lv in range(len(col[k]))], 1), codes), \
        "recomputed argmin codes differ from the tokenizer's codes"

    new = codes.copy()
    used = collections.Counter(tuple(r) for r in new.tolist())
    groups = collections.defaultdict(list)
    for i, r in enumerate(new.tolist()):
        groups[tuple(r)].append(i)
    changes = []
    for sid in sorted((g for g in groups if len(groups[g]) > 1), key=lambda g: min(groups[g])):
        members = sorted(groups[sid], key=lambda i: (own[i], i))
        for i in sorted(members[1:]):
            done = False
            for depth in range(L):                      # 0 = last level, 1 = second-to-last, ...
                best = None
                for ch in ("t", "v"):
                    lv = L - 1 - depth
                    j = col[ch][lv]
                    d = dist[ch][lv][i]
                    for k in np.argsort(d, kind="stable"):
                        if k == new[i, j]:
                            continue
                        cand = new[i].copy(); cand[j] = k
                        if used[tuple(cand)] == 0:
                            inc = float(d[k] - d[new[i, j]])
                            if best is None or inc < best[0]:
                                best = (inc, ch, lv, j, int(k))
                            break                       # nearest free code of this chain level
                if best is not None:
                    inc, ch, lv, j, k = best
                    used[tuple(new[i])] -= 1
                    changes.append({"item": i, "chain": {"t": "text", "v": "image"}[ch], "level": lv,
                                    "is_last_level": lv == L - 1, "old_code": int(new[i, j]), "new_code": k,
                                    "distance_increase": inc, "group": list(map(int, members))})
                    new[i, j] = k
                    used[tuple(new[i])] += 1
                    done = True
                    break
            assert done, f"item {i}: no free code on any modality level"
    assert np.array_equal(new[:, :S], codes[:, :S]), "a shared code changed"
    tup = [tuple(r) for r in new.tolist()]
    assert len(set(tup)) == N, "SIDs are not unique after reassignment"

    names = [f"s{j}" for j in range(S)] + [f"t{j}" for j in range(L)] + [f"v{j}" for j in range(L)]
    index = {str(i): [f"<{n}_{k}>" for n, k in zip(names, r)] for i, r in enumerate(new.tolist())}
    path = f"{c['data_root']}/{c['dataset']}/{c['dataset']}.index_{a.tag}.json"
    json.dump(index, open(path, "w"), indent=1)
    out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    np.save(f"{out_dir}/codes_{a.tag}.npy", new)
    incs = [ch["distance_increase"] for ch in changes]
    summary = {"ckpt": a.ckpt, "epoch": epoch, "index_file": path, "items": N, "sid_len": S + 2 * L,
               "collision_before": (N - len(groups)) / N, "groups_with_collision": sum(len(g) > 1 for g in groups.values()),
               "items_changed": len(changes),
               "changed_by": dict(collections.Counter(f"{ch['chain']}_level{ch['level']}" for ch in changes)),
               "distance_increase_mean": float(np.mean(incs)) if incs else 0.0,
               "distance_increase_max": float(np.max(incs)) if incs else 0.0,
               "own_distance_mean_all_items": float(own.mean()),
               "all_unique": True, "shared_unchanged": True, "deterministic": "no randomness; groups by first item id, items by id",
               "changes": changes}
    json.dump(summary, open(f"{out_dir}/reassign_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "changes"}))


if __name__ == "__main__":
    main()
