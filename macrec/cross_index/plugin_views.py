"""mmgen addition: MD-RQ-VAE SIDs as two MQL4GRec modality SIDs ("plug-in", docs/PLUGIN_REPORT.md).

Each item's locked v1 MD SID [f, t_1..t_L, i_1..i_L] is split into two views:
    text view  [f, t_1..t_L]      image view [f, i_1..i_L]
Different MD items can share a view (same [f, t...] with different image chains). Each view is made unique with
MQL4GRec's own rule, transcribed from generate_indices_single.py (= MQL4GRec index/generate_indices_distance.py),
applied to that view's codes:
  - collision groups in order of their first item id; at most 2 passes (MQL4GRec's tt == 2 stop);
  - in a group, items sorted by their distance at the view's LAST level (stable, so ties keep the lower item id);
    the first keeps its code;
  - the item at sorted position i tries last-level codes in increasing distance, starting at the i-th nearest,
    until the view SID is unused; if none, the second-to-last level is changed to its 1st, 2nd, ... nearest code
    and the last level retried from the nearest (MQL4GRec's fallback);
  - the set of used view SIDs only grows (MQL4GRec keeps the old SIDs of moved items in the set).
Distances are squared L2 between the item's residual entering that level and every code of that level, with the
residual taken along the item's ACTUAL v1 codes (MQL4GRec uses the argmin path; identical where v1 codes are the
argmin codes). "Its distance at the last level" = the distance to its own v1 code there (MQL4GRec: the min, which
is the own code for argmin codes). f is never touched (asserted); only items in a view collision are touched
(asserted); no randomness.

Token format:
  main (pure drop-in): MQL4GRec's prefixes, text <a_><b_><c_><d_>, image <A_><B_><C_><D_>, so f is <a_k> in the
                       text view and <A_k> in the image view (two different tokens);
  shared-f:            identical codes, but f is one shared token <s_k> in both views.
The locked v1 files are only read. Outputs (data dir): <D>.index_plug<cfg>_{text,image}.json (main),
<D>.index_plug<cfg>s_{text,image}.json (shared-f); provenance + every change: --out_dir/views.json.
"""
import argparse
import collections
import copy
import hashlib
import json
import os

import numpy as np
import torch

from generate_indices_md import load, prep

TEXT_P, IMAGE_P = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"], ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@torch.no_grad()
def chain_distances(model, c, codes, S, L):
    """dist[ch][lv]: (N, K) squared distances of the residual entering level lv (along the v1 codes) to every code."""
    d = f"{c['data_root']}/{c['dataset']}/{c['dataset']}"
    xt = prep(np.load(f"{d}.emb-llama-td.npy"), c["input_mode"])
    xi = prep(np.load(f"{d}.emb-ViT-L-14.npy"), c["input_mode"])
    enc = [model.encode(xt[s:s + 4096], xi[s:s + 4096]) for s in range(0, xt.shape[0], 4096)]
    assert np.array_equal(torch.cat([e["codes"] for e in enc]).numpy()[:, :S], codes[:, :S]), "f differs from the tokenizer"
    out = {}
    for ch, rq, inp, cols in (("t", model.text_rq, "v_t", range(S, S + L)), ("v", model.image_rq, "v_i", range(S + L, S + 2 * L))):
        res = torch.cat([e[inp] for e in enc])
        out[ch] = []
        for vq, j in zip(rq.vq_layers, cols):
            w = vq.embedding.weight
            out[ch].append(torch.cdist(res, w).pow(2).numpy())
            res = res - w[torch.as_tensor(codes[:, j])]
    return out


def mql_reassign(view, dist_last, dist_prev):
    """MQL4GRec's collision loop on one view. view (N, 1+L) int codes; dist_*: (N, K) at the last / 2nd-to-last level."""
    all_indices = [list(map(int, r)) for r in view]
    all_indices_str = [str(r) for r in all_indices]
    all_indices_str_set = set(all_indices_str)
    level = view.shape[1] - 1
    max_num = dist_last.shape[1]
    sort_last, sort_prev = np.argsort(dist_last, axis=1, kind="stable"), np.argsort(dist_prev, axis=1, kind="stable")
    own_last = dist_last[np.arange(len(view)), view[:, level]]
    changes, passes = [], 0
    while True:
        if len(set(all_indices_str)) == len(all_indices_str) or passes == 2:
            break
        index2id = collections.OrderedDict()
        for i, s in enumerate(all_indices_str):
            index2id.setdefault(s, []).append(i)
        groups = [g for g in index2id.values() if len(g) > 1]
        for group in groups:
            order = np.argsort(np.array([own_last[it] for it in group]), kind="stable")
            for i, m in enumerate(order):
                if i == 0:
                    continue
                item = group[m]
                ori = copy.deepcopy(all_indices[item])
                num = i
                while str(ori) in all_indices_str_set and num < max_num:
                    ori[level] = int(sort_last[item][num]); num += 1
                fallback = False
                for k in range(1, max_num):                               # second-to-last level
                    if str(ori) in all_indices_str_set:
                        ori = copy.deepcopy(all_indices[item]); ori[level - 1] = int(sort_prev[item][k]); fallback = True
                    num = 0
                    while str(ori) in all_indices_str_set and num < max_num:
                        ori[level] = int(sort_last[item][num]); num += 1
                    if str(ori) not in all_indices_str_set:
                        break
                changes.append({"item": int(item), "pass": passes, "old": all_indices[item], "new": ori,
                                "second_to_last_changed": ori[level - 1] != all_indices[item][level - 1],
                                "fallback_used": fallback, "group": list(map(int, group))})
                all_indices[item] = ori
                all_indices_str[item] = str(ori)
                all_indices_str_set.add(str(ori))
        passes += 1
    return np.array(all_indices), changes, passes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1_dir", required=True, help="mmgen_runs/md/sids/v1/<D>/<cfg>")
    p.add_argument("--tag", required=True, help="plug122 | plug133")
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    v1_index = os.path.join(a.v1_dir, "index.json")
    expect = open(os.path.join(a.v1_dir, "index.sha256")).read().split()[0]
    got = sha(v1_index)
    assert got == expect, f"v1 index hash {got} != {expect}"
    model, c, epoch = load(os.path.join(a.v1_dir, "tokenizer.pth"))
    S, L = c["n_shared"], c["n_modal"]
    assert S == 1 and L in (2, 3), (S, L)
    idx = json.load(open(v1_index))
    N = len(idx)
    codes = np.array([[int(t.split("_")[1].rstrip(">")) for t in idx[str(i)]] for i in range(N)])
    names = [t.split("_")[0] for t in idx["0"]]
    assert names == ["<s0"] + [f"<t{j}" for j in range(L)] + [f"<v{j}" for j in range(L)], names
    dist = chain_distances(model, c, codes, S, L)
    D = c["dataset"]
    data_dir = f"{c['data_root']}/{D}"
    res = {"dataset": D, "cfg": f"1+{L}+{L}", "tag": a.tag, "v1_index": v1_index, "v1_sha256": got,
           "tokenizer": os.path.join(a.v1_dir, "tokenizer.pth"), "tokenizer_epoch": epoch, "items": N,
           "rule": __doc__.split("Token format:")[0].strip(), "views": {}, "files": {}}
    views = {}
    for name, ch, cols in (("text", "t", list(range(0, 1 + L))), ("image", "v", [0] + list(range(1 + L, 1 + 2 * L)))):
        view = codes[:, cols]
        before = collections.Counter(map(tuple, view.tolist()))
        colliding = sum(n for n in before.values() if n > 1)
        new, changes, passes = mql_reassign(view, dist[ch][L - 1], dist[ch][L - 2])
        assert np.array_equal(new[:, 0], view[:, 0]), "f changed"
        moved = set(np.nonzero((new != view).any(1))[0].tolist())
        in_collision = {i for i, r in enumerate(view.tolist()) if before[tuple(r)] > 1}
        assert moved <= in_collision, "an item outside a collision group was changed"
        unique = len(set(map(tuple, new.tolist()))) == N
        views[name] = new
        res["views"][name] = {"items_in_collision_before": colliding,
                              "collision_rate_before": colliding / N,
                              "groups_before": sum(n > 1 for n in before.values()),
                              "max_group": max(before.values()),
                              "items_reassigned": len(moved), "passes": passes,
                              "second_to_last_level_changed": sum(ch_["second_to_last_changed"] for ch_ in changes),
                              "all_unique_after": unique, "changes": changes}
        assert unique, f"{name} view not unique after MQL4GRec's 2 passes"
    for variant, suffix in (("main", ""), ("shared_f", "s")):
        for name, pre in (("text", TEXT_P), ("image", IMAGE_P)):
            toks = (["<s_{}>"] + pre[1:]) if variant == "shared_f" else pre
            index = {str(i): [toks[j].format(int(k)) for j, k in enumerate(r)] for i, r in enumerate(views[name].tolist())}
            fn = f"{D}.index_{a.tag}{suffix}_{name}.json"
            path = os.path.join(data_dir, fn)
            json.dump(index, open(path, "w"), indent=1)
            res["files"][f"{variant}/{name}"] = {"file": fn, "sha256": sha(path),
                                                 "vocab": len({t for v in index.values() for t in v})}
    json.dump(res, open(os.path.join(a.out_dir, "views.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("views", "rule")}, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "changes"} for k, v in res["views"].items()}, indent=1))


if __name__ == "__main__":
    main()
