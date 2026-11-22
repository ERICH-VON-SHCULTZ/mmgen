"""mmgen addition (docs/MFD_REPORT.md §23): P1a-opt == P1a on the full test set (one dataset per call, batch 64, K_f 20 / 50 / 100).

Per batch the encoder output is computed ONCE and decoded twice: by P1a's reference path (mfd_mv_infer.f_logprobs +
branch_scores + infer's plain-cfg score, unchanged code) and by r23_p1aopt.P1aOpt (exact) — so the comparison isolates the
decode (P1a's item-level encoder itself is not bit-reproducible: scatter_add noise ~1e-5, measured as the noise floor below).
Reported per K_f:
  - same kept f codes, same candidate set (user, item) per user;
  - max |score_opt - score_ref| over aligned candidates;
  - match rate: share of users whose top-10 list (ids, in order) is identical; same target rank (capped at 10);
  - HR@10 / NDCG@10 of both; for users whose lists differ: the largest reference-score gap between the items that swap
    (rounding-level near-ties if ~1e-5);
  - depth-3 duplicate prefixes (expected 0);
  - the restricted-normaliser variant (P1aoptr, not exact): its match rate vs the reference;
  - noise floor (K_f 50): reference vs reference with a fresh encoder pass.
Out: mmgen_runs/timing/r23/<D>/p1aopt_verify.json; exit 1 unless every exact check passes.
"""
import argparse
import json
import math
import os

import torch

import r23_timing as RT
from mfd_eval import rank_lists
from mfd_mv_infer import branch_scores, encode, f_logprobs
from r23_p1aopt import dedupe_check


def ref_decode(m, enc, mask, K):
    """= mfd_mv_infer.infer with the plain cfg (a = b = (1, 0, 0), w = .5/.5), on a given encoder output."""
    lpf = f_logprobs(m.model, m.cat, enc, mask)
    fterm = sum(a * x for a, x in zip((1,), (lpf,)))
    Kk = min(K, m.cat.nF)
    fsel = fterm.topk(Kk, dim=1).indices
    user, item, frank, st, si = branch_scores(m.model, m.cat, enc, mask, fsel, m.T, m.I)
    fv = fterm.gather(1, fsel).reshape(-1)[user * Kk + frank]
    score = fv + sum(b * (0.5 * st + 0.5 * si) for b in (1,))
    return user, item, score, fsel


def lists(m, user, item, score, B):
    ids, sc = rank_lists(user, item, score, B, m.cat.N)
    return ids[:, :10], sc[:, :10]


def metrics(top, tg):
    hit = (top == tg[:, None])
    r = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(tg, 10))
    return r, float((r < 10).float().mean()), float(torch.where(r < 10, 1 / torch.log2(r.float() + 2), torch.zeros_like(r, dtype=torch.float)).mean())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--Ks", default="20,50,100")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=64)
    a = p.parse_args()
    D = a.dataset
    out = {"env": RT.env_info(), "dataset": D, "batch": a.bs, "K": {}}
    f = f"{RT.OUT_DIR}/{D}/p1aopt_verify.json"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    ok_all = True
    for K in [int(x) for x in a.Ks.split(",")]:
        m = RT.MFDMethod(D, f"P1aopt:{K}")
        order = list(range(m.U))[: a.limit or None]
        tg_all = torch.tensor(m.targets)
        acc = dict(users=0, same_fsel=0, same_cand_users=0, same_top10=0, same_rank=0, same_top10_r=0, dup=0, max_abs=0.0,
                   max_gap_diff_users=0.0, noise_same_top10=0, noise_users=0)
        R = {"ref": [], "opt": [], "optr": []}
        for s0 in range(0, len(order), a.bs):
            idx = order[s0:s0 + a.bs]
            ids, mask = m.collate(idx)
            ids, mask = ids.cuda(), mask.cuda()
            B = len(idx)
            enc, mk = encode(m.model, ids, mask)
            ur, ir, sr, fr = ref_decode(m, enc, mk, K)
            uo, io, so = m.popt.infer(ids, mask, K, enc=(enc, mk))
            uq, iq, sq = m.popt.infer(ids, mask, K, enc=(enc, mk), restrict=True)
            Kk = fr.shape[1]
            lpf_o = m.popt.f_logprobs(enc, mk)
            fo = lpf_o.topk(Kk, dim=1).indices
            acc["same_fsel"] += int((fo == fr).all(1).sum())
            acc["dup"] += dedupe_check(m.cat, fr)
            kr = ur * 10 ** 7 + ir; ko = uo * 10 ** 7 + io
            orr, oo = torch.argsort(kr), torch.argsort(ko)
            if torch.equal(kr[orr], ko[oo]):
                acc["same_cand_users"] += B
                acc["max_abs"] = max(acc["max_abs"], float((sr[orr] - so[oo]).abs().max()))
            else:                                                   # per-user candidate-set equality
                for b in range(B):
                    a1 = torch.sort(ir[ur == b]).values; a2 = torch.sort(io[uo == b]).values
                    acc["same_cand_users"] += int(torch.equal(a1, a2))
            tr, scr = lists(m, ur, ir, sr, B)
            to, _ = lists(m, uo, io, so, B)
            tq, _ = lists(m, uq, iq, sq, B)
            tg = tg_all[idx].cuda()
            same = (tr == to).all(1)
            acc["same_top10"] += int(same.sum()); acc["same_top10_r"] += int((tr == tq).all(1).sum())
            rr, _, _ = metrics(tr, tg); ro, _, _ = metrics(to, tg)
            acc["same_rank"] += int((rr == ro).sum())
            for b in torch.nonzero(~same).flatten().tolist():         # differing lists: reference-score gap of swapped items
                sc_map = {int(i): float(s) for i, s in zip(ir[ur == b].tolist(), sr[ur == b].tolist())}
                x, y = tr[b].tolist(), to[b].tolist()
                gap = max(abs(sc_map.get(i, math.inf) - sc_map.get(j, math.inf)) for i, j in zip(x, y) if i != j)
                acc["max_gap_diff_users"] = max(acc["max_gap_diff_users"], gap)
            if K == 50:                                             # noise floor: fresh encoder pass, reference decode
                enc2, mk2 = encode(m.model, ids, mask)
                u2, i2, s2, _ = ref_decode(m, enc2, mk2, K)
                t2, _ = lists(m, u2, i2, s2, B)
                acc["noise_same_top10"] += int((tr == t2).all(1).sum()); acc["noise_users"] += B
            for k, t in (("ref", tr), ("opt", to), ("optr", tq)):
                R[k].append(t.cpu())
            acc["users"] += B
        tgc = tg_all[order]
        res = {k: metrics(torch.cat(v), tgc)[1:] for k, v in R.items()}
        U = acc["users"]
        r = {"users": U, "share_same_fsel": acc["same_fsel"] / U, "share_same_candidates": acc["same_cand_users"] / U,
             "max_abs_score_diff": acc["max_abs"], "match_rate_top10": acc["same_top10"] / U,
             "share_same_target_rank": acc["same_rank"] / U, "max_ref_score_gap_in_differing_lists": acc["max_gap_diff_users"],
             "hr10_ndcg10": {"ref": res["ref"], "opt": res["opt"], "optr_restricted": res["optr"]},
             "depth3_duplicate_prefixes": acc["dup"], "restricted_variant_match_rate_top10": acc["same_top10_r"] / U}
        if acc["noise_users"]:
            r["noise_floor_ref_vs_ref_fresh_encoder_top10"] = acc["noise_same_top10"] / acc["noise_users"]
        r["ok"] = (r["share_same_target_rank"] == 1.0 and res["ref"] == res["opt"] and acc["dup"] == 0
                   and r["max_abs_score_diff"] < 1e-3 and (r["match_rate_top10"] == 1.0 or acc["max_gap_diff_users"] < 1e-4))
        ok_all &= r["ok"]
        out["K"][str(K)] = r
        print(("PASS" if r["ok"] else "FAIL"), f"{D} K_f {K}", json.dumps(r), flush=True)
        json.dump(out, open(f, "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    out["all_pass"] = ok_all
    json.dump(out, open(f, "w"), indent=1)
    print("ALL PASS" if ok_all else "SOME CHECK FAILED", flush=True)
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
