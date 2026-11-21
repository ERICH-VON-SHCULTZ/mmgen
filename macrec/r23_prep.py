"""mmgen addition (docs/MFD_REPORT.md §23): preparation and verification step BEFORE the timing jobs (one dataset per call, GPU).
 0 batch-1 subset: 2,000 test users (dataset indices), numpy default_rng(0), sorted; same users for every method.
   Check: every method's dataset has the same users in the same order with the same target item.
 1 C-fast verification, FULL test set, test_ddp_save's order and batch 64: for both lists (seqrec, seqimage) the 20 returned
   sequences and their sequence scores of C-fast are compared with original C (Python trie) bitwise (torch.equal per user);
   the merged top-10 lists are compared per user. Also: original C (this harness) vs the reported save files
   (baselines/<D>/mql/seed42/eval/test/save_*_20.json, aligned by user): same strings, same scores; HR@10 of both vs the reported
   merged HR@10.
 2 merge / top-10 reimplementation (r23_timing.merge_top10, single_top10) vs MACRec's ensemble.get_topk_results_ensemble /
   evaluate.get_topk_results on the reported save files of C, D, TIGER, chain: identical hit vectors for every user.
 3 instrumented paths = plain paths: P1a stage-timed copy vs infer() (top-10 lists), instrumented generate (stage timers) vs
   plain generate (sequences, scores), on the first 4 batches.
 4 P1a reference ranks: every item under every f scored once (full view), exact target rank (mfd_mv_infer.ranks_from_terms) for
   plain K_f in {20, 50, 100}, w = 1 / .5 / .5, seeds 42 / 43 / 44 (+ A01 K_f 50 seed 42 on Arts); K_f 50 compared with
   r21_eval's reported ranks, K_f 100 with eval_ii's (seed 42).
Out: mmgen_runs/timing/r23/<D>/prep.json; exit 1 if C-fast differs anywhere or a check fails (the timing jobs depend afterok).
"""
import argparse
import json
import os

import numpy as np
import torch

import r23_timing as RT
from ensemble import get_topk_results_ensemble
from evaluate import get_topk_results
from mfd_mv_infer import branch_scores, encode, f_logprobs, metrics, ranks_from_terms


def check(res, name, cond, **info):
    res["checks"][name] = dict(ok=bool(cond), **info)
    print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str)[:2000], flush=True)


def save_lists(D, b, task):
    return json.load(open(f"{RT.R}/baselines/{D}/{b}/seed42/eval/test/save_{task}_20.json"))


@torch.no_grad()
def cfast_verify(D, res, limit=0):
    m = RT.BeamMethod(D, "C")
    for l in m.lists:
        l.gpu = RT.GPUTrieProcessor(l.trie, m.model.config.vocab_size, "cuda")
    order = m.order[:limit] if limit else m.order
    U = len(order)
    same_seq = {l.task: 0 for l in m.lists}
    same_score = {l.task: 0 for l in m.lists}
    same_top_merge, orig_lists, fast_lists = 0, {}, {}
    orig_raw = {l.task: {} for l in m.lists}
    for idx in RT.batches(m, order, 64):
        b = m.collate(idx)
        out = {"orig": [], "fast": []}
        for l, (ids, mask) in zip(m.lists, b):
            ids, mask = ids.cuda(), mask.cuda()
            for kind in ("orig", "fast"):
                gpu = l.gpu
                if kind == "orig":
                    l.gpu = None
                s, sc = m.gen(l, ids, mask, RT.NULL)
                l.gpu = gpu
                out[kind].append((s.cpu(), sc.cpu()))
            (so, sco), (sf, scf) = out["orig"][-1], out["fast"][-1]
            B = len(idx)
            if so.shape == sf.shape:
                eq_s = (so.view(B, RT.NB, -1) == sf.view(B, RT.NB, -1)).all(-1).all(-1)
            else:
                eq_s = torch.zeros(B, dtype=torch.bool)
            eq_sc = (sco.view(B, RT.NB) == scf.view(B, RT.NB)).all(-1)
            same_seq[l.task] += int(eq_s.sum()); same_score[l.task] += int(eq_sc.sum())
            pred = [x.strip().replace(" ", "") for x in m.tok.batch_decode(so, skip_special_tokens=True)]
            for k, u in enumerate(idx):
                orig_raw[l.task][u] = (pred[k * RT.NB:(k + 1) * RT.NB], sco[k * RT.NB:(k + 1) * RT.NB].tolist())
        tops = {}
        for kind in ("orig", "fast"):
            r = []
            for l, (s, sc) in zip(m.lists, out[kind]):
                r.append(m.post_list(l, s, sc, len(idx), RT.NULL))
            tops[kind] = RT.merge_top10(m.lists, r)
        for k, u in enumerate(idx):
            orig_lists[u], fast_lists[u] = tops["orig"][k], tops["fast"][k]
            same_top_merge += tops["orig"][k] == tops["fast"][k]
    rep_hr = RT.reported_hr10(D, "C")[0]
    tg = [m.targets[u] for u in order]
    hr_o, hr_f = RT.hr10([orig_lists[u] for u in order], tg), RT.hr10([fast_lists[u] for u in order], tg)
    check(res, "1_cfast_equals_C_full_test", all(same_seq[t] == U and same_score[t] == U for t in same_seq) and same_top_merge == U,
          users=U, same_20_sequences=same_seq, same_20_scores_bitwise=same_score, same_merged_top10=same_top_merge,
          match_rate_top10=same_top_merge / U)
    res["cfast_match_rate_top10"] = same_top_merge / U
    # original C (this harness) vs the reported save files
    agree = {}
    for l in m.lists:
        sv = save_lists(D, "mql", l.task)
        n_s = n_sc = 0
        for j, u in enumerate(sv["all_users"]):
            if u not in orig_raw[l.task]:
                continue
            p, s = orig_raw[l.task][u]
            rp = [x.strip().replace(" ", "") for x in sv["all_outputs"][j * RT.NB:(j + 1) * RT.NB]]
            n_s += p == rp
            n_sc += s == sv["all_scores"][j * RT.NB:(j + 1) * RT.NB]
        agree[l.task] = {"same_strings": n_s / U, "same_scores": n_sc / U, "users_in_sampler_order": sv["all_users"][:U] == order}
    check(res, "1b_C_harness_equals_reported_save_files", all(v["same_strings"] == 1 and v["users_in_sampler_order"] for v in agree.values()),
          per_list=agree, hr10_harness=hr_o, hr10_cfast=hr_f, hr10_reported=rep_hr,
          hr10_match=abs(hr_o - rep_hr) < 0.5 / U and abs(hr_f - rep_hr) < 0.5 / U)
    del m
    torch.cuda.empty_cache()


def reimpl_verify(D, res):
    out = {}
    for name in ("C", "D", "TIGER", "chain"):
        b, idx, iidx, tasks = RT.BEAM[name]
        lists = [RT.BeamList.__new__(RT.BeamList) for _ in tasks]
        for l, t in zip(lists, tasks):
            args = RT.mk_args(D, idx, iidx, ["--test_task", t])
            ds = RT.SeqRecDataset(args, task=t, mode="test", sample_num=args.sample_num)
            l.task, l.all_items = t, ds.get_all_items()
            l.sid2item = {"".join(v): int(k) for k, v in ds.indices.items()}
        sv = [save_lists(D, b, t) for t in tasks]
        users = sv[0]["all_users"]
        preds = [[x.strip().replace(" ", "") for x in s["all_outputs"]] for s in sv]
        mine = []
        for j in range(len(users)):
            sl = slice(j * RT.NB, (j + 1) * RT.NB)
            if len(tasks) == 2:
                us = sv[1]["all_users"].index(users[j]) if sv[1]["all_users"][j] != users[j] else j
                sl2 = slice(us * RT.NB, (us + 1) * RT.NB)
                mine.append(RT.merge_top10(lists, [(preds[0][sl], sv[0]["all_scores"][sl]), (preds[1][sl2], sv[1]["all_scores"][sl2])])[0])
            else:
                mine.append(RT.single_top10(lists[0], preds[0][sl], sv[0]["all_scores"][sl])[0])
        tg = [lists[0].sid2item[t] for t in sv[0]["all_targets"]]
        my_hits = [[int(x == t) for x in l] for l, t in zip(mine, tg)]
        if len(tasks) == 2:
            def info(s, l):
                p = [x.strip().replace(" ", "") for x in s["all_outputs"]]
                return sorted([(s["all_users"][j], [l.sid2item[s["all_targets"][j]]],
                                [[l.sid2item[x]] if x in l.sid2item else [-1] for x in p[j * RT.NB:(j + 1) * RT.NB]],
                                s["all_scores"][j * RT.NB:(j + 1) * RT.NB]) for j in range(len(s["all_users"]))], key=lambda x: x[0])
            ref = get_topk_results_ensemble(info(sv[0], lists[0]), info(sv[1], lists[1]))
            order = np.argsort(users)
            my_sorted = [my_hits[j] for j in order]
        else:
            ref = get_topk_results(list(sv[0]["all_outputs"]), list(sv[0]["all_scores"]), sv[0]["all_targets"], RT.NB,
                                   all_items=lists[0].all_items)
            my_sorted = my_hits
        same = sum(a[:10] == r[:10] for a, r in zip(my_sorted, ref))
        out[name] = {"users": len(ref), "same_top10_hits": same, "hr10_mine": float(np.mean([sum(h[:10]) > 0 for h in my_hits])),
                     "hr10_ref_fn": float(np.mean([sum(h[:10]) > 0 for h in ref])), "hr10_reported": RT.reported_hr10(D, name)[0]}
    check(res, "2_merge_top10_reimplementation", all(v["same_top10_hits"] == v["users"] for v in out.values()), per_method=out)


@torch.no_grad()
def instrumented_verify(D, res):
    # P1a: the item-level encoder's scatter_add is not bit-reproducible (§17), so two plain runs can differ by float noise;
    # compare candidates exactly and scores to 1e-4, with plain-vs-plain as the noise reference
    m = RT.make(D, "P1a:50")
    same_cand, d_stage, d_rep, top_same, top_same_rep, n = True, 0.0, 0.0, 0, 0, 0
    for idx, b in RT.precollate(m, m.order[:256], 64):
        ra, rb, rc = [], [], []
        a = m.run(b, RT.NULL, ra)
        s = m.run(b, RT.Stages(), rb)
        a2 = m.run(b, RT.NULL, rc)
        # align candidates by (user, item): float noise can swap two near-tied f codes in top-K (permutes the candidate order)
        def key(r):
            k = r[0][0] * 10 ** 7 + r[0][1]
            o = torch.argsort(k)
            return k[o], r[0][2][o]
        (ka, sa), (kb, sb), (kc, sc) = key(ra), key(rb), key(rc)
        same_cand &= torch.equal(ka, kb)
        if torch.equal(ka, kb):
            d_stage = max(d_stage, float((sa - sb).abs().max()))
        if torch.equal(ka, kc):
            d_rep = max(d_rep, float((sa - sc).abs().max()))
        top_same += sum(x == y for x, y in zip(a, s)); top_same_rep += sum(x == y for x, y in zip(a, a2)); n += len(a)
    del m
    out = {"P1a_stage_copy_equals_infer": same_cand and d_stage < 1e-4, "P1a_same_candidates": same_cand,
           "P1a_max_abs_score_diff_stage_vs_plain": d_stage, "P1a_max_abs_score_diff_plain_vs_plain": d_rep,
           "P1a_share_same_top10_stage_vs_plain": top_same / n, "P1a_share_same_top10_plain_vs_plain": top_same_rep / n}
    for name in ("C", "Cfast"):
        m = RT.make(D, name)
        same = True
        for idx, b in RT.precollate(m, m.order[:256], 64):
            raw_a, raw_b = [], []
            ta = m.run(b, RT.NULL, raw_a)
            st = RT.Stages()
            with RT.instrument_beam(m, st):
                tb = m.run(b, st, raw_b)
            same &= ta == tb and all(torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) for x, y in zip(raw_a, raw_b))
        out[f"{name}_instrumented_equals_plain"] = same
        out[f"{name}_stage_names"] = sorted(st.t)
        del m
        torch.cuda.empty_cache()
    check(res, "3_instrumented_equals_plain", all(v for k, v in out.items() if k.endswith("plain") or k.endswith("infer")), **out)


@torch.no_grad()
def p1a_ref(D, res, limit=0):
    cfgs = [dict(variant="plain", K=K, w_t=0.5, w_i=0.5, a=(1, 0, 0), b=(1, 0, 0), name=f"K{K}") for K in (20, 50, 100)]
    ref, cmp = {}, {}
    runs = [("P1a", s) for s in (42, 43, 44)] + ([("A01", 42)] if D == "Arts" else [])
    for run, S in runs:
        m = RT.MFDMethod(D, f"{run}:50", seed=S)
        allf = torch.arange(m.cat.nF, device="cuda")
        Rk = []
        for idx in RT.batches(m, m.order[:limit] if limit else m.order, 64):
            ids, mask = m.collate(idx)
            tgt = torch.tensor([m.targets[j] for j in idx], device="cuda")
            enc, mk = encode(m.model, ids.cuda(), mask.cuda())
            lpf = f_logprobs(m.model, m.cat, enc, mk)
            user, item, _, s_t, s_i = branch_scores(m.model, m.cat, enc, mk, allf[None].expand(enc.shape[0], -1), m.T, m.I)
            z, zf = torch.zeros_like(s_t), torch.zeros_like(lpf)          # views 2 / 3 carry weight 0 (a = b = (1, 0, 0))
            Rk.append(ranks_from_terms(cfgs, torch.stack([lpf, zf, zf]), user, item, torch.stack([s_t, z, z]),
                                       torch.stack([s_i, z, z]), tgt, m.cat))
        Rk = torch.cat(Rk, 1)
        for j, c in enumerate(cfgs):
            ref[f"{run}:{c['K']}:s{S}"] = metrics(Rk[j])
        rr = torch.load(f"{m.ck}/r21_eval/ranks.pt", weights_only=False)["test"]["plain"][:Rk.shape[1]]
        cmp[f"{run} s{S} K50 vs r21_eval"] = {"share_same_rank_capped10": float((rr.clamp(max=10) == Rk[1].clamp(max=10)).float().mean()),
                                              "hr10_exact": ref[f"{run}:50:s{S}"]["hr10"], "hr10_reported": metrics(rr)["hr10"]}
        ii = f"{m.ck}/eval_ii/ranked_test.pt"
        if run == "P1a" and os.path.exists(ii):
            d = torch.load(ii, weights_only=False)
            n = Rk.shape[1]
            hit = (d["joint"][0][:n, :10] == d["targets"][:n, None]).any(1).float().mean().item()
            cmp[f"{run} s{S} K100 vs eval_ii"] = {"hr10_exact": ref[f"{run}:100:s{S}"]["hr10"], "hr10_reported": hit}
        del m
        torch.cuda.empty_cache()
    res["p1a_ref"] = ref
    check(res, "4_p1a_exact_ranks_vs_reported", all(abs(v["hr10_exact"] - v["hr10_reported"]) < 2e-4 for v in cmp.values()),
          comparisons=cmp, note="exact all-f scoring vs the batched deployment path; tolerance 2e-4 HR@10 (numerical ties)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--skip", default="", help="comma list of steps to skip (debug only)")
    p.add_argument("--limit", type=int, default=0, help="smoke test only: first N users in steps 1 and 4")
    a = p.parse_args()
    D = a.dataset
    os.makedirs(f"{RT.OUT_DIR}/{D}", exist_ok=True)
    f = f"{RT.OUT_DIR}/{D}/prep.json"
    res = {"env": RT.env_info(), "dataset": D, "checks": {}}
    # 0 subset + same users / targets in every method's dataset
    mfd = RT.MFDMethod(D, "P1a:50")
    U = mfd.U
    sub = sorted(np.random.default_rng(0).choice(U, 2000, replace=False).tolist())
    res["batch1_subset"] = sub
    tg = {"P1a": mfd.targets}
    del mfd
    for name in ("C", "D", "TIGER", "chain"):
        b, idx, iidx, tasks = RT.BEAM[name]
        for t in tasks:
            args = RT.mk_args(D, idx, iidx, ["--test_task", t])
            ds = RT.SeqRecDataset(args, task=t, mode="test", sample_num=args.sample_num)
            s2i = {"".join(v): int(k) for k, v in ds.indices.items()}
            tg[f"{name}/{t}"] = [s2i[d["item"]] for d in ds.inter_data]
    check(res, "0_same_users_and_targets", all(v == tg["P1a"] for v in tg.values()), users=U, lists=sorted(tg))
    json.dump(res, open(f, "w"), indent=1)
    steps = [("p1a_ref", p1a_ref), ("reimpl", reimpl_verify), ("instrumented", instrumented_verify), ("cfast", cfast_verify)]
    for k, fn in steps:
        if k in a.skip.split(","):
            continue
        fn(D, res, a.limit) if k in ("p1a_ref", "cfast") else fn(D, res)
        json.dump(res, open(f, "w"), indent=1)
    res["all_pass"] = all(v["ok"] for v in res["checks"].values())
    json.dump(res, open(f, "w"), indent=1)
    print("ALL PASS" if res["all_pass"] else "SOME CHECK FAILED", flush=True)
    if not res["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
