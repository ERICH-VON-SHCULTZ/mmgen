"""mmgen addition: sanity checks of codebook-geometry-aware training before any full run (docs/MFD_REPORT.md §14). CPU.

0. Provenance: the SID index equals the locked v1 file (sha256), the tokenizer checkpoint is the one that produced
   it (v1 tokenizer.pth == the attempt's final.pth), and re-encoding every item with it gives the index codes except
   for the items moved by the distance reassignment (reassign.json): the token -> codebook-row mapping is right.
1. At init (delta = 0) code-embedding cosines equal the cosines of the projected codebook vectors, and (orthogonal
   P) of the raw codebook vectors; for the shuffled control they equal the permuted vectors' cosines. Mean norm per
   level = T5's default embedding norm. Tied head. Gradients reach P and delta. eps = 0 soft loss = HF's loss.
   Save / reload (the Trainer's path) and bake() leave the logits unchanged. Chain (PlainT5) and MFD (MFDT5).
2. Soft labels: every target row sums to 1 with mass 1 - eps on the true f; T; top-5 neighbours of a few f codes.
Writes --out_file (JSON); exits 1 on any failure.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import torch
from safetensors.torch import load_file
from transformers import T5Config, T5Tokenizer

import geo_embed as ge
from finetune_ext import PlainT5
from mfd_model import BRANCH_TOKENS, MFDT5

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_index"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cos(x):
    x = torch.nn.functional.normalize(x.float(), dim=1)
    return x @ x.t()


def build(cls, tok, ckpt, geo, eps, seed=0):
    torch.manual_seed(0)
    m = cls(T5Config.from_pretrained("./config/ckpt"))
    m.resize_token_embeddings(len(tok))
    info = m.install_geometry(tok, ckpt, geo, eps, seed)
    return m.eval(), info


def logits(m, mfd, V):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(32100, V, (3, 15), generator=g)
    y = torch.randint(32100, V, (3, 4), generator=g)
    y = y.unsqueeze(1).repeat(1, 2, 1) if mfd else y
    with torch.no_grad():
        return m(input_ids=x, attention_mask=torch.ones_like(x), decoder_input_ids=y).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Instruments")
    p.add_argument("--index", required=True)
    p.add_argument("--v1_dir", required=True)
    p.add_argument("--out_file", required=True)
    a = p.parse_args()
    rep, fails = {"checks": {}}, []
    ok = lambda name, cond, **kw: (rep["checks"].__setitem__(name, {"ok": bool(cond), **kw}), None if cond else fails.append(name))

    # ---- 0. provenance ----
    ckpt = f"{a.v1_dir}/tokenizer.pth"
    rc = json.load(open(f"{a.v1_dir}/reassign.json"))
    ok("index_is_v1", sha(a.index) == open(f"{a.v1_dir}/index.sha256").read().split()[0], index=a.index)
    ok("tokenizer_is_producer", sha(ckpt) == sha(rc["ckpt"]), tokenizer=ckpt, producer=rc["ckpt"], sha256=sha(ckpt))
    from generate_indices_md import compute_codes, load
    model, c, _ = load(ckpt)
    codes, _, _ = compute_codes(model, c, device="cpu")
    idx = json.load(open(a.index))
    moved = {ch["item"] for ch in rc["changes"]}
    diff_items = set()
    for k, toks in idx.items():
        got = [int(t.split("_")[1][:-1]) for t in toks]
        if got != [int(x) for x in codes[int(k)]]:
            diff_items.add(int(k))
    ok("tokens_match_codebook_argmin", diff_items == moved, items_differing=len(diff_items), reassigned=len(moved))

    books = ge.load_codebooks(ckpt)
    rep["codebooks"] = {k: list(v.shape) for k, v in books.items()}
    tok = T5Tokenizer.from_pretrained("./config/ckpt", model_max_length=512)
    tok.add_tokens(sorted({t for v in idx.values() for t in v}))
    V = len(tok)
    toks = ge.code_tokens(tok)

    # ---- 1. embeddings ----
    for name, cls, mfd in (("chain", ge.make_geo_class(PlainT5), False), ("mfd", ge.make_geo_class(MFDT5), True)):
        tk = tok
        if mfd:
            tk = T5Tokenizer.from_pretrained("./config/ckpt", model_max_length=512)
            tk.add_tokens(sorted({t for v in idx.values() for t in v} | set(BRANCH_TOKENS)))
        for geo in ("codebook", "shuffled"):
            m, info = build(cls, tk, ckpt, geo, 0.0, seed=0)
            W = m.shared.full_weight().detach()
            r = {"tied_head": info["tied_head"], "levels": {}}
            for lv in m.shared.levels:
                ids = getattr(m.shared, f"ids_{lv}")
                cv = getattr(m.shared, f"cv_{lv}")
                proj = cv @ m.shared.P[lv].detach().t()
                true_cv = books[lv][torch.tensor([cc for i, l, cc in ge.code_tokens(tk) if l == lv])]
                e = W[ids]
                r["levels"][lv] = {
                    "max_abs_cos_diff_vs_projected": (cos(e) - cos(proj)).abs().max().item(),
                    "max_abs_cos_diff_vs_codebook_used": (cos(e) - cos(cv)).abs().max().item(),
                    "corr_cos_vs_true_codebook": torch.corrcoef(torch.stack([cos(e).flatten(), cos(true_cv).flatten()]))[0, 1].item(),
                    "mean_norm": e.norm(dim=1).mean().item(), "target_norm": info["target_mean_norm"]}
                lvr = r["levels"][lv]
                ok(f"{name}_{geo}_{lv}_cos", lvr["max_abs_cos_diff_vs_projected"] < 1e-4 and lvr["max_abs_cos_diff_vs_codebook_used"] < 1e-4)
                ok(f"{name}_{geo}_{lv}_norm", abs(lvr["mean_norm"] / lvr["target_norm"] - 1) < 1e-3)
                if geo == "shuffled":
                    ok(f"{name}_shuffled_{lv}_is_permuted", not torch.equal(cv, true_cv) and abs(lvr["corr_cos_vs_true_codebook"]) < 0.2)
            # cross-level pair (shared vs text level 0): cosine of the projected vectors
            ids_all, rows = m.shared.code_rows()
            ok(f"{name}_{geo}_all_code_rows_cos", (cos(W[ids_all]) - cos(rows.detach())).abs().max().item() < 1e-4)
            other = torch.tensor([i for i in range(V) if i not in set(ids_all.tolist())])
            ok(f"{name}_{geo}_other_rows_unchanged", torch.equal(W[other], m.shared.base.weight[other].detach()))
            # gradients reach P and delta; tied head uses the same table
            m.train()
            x = torch.randint(32100, V, (2, 10))
            if mfd:
                y = torch.randint(32100, V, (2, 2, 4))
                out = m(input_ids=x, attention_mask=torch.ones_like(x), decoder_input_ids=y, labels=y)
            else:
                y = torch.randint(32100, V, (2, 5))
                out = m(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
            out.loss.backward()
            ok(f"{name}_{geo}_grads", all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.shared.P.values())
               and m.shared.delta.grad.abs().sum() > 0)
            m.eval()
            # save / reload as the Trainer does (safetensors, strict=False), then bake
            before = logits(m, mfd, V)
            with tempfile.TemporaryDirectory() as d:
                m.save_pretrained(d)
                m2, _ = build(cls, tk, ckpt, "codebook", 0.0)
                res = m2.load_state_dict(load_file(f"{d}/model.safetensors"), strict=False)
                bad = [k for k in res.missing_keys if not any(s in k for s in ge.GeoMixin._tied_weights_keys)]
                ok(f"{name}_{geo}_reload", not bad and not res.unexpected_keys and (logits(m2, mfd, V) - before).abs().max() < 1e-5,
                   missing=res.missing_keys, unexpected=res.unexpected_keys)
            m.bake()
            ok(f"{name}_{geo}_bake", (logits(m, mfd, V) - before).abs().max().item() < 1e-4
               and m.lm_head.weight is m.shared.weight)
            rep[f"{name}_{geo}"] = r | {"info": info}
        # soft loss with eps = 0 targets reproduces the model's own loss
        m, _ = build(cls, tk, ckpt, "codebook", 0.1)
        m.soft_Q = torch.eye(len(m.soft_ids))
        g = torch.Generator().manual_seed(2)
        x = torch.randint(32100, V, (4, 10), generator=g)
        f_ids = m.soft_ids
        if mfd:
            y = torch.randint(32100, V, (4, 2, 4), generator=g)
            y[:, 0, 0] = f_ids[torch.randint(len(f_ids), (4,), generator=g)]
            y[:, 1, :2] = -100
            kw = dict(decoder_input_ids=torch.randint(32100, V, (4, 2, 4), generator=g))
        else:
            y = torch.randint(32100, V, (4, 6), generator=g)
            y[:, 0] = f_ids[torch.randint(len(f_ids), (4,), generator=g)]
            y[:, -1] = -100
            kw = {}
        with torch.no_grad():
            o = m(input_ids=x, attention_mask=torch.ones_like(x), labels=y, **kw)
            mine = o["loss"].item()                              # as the Trainer reads it
            ok(f"{name}_soft_loss_is_output_key", "loss" in o.keys() and o.to_tuple()[0] is o["loss"])
            m.soft_Q = None
            ref = m(input_ids=x, attention_mask=torch.ones_like(x), labels=y, **kw)["loss"].item()
        ok(f"{name}_soft_loss_eps0_equals_model_loss", abs(mine - ref) < 1e-4, soft=mine, model=ref)

    # ---- 2. soft labels ----
    for eps in (0.1, 0.2):
        ids, Q, T, sinfo = ge.soft_f_targets(tok, books, eps)
        ok(f"soft_eps{eps}_rows_sum_1", (Q.sum(1) - 1).abs().max().item() < 1e-5)
        ok(f"soft_eps{eps}_true_mass", (Q.diagonal() - (1 - eps)).abs().max().item() < 1e-6)
        fcodes = [int(tok.convert_ids_to_tokens(int(i)).split("_")[1][:-1]) for i in ids]
        c = books["s0"][torch.tensor(fcodes)]
        d2 = torch.cdist(c, c).pow(2)
        ex = {}
        for j in (0, 50, 100, 150, 200):
            q = Q[j].clone()
            q[j] = 0
            top = q.topk(5)
            ex[f"<s0_{fcodes[j]}>"] = [{"code": f"<s0_{fcodes[k]}>", "mass": round(v.item(), 4),
                                         "sq_dist": round(d2[j, k].item(), 3)} for v, k in zip(top.values, top.indices)]
            ex[f"<s0_{fcodes[j]}>"].append({"mass_on_true": round(Q[j, j].item(), 4),
                                             "mass_top5_neighbours": round(top.values.sum().item(), 4)})
        rep[f"soft_eps{eps}"] = sinfo | {"examples": ex,
                                         "median_sq_dist_all_pairs": d2[~torch.eye(len(fcodes), dtype=bool)].median().item()}
    # §15 V3: plain label smoothing on f; V4: geometric targets with 70% of the eps mass on the 10 nearest f
    ids, Q, _, sinfo = ge.soft_f_targets(tok, books, 0.2, "uniform")
    K = len(ids)
    off = ~torch.eye(K, dtype=torch.bool)
    ok("V3_uniform_true_mass", (Q.diagonal() - 0.8).abs().max().item() < 1e-6)
    ok("V3_uniform_other_mass", (Q[off] - 0.2 / (K - 1)).abs().max().item() < 1e-7, other=0.2 / (K - 1), f_codes=K)
    ok("V3_rows_sum_1", (Q.sum(1) - 1).abs().max().item() < 1e-5)
    rep["V3_uniform"] = sinfo | {"true_mass": 0.8, "other_mass": 0.2 / (K - 1)}
    ids, Q, T4, sinfo = ge.soft_f_targets(tok, books, 0.2, "geometric", 0.7)
    fcodes = [int(tok.convert_ids_to_tokens(int(i)).split("_")[1][:-1]) for i in ids]
    c = books["s0"][torch.tensor(fcodes)]
    d2 = torch.cdist(c, c).pow(2)
    d2o = d2.clone(); d2o.fill_diagonal_(float("inf"))
    order = d2o.argsort(1)
    per = {}
    for j in (0, 50, 100, 150, 200):
        m = Q[j, order[j]] / 0.2
        per[f"<s0_{fcodes[j]}>"] = {"eps_mass_5nn": round(m[:5].sum().item(), 4), "eps_mass_10nn": round(m[:10].sum().item(), 4)}
    ok("V4_mean_10nn_mass_0.7", abs(sinfo["eps_mass_10nn"] - 0.7) < 1e-3, T=T4, mean_5nn=sinfo["eps_mass_5nn"])
    ok("V4_rows_sum_1", (Q.sum(1) - 1).abs().max().item() < 1e-5)
    rep["V4_geometric"] = sinfo | {"examples": per}
    print(f"V4: T={T4:.3f}  mean eps mass on 5-NN {sinfo['eps_mass_5nn']:.3f}, on 10-NN {sinfo['eps_mass_10nn']:.3f}")
    for code, v in per.items():
        print("  ", code, v)
    print(f"V3: true f {0.8}, every other f code {0.2 / (K - 1):.6f} ({K} f codes)")
    # §17 on V2-MFD: P1a item-level encoder, P1b/c f weight, P1d/e offset penalty
    from argparse import Namespace
    from enc_struct import ItemMFDT5
    from mfd_data import Catalog, branch_tensors
    from mfd_eval import score_batch
    tk = T5Tokenizer.from_pretrained("./config/ckpt", model_max_length=512)
    tk.add_tokens(sorted({t for v in idx.values() for t in v} | set(BRANCH_TOKENS)))
    keys = list(idx)[:60]
    full = lambda k: "".join(idx[k])
    txt = lambda k: "".join(idx[k][:3])
    n_it = [1, 3, 5, 12, 20, 7]
    hist = [[keys[(j + q) % 60] for q in range(n)] for j, n in enumerate(n_it)]
    enc_t = tk(["".join(full(k) for k in h) for h in hist] + ["".join(txt(k) for k in h) for h in hist], return_tensors="pt", padding="longest")
    x, xm = enc_t["input_ids"], enc_t["attention_mask"]
    T_, I_ = tk.convert_tokens_to_ids(list(BRANCH_TOKENS))
    tab = torch.tensor([tk.convert_tokens_to_ids(idx[k]) for k in keys[:12]])
    y, labels = branch_tensors(tab, torch.arange(12), T_, I_)
    cls = ge.make_geo_class(ItemMFDT5)
    torch.manual_seed(0)
    m = cls(T5Config.from_pretrained("./config/ckpt"))
    m.resize_token_embeddings(len(tk))
    m.install_geometry(tk, ckpt, "codebook")
    m.install_struct(tk, "E2")
    m.eval()
    enc, em = m.struct_encode(x, xm)
    ok("P1a_encoder_length_is_items", em.sum(1).tolist() == n_it + n_it and enc.last_hidden_state.shape[1] == 20,
       items=n_it, encoder_lengths=em.sum(1).tolist(), token_lengths=xm.sum(1).tolist())
    out = m(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y)
    ok("P1a_mfd_decoder_rows_unchanged", out.logits.shape[:3] == (12, 2, y.shape[2]), shape=list(out.logits.shape))
    ds_name = os.path.basename(os.path.dirname(a.index))
    cat = Catalog(Namespace(data_path=os.path.dirname(os.path.dirname(a.index)), dataset=ds_name, index_file=os.path.basename(a.index)[len(ds_name):]), tk, "cpu")
    m.config.use_cache = False
    with torch.no_grad():
        sb = score_batch(m, cat, x[:2], xm[:2], 5, T_, I_)
    ok("P1a_score_batch_runs", len(sb["item"]) > 0 and torch.isfinite(sb["s_t"]).all())
    with torch.no_grad():
        before = m(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y).logits
        m.bake()
        after = m(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y).logits
    m.__class__ = ItemMFDT5
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = ItemMFDT5.from_pretrained(d).eval()
        with torch.no_grad():
            re_ = m2(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y).logits
    ok("P1a_bake_and_reload", (before - after).abs().max() < 1e-4 and (before - re_).abs().max() < 1e-4,
       bake=float((before - after).abs().max()), reload=float((before - re_).abs().max()))
    base = ge.make_geo_class(MFDT5)

    def build_w(fw, lam):
        torch.manual_seed(0)
        z = base(T5Config.from_pretrained("./config/ckpt"))
        z.resize_token_embeddings(len(tk))
        z.install_geometry(tk, ckpt, "codebook", f_weight=fw, offset_lambda=lam)
        return z

    def loss_of(z, train):
        z.train(train)
        torch.manual_seed(1)
        return z(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y, labels=labels)
    for name, fw in (("P1b", 2.0), ("P1c", 3.0)):
        z = build_w(fw, 0.0)
        o = loss_of(z, True)
        lp = o.logits.float().log_softmax(-1)
        v = labels != -100
        ce = -lp[v].gather(1, labels[v][:, None]).squeeze(1)
        isf = torch.zeros_like(labels, dtype=torch.bool)
        isf[:, 0, 0] = True
        manual = (fw * (-lp[isf].gather(1, labels[isf][:, None])).sum() + ce.sum() - (-lp[isf].gather(1, labels[isf][:, None])).sum()) / v.sum()
        ok(f"{name}_loss_equals_weighted_sum", abs(o["loss"].item() - manual.item()) < 1e-4, loss=o["loss"].item(), manual=manual.item())
        with torch.no_grad():
            z.eval()
            ev = z(input_ids=x[:12], attention_mask=xm[:12], decoder_input_ids=y, labels=labels)
            ok(f"{name}_eval_loss_is_plain_ce", abs(ev["loss"].item() - (-ev.logits.float().log_softmax(-1)[v].gather(1, labels[v][:, None])).mean().item()) < 1e-4)
    for name, lam in (("P1d", 1e-3), ("P1e", 1e-2)):
        z = build_w(1.0, lam)
        plain = build_w(1.0, 0.0)
        a0, b0 = loss_of(z, True), loss_of(plain, True)
        ok(f"{name}_penalty_zero_at_init", z.offset_penalty().item() == 0 and abs(a0["loss"].item() - b0["loss"].item()) < 1e-5,
           with_penalty=a0["loss"].item(), without=b0["loss"].item())
        with torch.no_grad():
            z.shared.delta[:] = 1.0
        ok(f"{name}_penalty_value", abs(z.offset_penalty().item() - z.shared.delta.shape[1]) < 1e-4, value=z.offset_penalty().item())
    rep["all_pass"] = not fails
    rep["failures"] = fails
    json.dump(rep, open(a.out_file, "w"), indent=1)
    for k, v in rep["checks"].items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k}")
    for eps in (0.1, 0.2):
        print(f"soft labels eps={eps}: T={rep[f'soft_eps{eps}']['T']:.3f}")
        for code, lst in list(rep[f"soft_eps{eps}"]["examples"].items())[:3]:
            print("  ", code, lst)
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
