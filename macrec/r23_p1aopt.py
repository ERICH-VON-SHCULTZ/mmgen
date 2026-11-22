"""mmgen addition (docs/MFD_REPORT.md §23): P1a-opt, an exact faster implementation of P1a's plain inference (K_f, w = 1/.5/.5,
full view; mfd_mv_infer.infer with the plain cfg). Same model, same weights, same scores up to float rounding; nothing in the
model or in the training code changes.

What P1a's reference path does (mfd_model.last_logprobs): every decoder call recomputes all positions of every row without
a KV cache, replicates the user's encoder memory per row (and with it the cross-attention keys / values), and applies the
full-vocabulary output projection at every position although only the last one is used.

P1a-opt evaluates the same decoder as a prefix tree, one new token per node, each node once:
  depth 0  [start]                         per user               -> log P(f)            (f codes)
  depth 1  [start, f]                      per (user, kept f)     -> no output needed
  depth 2  [start, f, <T>|<I>]             per (user, f, branch)  -> log P(t1 | f), log P(i1 | f)
  depth 3  [start, f, <T>|<I>, c1]         per catalog prefix     -> log P(t2 | f, t1), log P(i2 | f, i1)
  (1+2+2 SIDs: L = 2; both branches of depth 3 in one decoder call)
- self-attention keys / values of a node are computed once and read by its descendants (KV cache over the tree);
- cross-attention keys / values are computed once per user and every row attends to its user's memory in a per-user group
  (padded batched matmul), so the memory is never copied per row;
- depths 1 and 2 run as one decoder call (2 new positions per (f, branch) row, causal); fused q/k/v projections, one
  einsum per attention, the cross-attention keys / values of all layers in one matmul;
- the output projection runs only at the node's own (last) position and only at the depths whose log-probabilities are
  used (fused full-vocabulary log_softmax, then the needed columns). The normaliser stays the FULL vocabulary, as in P1a: restricting it to the current level's codes changes
  every row's normaliser by its own log-mass outside those codes, so it is not ranking-exact (`restrict=True` = that variant,
  reported separately with its own match rate, never as P1a-opt).
- duplicate prefixes: the depth-3 rows are the distinct catalog prefixes (f, c1) under each kept f, already unique per user;
  rows of different users differ in their encoder memory, so there is nothing to deduplicate (checked: `dedupe_check`).
Candidate enumeration, scoring and top-10 are P1a's (mfd_eval.rank_lists, ties by item id).

T5 details replicated from transformers 4.47 modeling_t5 (fp32): RMS layer norm, unscaled dot-product attention, the
decoder's relative position bias (block 0, bidirectional=False) added in every layer, cross-attention with the inverted
encoder mask (finfo.min) and no position bias, relu FF, final layer norm, d_model**-0.5 before the tied output projection.
"""
import torch

from mfd_eval import excl_cumsum
from mfd_mv_infer import encode


def _rms(x, w, eps):
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


class P1aOpt:
    def __init__(self, model, cat, T, I, chunk=32768, attn="auto", out="lsm"):   # defaults = fastest exact variant (r23_p1aopt_bench, Arts)
        assert cat.L == 2, "P1a-opt is written for 1+2+2 SIDs (depth-3 nodes are the last branch level)"
        self.model, self.cat, self.T, self.I, self.chunk = model, cat, T, I, chunk
        self.attn, self.outm = attn, out        # attn: auto (elementwise for 1 new position, einsum otherwise) | einsum | ew; out: lsm | lse
        cfg = model.config
        self.H, self.dk, self.eps = cfg.num_heads, cfg.d_kv, cfg.layer_norm_epsilon
        self.inner = self.H * self.dk
        self.scale = cfg.d_model ** -0.5 if cfg.tie_word_embeddings else 1.0
        dec = model.decoder
        self.E = dec.embed_tokens.weight
        self.W = model.lm_head.weight
        self.lnf = dec.final_layer_norm.weight
        self.layers, ckv = [], []
        for blk in dec.block:
            sa, ca, ff = blk.layer[0], blk.layer[1], blk.layer[2]
            assert not getattr(ff, "is_gated_act", False) and hasattr(ff.DenseReluDense, "wi")
            assert ff.DenseReluDense.act.__class__.__name__.lower().startswith("relu")
            A, C = sa.SelfAttention, ca.EncDecAttention
            self.layers.append(dict(ln_s=sa.layer_norm.weight, qkv=torch.cat([A.q.weight, A.k.weight, A.v.weight]).t().contiguous(),
                                    o=A.o.weight.t().contiguous(), ln_c=ca.layer_norm.weight, cq=C.q.weight.t().contiguous(),
                                    co=C.o.weight.t().contiguous(), ln_f=ff.layer_norm.weight,
                                    wi=ff.DenseReluDense.wi.weight.t().contiguous(), wo=ff.DenseReluDense.wo.weight.t().contiguous()))
            ckv += [C.k.weight, C.v.weight]
        self.ckv = torch.cat(ckv).t().contiguous()                          # (d, n_layers * 2 * inner)
        self.Wt = self.W.t().contiguous()
        with torch.no_grad():                                  # (H, 4, 4): bias of query position p over keys 0..p
            self.pos_bias = dec.block[0].layer[0].SelfAttention.compute_bias(4, 4)[0].float()
        neg = torch.finfo(torch.float32).min                   # decoder causal mask as HF adds it (finfo.min)
        self.causal = torch.triu(torch.full((4, 4), neg, device=self.pos_bias.device), 1)
        self.start = cfg.decoder_start_token_id

    # ---------------------------------------------------------------- building blocks
    def _crosskv(self, enc, m):
        B, E, _ = enc.shape
        kv = (enc @ self.ckv).view(B, E, len(self.layers), 2, self.H, self.dk)
        Kc = [kv[:, :, l, 0].transpose(1, 2) for l in range(len(self.layers))]   # (B, H, E, dk)
        Vc = [kv[:, :, l, 1].transpose(1, 2) for l in range(len(self.layers))]
        mb = (1.0 - m[:, None, None, :].to(enc.dtype)) * torch.finfo(enc.dtype).min
        return Kc, Vc, mb

    @staticmethod
    def _group(u):
        """rows sorted by user u (N,) -> (first user, number of users, local user, slot, width); one host sync."""
        u0 = u[0]
        uu = u - u0
        cnt = torch.bincount(uu)
        pos = torch.arange(len(u), device=u.device) - excl_cumsum(cnt)[uu]
        a, R = torch.stack([u0, cnt.max()]).tolist()
        return a, len(cnt), uu, pos, R

    def _cross(self, q, g, Kc, Vc, mb):
        """q (N, H, dk) sorted by user; per-user batched matmul against the user's memory (never copied per row)."""
        u0, nb, uu, pos, R = g
        if nb == 1:                                                         # one user: no padding needed
            qp = q.permute(1, 0, 2)[None]                                   # (1, H, N, dk)
        else:
            qp = q.new_zeros(nb, R, self.H, self.dk)
            qp[uu, pos] = q
            qp = qp.permute(0, 2, 1, 3)                                     # (nb, H, R, dk)
        s = torch.matmul(qp, Kc[u0:u0 + nb].transpose(3, 2)) + mb[u0:u0 + nb]
        a = torch.softmax(s.float(), dim=-1).type_as(s)
        o = torch.matmul(a, Vc[u0:u0 + nb])                                 # (nb, H, R|N, dk)
        return o[0].permute(1, 0, 2) if nb == 1 else o.permute(0, 2, 1, 3)[uu, pos]

    def _depth(self, tok, p0, anc, u, cx):
        """tok (N, P): P new tokens at positions p0 .. p0+P-1 of every row (causal among them); anc = [(K_l, V_l, idx)]
        for positions 0 .. p0-1 (K_l (M, H, dk)); u (N,) user of every row, sorted. Returns the final hidden states
        (N, P, d) and this call's per-layer (K, V) of shape (N, P, H, dk)."""
        Kc, Vc, mb = cx
        N, P = tok.shape
        H, dk, inner = self.H, self.dk, self.inner
        h = self.E[tok]                                                     # (N, P, d)
        bias = self.pos_bias[:, p0:p0 + P, :p0 + P] + self.causal[p0:p0 + P, :p0 + P]   # (H, P, p0+P)
        uP = u.repeat_interleave(P)
        g = self._group(uP)
        Ks, Vs = [], []
        for l, L in enumerate(self.layers):
            x = _rms(h, L["ln_s"], self.eps)
            q, k, v = (x @ L["qkv"]).view(N, P, 3, H, dk).unbind(2)          # (N, P, H, dk)
            Ks.append(k); Vs.append(v)
            if P == 1 and self.attn != "einsum":                            # one new position: elementwise, no key copies
                q1 = q[:, 0]
                kk = [a[0][l][a[2]] for a in anc] + [k[:, 0]]
                vv = [a[1][l][a[2]] for a in anc] + [v[:, 0]]
                s = torch.stack([(q1 * x).sum(-1) for x in kk], -1) + bias[:, 0]     # (N, H, p0+1)
                w = torch.softmax(s.float(), dim=-1).type_as(s)
                o = sum(w[..., j:j + 1] * vv[j] for j in range(len(vv))).reshape(N, 1, inner)
            else:
                keys = torch.cat([a[0][l][a[2]][:, None] for a in anc] + [k], 1)  # (N, p0+P, H, dk)
                vals = torch.cat([a[1][l][a[2]][:, None] for a in anc] + [v], 1)
                s = torch.einsum("nphd,njhd->nhpj", q, keys) + bias
                w = torch.softmax(s.float(), dim=-1).type_as(s)
                o = torch.einsum("nhpj,njhd->nphd", w, vals).reshape(N, P, inner)
            h = h + o @ L["o"]
            x = _rms(h, L["ln_c"], self.eps)
            cq = (x @ L["cq"]).view(N * P, H, dk)
            h = h + (self._cross(cq, g, Kc[l], Vc[l], mb).reshape(N, P, inner) @ L["co"])
            x = _rms(h, L["ln_f"], self.eps)
            h = h + torch.relu(x @ L["wi"]) @ L["wo"]
        return _rms(h, self.lnf, self.eps), Ks, Vs

    def _out(self, hf, cols, restrict):
        """log P(cols) at the rows' last position; hf (N, d) final hidden states; cols = list of column tensors (same
        rows) -> list of (N, len(c)). Exact: full-vocabulary log-softmax, then the columns."""
        z = hf * self.scale
        if restrict:                                                         # NOT exact (see module doc)
            return [(z @ self.Wt[:, c]).float().log_softmax(-1) for c in cols]
        lg = (z @ self.Wt).float()
        if self.outm == "lse":
            lse = torch.logsumexp(lg, -1, keepdim=True)
            return [lg[:, c] - lse for c in cols]
        lsm = lg.log_softmax(-1)                                             # one fused kernel (as P1a)
        return [lsm[:, c] for c in cols]

    @torch.no_grad()
    def f_logprobs(self, enc, m, restrict=False):
        """depth 0 alone: log P(f) of every f code (for the verification of the kept f codes)."""
        B = enc.shape[0]
        cx = self._crosskv(enc, m)
        ar = torch.arange(B, device=enc.device)
        h0, _, _ = self._depth(torch.full((B, 1), self.start, device=enc.device, dtype=torch.long), 0, [], ar, cx)
        return self._out(h0[:, 0], [self.cat.tok[0]], restrict)[0]

    # ---------------------------------------------------------------- full inference (= infer(), plain cfg)
    @torch.no_grad()
    def infer(self, ids, mask, K, w_t=0.5, w_i=0.5, restrict=False, enc=None, st=None, counts=None):
        st = st or (lambda name: _null())
        cat, dev = self.cat, ids.device
        with st("encoder"):
            if enc is None:
                enc, m = encode(self.model, ids, mask)
            else:
                enc, m = enc
            B = enc.shape[0]
            cx = self._crosskv(enc, m)
        ar = torch.arange(B, device=dev)
        with st("f_step"):                                                  # depth 0
            h0, K0, V0 = self._depth(torch.full((B, 1), self.start, device=dev, dtype=torch.long), 0, [], ar, cx)
            K0, V0 = [k[:, 0] for k in K0], [v[:, 0] for v in V0]
            lpf = self._out(h0[:, 0], [cat.tok[0]], restrict)[0]
            K = min(K, cat.nF)
            fsel = lpf.topk(K, dim=1).indices
        BK = B * K
        fflat = fsel.reshape(-1)
        rows = ar.repeat_interleave(K)
        ftok = cat.tok[0][fflat]
        lv_t, lv_i = cat.levels["t"], cat.levels["i"]
        with st("branch_step1"):                                            # depths 1 + 2: [f, <T>|<I>] per (slot, branch)
            slot2 = torch.arange(BK, device=dev).repeat_interleave(2)        # row = slot * 2 + branch (user-major)
            br = torch.tensor([self.T, self.I], device=dev).repeat(BK)
            h12, K12, V12 = self._depth(torch.stack([ftok[slot2], br], 1), 1, [(K0, V0, rows[slot2])], rows[slot2], cx)
            K1, V1 = [k[0::2, 0] for k in K12], [v[0::2, 0] for v in V12]   # depth-1 node = (slot): its <T> row's position 1
            K2, V2 = [k[:, 1] for k in K12], [v[:, 1] for v in V12]
            h2 = h12[:, 1]
            lp = {("t", 0): self._out(h2[0::2], [cat.tok[lv_t[0]]], restrict)[0],
                  ("i", 0): self._out(h2[1::2], [cat.tok[lv_i[0]]], restrict)[0]}
        with st("prefix_enumeration"):                                      # catalog prefixes (f, c1) under the kept f
            parts, meta = [], {}
            for b_i, (name, lv) in enumerate((("t", lv_t), ("i", lv_i))):
                P = cat.pref[(name, 1)]
                cnt = P["cnt"][fflat]
                slot = torch.arange(BK, device=dev).repeat_interleave(cnt)
                stt = excl_cumsum(cnt)
                local = torch.arange(len(slot), device=dev) - stt[slot]
                codes = P["code"][P["ptr"][fflat[slot]] + local]
                meta[name] = stt
                parts.append((slot, cat.tok[lv[0]][codes[:, 0]], slot * 2 + b_i))
            n_t = len(parts[0][0])
            slot3 = torch.cat([x[0] for x in parts])
            tok3 = torch.cat([x[1] for x in parts])
            par3 = torch.cat([x[2] for x in parts])
            order = torch.argsort(rows[slot3], stable=True)                 # depth-3 rows sorted by user (both branches)
            n3 = len(order)
            is_t = order < n_t
            dest = torch.where(is_t, order, order - n_t)                     # index within its branch
        with st("branch_step2"):                                            # depth 3, both branches in one decoder pass
            ct, ci = cat.tok[lv_t[1]], cat.tok[lv_i[1]]
            out_t = torch.empty(n_t, len(ct), device=dev)
            out_i = torch.empty(n3 - n_t, len(ci), device=dev)
            for s0 in range(0, n3, self.chunk):
                o = order[s0:s0 + self.chunk]
                so = slot3[o]
                h3, _, _ = self._depth(tok3[o][:, None], 3, [(K0, V0, rows[so]), (K1, V1, so), (K2, V2, par3[o])], rows[so], cx)
                tm, d = is_t[s0:s0 + self.chunk], dest[s0:s0 + self.chunk]
                if restrict:
                    out_t[d[tm]] = self._out(h3[tm, 0], [ct], True)[0]
                    out_i[d[~tm]] = self._out(h3[~tm, 0], [ci], True)[0]
                else:
                    a_t, a_i = self._out(h3[:, 0], [ct, ci], False)
                    out_t[d[tm]] = a_t[tm]
                    out_i[d[~tm]] = a_i[~tm]
            lp[("t", 1)], lp[("i", 1)] = out_t, out_i
        with st("candidates_scoring"):                                      # = branch_scores + infer (plain cfg)
            cnt_c = cat.f_item_cnt[fflat]
            slot_c = torch.arange(BK, device=dev).repeat_interleave(cnt_c)
            local_c = torch.arange(len(slot_c), device=dev) - excl_cumsum(cnt_c)[slot_c]
            item = cat.items_by_f[cat.f_item_ptr[fflat[slot_c]] + local_c]
            c = cat.code[item]
            s = {}
            for name in ("t", "i"):
                lv = cat.levels[name]
                v = lp[(name, 0)][slot_c, c[:, lv[0]]]
                row = meta[name][slot_c] + cat.pref[(name, 1)]["item_local"][item]
                v = v + lp[(name, 1)][row, c[:, lv[1]]]
                s[name] = v
            user, frank = slot_c // K, slot_c % K
            fterm = 0 + 1 * lpf                                              # infer(): sum(a_v * lpf_v), a = (1, 0, 0)
            fv = fterm.gather(1, fsel).reshape(-1)[user * K + frank]
            score = fv + (0 + 1 * (w_t * s["t"] + w_i * s["i"]))             # infer(): sum(b_v * (...)), b = (1, 0, 0)
        if counts is not None:
            counts["rows_depth"] = [B, BK, 2 * BK, n3]
            counts["positions_computed"] = B + 4 * BK + n3                   # depth-1 position computed in both branch rows
            counts["candidates"] = len(item)
            counts["enc_len"] = float(m.sum())
        return user, item, score


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def dedupe_check(cat, fsel):
    """Depth-3 prefixes (user, f, branch, c1) are distinct: returns the number of duplicate rows (expected 0)."""
    B, K = fsel.shape
    fflat = fsel.reshape(-1)
    dup = 0
    for name in ("t", "i"):
        P = cat.pref[(name, 1)]
        cnt = P["cnt"][fflat]
        slot = torch.arange(B * K, device=fsel.device).repeat_interleave(cnt)
        local = torch.arange(len(slot), device=fsel.device) - excl_cumsum(cnt)[slot]
        codes = P["code"][P["ptr"][fflat[slot]] + local][:, 0]
        key = (slot // K) * (10 ** 9) + fflat[slot] * 10 ** 4 + codes
        dup += len(key) - len(torch.unique(key))
    return dup
