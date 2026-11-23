"""mmgen addition (docs/MFD_REPORT.md §23): inference-speed measurement of P1a vs the generative baselines (no training, no change
to any model code path; the timed calls are the evaluated ones).

Methods (seed-42 checkpoints):
  P1a:K / A01:K  MFD deployment path of r21_eval (mfd_mv_infer.infer, plain full view, w = 1 / .5 / .5, K_f = K), then the top 10
                 per user (mfd_eval.rank_lists; ties by item id). Dataset order (as evaluated).
  C, D           MQL4GRec / MACRec (paper setting): two constrained beam searches (seqrec = text SIDs, seqimage = image SIDs; HF
                 generate exactly as test_ddp_save: 20 beams, 20 returned, max_new_tokens 10, early_stopping, output_scores,
                 prefix_allowed_tokens_fn = the Python trie), then the MQL4GRec merge (ensemble.py rule) -> top 10.
  TIGER          tiger_text: one constrained beam search (same call), top 10 of the 20 beams (get_topk_results rule, filter_items).
  chain          MD 1+2+2 chain decoder (md122_a05_ep500_reassign, plain T5, 6 decoding steps), same call as TIGER.
  Cfast          C with r23_cfast.GPUTrieProcessor instead of the Python trie (verified identical outputs: r23_prep.py).
  Beam methods use test_ddp_save's user order (DistributedSampler, 1 replica, seed 0 -> randperm) so batch 64 reproduces the
  evaluated batches exactly.
Protocol: inputs pre-tokenised on the CPU before timing (identical for every method: tokenizer collation excluded); timed region
per batch = host->device copy of the inputs + inference + top-10 item ids on the host (incl. C/D's string decode and merge);
fp32 everywhere (checkpoint dtype, torch default matmul precision = no TF32), no torch.compile, no CUDA graphs; HF beam search
uses its KV cache (as evaluated), MFD re-runs its short decoder sequences (as evaluated). Per (method, batch size): warm-up on
the first >= 200 users, then R full timed passes (torch.cuda.synchronize at both ends), ms/user = pass seconds / users.
Batch 1: the fixed 2,000-user subset (r23_prep.py, same users for every method). Batch 256 falls back to 192 / 128 on OOM.
--mode time     timed passes (+ batch-64 lists, HR@10 vs the reported number) -> <out_file>
--mode profile  stage breakdown (stage timers with a sync at every boundary + torch.profiler device time), hardware-independent
                counts and FLOPs (torch.utils.flop_counter, measured on the first 256 users) -> <out_file>
Every job first runs a short anchor (P1a K_f 50 batch 64 on 2,048 users, C seqrec batch 64 on 512 users) to compare jobs.
"""
import argparse
import contextlib
import functools
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import LogitsProcessorList, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

from collator import TestCollatorSave
from data import SeqRecDataset
from generation_trie import Trie
from mfd_data import Catalog, view_maps
from mfd_eval import excl_cumsum, load_mfd_model, load_split, rank_lists
from mfd_model import BRANCH_TOKENS
from mfd_mv_infer import encode, f_logprobs, infer
from r23_cfast import GPUTrieProcessor
from r23_p1aopt import P1aOpt
from utils import parse_dataset_args, parse_global_args, parse_test_args, prefix_allowed_tokens_fn

R = "/scratch/yw8866/mmgen_runs"
I122 = ".index_md122_a05_ep500_reassign.json"
BEAM = {   # method -> (baseline dir, text index, image index, lists)
    "C": ("mql", ".index_lemb_mql.json", ".index_vitemb_mql.json", ("seqrec", "seqimage")),
    "D": ("macrecpaper", ".index_lemb_macrecpaper.json", ".index_vitemb_macrecpaper.json", ("seqrec", "seqimage")),
    "TIGER": ("tiger_text", ".index_lemb_tiger.json", ".index_lemb_tiger.json", ("seqrec",)),
    "chain": ("md122_a05_ep500_reassign", I122, I122, ("seqrec",)),
}
MFD = {"P1a": "mfd122_mv_geoP1a", "A01": "r21_A01", "P1aopt": "mfd122_mv_geoP1a", "P1aoptr": "mfd122_mv_geoP1a"}
# P1aopt = r23_p1aopt.P1aOpt (exact, same scores as P1a up to rounding); P1aoptr = its restricted-normaliser variant (not exact)
NB = 20
OUT_DIR = os.environ.get("R23_OUT", f"{R}/timing/r23")          # R23_OUT: smoke-test output dir


# ------------------------------------------------------------------ stage timers
class Stages:
    """Wall time per stage, synchronised at both boundaries; nested names ('a/b') are sub-stages of 'a'. Each range is also a
    torch.profiler record_function (device time when a profiler is active)."""

    def __init__(self):
        self.t = defaultdict(float)
        self.prefix = ""
        self.step = 0

    @contextlib.contextmanager
    def __call__(self, name):
        full = self.prefix + name
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.record_function(full):
            yield
            torch.cuda.synchronize()
        self.t[full] += time.perf_counter() - t0


NULL = lambda name: contextlib.nullcontext()


def mk_args(D, idx, iidx, extra=()):
    p = argparse.ArgumentParser()
    p = parse_global_args(p); p = parse_dataset_args(p); p = parse_test_args(p)
    return p.parse_args(["--data_path", f"{R}/data/", "--dataset", D, "--index_file", idx, "--image_index_file", iidx,
                         "--test_batch_size", "64", "--num_beams", str(NB), "--filter_items", *extra])


def sampler_order(n):
    """test_ddp_save's DistributedSampler(num_replicas=1, shuffle=True, seed=0, drop_last=True) order."""
    g = torch.Generator(); g.manual_seed(0)
    return torch.randperm(n, generator=g).tolist()


# ------------------------------------------------------------------ beam-search methods (C, D, TIGER, chain, Cfast)
class BeamList:
    def __init__(self, D, ck, idx, iidx, task, tok):
        self.task = task
        args = mk_args(D, idx, iidx, ["--test_task", task])
        self.ds = SeqRecDataset(args, task=task, mode="test", sample_num=args.sample_num)
        self.coll = TestCollatorSave(args, tok)
        self.all_items = self.ds.get_all_items()
        self.trie = Trie([[0] + tok.encode(c) for c in self.all_items])
        self.prefix_fn = prefix_allowed_tokens_fn(self.trie)
        self.sid2item = {"".join(v): int(k) for k, v in self.ds.indices.items()}
        assert len(self.sid2item) == len(self.ds.indices)
        self.targets = [self.sid2item[d["item"]] for d in self.ds.inter_data]
        self.gpu = None

    def collate(self, idx):
        inp, tg, users = self.coll([self.ds[j] for j in idx])
        assert users == list(idx)
        return inp["input_ids"].pin_memory(), inp["attention_mask"].pin_memory()


class BeamMethod:
    kind = "beam"

    def __init__(self, D, name):
        self.name = name
        self.fast = name.endswith("fast")
        base = name[:-4] if self.fast else name
        b, idx, iidx, tasks = BEAM[base]
        self.ck = f"{R}/baselines/{D}/{b}/seed42"
        self.tok = T5Tokenizer.from_pretrained(self.ck)
        self.model = T5ForConditionalGeneration.from_pretrained(self.ck).cuda().eval()
        self.lists = [BeamList(D, self.ck, idx, iidx, t, self.tok) for t in tasks]
        assert all(l.targets == self.lists[0].targets for l in self.lists)
        self.targets = self.lists[0].targets
        self.U = len(self.targets)
        self.order = sampler_order(self.U)
        if self.fast:
            for l in self.lists:
                l.gpu = GPUTrieProcessor(l.trie, self.model.config.vocab_size, "cuda")
        self.counts = None

    def collate(self, idx):
        return [l.collate(idx) for l in self.lists]

    def gen(self, l, ids, mask, st):
        kw = dict(input_ids=ids, attention_mask=mask, max_new_tokens=10, num_beams=NB, num_return_sequences=NB,
                  output_scores=True, return_dict_in_generate=True, early_stopping=True)
        if l.gpu is not None:
            kw["logits_processor"] = LogitsProcessorList([l.gpu])
        else:
            kw["prefix_allowed_tokens_fn"] = l.prefix_fn
        with st("generate"):
            out = self.model.generate(**kw)
        return out["sequences"], out["sequences_scores"]

    def post_list(self, l, seqs, scores, B, st):
        with st("decode"):
            pred = [s.strip().replace(" ", "") for s in self.tok.batch_decode(seqs, skip_special_tokens=True)]
            sc = scores.tolist()
        return pred, sc

    def run(self, batch, st=NULL, raw=None):
        res = []
        for l, (ids, mask) in zip(self.lists, batch):
            st_prefix = getattr(st, "prefix", None)
            if st_prefix is not None:
                st.prefix = l.task + "/"; st.step = 0
            with st("input_copy"):
                ids_d, mask_d = ids.to("cuda", non_blocking=True), mask.to("cuda", non_blocking=True)
            seqs, scores = self.gen(l, ids_d, mask_d, st)
            B = ids.shape[0]
            pred, sc = self.post_list(l, seqs, scores, B, st)
            if raw is not None:
                raw.append((l.task, seqs.cpu(), scores.cpu()))
            res.append((pred, sc))
            if st_prefix is not None:
                st.prefix = ""
        with st("rank_merge"):
            top = merge_top10(self.lists, res) if len(self.lists) == 2 else single_top10(self.lists[0], *res[0])
        return top



class OptMethod(BeamMethod):
    """C-opt lower bound (round-23 addition): only the model forwards of the beam search, replayed at the real shapes:
    per list the encoder on the B users, then n_steps decoder forwards (model.forward with KV cache, one new token) on the
    B x 20 beam rows, exactly the calls HF generate makes (n_steps counted from a real generate of this list). Excluded:
    trie constraint, log_softmax / top-k / BeamSearchScorer, cache reordering, finalize, beam-to-text decoding, merge.
    'optk' adds the minimal GPU beam step back (log_softmax + top-2k over B x 20 x V, no constraint).
    Token values are the decoder start token (compute does not depend on them). Returns no lists."""
    def __init__(self, D, name):
        self.with_topk = name.endswith("optk")
        base = name[:-4] if self.with_topk else name[:-3]
        super().__init__(D, base)
        self.name, self.base = name, base
        self.replay = True
        self.steps = {}
        for l in self.lists:                                 # count model.forward calls of one real generate per list
            ids, mask = l.collate(self.order[:8])
            n = [0]
            f0 = self.model.forward
            def cnt(*a, **k):
                n[0] += 1
                return f0(*a, **k)
            self.model.forward = cnt
            try:
                self.gen(l, ids.cuda(), mask.cuda(), NULL)
            finally:
                del self.model.forward
            self.steps[l.task] = n[0]

    @torch.no_grad()
    def run(self, batch, st=NULL, raw=None):
        if not self.replay:
            return BeamMethod.run(self, batch, st, raw)
        from transformers.modeling_outputs import BaseModelOutput
        model, start = self.model, self.model.config.decoder_start_token_id
        B = batch[0][0].shape[0]
        for l, (ids, mask) in zip(self.lists, batch):
            ids, mask = ids.to("cuda", non_blocking=True), mask.to("cuda", non_blocking=True)
            h = model.encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            eo = BaseModelOutput(last_hidden_state=h.repeat_interleave(NB, 0))
            am = mask.repeat_interleave(NB, 0)
            dec = torch.full((B * NB, 1), start, dtype=torch.long, device="cuda")
            past = None
            for t in range(self.steps[l.task]):
                out = model(encoder_outputs=eo, attention_mask=am, decoder_input_ids=dec, past_key_values=past,
                            use_cache=True, return_dict=True)
                past = out.past_key_values
                if self.with_topk:
                    lp = torch.log_softmax(out.logits[:, -1, :], dim=-1).view(B, -1)
                    lp.topk(2 * NB, dim=1)
        return [[-1] * 10 for _ in range(B)]


def single_top10(l, pred, sc):
    """evaluate.get_topk_results with filter_items: beams not in the catalog get -1000; stable sort by score; top 10."""
    out = []
    for b in range(len(pred) // NB):
        pairs = [(p, s if p in l.all_items else -1000) for p, s in zip(pred[b * NB:(b + 1) * NB], sc[b * NB:(b + 1) * NB])]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        out.append([l.sid2item.get(p, -1) for p, _ in pairs])
    return out


def merge_top10(lists, res):
    """ensemble.get_sort_results + get_topk_results_ensemble (MQL4GRec merge of the two 20-beam lists), returning item ids."""
    (pt, st_), (pi, si) = res
    out = []
    for b in range(len(pt) // NB):
        d = {}
        for i in range(NB):
            for l, p, s in ((lists[0], pt, st_), (lists[1], pi, si)):
                item = l.sid2item.get(p[b * NB + i], -1)
                score = s[b * NB + i]
                if item == -1:
                    score = -1000
                if item in d and item != -1:
                    d[item] = (score + d[item]) / 2 + 1
                else:
                    d[item] = score
        out.append([x for x, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:10]])
    return out


@contextlib.contextmanager
def instrument_beam(m, st):
    """Stage timers inside HF generate (identical computation): encoder, each decoder step (model forward), constraint
    (Python trie callback / per-row mask writes / rest, or the GPU trie), beam bookkeeping (scorer process / finalize, KV-cache
    reorder); the rest of generate (log-softmax, top-k, concatenation, setup) = generate - these."""
    model = m.model
    enc_f, fwd_f, reo_f = model.encoder.forward, model.forward, model._temporary_reorder_cache
    proc_f, fin_f, pc_f, gt_f = BeamSearchScorer.process, BeamSearchScorer.finalize, PrefixConstrainedLogitsProcessor.__call__, GPUTrieProcessor.__call__

    @functools.wraps(enc_f)
    def enc_w(*a, **k):
        with st("encoder"):
            return enc_f(*a, **k)

    @functools.wraps(fwd_f)
    def fwd_w(*a, **k):
        st.step += 1
        with st(f"decoder_step{st.step}"):
            return fwd_f(*a, **k)

    @functools.wraps(reo_f)
    def reo_w(*a, **k):
        with st("beam/cache_reorder"):
            return reo_f(*a, **k)

    def proc_w(self, *a, **k):
        with st("beam/scorer_process"):
            return proc_f(self, *a, **k)

    def fin_w(self, *a, **k):
        with st("beam/scorer_finalize"):
            return fin_f(self, *a, **k)

    def pc_w(self, input_ids, scores):                       # = PrefixConstrainedLogitsProcessor.__call__ (HF 4.47) + timers
        with st("constraint"):
            mask = torch.full_like(scores, -math.inf)
            tcb = tw = 0.0
            for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
                for beam_id, sent in enumerate(beam_sent):
                    t0 = time.perf_counter()
                    allowed = self._prefix_allowed_tokens_fn(batch_id, sent)
                    t1 = time.perf_counter()
                    if len(allowed) == 0:
                        raise ValueError("empty prefix_allowed_tokens")
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0
                    tw += time.perf_counter() - t1
                    tcb += t1 - t0
            out = scores + mask
        st.t[st.prefix + "constraint/trie_callback_cpu"] += tcb
        st.t[st.prefix + "constraint/mask_row_writes_launch"] += tw
        return out

    def gt_w(self, input_ids, scores):
        with st("constraint"):
            return gt_f(self, input_ids, scores)

    model.encoder.forward, model.forward, model._temporary_reorder_cache = enc_w, fwd_w, reo_w
    BeamSearchScorer.process, BeamSearchScorer.finalize = proc_w, fin_w
    PrefixConstrainedLogitsProcessor.__call__, GPUTrieProcessor.__call__ = pc_w, gt_w
    try:
        yield
    finally:
        del model.encoder.forward, model.forward, model._temporary_reorder_cache
        BeamSearchScorer.process, BeamSearchScorer.finalize = proc_f, fin_f
        PrefixConstrainedLogitsProcessor.__call__, GPUTrieProcessor.__call__ = pc_f, gt_f


# ------------------------------------------------------------------ MFD methods (P1a, A01)
class MFDMethod:
    kind = "mfd"

    def __init__(self, D, name, seed=42):
        self.name = name
        run, K = name.split(":")
        self.K = int(K)
        # "opt@<run dir>": any 1+2+2 MFD checkpoint mfd/<D>/<run dir>/seed<S> on the P1a-opt path (§24 A2 latency)
        self.ck = f"{R}/mfd/{D}/{run[4:]}/seed{seed}" if run.startswith("opt@") else f"{R}/mfd/{D}/{MFD[run]}/seed{seed}"
        self.args = mk_args(D, I122, I122, ["--max_his_len", "20", "--tasks", "seqrec_mv", "--prompt_num", "4"])
        self.tok = T5Tokenizer.from_pretrained(self.ck)
        self.model = load_mfd_model(self.ck).cuda().eval()
        self.model.config.use_cache = False
        self.cat = Catalog(self.args, self.tok, "cuda")
        self.T, self.I = self.tok.convert_tokens_to_ids(list(BRANCH_TOKENS))
        self.ds, self.coll = load_split(self.args, "test", self.tok)
        self.vmap = view_maps(self.ds.indices)
        self.targets = [self.ds.sid2item[d["item"]] for d in self.ds.inter_data]
        self.U = len(self.targets)
        self.order = list(range(self.U))                          # dataset order (r21_eval)
        self.cfg = dict(variant="plain", K=self.K, w_t=0.5, w_i=0.5, a=(1, 0, 0), b=(1, 0, 0))
        self.popt = P1aOpt(self.model, self.cat, self.T, self.I) if run.startswith(("P1aopt", "opt@")) else None
        self.restrict = run == "P1aoptr"

    def collate(self, idx):
        b = [dict(input_ids="".join(self.vmap["full"][h] for h in self.ds.inter_data[j]["inters"]),
                  labels=self.ds.sid2item[self.ds.inter_data[j]["item"]]) for j in idx]
        x = self.coll(b)
        return x["input_ids"].pin_memory(), x["attention_mask"].pin_memory()

    def top10(self, user, item, score, B):
        ids = rank_lists(user, item, score, B, self.cat.N)[0][:, :10]
        ids = torch.where(ids < self.cat.N, ids, torch.full_like(ids, -1))
        return ids.cpu().tolist()

    def run(self, batch, st=NULL, raw=None):
        ids, mask = batch
        if self.popt is not None:
            with st("input_copy"):
                ids, mask = ids.to("cuda", non_blocking=True), mask.to("cuda", non_blocking=True)
            user, item, score = self.popt.infer(ids, mask, self.K, restrict=self.restrict, st=st if isinstance(st, Stages) else None)
            if raw is not None:
                raw.append((user.cpu(), item.cpu(), score.cpu()))
            with st("top10"):
                return self.top10(user, item, score, ids.shape[0])
        if isinstance(st, Stages):
            return self.run_stages(ids, mask, st, raw)
        vb = {"full": {"input_ids": ids, "attention_mask": mask}}           # infer() moves them to the GPU
        user, item, score = infer(self.model, self.cat, vb, self.cfg, self.T, self.I, False)
        if raw is not None:
            raw.append((user.cpu(), item.cpu(), score.cpu()))
        return self.top10(user, item, score, ids.shape[0])

    def run_stages(self, ids, mask, st, raw=None, counts=None):
        """infer() (plain, full view) + mfd_mv_infer.branch_scores with a timer around every stage; identical operations
        (the prefix enumeration of both branches is done before their decoder calls; independent tensors)."""
        model, cat, K, T, I = self.model, self.cat, self.K, self.T, self.I
        with st("input_copy"):
            ids, mask = ids.to("cuda", non_blocking=True), mask.to("cuda", non_blocking=True)
        with st("encoder"):
            enc, m = encode(model, ids, mask)
        with st("f_step"):
            lpf = f_logprobs(model, cat, enc, m)
            fsel = lpf.topk(min(K, cat.nF), dim=1).indices
        dev = enc.device
        B, K = fsel.shape
        L = cat.L
        start = model.config.decoder_start_token_id
        ar = torch.arange(B, device=dev)
        BK = B * K
        fflat = fsel.reshape(-1)
        rows = ar.repeat_interleave(K)
        ftok = cat.tok[0][fflat]
        z = torch.full_like(ftok, start)
        lv_t, lv_i = cat.levels["t"], cat.levels["i"]
        with st("branch_step1"):                                  # decoder pass 2: t1 and i1 after every kept f
            dec2 = torch.cat([torch.stack([z, ftok, torch.full_like(ftok, T)], 1), torch.stack([z, ftok, torch.full_like(ftok, I)], 1)])
            lp2t, lp2i = model.last_logprobs(enc, m, torch.cat([rows, rows]), dec2, cols=[cat.tok[lv_t[0]], cat.tok[lv_i[0]]])
            lp = {("t", 0): lp2t[:BK], ("i", 0): lp2i[BK:]}
        meta, decs = {}, {}
        n_rows = {"pass1": B, "pass2": 2 * BK, "pass3": 0}
        with st("prefix_enumeration"):                            # catalog prefixes (f, c_1) under the kept f
            for d in range(1, L):
                for name, branch_tok in (("t", T), ("i", I)):
                    lv = cat.levels[name]
                    P = cat.pref[(name, d)]
                    cnt = P["cnt"][fflat]
                    slot = torch.arange(BK, device=dev).repeat_interleave(cnt)
                    stt = excl_cumsum(cnt)
                    local = torch.arange(len(slot), device=dev) - stt[slot]
                    codes = P["code"][P["ptr"][fflat[slot]] + local]
                    toks = torch.stack([cat.tok[lv[j]][codes[:, j]] for j in range(d)], 1)
                    decs[(name, d)] = (slot, torch.cat([torch.full_like(slot, start)[:, None], ftok[slot][:, None],
                                                        torch.full_like(slot, branch_tok)[:, None], toks], 1))
                    meta[(name, d)] = stt
                    n_rows["pass3"] += len(slot)
        with st("branch_step2"):                                  # decoder pass 3: t2 / i2 for every kept prefix
            for d in range(1, L):
                for name in ("t", "i"):
                    slot, dec = decs[(name, d)]
                    lp[(name, d)] = model.last_logprobs(enc, m, rows[slot], dec, cols=[cat.tok[cat.levels[name][d]]])[0]
        with st("candidates_scoring"):                            # every item under a kept f, exact score
            cnt_c = cat.f_item_cnt[fflat]
            slot_c = torch.arange(BK, device=dev).repeat_interleave(cnt_c)
            local_c = torch.arange(len(slot_c), device=dev) - excl_cumsum(cnt_c)[slot_c]
            item = cat.items_by_f[cat.f_item_ptr[fflat[slot_c]] + local_c]
            c = cat.code[item]
            s = {}
            for name in ("t", "i"):
                lv = cat.levels[name]
                v = lp[(name, 0)][slot_c, c[:, lv[0]]]
                for d in range(1, L):
                    row = meta[(name, d)][slot_c] + cat.pref[(name, d)]["item_local"][item]
                    v = v + lp[(name, d)][row, c[:, lv[d]]]
                s[name] = v
            user, frank = slot_c // K, slot_c % K
            fv = lpf.gather(1, fsel).reshape(-1)[user * K + frank]
            score = fv + 1 * (self.cfg["w_t"] * s["t"] + self.cfg["w_i"] * s["i"])
        with st("top10"):
            top = self.top10(user, item, score, B)
        if raw is not None:
            raw.append((user.cpu(), item.cpu(), score.cpu()))
        if counts is not None:
            counts["enc_len"] += float(m.sum()); counts["enc_len_padded"] += float(m.shape[1] * B)
            counts["rows_pass1"] += n_rows["pass1"]; counts["rows_pass2"] += n_rows["pass2"]; counts["rows_pass3"] += n_rows["pass3"]
            counts["candidates"] += len(item); counts["users"] += B
        return top


def make(D, name):
    if ":" in name:
        return MFDMethod(D, name)
    return OptMethod(D, name) if name.endswith(("opt", "optk")) else BeamMethod(D, name)


# ------------------------------------------------------------------ timing
def batches(m, order, bs):
    return [order[s:s + bs] for s in range(0, len(order), bs)]


def precollate(m, order, bs):
    return [(idx, m.collate(idx)) for idx in batches(m, order, bs)]


@torch.no_grad()
def timed_passes(m, order, bs, repeats, warmup):
    pre = precollate(m, order, bs)
    nw = max(1, math.ceil(warmup / bs))
    for idx, b in pre[:nw]:
        m.run(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    secs, lists = [], None
    for r in range(repeats):
        out = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for idx, b in pre:
            out.extend(m.run(b))
        torch.cuda.synchronize(); secs.append(time.perf_counter() - t0)
        if r == 0:
            lists = out
    return secs, lists, torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9


def run_bs(m, order, bs, repeats, warmup):
    """bs, with the OOM fallback 256 -> 192 -> 128 (state it in the output)."""
    for b in [bs] + ([192, 128] if bs == 256 else []):
        try:
            return b, timed_passes(m, order, b, repeats, warmup)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[r23] {m.name} OOM at batch {b}", flush=True)
    raise RuntimeError("OOM at every fallback batch size")


def reported_hr10(D, name):
    if ":" in name:                     # K_f 50: r21_eval (the reported main setting); K_f 100: eval_ii (§19 setting (ii));
        run, K = name.split(":")        # K_f 20 at w = 1/.5/.5 was never reported: r23_prep's exact-scoring reference
        run = "P1a" if run.startswith("P1aopt") else run   # P1a-opt must reproduce P1a's reported numbers
        if run.startswith("opt@"):                          # a round-24 run on the P1a-opt path: its own r21_eval (K_f 50)
            f = f"{R}/mfd/{D}/{run[4:]}/seed42/r21_eval/eval.json"
            return (json.load(open(f))["test"]["plain"]["hr10"] if K == "50" else None), f
        ck = f"{R}/mfd/{D}/{MFD[run]}/seed42"
        if K == "50":
            f = f"{ck}/r21_eval/eval.json"
            return json.load(open(f))["test"]["plain"]["hr10"], f
        if K == "100" and run == "P1a":
            f = f"{ck}/eval_ii/results_test.json"
            return json.load(open(f))["modes"]["joint"]["hit@10"], f
        ref = json.load(open(f"{OUT_DIR}/{D}/prep.json"))["p1a_ref"]
        key = f"{run}:{K}:s42"
        return (ref[key]["hr10"] if key in ref else None), f"r23_prep exact-scoring ranks ({key}; not a reported run)"
    base = name[:-4] if name.endswith("fast") else name
    b, _, _, tasks = BEAM[base]
    e = f"{R}/baselines/{D}/{b}/seed42/eval/test"
    f = f"{e}/results_ensemble_20.json" if len(tasks) == 2 else f"{e}/results_seqrec_20.json"
    d = json.load(open(f))
    return d.get("mean_results", d)["hit@10"], f


def hr10(lists, targets):
    return float(np.mean([t in l[:10] for l, t in zip(lists, targets)]))


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "nvidia_smi": smi, "torch": torch.__version__,
            "cuda_torch": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
            "transformers": __import__("transformers").__version__, "cpu": cpu, "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job": os.environ.get("SLURM_JOB_ID"), "precision": "fp32 (tf32 matmul %s)" % torch.backends.cuda.matmul.allow_tf32,
            "torch_compile": False, "cuda_graphs": False}


def anchor(D):
    """Short fixed measurement repeated in every job (compare hardware / node between jobs)."""
    out = {}
    for name, n in (("P1a:50", 2048), ("C", 512)):
        m = make(D, name)
        secs, _, _, _ = timed_passes(m, m.order[:n], 64, 1, 200)
        out[name] = {"users": n, "ms_per_user": 1000 * secs[0] / n}
        del m
        torch.cuda.empty_cache()
    return out


def subset(D):
    return json.load(open(f"{OUT_DIR}/{D}/prep.json"))["batch1_subset"]


def mode_time(a):
    res = {"env": env_info(), "dataset": a.dataset, "repeats": a.repeats, "warmup_users": a.warmup, "runs": {}}
    if a.anchor:
        res["anchor"] = anchor(a.dataset)
        print("[anchor]", json.dumps(res["anchor"]), flush=True)
    os.makedirs(f"{OUT_DIR}/{a.dataset}/lists", exist_ok=True)
    sub = subset(a.dataset)
    for name in a.methods.split(","):
        m = make(a.dataset, name)
        for bs in [int(x) for x in a.batches.split(",")]:
            order = sub if bs == 1 else m.order
            if a.limit:
                order = order[:a.limit]
            used, (secs, lists, peak, peak_res) = run_bs(m, order, bs, a.repeats, a.warmup)
            ms = [1000 * s / len(order) for s in secs]
            r = {"method": name, "batch": bs, "batch_used": used, "users": len(order), "seconds": secs, "ms_per_user": ms,
                 "ms_mean": float(np.mean(ms)), "ms_std": float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
                 "users_per_s": float(len(order) / np.mean(secs)), "peak_alloc_gb": peak, "peak_reserved_gb": peak_res}
            tg = [m.targets[j] for j in order]
            r["hr10"] = hr10(lists, tg)
            if isinstance(m, OptMethod):                     # no outputs; check the replay does the same model FLOPs as the real run
                r["hr10"] = None
                r["replay_steps_per_list"] = m.steps
                if bs == 64 and "flop_check" not in res:
                    res["flop_check"] = {}
                if bs == 64 and m.base not in res["flop_check"]:
                    first = m.order[:256]
                    fr = flops(m, first); m.replay = False; fb = flops(m, first); m.replay = True
                    ok = all(abs(fr[k] - fb[k]) <= 1e-6 * max(1.0, fb[k]) for k in ("encoder_gflop_per_user", "decoder_incl_lm_head_gflop_per_user"))
                    res["flop_check"][m.base] = {"replay": fr, "real": fb, "model_flops_equal": ok}
                    print(f"[copt] {m.base} replay model FLOPs == real beam search: {ok} "
                          f"(enc {fr['encoder_gflop_per_user']:.4f}/{fb['encoder_gflop_per_user']:.4f}, "
                          f"dec {fr['decoder_incl_lm_head_gflop_per_user']:.4f}/{fb['decoder_incl_lm_head_gflop_per_user']:.4f} GFLOP/user)", flush=True)
            elif bs != 1 and not a.limit:
                rep, src = reported_hr10(a.dataset, name)
                r["hr10_reported"], r["hr10_reported_source"] = rep, src
                r["hr10_match"] = None if rep is None else abs(r["hr10"] - rep) < 0.5 / len(order)
                aligned = [None] * m.U
                for j, l in zip(order, lists):
                    aligned[j] = l
                lf = f"{OUT_DIR}/{a.dataset}/lists/{name.replace(':', '_K')}_b{bs}.pt"
                torch.save({"top10": aligned, "targets": m.targets}, lf)
                r["lists_file"] = lf
            res["runs"][f"{name}@{bs}"] = r
            print(f"[time] {a.dataset} {name} batch {used}: {r['ms_mean']:.3f} +- {r['ms_std']:.3f} ms/user, "
                  f"HR@10 {r['hr10'] if r['hr10'] is None else round(r['hr10'], 4)} (reported {r.get('hr10_reported')}, match {r.get('hr10_match')}), peak {peak:.2f} GB", flush=True)
            json.dump(res, open(a.out_file, "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    json.dump(res, open(a.out_file, "w"), indent=1)


# ------------------------------------------------------------------ profile: stages, counts, FLOPs
def flops(m, order):
    """FLOPs split by module with forward hooks on model.encoder / model.decoder / model.lm_head (P1a's inference calls the
    T5 stacks directly, where FlopCounterMode's module names cannot tell encoder from decoder). other = outside those modules
    (MFD candidate scoring, beam bookkeeping)."""
    from torch.utils.flop_counter import FlopCounterMode
    pre = precollate(m, order, 64)
    fc = FlopCounterMode(display=False, depth=None)
    acc, start, hs = defaultdict(int), {}, []
    po = getattr(m, "popt", None)
    if po is not None:                                       # P1a-opt never calls model.decoder / lm_head: wrap its own parts
        def wrap(fn, part):
            def w(*a, **k):
                s0 = fc.get_total_flops()
                r = fn(*a, **k)
                acc[part] += fc.get_total_flops() - s0
                return r
            return w
        po._depth, po._crosskv, po._out = wrap(po._depth, "decoder"), wrap(po._crosskv, "decoder"), wrap(po._out, "lm_head")
    for part in ("encoder", "decoder", "lm_head"):
        mod = getattr(m.model, part)
        def pre_h(mod, args, part=part):
            start[part] = fc.get_total_flops()
        def post_h(mod, args, out, part=part):
            acc[part] += fc.get_total_flops() - start.pop(part)
        hs += [mod.register_forward_pre_hook(pre_h), mod.register_forward_hook(post_h)]
    try:
        with torch.no_grad(), fc:
            for idx, b in pre:
                m.run(b)
    finally:
        for h in hs:
            h.remove()
        if po is not None:
            del po._depth, po._crosskv, po._out
    glob, n = fc.get_total_flops(), len(order)
    enc, dec, lm = acc["encoder"], acc["decoder"], acc["lm_head"]
    return {"users": n, "encoder_gflop_per_user": enc / n / 1e9, "decoder_incl_lm_head_gflop_per_user": (dec + lm) / n / 1e9,
            "total_gflop_per_user": glob / n / 1e9, "other_gflop_per_user": (glob - enc - dec - lm) / n / 1e9,
            "lm_head_gflop_per_user": lm / n / 1e9,
            "method": "measured: torch.utils.flop_counter.FlopCounterMode (matmul / bmm / addmm / sdpa ops; 2 FLOP per MAC), "
                      "split by forward hooks on encoder / decoder / lm_head"}


@torch.no_grad()
def beam_counts(m, order):
    c = defaultdict(float)
    for idx, b in precollate(m, order, 64):
        for l, (ids, mask) in zip(m.lists, b):
            seqs, _ = m.gen(l, ids.cuda(), mask.cuda(), NULL)
            B = ids.shape[0]
            c[f"{l.task}/enc_len"] += float(mask.sum()); c[f"{l.task}/enc_len_padded"] += float(mask.shape[1] * B)
            c[f"{l.task}/steps"] += (seqs.shape[1] - 1) * B
        c["users"] += len(idx)
    U = c["users"]
    out = {"users": U, "lists": len(m.lists)}
    for l in m.lists:
        t = l.task
        steps = c[f"{t}/steps"] / U
        out[t] = {"sequential_decoder_steps": steps, "mean_encoder_len": c[f"{t}/enc_len"] / U,
                  "mean_encoder_len_padded": c[f"{t}/enc_len_padded"] / U, "cross_attention_memory_len": c[f"{t}/enc_len"] / U,
                  "decoder_rows_per_step": NB, "decoder_token_positions_per_user": NB * steps}
    out["sequential_decoder_steps_total"] = sum(out[l.task]["sequential_decoder_steps"] for l in m.lists)
    out["decoder_token_positions_per_user_total"] = sum(out[l.task]["decoder_token_positions_per_user"] for l in m.lists)
    return out


@torch.no_grad()
def popt_counts(m, order):
    """P1a-opt: one decoder position per tree node (KV cache), output projection only at depths 0, 2, 3."""
    c = defaultdict(float)
    for idx, (ids, mask) in precollate(m, order, 64):
        k = {}
        m.popt.infer(ids.cuda(), mask.cuda(), m.K, restrict=m.restrict, counts=k)
        for d, n in enumerate(k["rows_depth"]):
            c[f"d{d}"] += n
        c["candidates"] += k["candidates"]; c["users"] += ids.shape[0]
        c["enc_len"] += k["enc_len"]; c["positions"] += k["positions_computed"]
    U = c["users"]
    rows = [c[f"d{d}"] / U for d in range(4)]
    return {"users": U, "sequential_decoder_steps": 3, "mean_encoder_len": c["enc_len"] / U, "cross_attention_memory_len": c["enc_len"] / U,
            "decoder_rows_per_step": [rows[0], rows[1] + rows[2], rows[3]], "decoder_rows_per_depth": rows,
            "decoder_token_positions_per_user": c["positions"] / U, "output_projection_rows_per_user": rows[0] + rows[2] + rows[3],
            "candidates_scored_per_user": c["candidates"] / U, "K_f": m.K,
            "note": "tree KV cache: one new position per node; decoder calls = depth 0 | depths 1+2 | depth 3 (both branches)"}


@torch.no_grad()
def mfd_counts(m, order):
    if m.popt is not None:
        return popt_counts(m, order)
    c = defaultdict(float)
    st = Stages()
    for idx, (ids, mask) in precollate(m, order, 64):
        m.run_stages(ids, mask, st, counts=c)
    U = c["users"]
    r = {k: c[k] / U for k in ("rows_pass1", "rows_pass2", "rows_pass3", "candidates")}
    return {"users": U, "sequential_decoder_steps": 1 + m.cat.L, "mean_encoder_len": c["enc_len"] / U,
            "mean_encoder_len_padded": c["enc_len_padded"] / U, "cross_attention_memory_len": c["enc_len"] / U,
            "decoder_rows_per_step": [r["rows_pass1"], r["rows_pass2"], r["rows_pass3"]],
            "decoder_token_positions_per_user": r["rows_pass1"] * 1 + r["rows_pass2"] * 3 + r["rows_pass3"] * 4,
            "candidates_scored_per_user": r["candidates"], "K_f": m.K,
            "note": "no KV cache: pass 1 = [start], pass 2 = [start, f, <T>|<I>], pass 3 = [start, f, <T>|<I>, c1] (all positions recomputed)"}


@torch.no_grad()
def stages_run(m, order, bs, gpu_prof_users):
    pre = precollate(m, order, bs)
    for idx, b in pre[:max(1, math.ceil(200 / bs))]:
        m.run(b)
    st = Stages()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if m.kind == "beam":
        with instrument_beam(m, st):
            for idx, b in pre:
                m.run(b, st)
    else:
        for idx, b in pre:
            m.run(b, st)
    torch.cuda.synchronize(); total = time.perf_counter() - t0
    n = len(order)
    out = {"users": n, "batch": bs, "instrumented_total_ms_per_user": 1000 * total / n,
           "stages_ms_per_user": {k: 1000 * v / n for k, v in sorted(st.t.items())}}
    if m.kind == "beam":                                     # generate - (encoder, decoder steps, constraint, beam bookkeeping)
        for l in m.lists:
            p = l.task + "/"
            inner = sum(v for k, v in st.t.items() if k.startswith(p) and k.count("/") == 1 and k != p + "generate"
                        and k.split("/")[1] not in ("input_copy", "decode")) + sum(v for k, v in st.t.items() if k.startswith(p + "beam/"))
            out["stages_ms_per_user"][p + "generate_other"] = 1000 * (st.t[p + "generate"] - inner) / n
    # device (kernel) time per stage on a sub-sample, torch.profiler (record_function ranges)
    if gpu_prof_users:
        sub = pre[:max(1, gpu_prof_users // bs)]
        st2 = Stages()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            if m.kind == "beam":
                with instrument_beam(m, st2):
                    for idx, b in sub:
                        m.run(b, st2)
            else:
                for idx, b in sub:
                    m.run(b, st2)
        nu = sum(len(i) for i, _ in sub)
        dev_t = {}
        for e in prof.key_averages():
            if e.key in st2.t:
                dt = getattr(e, "device_time_total", None)
                if dt is None:
                    dt = getattr(e, "cuda_time_total", 0)
                dev_t[e.key] = dt / 1000 / nu                 # us -> ms per user
        out["profiler_users"] = nu
        out["device_kernel_ms_per_user"] = dict(sorted(dev_t.items()))
        out["profiled_wall_ms_per_user"] = {k: 1000 * v / nu for k, v in sorted(st2.t.items())}
    return out


def mode_profile(a):
    res = {"env": env_info(), "dataset": a.dataset, "methods": {}}
    sub = subset(a.dataset)
    for name in a.methods.split(","):
        m = make(a.dataset, name)
        r = {}
        first = m.order[:256]
        t0 = time.time()
        r["counts"] = mfd_counts(m, first) if m.kind == "mfd" else beam_counts(m, first)
        r["flops"] = flops(m, first)
        print(f"[profile] {name} counts + flops {time.time() - t0:.0f} s", flush=True)
        if a.stage_batches:
            r["stages"] = {}
            for bs in [int(x) for x in a.stage_batches.split(",")]:
                order = sub[:a.limit or a.b1_stage_users] if bs == 1 else m.order[:a.limit or a.stage_users]
                gp = a.gpu_prof_users if bs != 1 else min(a.gpu_prof_users, 200)
                if m.kind == "beam":                            # Python-trie beam search: ~1M profiler events per 256 users
                    gp = min(gp, 256 if bs != 1 else 100)
                t0 = time.time()
                r["stages"][str(bs)] = stages_run(m, order, bs, gp)
                print(f"[profile] {name} stages batch {bs} ({len(order)} users, profiler {gp}) {time.time() - t0:.0f} s", flush=True)
        res["methods"][name] = r
        print(f"[profile] {a.dataset} {name}", json.dumps(r)[:3000], flush=True)
        json.dump(res, open(a.out_file, "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    json.dump(res, open(a.out_file, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("time", "profile"), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--methods", required=True, help="comma list: P1a:20,P1a:50,P1a:100,A01:50,C,Cfast,D,TIGER,chain; Copt / Coptk etc. = model forwards only (C-opt lower bound)")
    p.add_argument("--batches", default="1,64,256")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--anchor", action="store_true")
    p.add_argument("--stage_batches", default="", help="profile: batch sizes of the stage breakdown (e.g. 64,1)")
    p.add_argument("--stage_users", type=int, default=2560, help="profile: users of the batch-64 stage breakdown")
    p.add_argument("--b1_stage_users", type=int, default=500, help="profile: users of the batch-1 stage breakdown (first of the fixed subset)")
    p.add_argument("--gpu_prof_users", type=int, default=640)
    p.add_argument("--limit", type=int, default=0, help="smoke test only: first N users of every order")
    p.add_argument("--out_file", required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False                # torch default, stated explicitly (fp32 for every method)
    os.makedirs(os.path.dirname(a.out_file), exist_ok=True)
    (mode_time if a.mode == "time" else mode_profile)(a)


if __name__ == "__main__":
    main()
