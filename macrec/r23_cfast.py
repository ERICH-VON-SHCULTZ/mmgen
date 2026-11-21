"""mmgen addition (docs/MFD_REPORT.md §23): C-fast, GPU-side trie masking for MACRec's constrained beam search.

Original C (test_ddp_save.py): generate(prefix_allowed_tokens_fn=...) -> HF 4.47 PrefixConstrainedLogitsProcessor, which per
decoding step loops over every (user, beam) row in Python, calls the trie (generation_trie.Trie.get on sentence.tolist(), one
device->host copy per row) and writes one indexed row of the mask (one small kernel launch + host->device copy per row):
    mask = full_like(scores, -inf); mask[row, allowed] = 0 (per row); return scores + mask
C-fast: the SAME trie (built from the same Trie object), flattened once into GPU tensors:
    edge table  key = node * V + token (sorted), child node per edge  -> a beam row's node = walk of its decoder input ids
                with torch.searchsorted (one batched lookup per position, all rows at once)
    CSR         children of every node (ptr, tokens)                    -> allowed (row, token) pairs by repeat_interleave
and the same mask arithmetic (full_like -inf, set allowed to 0, scores + mask), so the processed scores are bitwise identical
to the original's whenever the allowed sets are identical. Same model, beams, max_new_tokens, early stopping, output scores.
Empty allowed set -> ValueError, as HF's processor. Passed as generate(logits_processor=[GPUTrieProcessor]) with no
prefix_allowed_tokens_fn (the only other active processor list entries for these configs: none; verified by output equality
on the full test set, r23_prep.py).
"""
import math

import torch
from transformers.generation.logits_process import LogitsProcessor


class GPUTrieProcessor(LogitsProcessor):
    def __init__(self, trie, vocab_size, device):
        V = int(vocab_size)
        nodes = [trie.trie_dict]                      # node 0 = root (before the decoder start token)
        e_par, e_tok, e_child = [], [], []
        ptr, kids = [0], []
        i = 0
        while i < len(nodes):                          # BFS; children in the trie dict's insertion order
            d = nodes[i]
            for t, sub in d.items():
                assert 0 <= t < V
                e_par.append(i); e_tok.append(t); e_child.append(len(nodes)); kids.append(t)
                nodes.append(sub)
            ptr.append(len(kids))
            i += 1
        n = len(nodes)
        self.V, self.n_nodes, self.dead = V, n, n      # dead = off-trie state (no children)
        key = torch.tensor(e_par, dtype=torch.long) * V + torch.tensor(e_tok, dtype=torch.long)
        o = torch.argsort(key)
        self.keys = key[o].to(device)
        self.child = torch.tensor(e_child, dtype=torch.long)[o].to(device)
        p = torch.tensor(ptr, dtype=torch.long)
        cnt = p[1:] - p[:-1]
        self.cnt = torch.cat([cnt, torch.zeros(1, dtype=torch.long)]).to(device)        # dead node: 0 children
        self.ptr = torch.cat([p[:-1], torch.zeros(1, dtype=torch.long)]).to(device)
        self.kids = torch.tensor(kids, dtype=torch.long).to(device)
        self.E = len(e_par)

    def nodes_of(self, input_ids):
        N, t = input_ids.shape
        node = torch.zeros(N, dtype=torch.long, device=input_ids.device)
        for j in range(t):
            key = node * self.V + input_ids[:, j]
            pos = torch.searchsorted(self.keys, key).clamp(max=self.E - 1)
            ok = (self.keys[pos] == key) & (node != self.dead)
            node = torch.where(ok, self.child[pos], torch.full_like(node, self.dead))
        return node

    def allowed(self, input_ids):
        """(rows, cols) of every allowed (beam row, token) pair; same sets as Trie.get(sentence.tolist())."""
        node = self.nodes_of(input_ids)
        cnt = self.cnt[node]
        tot, n_empty = torch.stack([cnt.sum(), (cnt == 0).sum()]).tolist()             # one device->host sync per step
        if n_empty:
            raise ValueError("GPUTrieProcessor: empty allowed set (the constraint is unsatisfiable), as HF's processor")
        N = input_ids.shape[0]
        rows = torch.repeat_interleave(torch.arange(N, device=input_ids.device), cnt, output_size=tot)
        start = (torch.cumsum(cnt, 0) - cnt)[rows]
        local = torch.arange(tot, device=input_ids.device) - start
        cols = self.kids[self.ptr[node][rows] + local]
        return rows, cols

    def __call__(self, input_ids, scores):
        rows, cols = self.allowed(input_ids)
        mask = torch.full_like(scores, -math.inf)
        mask[rows, cols] = 0
        return scores + mask
