"""mmgen addition (docs/MFD_REPORT.md §22.9, E9): mixture of softmaxes (Yang et al., 2018) at the f step, on A01.

At decoder position 0 (decoder input = decoder_start_token: the position whose output is P(f), in training and in every
inference path) with final decoder state h (T5's tied-head scaling d^-0.5 applied, as for the plain logits):
  h_k = W_k h (k = 1..K),  logits_k = h_k E^T (tied output table),  pi = softmax(W_pi h),
  P(.) = sum_k pi_k softmax(logits_k)                               (over the full vocabulary, as A01's f softmax)
Init: W_k = I + N(0, 0.01^2), W_pi = 0 (pi uniform) -> P(f) ~ A01's at init.
The position-0 logits are REPLACED by log P (a valid logit vector: log-softmax(log P) = log P), so everything downstream uses
the mixture unchanged: the f cross-entropy is -log P(true f); A01's teacher KL reads logits[:, 0, 0, f codes] (the mixture,
renormalised over f as in A01; teacher = mean over views of the mixture); inference reads the full-vocabulary log-softmax at
the start position (mfd_eval.score_batch pass 1 / mfd_mv_infer.f_logprobs) = log P. Every other position is unchanged.
MoSHead replaces the tied lm_head (reads the model's current table: the GeoEmbedding during training, the baked table after);
W (K, d, d) and W_pi live on the model, are saved with it (config.r22_e9 = K) and mfd_eval.load_mfd_model rebuilds this class.
Inference cost: in MFDT5.last_logprobs (inference) only the LAST position is read, so for decoder inputs longer than one token
the mixture at position 0 is skipped (it would be discarded).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from r21_views import GroupedItemMFDT5


def table_of(emb):
    return emb.full_weight() if hasattr(emb, "full_weight") else emb.weight


class MoSHead(nn.Module):
    """Tied output head with the mixture at position 0 (owner held outside the module tree: no duplicate parameters)."""

    def __init__(self, owner):
        super().__init__()
        self.__dict__["owner"] = owner

    def forward(self, x):
        o = self.owner
        T = table_of(o.shared)
        lg = F.linear(x, T)
        if not o.r22_e9 or (o._e9_last_only and x.shape[1] > 1):
            return lg
        lp = o.mos_logprobs(x[:, 0], T)
        lg = lg.clone()
        lg[:, 0] = lp.to(lg.dtype)
        return lg


class E9Mixin:
    r22_e9 = 0
    _e9_last_only = False

    def __init__(self, config):
        super().__init__(config)
        self.r22_e9 = int(getattr(config, "r22_e9", 0) or 0)
        if self.r22_e9:
            self._build_e9()
            self.lm_head = MoSHead(self)

    def _build_e9(self, noise=0.0):
        d, K = self.config.d_model, self.r22_e9
        W = torch.eye(d).repeat(K, 1, 1)
        if noise > 0:
            W = W + noise * torch.randn(K, d, d)
        self.e9_W = nn.Parameter(W)
        self.e9_pi = nn.Linear(d, K, bias=False)
        nn.init.zeros_(self.e9_pi.weight)

    def install_e9(self, K, noise=0.01):
        assert K >= 2 and getattr(self, "struct_mode", None) == "E2", "E9 is defined on A01 (item-level encoder)"
        dev = next(self.parameters()).device
        self.r22_e9 = int(K)
        self.config.r22_e9 = int(K)
        self._build_e9(noise)
        self.e9_W.data = self.e9_W.data.to(dev)
        self.e9_pi.to(dev)
        self.lm_head = MoSHead(self)
        return {"r22_e9": K, "e9_params": self.e9_W.numel() + self.e9_pi.weight.numel(), "e9_W_init": f"I + N(0, {noise}^2)",
                "e9_pi_init": "zero (uniform)", "position": 0}

    def mos_logprobs(self, h, T):
        """h (N, d) scaled decoder state at position 0, T (V, d) table -> (N, V) log P (float32)."""
        hk = torch.einsum("kde,ne->knd", self.e9_W, h)                     # W_k h
        comp = torch.einsum("knd,vd->knv", hk, T).float().log_softmax(-1)
        lpi = self.e9_pi(h).float().log_softmax(-1).t()                     # (K, N)
        return torch.logsumexp(comp + lpi[..., None], 0)

    def tie_weights(self):
        super().tie_weights()
        if self.r22_e9 and not isinstance(self.lm_head, MoSHead):          # after geo_embed bake(): re-wrap the new plain head
            self.lm_head = MoSHead(self)

    @torch.no_grad()
    def last_logprobs(self, *a, **k):
        self._e9_last_only = True
        try:
            return super().last_logprobs(*a, **k)
        finally:
            self._e9_last_only = False

    def e9_stats(self):
        if not self.r22_e9:
            return None
        eye = torch.eye(self.config.d_model, device=self.e9_W.device)
        return {"W_minus_I_norm": [float((w - eye).norm()) for w in self.e9_W.detach()], "W_pi_norm": float(self.e9_pi.weight.norm())}


class E9GroupedItemMFDT5(E9Mixin, GroupedItemMFDT5):
    pass
