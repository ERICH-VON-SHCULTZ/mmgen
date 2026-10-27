"""mmgen addition: MD-RQ-VAE (multi-modal decomposed RQ-VAE) for MACRec-format data.

Port of the Tmall tokenizer (legacy/grid/src/modules/clustering/md_rqvae.py, see docs/MDRQVAE.md),
rebuilt from MACRec's own pieces (MLPLayers, ResidualVectorQuantizer / VectorQuantizer) at the
Amazon scale (256 codes per level, 32-d codes). Design principles kept from Tmall:

  1. SID = [shared codes f_1..f_S, text chain t_1..t_L, image chain i_1..i_L].
  2. The shared codes are quantized first, from a fusion of both modality latents (z_f).
  3. Each modality branch quantizes Linear([r_f ; z_m]) where r_f = z_f - q_f.detach() is the
     residual left by the shared quantization (hard code, as on Tmall).
  4. x_t is decoded from q_f + q_t only, x_i from q_f + q_i only.
  5. Conditional independence of the two chains given f is measured afterwards (md_diagnostics.py).

Everything that may be adjusted is a constructor argument (input scaling, fusion type, loss
weights, codebook lookup, dead-code reset); defaults are attempt 1 in docs/MDRQVAE_LOG.md.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class MDRQVAE(nn.Module):
    def __init__(self, text_dim, image_dim, n_shared=1, n_modal=2, codebook_size=256, e_dim=32,
                 layers=(2048, 1024, 512, 256, 128, 64), fusion="mlp", info_nce_weight=1.0,
                 quant_loss_weight=1.0, kmeans_init=True, kmeans_iters=100, dropout_prob=0.0,
                 dead_code_reset=False):
        super().__init__()
        self.n_shared, self.n_modal, self.e_dim = n_shared, n_modal, e_dim
        self.info_nce_weight, self.quant_loss_weight = info_nce_weight, quant_loss_weight
        self.dead_code_reset = dead_code_reset
        layers = list(layers)
        enc = lambda d: MLPLayers([d] + layers + [e_dim], dropout=dropout_prob)
        dec = lambda d: MLPLayers([e_dim] + layers[::-1] + [d], dropout=dropout_prob)
        self.text_encoder, self.image_encoder = enc(text_dim), enc(image_dim)
        self.text_decoder, self.image_decoder = dec(text_dim), dec(image_dim)

        self.fusion = fusion
        if fusion == "mlp":                          # [z_t ; z_i] -> 2e -> e
            self.fuse = MLPLayers([2 * e_dim, 2 * e_dim, e_dim])
        elif fusion == "xattn":                      # Tmall CrossAttentionFusion with two length-1 tokens
            self.t2i = nn.MultiheadAttention(e_dim, 4, batch_first=True)
            self.i2t = nn.MultiheadAttention(e_dim, 4, batch_first=True)
            self.ln_t, self.ln_i, self.ln_f = nn.LayerNorm(e_dim), nn.LayerNorm(e_dim), nn.LayerNorm(e_dim)
            self.fuse = nn.Linear(2 * e_dim, e_dim)
        else:
            raise ValueError(fusion)
        self.text_branch = nn.Linear(2 * e_dim, e_dim)     # Tmall: single Linear, no activation
        self.image_branch = nn.Linear(2 * e_dim, e_dim)

        rq = lambda n: ResidualVectorQuantizer([codebook_size] * n, e_dim, sk_epsilons=[0.0] * n,
                                               kmeans_init=kmeans_init, kmeans_iters=kmeans_iters)
        self.shared_rq, self.text_rq, self.image_rq = rq(n_shared), rq(n_modal), rq(n_modal)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))   # Tmall InfoNCE temperature

    # ---- pieces ----
    def fuse_latents(self, z_t, z_i):
        if self.fusion == "mlp":
            return self.fuse(torch.cat([z_t, z_i], dim=-1))
        t, i = z_t.unsqueeze(1), z_i.unsqueeze(1)
        a_t = self.ln_t(t + self.t2i(t, i, i, need_weights=False)[0]).squeeze(1)
        a_i = self.ln_i(i + self.i2t(i, t, t, need_weights=False)[0]).squeeze(1)
        return self.ln_f(self.fuse(torch.cat([a_t, a_i], dim=-1)))

    def info_nce(self, a, p):
        """In-batch InfoNCE, one direction (rows of a.p^T), learnable temperature: as on Tmall."""
        a, p = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
        logits = a @ p.t() / self.log_temperature.exp().clamp(min=1e-4)
        return F.cross_entropy(logits, torch.arange(a.shape[0], device=a.device))

    def encode(self, x_t, x_i):
        z_t, z_i = self.text_encoder(x_t), self.image_encoder(x_i)
        z_f = self.fuse_latents(z_t, z_i)
        q_f, l_f, c_f, _ = self.shared_rq(z_f, use_sk=False)
        r_f = z_f - q_f.detach()                      # residual against the hard shared code
        v_t = self.text_branch(torch.cat([r_f, z_t], dim=-1))
        v_i = self.image_branch(torch.cat([r_f, z_i], dim=-1))
        q_t, l_t, c_t, _ = self.text_rq(v_t, use_sk=False)
        q_i, l_i, c_i, _ = self.image_rq(v_i, use_sk=False)
        return dict(z_t=z_t, z_i=z_i, z_f=z_f, v_t=v_t, v_i=v_i, q_f=q_f, q_t=q_t, q_i=q_i,
                    codes=torch.cat([c_f, c_t, c_i], dim=-1), quant_loss=(l_f + l_t + l_i) / 3)

    def forward(self, x_t, x_i):
        e = self.encode(x_t, x_i)
        e["xh_t"] = self.text_decoder(e["q_f"] + e["q_t"])
        e["xh_i"] = self.image_decoder(e["q_f"] + e["q_i"])
        return e

    def compute_loss(self, e, x_t, x_i):
        rec_t, rec_i = F.mse_loss(e["xh_t"], x_t), F.mse_loss(e["xh_i"], x_i)
        loss = rec_t + rec_i + self.quant_loss_weight * e["quant_loss"]
        nce = torch.zeros((), device=x_t.device)
        if self.info_nce_weight > 0:
            nce = self.info_nce(e["z_f"], e["z_t"]) + self.info_nce(e["z_f"], e["z_i"])
            loss = loss + self.info_nce_weight * nce
        return loss, dict(rec_t=rec_t.item(), rec_i=rec_i.item(), quant=e["quant_loss"].item(), nce=nce.item())

    @torch.no_grad()
    def reset_dead_codes(self, e):
        """Re-seed codes unused in this batch from random batch inputs of the same quantizer."""
        n = 0
        for rq, inp in ((self.shared_rq, e["z_f"]), (self.text_rq, e["v_t"]), (self.image_rq, e["v_i"])):
            res = inp
            for vq in rq.vq_layers:
                w = vq.embedding.weight
                idx = torch.cdist(res, w).argmin(-1)
                dead = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
                dead[idx] = False
                k = int(dead.sum())
                if k:
                    w[dead] = res[torch.randint(0, res.shape[0], (k,), device=w.device)]
                    n += k
                res = res - w[torch.cdist(res, w).argmin(-1)]
        return n

    @torch.no_grad()
    def decode_parts(self, q_f, q_t, q_i):
        """Reconstructions from the sum and from each part alone (additivity diagnostic)."""
        return dict(t_sum=self.text_decoder(q_f + q_t), t_f=self.text_decoder(q_f), t_m=self.text_decoder(q_t),
                    i_sum=self.image_decoder(q_f + q_i), i_f=self.image_decoder(q_f), i_m=self.image_decoder(q_i))

    @torch.no_grad()
    def code_vectors(self, codes):
        """codes: (N, S+2L) -> q_f, q_t, q_i (sums of the codebook vectors)."""
        S, L = self.n_shared, self.n_modal
        def summed(rq, cols):
            return sum(vq.embedding(cols[:, j]) for j, vq in enumerate(rq.vq_layers))
        return (summed(self.shared_rq, codes[:, :S]), summed(self.text_rq, codes[:, S:S + L]),
                summed(self.image_rq, codes[:, S + L:]))
