"""Feature sanity: shapes per dataset; Instruments vs MQL4GRec's published embeddings (row cosine)."""
import numpy as np, os
DATA = "/scratch/yw8866/mmgen_runs/data"
REF = "/scratch/yw8866/tmp/ref/MQL4GRec/data/Instruments"
for D in ["Instruments", "Arts", "Games"]:
    for tag in ["llama-td", "ViT-L-14"]:
        p = f"{DATA}/{D}/{D}.emb-{tag}.npy"
        if os.path.exists(p):
            e = np.load(p); print(D, tag, e.shape, e.dtype, "nan" if np.isnan(e.astype(np.float32)).any() else "ok")
for tag in ["llama-td", "ViT-L-14"]:
    a = np.load(f"{REF}/Instruments.emb-{tag}.npy").astype(np.float32)
    b = np.load(f"{DATA}/Instruments/Instruments.emb-{tag}.npy").astype(np.float32)
    an = a / np.linalg.norm(a, axis=1, keepdims=True); bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    c = (an * bn).sum(1)
    print(f"Instruments {tag} vs MQL4GRec: row cosine min {c.min():.4f} p1 {np.percentile(c,1):.4f} "
          f"median {np.median(c):.5f}; rows with cos<0.99: {(c<0.99).sum()}")
