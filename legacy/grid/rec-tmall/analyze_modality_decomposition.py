"""Empirically test whether MD-RQ-VAE codebooks are modality-decomposed, and whether a chain
RQ-VAE's levels are not. Pure offline analysis: no training, no generation, no downstream model.

Three tests on a FIXED random sample of items with both modalities present (same sample everywhere):
  TEST 1  modality attribution under code substitution
          (a) MD: decoder intervention -- 3x2 table of cosine drops (subbed code x damaged modality)
              CAVEAT reported inline: x_t_hat=D_t(e_f+e_t), x_i_hat=D_i(e_f+e_i); the image decoder
              never sees the text code, so subbing text leaves image recon EXACTLY unchanged (the
              cross-drop is 0 by construction, not by learning). The informative cells are the
              own-modality drops (is e_t actually used?) and the shared sub (damages both).
          (b) representation-level analogue on BOTH MD and chain (the fair, non-architectural test):
              among item pairs that share the other two codes but differ at one code, does the
              SigLIP TEXT distance move more than the IMAGE distance for the text code, etc.
  TEST 2  additivity: does D_t(e_f+e_t) beat D_t(e_f) and D_t(e_t)? shared-code balance.  (MD only)
  TEST 3  codebook semantic purity: per code, SigLIP text-spread vs image-spread; scatter per
          codebook (MD C_f/C_t/C_i) and per chain level. Codes with >= min_items assignments.

No tuning toward a positive result. If subbing the text code damages both modalities, or the chain
levels show the same selectivity as MD, that is reported plainly.
"""

import argparse
import os
import pickle
import sys

# The MD-RQ-VAE checkpoint pickles GRID's src.* classes, so 'src' must be importable to unpickle it.
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ----------------------------------------------------------------------------- IO
def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n"); fh.flush()


def load_id_map(pkl_path):
    """_ids.pkl is a row-ordered list of item_ids -> return dict item_id -> row."""
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)
    if isinstance(m, dict):
        # already item_id -> row (or row -> item_id); detect by checking values
        return {int(k): int(v) for k, v in m.items()}
    return {int(iid): row for row, iid in enumerate(m)}


def open_embeddings(path, n_rows):
    """Open a (N, d) float32 embedding file, mirroring GRID's loader: a standard .npy is memmapped,
    but these exports are RAW float32 binary dumps (no NUMPY magic) -> np.memmap with shape derived
    from the id-map length (np.load would try to unpickle them and fail)."""
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        try:
            return np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError:
            return np.load(path, allow_pickle=True)
    total = os.path.getsize(path) // 4
    dim = total // n_rows                       # GRID convention: floor; id map maps the FIRST n_rows
    assert dim > 0 and n_rows * dim <= total, f"bad shape: total={total} n_rows={n_rows} dim={dim}"
    leftover = total - n_rows * dim             # trailing rows beyond the id map (ignored, as GRID does)
    if leftover:
        print(f"  [open_embeddings] {path.split('/')[-1]}: dim={dim}, n_rows={n_rows}, "
              f"{leftover} trailing floats (~{leftover // max(dim,1)} extra rows) ignored", flush=True)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n_rows, dim))


def load_sid(path):
    """semantic_ids.pt -> LongTensor [n_levels, num_items] (column == item_id)."""
    o = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(o, dict):
        for k in ("semantic_ids", "sid", "ids"):
            if k in o and torch.is_tensor(o[k]):
                o = o[k]; break
        else:
            o = next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2)
    t = o.long()
    if t.shape[0] > t.shape[1]:      # levels are the SHORT axis
        t = t.t().contiguous()
    return t


# ----------------------------------------------------------------- MD-RQ-VAE weights
def rebuild_mlp(sd, prefix, device):
    """Rebuild an MLP (Linear/ReLU/Linear/.../Linear) from its state_dict weights."""
    # collect Linear layer indices (keys like '<prefix>0.weight')
    lin = sorted(int(k[len(prefix):].split(".")[0]) for k in sd
                 if k.startswith(prefix) and k.endswith(".weight") and sd[k].dim() == 2)
    assert lin, (f"no Linear weights under prefix '{prefix}'; decoder keys present: "
                 f"{[k for k in sd if 'decoder' in k][:12]}")
    layers = []
    for j, li in enumerate(lin):
        w = sd[f"{prefix}{li}.weight"]
        b = sd.get(f"{prefix}{li}.bias")
        lyr = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        with torch.no_grad():
            lyr.weight.copy_(w)
            if b is not None:
                lyr.bias.copy_(b)
        layers.append(lyr)
        if j < len(lin) - 1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers).to(device).eval()


def load_md(md_ckpt, device, fh):
    # Warm-import src.utils.utils BEFORE torch.load: unpickling the checkpoint cold triggers a
    # circular import (src.data...interfaces <-> src.utils.utils); importing it first breaks the cycle.
    import src.utils.utils  # noqa: F401
    ckpt = torch.load(md_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cb_keys = [k for k in sd if k.endswith(".centroids")]
    log(f"  MD state_dict: {len(sd)} tensors; codebook(.centroids) keys found: {cb_keys}", fh)

    def cb(name):
        # match a .centroids key whose path contains the codebook name (robust to nesting)
        cands = [k for k in cb_keys if name.split("_")[0] in k]
        assert len(cands) == 1, f"expected 1 '{name}' codebook, found {cands} among {cb_keys}"
        return sd[cands[0]].float().to(device)

    C_f, C_t, C_i = cb("shared_codebook"), cb("text_codebook"), cb("image_codebook")
    D_t = rebuild_mlp(sd, "text_decoder.model.", device)
    D_i = rebuild_mlp(sd, "image_decoder.model.", device)
    log(f"  codebook shapes  C_f={tuple(C_f.shape)}  C_t={tuple(C_t.shape)}  C_i={tuple(C_i.shape)}", fh)
    log(f"  decoder D_t: {D_t}", fh)
    return C_f, C_t, C_i, D_t, D_i


# --------------------------------------------------------------------------- helpers
def cos_rows(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def decode_cos(D, codesum, target, device, bs=8192):
    """cos(D(codesum), target) per row, batched."""
    out = torch.empty(codesum.shape[0])
    for s in range(0, codesum.shape[0], bs):
        e = slice(s, min(s + bs, codesum.shape[0]))
        rec = D(codesum[e].to(device))
        out[e] = cos_rows(rec, target[e].to(device)).cpu()
    return out


def boot_ci(x, n=1000, seed=0):
    r = np.random.default_rng(seed)
    m = x.mean()
    bs = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def random_pair_baseline(tf, imf, n_pairs, seed, fh):
    """Mean cosine distance over n_pairs RANDOM item pairs -- the 'no shared structure' reference
    scale for each modality (text and image SigLIP spreads differ globally, so the raw Test-1b
    text/image ratio must be read against this)."""
    rng = np.random.default_rng(seed + 12345)
    n = tf.shape[0]
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    same = a == b
    b[same] = (b[same] + 1) % n
    tfn = tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-12)
    imn = imf / (np.linalg.norm(imf, axis=1, keepdims=True) + 1e-12)
    bt = float(1 - np.einsum("ij,ij->i", tfn[a], tfn[b]).mean())
    bi = float(1 - np.einsum("ij,ij->i", imn[a], imn[b]).mean())
    log(f"\n=== RANDOM-PAIR BASELINE ({n_pairs:,} random item pairs, same sample) ===", fh)
    log(f"  text dist={bt:.4f}  image dist={bi:.4f}  text/image={bt / max(bi, 1e-9):.3f}", fh)
    log("  (Test 1b columns text/base & image/base normalize each modality by this; a code that", fh)
    log("   controls a modality drives that modality's dist FAR below its baseline (ratio << 1),", fh)
    log("   while the uncontrolled modality stays near baseline (ratio ~ 1).)", fh)
    return bt, bi


def mean_pairwise_cosdist(feats):
    """1 - mean_{i<j} cos(u_i,u_j) via the sum vector; feats (n,d), n>=2. O(n d)."""
    u = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    n = u.shape[0]
    s = u.sum(0)
    mean_cos = (float(s @ s) - n) / (n * (n - 1))
    return 1.0 - mean_cos


# ============================================================================ TEST 1a
def test1_md_intervention(C_f, C_t, C_i, D_t, D_i, sc, tc, ic, xt, xi, device, rng, fh):
    log("\n=== TEST 1a: MD-RQ-VAE modality attribution under code substitution (decoder) ===", fh)
    log("  CAVEAT: x_i_hat=D_i(e_f+e_i) does not depend on the text code (and vice versa), so the", fh)
    log("  cross-modality drop for a text/image substitution is 0 BY CONSTRUCTION. Read the own-", fh)
    log("  modality drops (does the code matter at all?) and the shared sub (must damage both).", fh)
    ef, et, ei = C_f[sc], C_t[tc], C_i[ic]
    base_t = decode_cos(D_t, ef + et, xt, device)
    base_i = decode_cos(D_i, ef + ei, xi, device)
    log(f"  baseline cos: text={base_t.mean():.4f}  image={base_i.mean():.4f}", fh)

    n = sc.shape[0]
    r_t = C_t[torch.as_tensor(rng.integers(0, C_t.shape[0], n))]
    r_i = C_i[torch.as_tensor(rng.integers(0, C_i.shape[0], n))]
    r_f = C_f[torch.as_tensor(rng.integers(0, C_f.shape[0], n))]

    # substitute text code -> only text recon changes
    dt_text = base_t - decode_cos(D_t, ef + r_t, xt, device)
    di_text = base_i - decode_cos(D_i, ef + ei, xi, device)   # unchanged (=0)
    # substitute image code -> only image recon changes
    dt_img = base_t - decode_cos(D_t, ef + et, xt, device)    # unchanged (=0)
    di_img = base_i - decode_cos(D_i, ef + r_i, xi, device)
    # substitute shared code -> both change
    dt_shared = base_t - decode_cos(D_t, r_f + et, xt, device)
    di_shared = base_i - decode_cos(D_i, r_f + ei, xi, device)

    rows = [("sub TEXT ", dt_text, di_text), ("sub IMAGE", dt_img, di_img),
            ("sub SHARE", dt_shared, di_shared)]
    log(f"  {'substitution':<12}{'Δcos text (CI)':>26}{'Δcos image (CI)':>26}{'selectivity':>13}", fh)
    for name, dt, di in rows:
        mt, lt, ht = boot_ci(dt.numpy())
        mi, li, hi = boot_ci(di.numpy())
        own, oth = (mt, mi) if "TEXT" in name else (mi, mt) if "IMAGE" in name else (max(mt, mi), min(mt, mi))
        sel = own / oth if abs(oth) > 1e-6 else float("inf")
        log(f"  {name:<12}{f'{mt:.4f} [{lt:.4f},{ht:.4f}]':>26}"
            f"{f'{mi:.4f} [{li:.4f},{hi:.4f}]':>26}{sel:>13.2f}", fh)


# ============================================================================ TEST 1b
def test1_repr_pairs(name, sid, sc_idx, tf, imf, sample_items, item2row_sid, device, fh,
                     level_names, base=None, max_pairs=40000, seed=0):
    """Representation-level analogue (fair, non-architectural): for each level L, among sampled
    items that share the OTHER levels but differ at L, mean SigLIP text vs image cosine distance.
    base=(bt,bi) is the random-pair baseline; columns text/base & image/base normalize by it."""
    log(f"\n=== TEST 1b: representation-level analogue -- {name} ===", fh)
    log("  For each level: pairs sharing the other two codes, differing at this one. If the level is", fh)
    log("  text-specific, differing there should move TEXT SigLIP distance >> IMAGE distance.", fh)
    bt, bi = base if base is not None else (None, None)
    rng = np.random.default_rng(seed)
    L = sid.shape[0]
    codes = sid[:, sample_items].numpy()           # (L, n_sample)
    tfn = tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-12)
    imn = imf / (np.linalg.norm(imf, axis=1, keepdims=True) + 1e-12)
    log(f"  {'level':<12}{'#pairs':>10}{'text dist':>14}{'image dist':>14}{'text/image':>12}"
        f"{'text/base':>11}{'image/base':>11}", fh)
    for lvl in range(L):
        others = [x for x in range(L) if x != lvl]
        key = codes[others[0]].astype(np.int64)
        for o in others[1:]:
            key = key * (codes[o].max() + 1) + codes[o]
        order = np.argsort(key, kind="stable")
        ksorted = key[order]
        bounds = np.flatnonzero(np.diff(ksorted)) + 1
        groups = np.split(order, bounds)
        pt, pi_, npair = [], [], 0
        for g in groups:
            if g.shape[0] < 2:
                continue
            cl = codes[lvl][g]
            # form up to a few differing pairs within the group
            for a in range(len(g)):
                for b in range(a + 1, min(a + 4, len(g))):
                    if cl[a] == cl[b]:
                        continue
                    ia, ib = g[a], g[b]
                    pt.append(1 - float(tfn[ia] @ tfn[ib]))
                    pi_.append(1 - float(imn[ia] @ imn[ib]))
                    npair += 1
            if npair >= max_pairs:
                break
        if npair < 50:
            log(f"  {level_names[lvl]:<12}{npair:>10}   (too few pairs -- unreliable; see TEST 3)", fh)
            continue
        mt, mi = float(np.mean(pt)), float(np.mean(pi_))
        tb = f"{mt/bt:>11.3f}" if bt else f"{'-':>11}"
        ib = f"{mi/bi:>11.3f}" if bi else f"{'-':>11}"
        log(f"  {level_names[lvl]:<12}{npair:>10}{mt:>14.4f}{mi:>14.4f}{mt/max(mi,1e-6):>12.3f}{tb}{ib}", fh)


# ============================================================================= TEST 2
def test2_additivity(C_f, C_t, C_i, D_t, D_i, sc, tc, ic, xt, xi, device, fh):
    log("\n=== TEST 2: additivity (does the sum beat its parts?) ===", fh)
    ef, et, ei = C_f[sc], C_t[tc], C_i[ic]
    z = torch.zeros_like(ef)
    branches = [("TEXT ", D_t, ef, et, xt), ("IMAGE", D_i, ef, ei, xi)]
    for nm, D, e_shared, e_mod, x in branches:
        c_sum = decode_cos(D, e_shared + e_mod, x, device)
        c_f = decode_cos(D, e_shared + z, x, device)     # shared only
        c_m = decode_cos(D, z + e_mod, x, device)        # modality only
        ms, ls, hs = boot_ci(c_sum.numpy()); mf, lf, hf = boot_ci(c_f.numpy()); mm, lm, hm = boot_ci(c_m.numpy())
        frac = float(((c_sum > c_f) & (c_sum > c_m)).float().mean())
        log(f"  {nm} branch:", fh)
        log(f"     cos(D(e_f+e_mod),x) = {ms:.4f} [{ls:.4f},{hs:.4f}]", fh)
        log(f"     cos(D(e_f),     x)  = {mf:.4f} [{lf:.4f},{hf:.4f}]   (shared only)", fh)
        log(f"     cos(D(e_mod),   x)  = {mm:.4f} [{lm:.4f},{hm:.4f}]   (modality only)", fh)
        log(f"     fraction sum beats BOTH parts = {frac:.4f}", fh)
    # shared-code balance: is D_t(e_f) closer to text than D_i(e_f) to image?
    ct_f = decode_cos(D_t, ef, xt, device)
    ci_f = decode_cos(D_i, ef, xi, device)
    gap = (ct_f - ci_f).numpy()
    mg, lg, hg = boot_ci(gap)
    log(f"  shared balance: cos(D_t(e_f),x_t) - cos(D_i(e_f),x_i) = {mg:+.4f} [{lg:+.4f},{hg:+.4f}]", fh)
    log("     (~0 => the shared code is genuinely modality-balanced; large => it leans one way)", fh)


# ============================================================================= TEST 3
def test3_purity(tag, codebooks, sid, level_names, sample_items, tf, imf, out_dir, min_items, fh):
    log(f"\n=== TEST 3: codebook semantic purity -- {tag} ===", fh)
    tfn = tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-12)
    imn = imf / (np.linalg.norm(imf, axis=1, keepdims=True) + 1e-12)
    codes = sid[:, sample_items].numpy()
    results = {}
    log(f"  {'codebook':<10}{'#codes>=min':>12}{'mean text spread':>18}{'mean image spread':>19}", fh)
    for lvl, lname in enumerate(level_names):
        c = codes[lvl]
        order = np.argsort(c, kind="stable")
        cs = c[order]
        bounds = np.flatnonzero(np.diff(cs)) + 1
        groups = np.split(order, bounds)
        tsp, isp = [], []
        for g in groups:
            if g.shape[0] < min_items:
                continue
            tsp.append(mean_pairwise_cosdist(tfn[g]))
            isp.append(mean_pairwise_cosdist(imn[g]))
        tsp, isp = np.array(tsp), np.array(isp)
        results[lname] = (tsp, isp)
        if len(tsp):
            log(f"  {lname:<10}{len(tsp):>12}{tsp.mean():>18.4f}{isp.mean():>19.4f}", fh)
        else:
            log(f"  {lname:<10}{0:>12}   (no code reached min_items={min_items})", fh)
    # scatter
    if HAVE_PLT and any(len(v[0]) for v in results.values()):
        allx = np.concatenate([v[0] for v in results.values() if len(v[0])])
        ally = np.concatenate([v[1] for v in results.values() if len(v[1])])
        lo = min(allx.min(), ally.min()); hi = max(allx.max(), ally.max())
        fig, ax = plt.subplots(figsize=(6, 6))
        cols = {level_names[0]: "#1f77b4", level_names[1]: "#d62728", level_names[2]: "#2ca02c"}
        for lname, (tsp, isp) in results.items():
            if len(tsp):
                ax.scatter(tsp, isp, s=8, alpha=0.5, label=f"{lname} (n={len(tsp)})",
                           color=cols.get(lname, None))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="diagonal (no selectivity)")
        ax.set_xlabel("TEXT SigLIP spread (1 - mean pairwise cos)")
        ax.set_ylabel("IMAGE SigLIP spread (1 - mean pairwise cos)")
        ax.set_title(f"Codebook modality purity -- {tag}")
        ax.legend(fontsize=8); ax.set_aspect("equal")
        p = os.path.join(out_dir, f"purity_scatter_{tag}.png")
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        log(f"  saved scatter -> {p}", fh)
    return results


# =================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_ckpt", required=True)
    ap.add_argument("--md_sid", required=True)
    ap.add_argument("--chain_sid", required=True)
    ap.add_argument("--text_npy", required=True)
    ap.add_argument("--text_ids", required=True)
    ap.add_argument("--image_npy", required=True)
    ap.add_argument("--image_ids", required=True)
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_items", type=int, default=50)
    ap.add_argument("--out_dir", default="/scratch/yw8866/rec-tmall/modality_decomp")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fh = open(os.path.join(args.out_dir, "report.txt"), "w")
    log(f"device={device}  sample={args.sample}  seed={args.seed}  min_items={args.min_items}", fh)

    # ---- SIDs (column == item_id) ----
    md_sid = load_sid(args.md_sid)
    ch_sid = load_sid(args.chain_sid)
    log(f"MD SID   shape={tuple(md_sid.shape)}  per-level widths(max+1)={[int(md_sid[l].max())+1 for l in range(md_sid.shape[0])]}", fh)
    log(f"chain SID shape={tuple(ch_sid.shape)}  per-level widths(max+1)={[int(ch_sid[l].max())+1 for l in range(ch_sid.shape[0])]}", fh)
    # identify MD shared level = the one with width > 1024 (only C_f is 2048)
    md_w = [int(md_sid[l].max()) + 1 for l in range(3)]
    shared_lvl = int(np.argmax(md_w))
    log(f"  MD shared level detected = row {shared_lvl} (width {md_w[shared_lvl]}); assuming rows = "
        f"[shared,text,image] with shared at {shared_lvl}", fh)
    if shared_lvl != 0:
        log("  ** WARNING: shared level is not row 0; reordering to [shared,text,image] **", fh)
        perm = [shared_lvl] + [l for l in range(3) if l != shared_lvl]
        md_sid = md_sid[perm].contiguous()

    # ---- feature id maps + memmaps ----
    id2row_t = load_id_map(args.text_ids)
    id2row_i = load_id_map(args.image_ids)
    tf_mm = open_embeddings(args.text_npy, len(id2row_t))
    if_mm = open_embeddings(args.image_npy, len(id2row_i))
    log(f"text npy {tf_mm.shape}  image npy {if_mm.shape}  |text_ids|={len(id2row_t)} |image_ids|={len(id2row_i)}", fh)

    # ---- candidate items: both modalities present AND valid SID column ----
    n_items = md_sid.shape[1]
    cand = np.array(sorted(set(id2row_t) & set(id2row_i) & set(range(n_items))), dtype=np.int64)
    # drop padded MD columns (all-zero across levels is the padding sentinel)
    nonpad = (md_sid[:, cand].abs().sum(0) > 0).numpy()
    cand = cand[nonpad]
    log(f"candidate items (both modalities + valid MD SID) = {len(cand):,}", fh)
    rng = np.random.default_rng(args.seed)
    if len(cand) > args.sample:
        sample = np.sort(rng.choice(cand, args.sample, replace=False))
    else:
        sample = cand
    log(f"using sample of {len(sample):,} items (seed {args.seed})", fh)

    # ---- gather features for the sample (read only sampled rows from the memmaps) ----
    rows_t = np.fromiter((id2row_t[int(i)] for i in sample), dtype=np.int64, count=len(sample))
    rows_i = np.fromiter((id2row_i[int(i)] for i in sample), dtype=np.int64, count=len(sample))
    tf = np.ascontiguousarray(tf_mm[rows_t]).astype(np.float32)
    imf = np.ascontiguousarray(if_mm[rows_i]).astype(np.float32)
    xt = torch.from_numpy(tf); xi = torch.from_numpy(imf)
    log(f"gathered features: text {tf.shape}  image {imf.shape}", fh)

    # ---- MD codes for the sample ----
    sc = md_sid[0, sample].long(); tc = md_sid[1, sample].long(); ic = md_sid[2, sample].long()

    # ---- MD weights ----
    log("\nloading MD-RQ-VAE checkpoint ...", fh)
    C_f, C_t, C_i, D_t, D_i = load_md(args.md_ckpt, device, fh)

    # ---- TEST 1a (MD decoder intervention) ----
    test1_md_intervention(C_f, C_t, C_i, D_t, D_i, sc, tc, ic, xt, xi, device, rng, fh)

    # ---- random-pair baseline (reference scale for Test 1b) ----
    base = random_pair_baseline(tf, imf, 40000, args.seed, fh)

    # ---- TEST 1b (representation-level, MD and chain) ----
    test1_repr_pairs("MD-RQ-VAE", md_sid, None, tf, imf, sample, None, device, fh,
                     level_names=["shared", "text", "image"], base=base, seed=args.seed)
    test1_repr_pairs("chain RQ-VAE", ch_sid, None, tf, imf, sample, None, device, fh,
                     level_names=["level0", "level1", "level2"], base=base, seed=args.seed)

    # ---- TEST 2 (additivity, MD) ----
    test2_additivity(C_f, C_t, C_i, D_t, D_i, sc, tc, ic, xt, xi, device, fh)

    # ---- TEST 3 (purity, MD and chain) ----
    test3_purity("MD-RQ-VAE", (C_f, C_t, C_i), md_sid, ["shared", "text", "image"],
                 sample, tf, imf, args.out_dir, args.min_items, fh)
    test3_purity("chain", None, ch_sid, ["level0", "level1", "level2"],
                 sample, tf, imf, args.out_dir, args.min_items, fh)

    # ---- VERDICT ----
    log("\n=== VERDICT ===", fh)
    log("  Read TEST 2 own-vs-parts (is e_t actually used, does e_f+e_t beat both?), the TEST 1b", fh)
    log("  text/image ratios (MD should be strongly asymmetric per level; chain near 1.0), and the", fh)
    log("  TEST 3 scatters (MD C_t low-text/high-image spread, C_i the mirror, C_f balanced; chain", fh)
    log("  levels all on the diagonal). If MD is not asymmetric or chain is equally asymmetric, the", fh)
    log("  decomposition is NOT a measurable property and that explains the downstream failures.", fh)
    fh.close()


if __name__ == "__main__":
    main()
