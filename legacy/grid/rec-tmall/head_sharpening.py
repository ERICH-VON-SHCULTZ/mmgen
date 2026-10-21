"""Mechanism check for the within-branch contrastive fine-tune: did the item heads sharpen on the
exact ground truths that were being missed?

The component audit found that the MFD-ONLY ground truths the reranker misses have the right shared
code (f* median rank 1) but diffuse item heads (t* rank 6, i* rank 3), while the ones it rescues have
t* and i* at rank 0. The fine-tune trains the heads to rank the GT tuple above every other catalog
tuple in its branch. If the mechanism is real, then on THAT FIXED SET OF EVENTS (defined by the OLD
model's pools, so the set cannot shift under us) the new model's teacher-forced t* and i* ranks should
fall toward 0, while f* should not move (the loss never touches the shared-code head directly).

Both dumps carry the per-event teacher-forced GT factors; they are read with mmap and joined on user_id.
"""

import argparse

import numpy as np
import torch

KEYS = ("f_star_rank", "tf_t_rank", "tf_i_rank", "f_star_logp", "tf_t_logp", "tf_i_logp")


def by_uid(path):
    c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["cols"]
    uid = c["user_id"].long().numpy(); o = np.argsort(uid)
    out = {k: c[k].float().numpy()[o] for k in KEYS}
    out["uid"] = uid[o]; out["gt_b"] = c["gt_b"].long().numpy()[o]
    return out


def res_by_uid(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    o = np.argsort(r["uid"])
    return {k: (v[o] if isinstance(v, np.ndarray) and v.shape[:1] == r["uid"].shape else v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dump", required=True); ap.add_argument("--new_dump", required=True)
    ap.add_argument("--old_mfd", required=True, help="old MFD reranker result (defines the fixed event set)")
    ap.add_argument("--ch", required=True, help="chain reranker result (pool membership)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    old, new = by_uid(a.old_dump), by_uid(a.new_dump)
    m, ch = res_by_uid(a.old_mfd), res_by_uid(a.ch)
    assert np.array_equal(old["uid"], new["uid"]) and np.array_equal(old["uid"], m["uid"]) \
        and np.array_equal(m["uid"], ch["uid"]), "user_id join failed"
    assert np.array_equal(old["gt_b"], new["gt_b"]), "gt behavior differs between dumps"
    pv = old["gt_b"] == 0
    own = m["has_gt"].astype(bool) & ~ch["has_gt"].astype(bool)
    hit = m["hit_after"] > 0
    sets = {"MFD-ONLY missed (target)": own & ~hit & pv, "MFD-ONLY rescued": own & hit & pv,
            "BOTH (control)": m["has_gt"].astype(bool) & ch["has_gt"].astype(bool) & pv,
            "all pv": pv}
    log("=" * 104)
    log("=== HEAD SHARPENING: teacher-forced GT factors, OLD model vs NEW (fine-tuned), on FIXED event sets ===")
    log("=" * 104)
    log(f"  {'event set (defined by the OLD model)':<34}{'n':>7}   {'f* rank':>13}{'t* rank':>13}{'i* rank':>13}"
        f"{'logP(t*|f)':>16}{'logP(i*|f)':>16}")
    for nm, s in sets.items():
        if s.sum() == 0:
            continue
        cell = lambda k, fmt: f"{fmt.format(np.median(old[k][s]))}->{fmt.format(np.median(new[k][s]))}"
        log(f"  {nm:<34}{int(s.sum()):>7,}   {cell('f_star_rank','{:.0f}'):>13}{cell('tf_t_rank','{:.0f}'):>13}"
            f"{cell('tf_i_rank','{:.0f}'):>13}{cell('tf_t_logp','{:.2f}'):>16}{cell('tf_i_logp','{:.2f}'):>16}")
    t = sets["MFD-ONLY missed (target)"]
    log("\n  on the TARGET set (MFD-ONLY missed, pv):")
    for k, lab in (("tf_t_rank", "t*"), ("tf_i_rank", "i*"), ("f_star_rank", "f*")):
        for thr in (1, 10):
            log(f"    share with {lab} rank < {thr:>2}: {float((old[k][t] < thr).mean()):.3f} -> {float((new[k][t] < thr).mean()):.3f}")
    both0 = (old["tf_t_rank"][t] == 0) & (old["tf_i_rank"][t] == 0)
    both1 = (new["tf_t_rank"][t] == 0) & (new["tf_i_rank"][t] == 0)
    log(f"    share with BOTH t* and i* at rank 0 (the 'rescued' profile): {both0.mean():.3f} -> {both1.mean():.3f}"
        f"   = {int(both1.sum() - both0.sum()):+,} events moved into the profile the reranker rescues")
    log("\n  reading: the mechanism is confirmed if t*/i* ranks fall on the TARGET set while f* stays put and")
    log("  the BOTH control does not degrade. If the heads sharpen but the reranker still does not promote")
    log("  these events, the bottleneck is downstream of the heads; if they do not sharpen, the loss did not")
    log("  reach them and a higher weight or longer fine-tune is the next knob.")
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
