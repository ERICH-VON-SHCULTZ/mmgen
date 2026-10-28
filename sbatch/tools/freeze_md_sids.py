"""Freeze one MD-RQ-VAE SID set into a versioned directory (never overwritten once locked).

  /scratch/yw8866/mmgen_runs/md/sids/<version>/<D>/<cfg>/
      tokenizer.pth          copy of the tokenizer checkpoint (best.pth)
      run_config.json        resolved tokenizer config
      index.json             item -> SID (the reassigned, token-free index)
      index.sha256           SHA-256 of index.json
      reassign.json          reassignment summary (every changed item)
      diagnostics.json       tokenizer diagnostics on this index
      diagnostics.md         one-row diagnostics table
      STATUS                 "v1 (not locked)" until --lock; locking makes every file read-only
Usage: python3 freeze_md_sids.py --dataset Instruments --cfg 122 [--version v1] [--lock]
Refuses to touch a locked directory; a change needs a new --version and a --note.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat

RUNS = "/scratch/yw8866/mmgen_runs"
CFG = {"122": ("a05_122", "md122_a05_reassign"), "133": ("a06_133", "md133_a06_reassign")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cfg", required=True, choices=list(CFG))
    p.add_argument("--version", default="v1")
    p.add_argument("--lock", action="store_true")
    p.add_argument("--note", default="")
    a = p.parse_args()
    name, tag = CFG[a.cfg]
    D = a.dataset
    src = f"{RUNS}/md/{D}/tok/{name}"
    dst = f"{RUNS}/md/sids/{a.version}/{D}/1+{a.cfg[1]}+{a.cfg[2]}"
    status = f"{dst}/STATUS"
    if os.path.exists(status) and "locked" in open(status).read() and "not locked" not in open(status).read():
        raise SystemExit(f"{dst} is locked; create a new version instead")
    if not a.lock:
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(f"{src}/best.pth", f"{dst}/tokenizer.pth")
        shutil.copy2(f"{src}/run_config.json", f"{dst}/run_config.json")
        shutil.copy2(f"{RUNS}/data/{D}/{D}.index_{tag}.json", f"{dst}/index.json")
        shutil.copy2(f"{src}/reassign_{tag}.json", f"{dst}/reassign.json")
        shutil.copy2(f"{src}/diagnostics_{tag}.json", f"{dst}/diagnostics.json")
        h = hashlib.sha256(open(f"{dst}/index.json", "rb").read()).hexdigest()
        open(f"{dst}/index.sha256", "w").write(f"{h}  index.json\n")
        d = json.load(open(f"{dst}/diagnostics.json")); r = json.load(open(f"{dst}/reassign.json"))
        c, s = d["cmi_first_level"], d["separation"]
        row = ("| dataset | config | rec cos text / image (A / B) | util per level | collision before / after | items changed |"
               " CMI₁ plug-in / null (sd) | corrected % H(t₁|f) | AMI f~A0 / f~B0 | sep. shared / text / image | additivity t / i |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|\n"
               f"| {D} | 1+{a.cfg[1]}+{a.cfg[2]} | {d['recon']['md_text']:.4f} / {d['recon']['md_image']:.4f} "
               f"({d['recon']['A_text']:.4f} / {d['recon']['B_image']:.4f}) | {' '.join(map(str, d['utilization'].values()))} | "
               f"{r['collision_before']*100:.2f}% / {d['full_sid_collision']*100:.2f}% | {r['items_changed']} | "
               f"{c['cmi_plugin_bits']:.3f} / {c['null_mean']:.3f} ({c['null_std']:.4f}) | {c['frac_corrected']*100:.2f}% | "
               f"{d['ami_shared_vs_A0']:.3f} / {d['ami_shared_vs_B0']:.3f} | {s['shared']['normalized_ratio']:.3f} / "
               f"{s['text']['normalized_ratio']:.3f} / {s['image']['normalized_ratio']:.3f} | "
               f"{d['additivity']['text']['frac_sum_beats_both']:.3f} / {d['additivity']['image']['frac_sum_beats_both']:.3f} |\n")
        open(f"{dst}/diagnostics.md", "w").write(row)
        open(status, "w").write(f"{a.version} (not locked)\n{a.note}\n")
        print(f"{dst}  sha256 {h}")
    else:
        open(status, "w").write(f"{a.version} locked\n{a.note}\n")
        for f in os.listdir(dst):
            os.chmod(f"{dst}/{f}", stat.S_IRUSR | stat.S_IRGRP)
        os.chmod(dst, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
        print(f"{dst} locked")


if __name__ == "__main__":
    main()
