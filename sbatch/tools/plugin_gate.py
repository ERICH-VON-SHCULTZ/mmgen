"""Pre-training gate for one plug-in run (docs/PLUGIN_REPORT.md): config diff OK, v1 SID hash = locked hash,
view index files = the hashes recorded when they were built (plugin/<D>/<cfg>/views.json). Exit 1 on failure.
Usage: python plugin_gate.py <D>:<plug122|plug122s|plug133|plug133s>
"""
import hashlib
import json
import sys

RD = "/scratch/yw8866/mmgen_runs"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    r = sys.argv[1]
    D, B = r.split(":")
    C = B[4:7]
    cd = json.load(open(f"{RD}/plugin/config_diff.json"))
    assert cd["all_ok"] and cd["runs"][r]["ok"], f"{r}: config diff not OK"
    v = json.load(open(f"{RD}/plugin/{D}/{C}/views.json"))
    lock = open(f"{RD}/md/sids/v1/{D}/1+{C[1]}+{C[2]}/index.sha256").read().split()[0]
    assert sha(v["v1_index"]) == lock == v["v1_sha256"], f"{r}: v1 hash"
    var = "shared_f" if B.endswith("s") else "main"
    for name in ("text", "image"):
        f = v["files"][f"{var}/{name}"]
        assert f["file"] == f"{D}.index_{B}_{name}.json", f["file"]
        assert sha(f"{RD}/data/{D}/{f['file']}") == f["sha256"], f"{r}: view hash {f['file']}"
    print(f"gate ok: {r} (v1 {lock[:12]}, {var} views)")


if __name__ == "__main__":
    main()
