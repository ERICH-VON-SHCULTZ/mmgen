"""mmgen addition (§23, development tool): P1a-opt implementation variants on one batch size (exact variants only), plus the
top CUDA kernels of the chosen one. Prints ms/user per variant; no outputs are saved except the log."""
import argparse, time
import torch
import r23_timing as RT

p = argparse.ArgumentParser(); p.add_argument("--dataset", default="Arts"); p.add_argument("--users", type=int, default=1024)
p.add_argument("--bs", type=int, default=64); p.add_argument("--K", type=int, default=50)
a = p.parse_args()
m = RT.MFDMethod(a.dataset, f"P1aopt:{a.K}")
pre = RT.precollate(m, m.order[:a.users], a.bs)
pre = [(i, (x.cuda(), y.cuda())) for i, (x, y) in pre]
po = m.popt
for attn in ("auto", "einsum"):
    for out in ("lsm", "lse"):
        for chunk in (8192, 16384, 32768):
            po.attn, po.outm, po.chunk = attn, out, chunk
            for _, (x, y) in pre[:3]:
                po.infer(x, y, a.K)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _, (x, y) in pre:
                po.infer(x, y, a.K)
            torch.cuda.synchronize()
            print(f"bs {a.bs} attn {attn:6s} out {out} chunk {chunk:5d}: {1000 * (time.perf_counter() - t0) / a.users:.3f} ms/user", flush=True)
po.attn, po.outm, po.chunk = "auto", "lsm", 16384
from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _, (x, y) in pre[:4]:
        po.infer(x, y, a.K)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
