"""Bandwidth benchmark for the Triton FP16 paged decode kernel (single layer).

Random data, fixed seed, warmup + cuda events. Reports achieved KV-read GB/s and % of
the GPU's peak (from bpt.roofline). Prints one JSON line at the end.
"""
import json
import subprocess

import torch

from bpt.roofline import GPUS
from bpt.triton_fp16 import paged_decode_attention_triton

Hq, Hkv, D = 32, 8, 128
WARMUP, ITERS = 10, 50


def gpu_key():
    name = torch.cuda.get_device_name(0)
    for k in GPUS:
        if k.split("-")[0] in name:
            return k
    return name


def bench(B, ctx, block_size):
    torch.manual_seed(0)
    nb = (ctx + block_size - 1) // block_size
    k_cache = torch.randn(B * nb, block_size, Hkv, D, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    bt = torch.randperm(B * nb, device="cuda").reshape(B, nb).to(torch.int32)
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)

    for _ in range(WARMUP):
        paged_decode_attention_triton(q, k_cache, v_cache, bt, seq_lens)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        paged_decode_attention_triton(q, k_cache, v_cache, bt, seq_lens)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / ITERS
    nbytes = 2 * B * ctx * Hkv * D * 2  # K + V, fp16
    gbps = nbytes / (ms * 1e-3) / 1e9
    return {"B": B, "ctx": ctx, "block_size": block_size, "ms": ms,
            "kv_MB": nbytes / 1e6, "GBps": gbps}


def main():
    gpu = gpu_key()
    peak = GPUS.get(gpu)
    rows = []
    print(f"{gpu}  Hq={Hq} Hkv={Hkv} D={D} fp16, single layer")
    print(f"{'B':>3}{'ctx':>7}{'bs':>4}{'ms':>9}{'KV MB':>9}{'GB/s':>8}{'%peak':>7}")
    for B in (1, 8, 32):
        for ctx in (4096, 16384):
            for bs in (16, 32):
                r = bench(B, ctx, bs)
                r["pct_peak"] = 100 * r["GBps"] / peak if peak else None
                rows.append(r)
                print(f"{B:>3}{ctx:>7}{bs:>4}{r['ms']:>9.3f}{r['kv_MB']:>9.1f}"
                      f"{r['GBps']:>8.1f}{r['pct_peak']:>7.1f}")
    print(json.dumps({"gpu": gpu, "peak_GBps": peak, "torch": torch.__version__,
                      "warmup": WARMUP, "iters": ITERS, "rows": rows}))


if __name__ == "__main__":
    main()
