"""Bandwidth benchmark for the Triton FP16 paged decode kernel (single layer).

Sweeps batch, context, and num_splits (1 = single-pass baseline, None = heuristic).
Random data, fixed seed, warmup + cuda events. Reports achieved KV-read GB/s and % of
the GPU's peak (from bpt.roofline). Prints one JSON line at the end.
"""
import json

import torch

from bpt.roofline import GPUS
from bpt.triton_fp16 import _sm_count, choose_num_splits, paged_decode_attention_triton

Hq, Hkv, D = 32, 8, 128
BLOCK = 32  # page size made no difference at 16 vs 32 in milestone 2
WARMUP, ITERS = 10, 50
SPLITS = (1, 2, 4, 8, 16, 32, 64, None)
CONFIGS = [(1, 4096), (1, 16384), (1, 65536),
           (8, 4096), (8, 16384), (8, 65536),
           (32, 4096), (32, 16384)]  # (batch, ctx); B=32 @ 64k would not fit a T4


def gpu_key():
    name = torch.cuda.get_device_name(0)
    for k in GPUS:
        if k.split("-")[0] in name:
            return k
    return name


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def main():
    gpu, sms = gpu_key(), _sm_count(0)
    peak = GPUS.get(gpu)
    rows = []
    print(f"{gpu} ({sms} SMs, peak {peak} GB/s)  Hq={Hq} Hkv={Hkv} D={D} fp16 bs={BLOCK}")
    print(f"{'B':>3}{'ctx':>7}{'splits':>8}{'ms':>9}{'GB/s':>8}{'%peak':>7}")
    for B, ctx in CONFIGS:
        torch.manual_seed(0)
        nb = (ctx + BLOCK - 1) // BLOCK
        k = torch.randn(B * nb, BLOCK, Hkv, D, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        bt = torch.randperm(B * nb, device="cuda").reshape(B, nb).to(torch.int32)
        sl = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
        q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
        nbytes = 2 * B * ctx * Hkv * D * 2  # K + V, fp16
        auto = choose_num_splits(B, Hkv, BLOCK, nb, sms)
        for n in SPLITS:
            ms = time_ms(lambda: paged_decode_attention_triton(q, k, v, bt, sl, num_splits=n))
            gbps = nbytes / (ms * 1e-3) / 1e9
            r = {"B": B, "ctx": ctx, "num_splits": n, "auto_splits": auto if n is None else None,
                 "ms": ms, "kv_MB": nbytes / 1e6, "GBps": gbps,
                 "pct_peak": 100 * gbps / peak if peak else None}
            rows.append(r)
            label = f"auto={auto}" if n is None else str(n)
            print(f"{B:>3}{ctx:>7}{label:>8}{ms:>9.3f}{gbps:>8.1f}{r['pct_peak']:>7.1f}")
        del k, v
        torch.cuda.empty_cache()
    print(json.dumps({"gpu": gpu, "sms": sms, "peak_GBps": peak, "torch": torch.__version__,
                      "block_size": BLOCK, "warmup": WARMUP, "iters": ITERS, "rows": rows}))


if __name__ == "__main__":
    main()
