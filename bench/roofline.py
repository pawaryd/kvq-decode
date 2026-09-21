"""Print KV bytes read per decode step and the minimum time on each GPU.

    python bench/roofline.py --model llama3-8b --ctx 32768 --batch 8 --kv-dtype int4
"""
import argparse

from bpt.roofline import (GPUS, KV_DTYPES, MODELS, ModelShape, bytes_per_step,
                          kv_bytes_per_token, min_time_ms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="llama3-8b")
    p.add_argument("--layers", type=int, help="override (1 gives per-layer numbers)")
    p.add_argument("--q-heads", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--kv-dtype", choices=sorted(KV_DTYPES), default="fp16")
    p.add_argument("--group-size", type=int, default=128,
                   help="tokens per key-scale group (quantized dtypes)")
    a = p.parse_args()

    m = MODELS[a.model]
    shape = ModelShape(a.layers or m.layers, a.q_heads or m.q_heads,
                       a.kv_heads or m.kv_heads, a.head_dim or m.head_dim)

    tok = kv_bytes_per_token(shape, a.kv_dtype, a.group_size)
    step = bytes_per_step(shape, a.ctx, a.batch, a.kv_dtype, a.group_size)
    fp16_step = bytes_per_step(shape, a.ctx, a.batch, "fp16", a.group_size)

    print(f"shape: {shape.layers}L, {shape.q_heads}q/{shape.kv_heads}kv heads, "
          f"D={shape.head_dim} | ctx={a.ctx} batch={a.batch} kv={a.kv_dtype}")
    print(f"bytes per context token (all layers): {tok:,.0f} B ({tok / 1024:.2f} KiB)")
    print(f"bytes per sequence per step:          {step / a.batch / 2**20:,.1f} MiB")
    print(f"bytes per decode step (batch):        {step / 2**30:,.3f} GiB")
    print("(KV-cache reads only; weights, q and output excluded)\n")

    print(f"{'GPU':<10}{'peak GB/s':>10}{'min time ms':>13}{'min us/seq':>12}"
          f"{'max steps/s':>13}{'vs fp16':>9}")
    for gpu, bw in GPUS.items():
        t = min_time_ms(step, gpu)
        print(f"{gpu:<10}{bw:>10.0f}{t:>13.3f}{t * 1e3 / a.batch:>12.1f}"
              f"{1e3 / t:>13.1f}{fp16_step / step:>8.2f}x")


if __name__ == "__main__":
    main()
