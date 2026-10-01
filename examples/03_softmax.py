"""Triton example 3: softmax, from a plain row softmax to the *online* softmax that
flash attention (and our paged decode kernel) is built on.

Three kernels, each adding one idea:
  A. softmax_single_block_kernel : whole row in one block; reductions (tl.max / tl.sum)
                                   and why you subtract the max
  B. softmax_online_kernel       : the row is processed in CHUNKS with a running max and
                                   running sum that get *rescaled* when the max grows
  C. attn_row_kernel             : single-query attention. B's trick plus a running
                                   weighted sum of V = one pass, never materializing the
                                   full score row. This is the core of decode attention.

Run on a GPU (from this repo use Modal):
    cd /tmp && <repo>/.venv/bin/python -m modal run <repo>/modal_scripts/run_remote.py \
        --gpu T4 --cmd "python examples/03_softmax.py"
"""
# pyright: reportArgumentType=false
# (Triton annotates block sizes as `tl.constexpr`; passing a plain int is the normal idiom.)
from typing import cast

import torch
import triton
import triton.language as tl

# softmax(x)_i = exp(x_i) / sum_j exp(x_j)
# Problem 1: exp overflows. In fp32, exp(89) is already inf. Fix: subtract the row max m
#            first. softmax(x) == softmax(x - m) because the factor exp(-m) cancels, and
#            now every exponent is <= 0, so exp() is in (0, 1] and can't overflow.
# Problem 2: the row may be too long to hold in one block (128k tokens of attention
#            scores). Fix: the *online* algorithm (kernel B).


# --------------------------------------------------------------------------------------
# STAGE 1 (kernel A): the whole row in one block
# --------------------------------------------------------------------------------------
@triton.jit
def softmax_single_block_kernel(
    x_ptr, out_ptr,
    n_cols,                 # row length (runtime)
    row_stride,             # elements between the start of consecutive rows
    BLOCK: tl.constexpr,    # power of two >= n_cols: the whole row fits in one block
):
    row = tl.program_id(axis=0)                # one program per row
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    # Out-of-range lanes are loaded as -inf, NOT 0: exp(-inf) = 0 so they contribute
    # nothing to the sum, and -inf can never be the max. (Loading 0 would inject fake
    # elements into both.)
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=float("-inf"))

    # Reductions collapse a whole block to a scalar (axis=0 = across the block's lanes).
    m = tl.max(x, axis=0)          # row max
    e = tl.exp(x - m)              # all exponents <= 0 -> no overflow
    s = tl.sum(e, axis=0)          # denominator
    tl.store(out_ptr + row * row_stride + offs, e / s, mask=mask)


# --------------------------------------------------------------------------------------
# STAGE 2 (kernel B): online softmax over chunks
# --------------------------------------------------------------------------------------
# Walk along the row in chunks, keeping two running numbers:
#     m = max of everything seen so far
#     l = sum of exp(x_i - m) over everything seen so far
# When a new chunk arrives with a bigger max m_new, the old l was computed relative to
# the OLD max, so rescale it:  l <- l * exp(m_old - m_new) + sum(exp(x_chunk - m_new)).
# After the last chunk, m and l are exactly the row max and denominator. A second pass
# then writes exp(x - m) / l. (Two reads + one write instead of one read + one write:
# the price of not needing the row to fit in a block.)
@triton.jit
def softmax_online_kernel(
    x_ptr, out_ptr, n_cols, row_stride,
    BLOCK: tl.constexpr,    # chunk size (any power of two; independent of row length)
):
    row = tl.program_id(axis=0)
    base = x_ptr + row * row_stride

    # Running state, kept as shape-[1] blocks so the loop-carried types stay consistent.
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)

    # Pass 1: build (m, l).
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < n_cols, other=float("-inf"))
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        # exp(m_i - m_new) is <= 1: it shrinks the old sum to the new reference max.
        # On the first chunk m_i = -inf, so this is exp(-inf) = 0 and l_i * 0 = 0. Fine.
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Pass 2: normalize.
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(base + offs, mask=mask, other=float("-inf"))
        tl.store(out_ptr + row * row_stride + offs, tl.exp(x - m_i) / l_i, mask=mask)


# --------------------------------------------------------------------------------------
# STAGE 3 (kernel C): single-query attention, one pass over K and V
# --------------------------------------------------------------------------------------
# out = softmax(q . K^T * scale) @ V for ONE query vector q. In decode, that's every step.
# Instead of computing all N scores, then softmax, then the weighted sum of V, process
# N in chunks and keep three running things: m, l (as above) and acc, the running
# weighted sum of V rows. When the max grows, acc must be rescaled by the SAME factor as
# l, because acc is also "relative to the old max":
#     acc <- acc * exp(m_old - m_new) + sum_i exp(s_i - m_new) * V_i
# At the end out = acc / l. The score row is never stored anywhere. That is flash attention.
@triton.jit
def attn_row_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    n_tokens, scale,
    stride_qb, stride_kb, stride_kn, stride_vb, stride_vn, stride_ob,
    D: tl.constexpr,         # head dim (power of two)
    BLOCK_N: tl.constexpr,   # tokens per chunk
):
    b = tl.program_id(axis=0)                       # one program per (independent) query
    d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + d)          # [D]

    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    for start in range(0, n_tokens, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        valid = n < n_tokens

        # Scores for this chunk: s[j] = q . K[n_j]. K chunk is [BLOCK_N, D]; multiply
        # by q (broadcast over rows) and sum over D. We use multiply+sum instead of
        # tl.dot because tl.dot needs every tile dimension >= 16 and we have ONE query
        # row. (The real decode kernel batches the query heads of a KV head into >= 16
        # rows so it *can* use tl.dot -- at the cost of padded, wasted rows.)
        k = tl.load(k_ptr + b * stride_kb + n[:, None] * stride_kn + d[None, :],
                    mask=valid[:, None], other=0.0)
        s = tl.sum(k * q[None, :], axis=1) * scale             # [BLOCK_N]
        s = tl.where(valid, s, float("-inf"))                  # padded lanes get weight 0

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)                                  # [BLOCK_N] unnormalized weights
        alpha = tl.exp(m_i - m_new)                            # rescale factor for old state
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(v_ptr + b * stride_vb + n[:, None] * stride_vn + d[None, :],
                    mask=valid[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)     # [D]
        m_i = m_new

    tl.store(out_ptr + b * stride_ob + d, acc / l_i)


# --------------------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------------------
def softmax_single_block(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()
    rows, n = x.shape
    out = torch.empty_like(x)
    softmax_single_block_kernel[(rows,)](x, out, n, x.stride(0), BLOCK=triton.next_power_of_2(n))
    return out


def softmax_online(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()
    rows, n = x.shape
    out = torch.empty_like(x)
    softmax_online_kernel[(rows,)](x, out, n, x.stride(0), BLOCK=block)
    return out


def attn_row(q, k, v, block_n: int = 128) -> torch.Tensor:
    """q [B, D], k/v [B, N, D] (fp32, contiguous) -> [B, D]."""
    assert q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    B, N, D = k.shape
    out = torch.empty_like(q)
    attn_row_kernel[(B,)](
        q, k, v, out, N, 1.0 / D**0.5,
        q.stride(0), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0),
        D=D, BLOCK_N=block_n,
    )
    return out


# --------------------------------------------------------------------------------------
# DEMOS (the first two run on plain PyTorch; they show the ideas without any kernel)
# --------------------------------------------------------------------------------------
def trace_online_softmax():
    """Follow (m, l) chunk by chunk on a tiny vector, then merge two halves."""
    print("== online softmax by hand (float64, chunks of 3) ==")
    x = torch.tensor([1.0, 3.0, 2.0, 10.0, 4.0, 0.0, 7.0, 7.0], dtype=torch.float64)
    m, l = torch.tensor(float("-inf"), dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    for i in range(0, len(x), 3):
        chunk = x[i:i + 3]
        m_new = torch.maximum(m, chunk.max())
        l = l * torch.exp(m - m_new) + torch.exp(chunk - m_new).sum()
        print(f"chunk {chunk.tolist()!s:<16} m: {m.item():>5.1f} -> {m_new.item():>4.1f}   l = {l.item():.6f}")
        m = m_new
    m_ref, l_ref = x.max(), torch.exp(x - x.max()).sum()
    print(f"one-shot   : m = {m_ref.item():.1f}  l = {l_ref.item():.6f}   (matches: "
          f"{bool(torch.isclose(l, l_ref))})")

    # The same (m, l) pair can be MERGED across independent pieces. This is exactly what
    # split-KV does: different programs process different parts of the context, then a
    # merge kernel combines their (m, l) [we store lse = m + log l] and partial outputs.
    a, b = x[:4], x[4:]
    (m1, l1), (m2, l2) = [(p.max(), torch.exp(p - p.max()).sum()) for p in (a, b)]
    m12 = torch.maximum(m1, m2)
    l12 = l1 * torch.exp(m1 - m12) + l2 * torch.exp(m2 - m12)
    print(f"merge of two halves: l = {l12.item():.6f}   (matches: {bool(torch.isclose(l12, l_ref))})")


def demo_stability():
    print("\n== why subtract the max ==")
    x = torch.tensor([[100.0, 101.0, 102.0]], device="cuda")
    naive = torch.exp(x) / torch.exp(x).sum(-1, keepdim=True)
    print("naive exp(x)/sum      :", naive.tolist(), "  <- exp(100) = inf in fp32, so inf/inf = nan")
    print("softmax_single_block  :", [round(v, 4) for v in softmax_single_block(x)[0].tolist()])
    print("torch.softmax         :", [round(v, 4) for v in torch.softmax(x, -1)[0].tolist()])


# --------------------------------------------------------------------------------------
# CORRECTNESS
# --------------------------------------------------------------------------------------
def check_softmax():
    torch.manual_seed(0)
    print("\n== softmax correctness vs torch.softmax ==")
    for rows, n in ((1, 1), (4, 100), (37, 1000), (8, 5000)):
        x = 5 * torch.randn(rows, n, device="cuda")
        ref = torch.softmax(x, dim=-1)
        for name, fn in (("single_block", softmax_single_block), ("online(BLOCK=256)", lambda t: softmax_online(t, 256))):
            err = (fn(x) - ref).abs().max().item()
            print(f"rows={rows:<3} n={n:<5} {name:<18} max abs error = {err:.1e}")
            assert err < 1e-6


def check_attention():
    torch.manual_seed(0)
    print("\n== single-query attention correctness (fp32) ==")
    for B, N, D in ((1, 1, 64), (3, 100, 64), (4, 4096, 128), (2, 1000, 128)):
        q = torch.randn(B, D, device="cuda")
        k = torch.randn(B, N, D, device="cuda")
        v = torch.randn(B, N, D, device="cuda")
        ref = torch.softmax((k @ q[:, :, None]).squeeze(-1) / D**0.5, dim=-1)[:, None, :] @ v
        ref = ref.squeeze(1)
        out = attn_row(q, k, v)
        err = (out - ref).abs().max().item()
        print(f"B={B} N={N:<5} D={D:<3} max abs error = {err:.1e}")
        assert err < 1e-4
    # The chunk size is a tuning knob only: results must not depend on it (beyond fp
    # rounding), because rescaling makes the running state exact regardless of chunking.
    outs = [attn_row(q, k, v, bn) for bn in (16, 64, 256)]
    spread = max((o - outs[0]).abs().max().item() for o in outs)
    print(f"max difference across BLOCK_N in (16, 64, 256): {spread:.1e}")


# --------------------------------------------------------------------------------------
# SPEED: softmax is memory-bound (a few flops per element), so we report GB/s.
# --------------------------------------------------------------------------------------
def benchmark():
    print("\n== softmax bandwidth: 64 rows, growing row length ==")
    rows = 64

    def report(name, fn, passes):
        try:
            ms = cast(float, triton.testing.do_bench(fn))
        except Exception as e:  # e.g. a block too big to compile
            print(f"    {name:<26} failed: {type(e).__name__}")
            return
        gbps = passes * rows * n * 4 / (ms * 1e-3) / 1e9
        print(f"    {name:<26} {ms:8.3f} ms   {gbps:7.1f} GB/s")

    for n in (1024, 8192, 65536, 262144):
        x = torch.randn(rows, n, device="cuda")
        print(f"n = {n}")
        report("torch.softmax", lambda: torch.softmax(x, -1), 2)          # 1 read + 1 write
        report("single block (A)", lambda: softmax_single_block(x), 2)    # 1 read + 1 write
        report("online BLOCK=1024 (B)", lambda: softmax_online(x, 1024), 3)  # 2 reads + 1 write
    print("Notes: GB/s counts the MINIMUM bytes each algorithm must move (A/torch: 1 read + 1 write;\n"
          "B: 2 reads + 1 write), so torch's real traffic may be higher than shown. For n <= 8192 the\n"
          "input is only ~2 MB (fits in the T4's 4 MB L2) and the kernels take ~10-30 us, so those\n"
          "rows are cache/launch dominated: only the n >= 65536 rows say anything about memory bandwidth.")


if __name__ == "__main__":
    trace_online_softmax()
    demo_stability()
    check_softmax()
    check_attention()
    benchmark()
