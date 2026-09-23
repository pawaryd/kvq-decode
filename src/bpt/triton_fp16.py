"""Triton FP16 GQA decode attention over a paged KV cache (milestone 2).

One program per (sequence, kv head). It walks that sequence's block table one page
at a time with an online softmax, and handles all `group = Hq/Hkv` query heads that
share the kv head in a single tl.dot (padded up to 16 rows, the tensor-core minimum).
Layout matches bpt.reference.

Split-KV (milestone 3): with `num_splits > 1` each sequence's pages are divided into
contiguous ranges handled by separate programs (grid axis 2). Each writes a normalized
partial output plus its log-sum-exp; a second small kernel merges them. `num_splits=1`
runs the original single-pass kernel (the milestone-2 baseline).
"""
import functools
import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K, V, BT, SL, O,
    scale,
    stride_qb, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_bt,
    stride_ob, stride_oh,
    GROUP: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)  # kv head
    seq_len = tl.load(SL + b)

    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    t = tl.arange(0, BLOCK)
    g_ok = g < GROUP

    q = tl.load(
        Q + b * stride_qb + (h * GROUP + g)[:, None] * stride_qh + d[None, :],
        mask=g_ok[:, None], other=0.0,
    )

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, D], dtype=tl.float32)

    for i in range(0, tl.cdiv(seq_len, BLOCK)):
        phys = tl.load(BT + b * stride_bt + i).to(tl.int64)
        valid = (i * BLOCK + t) < seq_len
        k = tl.load(
            K + phys * stride_kb + t[:, None] * stride_kt + h * stride_kh + d[None, :],
            mask=valid[:, None], other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * scale  # [G_PAD, BLOCK] fp32
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + phys * stride_vb + t[:, None] * stride_vt + h * stride_vh + d[None, :],
            mask=valid[:, None], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        O + b * stride_ob + (h * GROUP + g)[:, None] * stride_oh + d[None, :],
        out.to(O.dtype.element_ty), mask=g_ok[:, None],
    )


@triton.jit
def _split_kernel(
    Q, K, V, BT, SL, OP, LSE,
    scale, pages_per_split, num_splits,
    stride_qb, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_bt,
    Hq,
    GROUP: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    """Stage 1: online softmax over pages [s*pps, (s+1)*pps) of one (seq, kv head)."""
    b = tl.program_id(0)
    h = tl.program_id(1)  # kv head
    s = tl.program_id(2)
    seq_len = tl.load(SL + b)

    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    t = tl.arange(0, BLOCK)
    g_ok = g < GROUP

    q = tl.load(
        Q + b * stride_qb + (h * GROUP + g)[:, None] * stride_qh + d[None, :],
        mask=g_ok[:, None], other=0.0,
    )

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, D], dtype=tl.float32)

    start = s * pages_per_split
    end = tl.minimum(start + pages_per_split, tl.cdiv(seq_len, BLOCK))
    for i in range(start, end):
        phys = tl.load(BT + b * stride_bt + i).to(tl.int64)
        valid = (i * BLOCK + t) < seq_len
        k = tl.load(
            K + phys * stride_kb + t[:, None] * stride_kt + h * stride_kh + d[None, :],
            mask=valid[:, None], other=0.0,
        )
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(valid[None, :], sc, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        p = tl.exp(sc - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + phys * stride_vb + t[:, None] * stride_vt + h * stride_vh + d[None, :],
            mask=valid[:, None], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # An empty split has l_i == 0: write o = 0, lse = -inf so the merge ignores it.
    nonempty = l_i > 0.0
    l_safe = tl.where(nonempty, l_i, 1.0)
    lse = tl.where(nonempty, m_i + tl.log(l_safe), float("-inf"))
    row = (b * Hq + h * GROUP + g) * num_splits + s  # [G_PAD]
    tl.store(OP + row[:, None] * D + d[None, :], acc / l_safe[:, None], mask=g_ok[:, None])
    tl.store(LSE + row, lse, mask=g_ok)


@triton.jit
def _merge_kernel(
    OP, LSE, O, stride_ob, stride_oh, Hq, num_splits,
    D: tl.constexpr, S_PAD: tl.constexpr,
):
    """Stage 2: out = sum_s w_s * o_s / sum_s w_s with w_s = exp(lse_s - max lse)."""
    b = tl.program_id(0)
    hq = tl.program_id(1)
    sp = tl.arange(0, S_PAD)
    d = tl.arange(0, D)
    ok = sp < num_splits
    row = (b * Hq + hq) * num_splits + sp
    lse = tl.load(LSE + row, mask=ok, other=float("-inf"))
    w = tl.exp(lse - tl.max(lse, axis=0))
    o = tl.load(OP + row[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
    out = tl.sum(w[:, None] * o, axis=0) / tl.sum(w, axis=0)
    tl.store(O + b * stride_ob + hq * stride_oh + d, out.to(O.dtype.element_ty))


TARGET_PROGRAMS_PER_SM = 8
MIN_TOKENS_PER_SPLIT = 512


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def choose_num_splits(batch: int, kv_heads: int, block_size: int, max_pages: int,
                      sm_count: int) -> int:
    """Enough programs to fill the GPU, without splits shorter than a few pages."""
    want = triton.cdiv(TARGET_PROGRAMS_PER_SM * sm_count, batch * kv_heads)
    cap = max(1, max_pages // max(1, MIN_TOKENS_PER_SPLIT // block_size))
    return max(1, min(triton.next_power_of_2(want), cap))


def paged_decode_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
    num_splits: Optional[int] = None,
) -> torch.Tensor:
    """num_splits: None = heuristic, 1 = single-pass kernel, n = split each sequence n ways."""
    B, Hq, D = q.shape
    _, block_size, Hkv, _ = k_cache.shape
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.float16
    assert Hq % Hkv == 0
    assert D == triton.next_power_of_2(D) and D >= 16
    assert block_size == triton.next_power_of_2(block_size) and block_size >= 16
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert block_table.dtype == seq_lens.dtype == torch.int32
    assert int(seq_lens.min()) >= 1, "empty sequences are not supported"
    group = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    max_pages = block_table.shape[1]
    if num_splits is None:
        num_splits = choose_num_splits(B, Hkv, block_size, max_pages,
                                       _sm_count(q.device.index or 0))
    num_splits = max(1, min(num_splits, max_pages))
    g_pad = max(16, triton.next_power_of_2(group))

    if num_splits > 1:
        pages_per_split = triton.cdiv(max_pages, num_splits)
        num_splits = triton.cdiv(max_pages, pages_per_split)  # drop always-empty tails
    if num_splits > 1:
        o_part = torch.empty(B, Hq, num_splits, D, dtype=torch.float32, device=q.device)
        lse = torch.empty(B, Hq, num_splits, dtype=torch.float32, device=q.device)
        _split_kernel[(B, Hkv, num_splits)](
            q, k_cache, v_cache, block_table, seq_lens, o_part, lse,
            scale, pages_per_split, num_splits,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_table.stride(0),
            Hq,
            GROUP=group, G_PAD=g_pad, D=D, BLOCK=block_size,
        )
        _merge_kernel[(B, Hq)](
            o_part, lse, out, out.stride(0), out.stride(1), Hq, num_splits,
            D=D, S_PAD=triton.next_power_of_2(num_splits),
        )
        return out

    _paged_decode_kernel[(B, Hkv)](
        q, k_cache, v_cache, block_table, seq_lens, out,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        out.stride(0), out.stride(1),
        GROUP=group, G_PAD=g_pad, D=D, BLOCK=block_size,
    )
    return out
