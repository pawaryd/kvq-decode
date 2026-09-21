"""Triton FP16 GQA decode attention over a paged KV cache (milestone 2).

One program per (sequence, kv head). It walks that sequence's block table one page
at a time with an online softmax, and handles all `group = Hq/Hkv` query heads that
share the kv head in a single tl.dot (padded up to 16 rows, the tensor-core minimum).
Layout matches bpt.reference. No split-KV yet: with few sequences this launches only
B*Hkv programs and under-fills the GPU; milestone 3 fixes that.
"""
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


def paged_decode_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
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
    _paged_decode_kernel[(B, Hkv)](
        q, k_cache, v_cache, block_table, seq_lens, out,
        scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        out.stride(0), out.stride(1),
        GROUP=group, G_PAD=max(16, triton.next_power_of_2(group)), D=D, BLOCK=block_size,
    )
    return out
