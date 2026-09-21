"""PyTorch reference for GQA decode attention over a paged KV cache.

Layout (vLLM-style):
    q            [B, Hq, D]                       one query token per sequence
    k_cache      [num_blocks, block_size, Hkv, D]
    v_cache      [num_blocks, block_size, Hkv, D]
    block_table  [B, max_blocks] int32            logical block -> physical block
    seq_lens     [B] int32                        valid tokens per sequence
    returns      [B, Hq, D]

Math is done in fp32 regardless of input dtype; output is cast back to q.dtype.
This is the correctness oracle for the kernels, so it favours clarity over speed.
"""
import math
from typing import Optional, Tuple

import torch


def paged_decode_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    B, Hq, D = q.shape
    _, block_size, Hkv, _ = k_cache.shape
    assert Hq % Hkv == 0, "q heads must be a multiple of kv heads"
    group = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    for b in range(B):
        n = int(seq_lens[b])
        n_blocks = (n + block_size - 1) // block_size
        blocks = block_table[b, :n_blocks].long()
        # [n_blocks, block_size, Hkv, D] -> [n, Hkv, D], dropping the partial-block tail
        k = k_cache[blocks].reshape(-1, Hkv, D)[:n].float()
        v = v_cache[blocks].reshape(-1, Hkv, D)[:n].float()
        # q head h reads kv head h // group
        k = k.repeat_interleave(group, dim=1)  # [n, Hq, D]
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hn,nhd->hd", probs, v).to(q.dtype)
    return out


def build_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter contiguous k/v [B, S, Hkv, D] into a paged cache.

    Physical blocks are randomly permuted (so block-table bugs show up), and slots
    past each sequence's length are filled with large garbage (so masking bugs
    show up). Returns (k_cache, v_cache, block_table).
    """
    B, S, Hkv, D = k.shape
    max_blocks = (S + block_size - 1) // block_size
    num_blocks = B * max_blocks
    perm = torch.randperm(num_blocks, generator=generator).to(k.device)
    block_table = perm.reshape(B, max_blocks).to(torch.int32)

    def fill(x: torch.Tensor) -> torch.Tensor:
        cache = 100 * torch.randn(
            num_blocks, block_size, Hkv, D, generator=generator
        ).to(device=x.device, dtype=x.dtype)
        for b in range(B):
            n = int(seq_lens[b])
            for i in range((n + block_size - 1) // block_size):
                lo, hi = i * block_size, min((i + 1) * block_size, n)
                cache[block_table[b, i].long(), : hi - lo] = x[b, lo:hi]
        return cache

    return fill(k), fill(v), block_table
