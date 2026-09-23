"""Triton FP16 paged decode kernel vs the PyTorch reference (needs a CUDA GPU)."""
import pytest
import torch

from bpt.reference import build_paged_cache, paged_decode_attention_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [(32, 8, 128), (4, 4, 64), (8, 1, 64)]  # Llama-3-8B, MHA, MQA


def make(Hq, Hkv, D, block_size, lens):
    g = torch.Generator().manual_seed(0)
    B, S = len(lens), max(lens)
    q = torch.randn(B, Hq, D, generator=g).cuda().half()
    k = torch.randn(B, S, Hkv, D, generator=g).cuda().half()
    v = torch.randn(B, S, Hkv, D, generator=g).cuda().half()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    k_cache, v_cache, bt = build_paged_cache(k, v, seq_lens, block_size, g)
    return q, k_cache, v_cache, bt, seq_lens


# 1 = single-pass kernel; 2, 3, 7 = split-KV (7 exceeds the page count of short
# sequences, so some splits are empty); None = heuristic
@pytest.mark.parametrize("num_splits", [1, 2, 3, 7, None])
@pytest.mark.parametrize("Hq,Hkv,D", SHAPES)
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("lens", [[1], [37], [100, 33, 1, 64], [4096, 1000, 17, 2049]])
def test_matches_reference(Hq, Hkv, D, block_size, lens, num_splits):
    from bpt.triton_fp16 import paged_decode_attention_triton

    q, k_cache, v_cache, bt, seq_lens = make(Hq, Hkv, D, block_size, lens)
    out = paged_decode_attention_triton(q, k_cache, v_cache, bt, seq_lens,
                                        num_splits=num_splits)
    ref = paged_decode_attention_ref(q, k_cache, v_cache, bt, seq_lens)

    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-3)).max().item()
    print(f"\n[Hq={Hq} Hkv={Hkv} D={D} bs={block_size} splits={num_splits} lens={lens}] "
          f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    assert max_abs < 2e-3


def test_split_kv_is_deterministic():
    from bpt.triton_fp16 import paged_decode_attention_triton

    args = make(32, 8, 128, 16, [4096, 1000, 17, 2049])
    a = paged_decode_attention_triton(*args, num_splits=8)
    b = paged_decode_attention_triton(*args, num_splits=8)
    assert torch.equal(a, b)


def test_splits_agree_with_single_pass():
    """Merging partial softmaxes must not change the answer beyond fp16 rounding."""
    from bpt.triton_fp16 import paged_decode_attention_triton

    args = make(32, 8, 128, 32, [4096, 1000, 17, 2049])
    base = paged_decode_attention_triton(*args, num_splits=1)
    for n in (2, 5, 16):
        out = paged_decode_attention_triton(*args, num_splits=n)
        assert (out.float() - base.float()).abs().max().item() < 2e-3
