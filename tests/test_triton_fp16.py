"""Triton FP16 paged decode kernel vs the PyTorch reference (needs a CUDA GPU)."""
import pytest
import torch

from bpt.reference import build_paged_cache, paged_decode_attention_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPES = [(32, 8, 128), (4, 4, 64), (8, 1, 64)]  # Llama-3-8B, MHA, MQA


@pytest.mark.parametrize("Hq,Hkv,D", SHAPES)
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("lens", [[1], [37], [100, 33, 1, 64], [4096, 1000, 17, 2049]])
def test_matches_reference(Hq, Hkv, D, block_size, lens):
    from bpt.triton_fp16 import paged_decode_attention_triton

    g = torch.Generator().manual_seed(0)
    B, S = len(lens), max(lens)
    q = torch.randn(B, Hq, D, generator=g).cuda().half()
    k = torch.randn(B, S, Hkv, D, generator=g).cuda().half()
    v = torch.randn(B, S, Hkv, D, generator=g).cuda().half()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    k_cache, v_cache, bt = build_paged_cache(k, v, seq_lens, block_size, g)

    out = paged_decode_attention_triton(q, k_cache, v_cache, bt, seq_lens)
    ref = paged_decode_attention_ref(q, k_cache, v_cache, bt, seq_lens)

    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-3)).max().item()
    print(f"\n[Hq={Hq} Hkv={Hkv} D={D} bs={block_size} lens={lens}] "
          f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    assert max_abs < 2e-3
