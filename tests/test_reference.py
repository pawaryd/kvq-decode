"""Check the paged GQA decode reference against torch SDPA on a non-paged cache."""
import pytest
import torch
import torch.nn.functional as F

from bpt.reference import build_paged_cache, paged_decode_attention_ref

# (Hq, Hkv, D): Llama-3-8B, MHA, MQA
SHAPES = [(32, 8, 128), (4, 4, 64), (8, 1, 64)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def sdpa_oracle(q, k, v, seq_lens):
    """Non-paged: per-sequence SDPA over the first seq_len tokens of contiguous k/v."""
    B, Hq, D = q.shape
    group = Hq // k.shape[2]
    outs = []
    for b in range(B):
        n = int(seq_lens[b])
        kb = k[b, :n].repeat_interleave(group, dim=1).permute(1, 0, 2)  # [Hq, n, D]
        vb = v[b, :n].repeat_interleave(group, dim=1).permute(1, 0, 2)
        o = F.scaled_dot_product_attention(q[b][:, None, :], kb, vb)  # [Hq, 1, D]
        outs.append(o[:, 0])
    return torch.stack(outs)


def make_inputs(B, lens, Hq, Hkv, D, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = max(lens)
    q = torch.randn(B, Hq, D, generator=g).to(device, dtype)
    k = torch.randn(B, S, Hkv, D, generator=g).to(device, dtype)
    v = torch.randn(B, S, Hkv, D, generator=g).to(device, dtype)
    return q, k, v, torch.tensor(lens, dtype=torch.int32, device=device), g


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("lens", [[1], [37], [100, 33, 1, 64]])
@pytest.mark.parametrize("Hq,Hkv,D", SHAPES)
def test_matches_sdpa(Hq, Hkv, D, lens, block_size, dtype, atol, device):
    q, k, v, seq_lens, g = make_inputs(len(lens), lens, Hq, Hkv, D, dtype, device)
    k_cache, v_cache, bt = build_paged_cache(k, v, seq_lens, block_size, g)

    out = paged_decode_attention_ref(q, k_cache, v_cache, bt, seq_lens)
    try:
        ref = sdpa_oracle(q, k, v, seq_lens)
    except RuntimeError:
        if dtype == torch.float16 and device == "cpu":
            pytest.skip("CPU SDPA lacks fp16 support in this torch build")
        raise

    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-3)).max().item()
    print(f"\n[{device} {dtype} Hq={Hq} Hkv={Hkv} D={D} bs={block_size} lens={lens}] "
          f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    assert max_abs < atol


def test_garbage_in_unused_slots_is_ignored():
    lens = [37, 5]
    q, k, v, seq_lens, g = make_inputs(2, lens, 8, 2, 32, torch.float32, "cpu")
    a = build_paged_cache(k, v, seq_lens, 16, torch.Generator().manual_seed(1))
    b = build_paged_cache(k, v, seq_lens, 16, torch.Generator().manual_seed(2))
    out_a = paged_decode_attention_ref(q, a[0], a[1], a[2], seq_lens)
    out_b = paged_decode_attention_ref(q, b[0], b[1], b[2], seq_lens)
    # different block permutations and different garbage -> identical result
    assert torch.allclose(out_a, out_b, atol=1e-6)
