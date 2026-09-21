# Notes

## Milestone 1: reference + roofline

### Reference (`src/bpt/reference.py`)
- vLLM-style paged layout: `k_cache/v_cache [num_blocks, block_size, Hkv, D]`, `block_table [B, max_blocks] int32`, `seq_lens [B]`.
- Math in fp32, cast to q dtype at the end. Deliberately slow (loop over batch, `repeat_interleave` for GQA) so it is easy to trust.
- Test oracle is torch SDPA on a contiguous cache with K/V expanded via `repeat_interleave` (not `enable_gqa`, to work on older torch).
- `build_paged_cache` randomly permutes physical blocks and fills unused slots with large garbage (100*randn), so block-table and masking bugs cannot pass by accident.
- Observed errors (CPU): fp32 max abs <= 6.6e-7; fp16 max abs ~5e-4.

### Roofline (`src/bpt/roofline.py`, `bench/roofline.py`)
- Counts KV-cache reads only. Weights, q, and output traffic are excluded.
- bytes/token = layers * Hkv * (2*D*elem_bytes + scale overhead). Llama-3-8B fp16 = 128 KiB/token.
- Quant overhead assumes asymmetric (fp16 scale + zero): K per-channel per `group_size` tokens (default 128), V per-token. This is more conservative than scale-only. FP8 is a plain cast, no scales. INT4 therefore gives 3.76x, not 4x, on Llama-3-8B.
- Peak BW (GB/s): T4 320, A100-80GB 2039, H100-SXM 3350, B200 8000. Spec-sheet peaks; achievable is typically ~80-90%.

## Milestone 2: Triton FP16 paged kernel (`src/bpt/triton_fp16.py`)

### Design
- Grid `(B, Hkv)`; each program serves the `group = Hq/Hkv` query heads that share its kv head with one `tl.dot` (rows padded to 16, the tensor-core minimum). Loop over the block table one page at a time with an online softmax (fp32 running max/sum/acc). Tile width = page size (16 or 32).
- fp16 only (T4 has no bf16/fp8). Requires seq_len >= 1, power-of-2 head_dim and block_size >= 16.
- Dev flow: no local GPU, so tests and benchmarks run on Modal T4 via `modal_scripts/run_remote.py`.
  Run Modal from outside the repo root or with a directory that is not called `modal/` (it shadowed the SDK; renamed to `modal_scripts/`).

### Correctness (T4, `tests/test_triton_fp16.py`, 24 cases)
- Llama-3-8B GQA, MHA, MQA; block size 16/32; ragged lengths up to 4096, incl. len 1 and non-multiples of the page size.
- Max abs error vs the fp32 reference: <= 9.8e-4 (about 1 fp16 ulp for outputs near 1-2). Max rel up to 0.11, only on outputs near zero (clamp 1e-3).

### Performance (T4, single layer, `results/triton_fp16_T4.json`, peak 320 GB/s; measured copy ~234 GB/s)
| batch | ctx | achieved GB/s | % of peak |
|---|---|---|---|
| 1 | 4k-16k | 6-15 | 2-5% |
| 8 | 4k-16k | ~37 | ~11.5% |
| 32 | 4k-16k | ~43 | ~13.5% |

This kernel is **slow, not a baseline to be proud of**. Diagnosis (hypotheses, not yet profiled):
- B=1 launches only `Hkv = 8` programs on a 40-SM GPU: the low numbers there are an occupancy problem (milestone 3, split-KV).
- Even at B=32 (256 programs) it tops out at ~43 GB/s, so it is also latency-bound per program: a serial loop over 16/32-token pages, one small dependent load-then-dot per iteration, no pipelining.
- With group=4 the 16-row `tl.dot` wastes 75% of the MMA rows (irrelevant if we are memory-bound, relevant while we are latency-bound).
- Page size 16 vs 32 made no difference at large batch.

Next: split-KV (milestone 3) for occupancy; then check larger tiles (several pages per iteration) and `num_stages` before concluding anything about the memory system.
