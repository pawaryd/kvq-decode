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

## Milestone 3: split-KV, execution log

Plan (approved 2026-09-21): third grid axis over contiguous page ranges; stage 1 writes a
normalized fp32 partial output plus log-sum-exp per (seq, q head, split); a small stage-2
kernel merges with w = exp(lse - max lse). `num_splits=1` keeps the milestone-2 kernel as the
baseline. `num_splits=None` uses a heuristic (target ~4 programs/SM, >= 256 tokens per
split, power of 2). Out of scope: multi-page tiles, `num_stages`, quantization. The A100
sweep and committing were left undecided by the user, so neither is done.

Log:
1. Wrote `_split_kernel`, `_merge_kernel`, `choose_num_splits` in `src/bpt/triton_fp16.py`.
   Empty splits (start >= end) emit o=0, lse=-inf; the merge weights them to 0. Effective
   split count is recomputed as `cdiv(max_pages, pages_per_split)` so trailing splits that
   are empty for *every* sequence are never launched.
2. Extended `tests/test_triton_fp16.py`: `num_splits` in {1, 2, 3, 7, auto} across the
   milestone-2 cases, plus determinism and split-vs-single-pass tests.
3. T4 correctness: 122 passed (120 parametrized + determinism + split-vs-single-pass).
   Max abs error is identical to milestone 2 (<= 9.8e-4), so the merge adds no visible error.
4. First T4 sweep (`results/triton_fp16_splitkv_T4.json`, before heuristic retune). Best GB/s per config, vs `num_splits=1`:
   B=1: 4k 23.0 (8 splits) vs 5.7; 16k 43.6 (64) vs 12.4; 64k 45.8 (64) vs 15.2.
   B=8: ~45-47 vs ~37. B=32: ~46.8-47.3 vs ~43.4. All 14-15% of the 320 GB/s peak at best.
   - Batch 1 improved 3.5-4x, as expected from the occupancy argument.
   - **But throughput plateaus at ~47 GB/s regardless of split count** once B*Hkv*splits exceeds ~100s of programs (B=32 gets nothing from splits). So occupancy was only part of the problem; the kernel itself is capped at ~15% of peak on T4.
   - First heuristic (4 programs/SM, >= 256 tokens/split) was mediocre: chose 16 splits at B=1/4k where 8 is ~60% faster (short splits pay the merge + launch overhead), and 4 at B=8 where 64 is ~8% faster.
   (Sweep numbers above were re-measured once more by an accidental rerun with the same old constants; results agreed within ~5%, the JSON on disk is that rerun.)
5. Retuned the two heuristic constants to 8 programs/SM and >= 512 tokens/split from that sweep (tuned on T4 only, on the same data it is evaluated on, so treat it as a T4 default, not a general result).
6. Re-ran the sweep with the retuned heuristic (JSON on disk is this run). Auto now picks: B=1: 8/32/64 splits for 4k/16k/64k; B=8: 8; B=32: 2. Achieved GB/s with auto:
   B=1 16k 39.5, 64k 44.2; B=8 4k-64k 42.5-44.3; B=32 41.8-42.3 (13-14% of peak). At large batch that is ~4-8% below the best fixed split count in the first sweep; between-run variation on separate T4 containers is of similar size (same split counts differed by up to ~5%), so the retune is not clearly better than the first heuristic at B>=8. It fixes the B=1/4k case *by construction* (8 splits), not measurably.
   - **B=1/4k is unreliable**: the same split counts gave 19-23 GB/s in the first sweep and 12-14 GB/s in this one. At ~1 ms per call this config is probably host launch-bound (two Triton launches + allocations per call). Hypothesis, not measured; don't read anything into B=1/4k.
7. Final T4 test run after the retune: 122 passed.

### Milestone 3 outcome
- Correct: 122/122 tests, max abs error <= 9.8e-4 (fp16 ulp), deterministic, splits agree with single-pass.
- Split-KV fixed the occupancy problem at small batch: B=1, 16k-64k context went from 12-15 GB/s to ~40-46 GB/s (~3x). It does nothing beyond that.
- **The kernel is still at ~13-15% of T4 peak (~42-47 GB/s vs 234 GB/s measured copy).** Splitting cannot fix that: at B>=8 there are already hundreds of programs and more splits do not help. The bottleneck is inside the per-program loop. Untested hypotheses, in the order I would check them:
  1. Triton's `tl.dot` on sm75 may not be using tensor cores efficiently (or at all) at M=16 tiles; check the generated PTX/SASS for `mma`.
  2. Per-page tile is tiny (16-32 tokens x 128) and the loop is serial with dependent loads and no software pipelining; try several pages per iteration / `num_stages`.
  3. 75% of the 16 padded rows are wasted at group=4 (irrelevant if memory-bound, but we are not).
  4. `tl.trans(k)` may be forcing a shared-memory round trip.
  Do one cheap experiment per hypothesis before changing the design. It is also unknown whether the ceiling is specific to T4/Triton-sm75; the A100 sweep would tell us and has not been run.

### A100-80GB run (2026-09-21, `results/triton_fp16_splitkv_A100.json`)
8. Correctness on A100-80GB: 122 passed, max abs error 9.8e-4 (same as T4).
9. Same sweep, same code (heuristic constants 8 programs/SM, 512 tokens/split), peak 2039 GB/s, 108 SMs.
   Best fixed split count per config vs `num_splits=1`, and `auto`:
   | B | ctx | splits=1 | best (splits) | auto (splits) |
   |---|---|---|---|---|
   | 1 | 4k | 70 GB/s (3.4%) | 118 (64) 5.8% | 113 (8) |
   | 1 | 16k | 70 (3.4%) | 480 (32) 23.6% | 452 (32) |
   | 1 | 64k | 88 (4.3%) | 1089 (64) 53.4% | 1039 (128) 50.9% |
   | 8 | 4k | 508 (24.9%) | 806 (8) 39.5% | 766 (8) |
   | 8 | 16k | 639 (31.3%) | 1334 (8) 65.4% | 1302 (16) |
   | 8 | 64k | 682 (33.5%) | 1636 (8) 80.2% | 1607 (16) |
   | 32 | 4k | 1267 (62.2%) | 1323 (4) 64.9% | 1304 (4) |
   | 32 | 16k | 1590 (78.0%) | 1638 (2) 80.3% | 1612 (4) |
10. Findings:
   - **On A100 this kernel reaches ~78-80% of peak at B>=8 with >=16k context** (B=8/64k: 80.2%, B=32/16k: 80.3%). So the ~15% T4 ceiling is *not* inherent to the algorithm/kernel structure; it is specific to T4 (sm75) under this Triton version. That makes hypothesis 1 (Triton's `tl.dot`/tensor-core path on sm75) the leading suspect for T4, but it is still untested. Practically: **T4 is a poor proxy for kernel performance; use it for correctness only.** (CLAUDE.md already says tuning happens on A100/H100/B200.)
   - Split-KV is decisive on A100 at small batch: B=1/64k goes 88 -> 1089 GB/s (12x, 4.3% -> 53%); B=8/64k 682 -> 1636 (2.4x).
   - Remaining gaps: B=1 is still far below B>=8 (4k: 6%, 16k: 24%, 64k: 53%). Short contexts have too little data to fill the GPU and each call (~0.14 ms) is close to launch overhead territory (not measured).
   - Caveat, L2: A100 L2 is 40 MB. B=1/4k reads 16.8 MB of KV per call, so it can be served from L2 across the 50 timed iterations; those rows are not pure HBM numbers. Larger configs (>= 67 MB) exceed L2 and are HBM-bound-ish. I did not flush L2 between iterations.
   - Heuristic (tuned on T4) transfers acceptably: `auto` is within ~1-6% of the best fixed split count for every config except two where it picks a value outside the swept set or is mildly off (B=1/64k auto=128 vs 64: -3.4 points; B=8/16k 16 vs 8: -1.5 points). Not retuned for A100.
   - Single run per config, one A100 container; differences of a few percent are within the run-to-run variation seen on T4.
11. Plot: `bench/plot_splitkv.py` reads the two split-KV result JSONs and writes `docs/img/splitkv_scaling.png` (achieved % of peak vs KV splits, one line per batch size, rows = GPU, columns = context 4k/16k/64k; separate y-scales per GPU row, so the T4 row is 0-20%). What it shows: on A100 the B=1 curve keeps rising with splits up to ~32 at long context while B>=8 saturates at 4-8 splits; B=32 gains almost nothing from splitting. On T4 all curves flatten at 13-15%. Batch 32 at 64k is absent (not benchmarked; would not fit a T4). Palette (blue/orange/aqua, slots 1-3) was checked with the dataviz validator (--pairs all, light): all gates pass; aqua is 2.74:1 on the surface, so series are also identified by marker shape and the legend. Direct labels are omitted in the T4 panels where lines converge. matplotlib is an optional dependency (`.[plot]`); the PNG is light-mode only.
