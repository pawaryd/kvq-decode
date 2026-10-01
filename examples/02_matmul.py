"""Triton example 2: matrix multiply, C[M, N] = A[M, K] @ B[K, N].

Builds on 01_vector_add.py. New ideas:
  1. a 2D grid: each program computes one BLOCK_M x BLOCK_N *tile* of C
  2. 2D offsets and 2D masks (rows x columns)
  3. a loop over K: walk along the shared dimension, accumulating partial products
  4. tl.dot: the block-level matrix multiply that maps onto tensor cores
  5. strides instead of assuming contiguity (also how the paged KV kernel addresses memory)

Vector add was *memory-bound*. Matmul with big square matrices is *compute-bound*: it
does ~N FLOPs per element it loads. Decode attention is the opposite (few query rows
against a huge KV cache), which is why we care about bytes there. Stage 5 shows both.

Run on a GPU (from this repo use Modal):
    cd /tmp && <repo>/.venv/bin/python -m modal run <repo>/modal_scripts/run_remote.py \
        --gpu T4 --cmd "python examples/02_matmul.py"
"""
# pyright: reportArgumentType=false, reportCallIssue=false
# (Triton annotates the tile sizes as `tl.constexpr`; passing a plain int is the normal idiom.
#  `num_warps=` is a launch option that Triton's stubs don't declare on the kernel signature.)
from typing import Any, cast

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# STAGE 1: the kernel
# --------------------------------------------------------------------------------------
# Picture C split into a grid of BLOCK_M x BLOCK_N tiles. Program (pid_m, pid_n) owns the
# tile in tile-row pid_m and tile-column pid_n. To compute it, it needs a horizontal strip
# of A (BLOCK_M rows x all K columns) and a vertical strip of B (all K rows x BLOCK_N
# columns). Those strips are too big to load at once, so we walk along K in chunks of
# BLOCK_K: load an A chunk [BLOCK_M, BLOCK_K] and a B chunk [BLOCK_K, BLOCK_N], multiply
# them (a small matmul), and ADD the result into a running accumulator.
#
#            K                     N
#        +---------+           +---------+          +---------+
#   M    |  A strip|  x     K  | B strip |   =   M  |  C tile |
#        +---------+           +---------+          +---------+
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,            # pointers to A, B, C
    M, N, K,                        # matrix sizes (runtime values)
    stride_am, stride_ak,           # A[m, k] lives at a_ptr + m*stride_am + k*stride_ak
    stride_bk, stride_bn,           # B[k, n] lives at b_ptr + k*stride_bk + n*stride_bn
    stride_cm, stride_cn,           # C[m, n] lives at c_ptr + m*stride_cm + n*stride_cn
    BLOCK_M: tl.constexpr,          # tile height  (rows of C per program)
    BLOCK_N: tl.constexpr,          # tile width   (columns of C per program)
    BLOCK_K: tl.constexpr,          # how far along K each loop step goes
):
    # 2D grid: axis 0 indexes tile-rows, axis 1 indexes tile-columns.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row / column indices this program is responsible for (1D blocks of indices).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # Build 2D pointer blocks by broadcasting. `offs_m[:, None]` is a column [BLOCK_M, 1],
    # `offs_k[None, :]` is a row [1, BLOCK_K]; adding them gives a [BLOCK_M, BLOCK_K]
    # block where element (i, j) points at A[offs_m[i], offs_k[j]]. Using strides means
    # this works for any layout (row-major, column-major, transposed views...).
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # The accumulator lives in registers for the whole loop. It is fp32 even though the
    # inputs are fp16: summing K products in fp16 would lose precision. This "low-precision
    # in, fp32 accumulate" pattern is what tensor cores do natively.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Walk along K. cdiv rounds up so a K that isn't a multiple of BLOCK_K still gets a
    # (partial) final step.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # How many valid K columns/rows remain from this step onward.
        k_left = K - k * BLOCK_K

        # Masks: a lane is valid only if its row (or column) is inside the matrix AND its
        # k index is inside K. `other=0.0` pads invalid lanes with 0, which is exactly
        # right: zeros add nothing to a dot product, so the tail contributes nothing.
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)

        # tl.dot multiplies [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N].
        # This is the tensor-core instruction path. Requirements: each tile dimension
        # >= 16 and a power of two. `acc += ...` folds this step into the running sum.
        acc += tl.dot(a, b)

        # Slide the pointer blocks BLOCK_K further along K for the next iteration.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert the fp32 accumulator to the output dtype (fp16) and write the tile out.
    # Mask: edge tiles may hang over the bottom/right of C.
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    # (Offsets here are int32, so this simple kernel breaks once an offset exceeds ~2^31
    #  elements. Real kernels cast to int64 -- the paged-KV kernel does for block ids.)


# --------------------------------------------------------------------------------------
# STAGE 2: the Python wrapper
# --------------------------------------------------------------------------------------
def matmul(a: torch.Tensor, b: torch.Tensor, block_m=64, block_n=64, block_k=32, num_warps=4):
    assert a.is_cuda and b.is_cuda and a.dtype == b.dtype == torch.float16
    assert a.shape[1] == b.shape[0], "inner dimensions must match"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 2D grid: one program per output tile.
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    # Note we pass .stride(i) for every tensor: the kernel never assumes contiguity.
    # `num_warps` = how many warps (groups of 32 threads) cooperate on one program.
    # (Typed `Any`: Triton's stubs say the launch returns None, but it returns the handle.)
    compiled: Any = matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps,
    )
    return c, compiled


# --------------------------------------------------------------------------------------
# STAGE 3: correctness
# --------------------------------------------------------------------------------------
def check_correctness():
    torch.manual_seed(0)
    print("== correctness ==")
    # Reference: multiply the same fp16 inputs in fp32, so the only error left is the final
    # rounding of C to fp16 (relative error ~5e-4 of the largest value).
    def check(name, a, b):
        c, _ = matmul(a, b)
        ref = a.float() @ b.float()
        err = (c.float() - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"{name:<34} max abs error = {err:.2e}   (relative to max |C|: {rel:.1e})")
        assert rel < 2e-3

    # Sizes that are NOT multiples of the tiles (64, 64, 32) exercise every mask.
    for M, N, K in ((16, 16, 16), (100, 70, 50), (257, 513, 129), (1024, 1024, 1024)):
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        check(f"M={M} N={N} K={K}", a, b)

    # A transposed (non-contiguous) A: same values, swapped strides. It works because the
    # kernel addresses memory through strides, not through an assumed layout.
    a_t = torch.randn(129, 100, device="cuda", dtype=torch.float16).t()  # shape [100, 129]
    b = torch.randn(129, 70, device="cuda", dtype=torch.float16)
    print(f"a_t strides = {a_t.stride()} (contiguous would be ({a_t.shape[1]}, 1))")
    check("transposed A (100x129 @ 129x70)", a_t, b)


# --------------------------------------------------------------------------------------
# STAGE 4: speed of a big square matmul (compute-bound)
# --------------------------------------------------------------------------------------
def benchmark_square():
    print("\n== compute-bound: 2048 x 2048 x 2048 ==")
    M = N = K = 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    flops = 2 * M * N * K  # one multiply + one add per (m, n, k)

    def report(name, fn):
        ms = cast(float, triton.testing.do_bench(fn))
        print(f"{name:<34} {ms:7.3f} ms   {flops / (ms * 1e-3) / 1e12:6.2f} TFLOP/s")

    report("torch (cuBLAS)", lambda: a @ b)
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps). Bigger tiles reuse each loaded element more
    # (fewer bytes per FLOP) but need more registers / shared memory. A tile that needs
    # more shared memory than the GPU has fails to compile -- caught and reported below.
    for bm, bn, bk, nw in ((16, 16, 32, 2), (32, 32, 32, 4), (64, 64, 32, 4),
                           (128, 64, 32, 4), (128, 128, 32, 8), (128, 128, 64, 8)):
        name = f"triton tile {bm}x{bn}x{bk} warps={nw}"
        try:
            report(name, lambda bm=bm, bn=bn, bk=bk, nw=nw: matmul(a, b, bm, bn, bk, nw))
        except Exception as e:  # e.g. OutOfResources
            print(f"{name:<34} failed: {type(e).__name__}")
    print("(T4 fp16 tensor-core peak is ~65 TFLOP/s; real kernels reach a fraction of it)")


# --------------------------------------------------------------------------------------
# STAGE 5: the contrast that motivates this whole project. A "skinny" matmul, only 16
# rows of A (like 16 query rows against a huge matrix), is memory-bound: the big matrix B
# must be streamed from memory once and each element is used only 16 times.
# --------------------------------------------------------------------------------------
def benchmark_skinny():
    print("\n== memory-bound: 16 x 4096 @ 4096 x 4096 ==")
    M, K, N = 16, 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    flops = 2 * M * N * K
    nbytes = (M * K + K * N + M * N) * 2  # fp16 = 2 bytes; B (32 MB) dominates
    print(f"arithmetic intensity: {flops / nbytes:.1f} FLOP/byte "
          f"(2048^3 square case: {2 * 2048**3 / (3 * 2048**2 * 2):.0f})")

    def report(name, fn):
        ms = cast(float, triton.testing.do_bench(fn))
        print(f"{name:<34} {ms:7.3f} ms   {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s   "
              f"{flops / (ms * 1e-3) / 1e12:5.2f} TFLOP/s")

    report("torch (cuBLAS)", lambda: a @ b)
    for bm, bn, bk, nw in ((16, 64, 64, 4), (16, 128, 64, 4), (16, 64, 128, 4)):
        name = f"triton tile {bm}x{bn}x{bk} warps={nw}"
        try:
            report(name, lambda bm=bm, bn=bn, bk=bk, nw=nw: matmul(a, b, bm, bn, bk, nw))
        except Exception as e:
            print(f"{name:<34} failed: {type(e).__name__}")
    print("Here TFLOP/s is tiny but GB/s is what matters: this is bandwidth-bound, like decode.")


# --------------------------------------------------------------------------------------
# STAGE 6: did the compiler use tensor cores?
# --------------------------------------------------------------------------------------
# tl.dot on fp16 should lower to `mma.sync` PTX instructions (the tensor-core op). If it
# fell back to plain fused multiply-adds (`fma`), you would see fma instead and speed
# would be far lower. Worth checking, especially on older GPUs like the T4 (sm75).
def show_ptx():
    print("\n== compiler output ==")
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _, compiled = matmul(a, b, 64, 64, 32, 4)
    ptx = compiled.asm["ptx"]
    lines = [ln.strip() for ln in ptx.splitlines()]
    mma = [ln for ln in lines if ln.startswith("mma.")]
    fma = [ln for ln in lines if ln.startswith("fma.")]
    print(f"mma (tensor core) instructions: {len(mma)}   fma (CUDA core) instructions: {len(fma)}")
    if mma:
        print("first mma:", mma[0][:110])
    print("num_warps =", compiled.metadata.num_warps)


if __name__ == "__main__":
    check_correctness()
    benchmark_square()
    benchmark_skinny()
    show_ptx()
