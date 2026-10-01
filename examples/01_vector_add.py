"""Triton example 1: vector add (out = x + y).

The smallest useful kernel. It introduces every idea the paged-decode kernel later uses:
  1. a *program* (a block of work) identified by tl.program_id
  2. computing the memory offsets that program owns
  3. masking, so the last block doesn't read/write out of bounds
  4. tl.load / tl.store on those offsets
  5. launching a grid of programs from Python

Run on a GPU (needs an NVIDIA GPU + Linux; from this repo use Modal):
    cd /tmp && <repo>/.venv/bin/python -m modal run <repo>/modal_scripts/run_remote.py \
        --gpu T4 --cmd "python examples/01_vector_add.py"
"""
# pyright: reportArgumentType=false
# (Triton annotates BLOCK_SIZE as `tl.constexpr`; passing a plain int is the normal idiom.)
from typing import Any, cast

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# STAGE 1: the kernel
# --------------------------------------------------------------------------------------
# @triton.jit compiles this Python function to GPU code the first time it is called
# (once per distinct set of constexpr values / dtypes). Inside the function you are NOT
# writing normal Python: only tl.* operations and simple control flow are allowed.
#
# Mental model: Triton runs MANY copies of this function in parallel. Each copy is a
# "program" (roughly a CUDA thread block). Each program handles one BLOCK_SIZE-long
# chunk of the vectors. Inside a program, operations act on whole *blocks* of values at
# once (e.g. `x + y` below adds BLOCK_SIZE numbers), not one element at a time. Triton
# decides how to spread that block over threads for you.
@triton.jit
def add_kernel(
    x_ptr,               # pointer to the first element of x in GPU memory
    y_ptr,               # pointer to y
    out_ptr,             # pointer to the output
    n_elements,          # total length: a runtime value, so one compiled kernel serves many n
                         # (Triton does specialize on n==1 and n%16==0, hence a few variants)
    BLOCK_SIZE: tl.constexpr,  # elements per program. constexpr = known at compile time,
                               # so the compiler can size registers/vector loads for it.
):
    # Which chunk am I? With a 1D grid, program_id(axis=0) is 0, 1, 2, ... num_programs-1.
    pid = tl.program_id(axis=0)

    # The first element this program owns.
    block_start = pid * BLOCK_SIZE

    # A *block* of indices: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1].
    # tl.arange(0, BLOCK_SIZE) makes [0, 1, ..., BLOCK_SIZE-1] (BLOCK_SIZE must be a
    # power of two); adding the scalar block_start shifts all of them.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # If n_elements isn't a multiple of BLOCK_SIZE, the last program's offsets run past
    # the end of the array. The mask marks which lanes are real (True) vs padding (False).
    mask = offsets < n_elements

    # Load: read x_ptr[offsets] for all lanes at once. Where mask is False nothing is
    # read (touching that memory could fault); those lanes just hold garbage/`other`.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # The actual math: elementwise add over the whole block.
    out = x + y

    # Store: write the results back. The mask again skips the out-of-range lanes.
    tl.store(out_ptr + offsets, out, mask=mask)


# --------------------------------------------------------------------------------------
# STAGE 2: the Python wrapper that launches the kernel
# --------------------------------------------------------------------------------------
def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)  # allocate the output; the kernel fills it in
    n = out.numel()

    # The grid = how many programs to launch (a tuple: 1D here). We need enough programs
    # that programs * BLOCK_SIZE >= n, hence the ceiling division. Example: n=1000,
    # BLOCK_SIZE=256 -> 4 programs; the last one has 24 real lanes and 232 masked ones.
    grid = (triton.cdiv(n, block_size),)

    # `kernel[grid](args...)` is the launch syntax. Tensors are passed as pointers.
    # Returns a handle to the compiled kernel, which we use in stage 5 to inspect PTX.
    # (Typed `Any`: Triton's stubs say the launch returns None, but it returns the kernel handle.)
    compiled: Any = add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out, compiled


# --------------------------------------------------------------------------------------
# STAGE 3: correctness (project rule: every kernel is checked against a reference)
# --------------------------------------------------------------------------------------
def check_correctness():
    torch.manual_seed(0)
    print("== correctness vs torch (x + y) ==")
    # Sizes that are NOT multiples of the block size exercise the mask; 1 is the extreme.
    for n in (1, 1000, 1024, 98432):
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out, _ = add(x, y, block_size=1024)
        err = (out - (x + y)).abs().max().item()
        print(f"n={n:>6}  max abs error = {err:.1e}")
        assert err == 0.0  # a single fp32 add is exact, so we expect bit-identical output


def demo_mask_matters():
    """What happens without the mask? Show that the 'tail' lanes really exist."""
    print("\n== how many lanes are padding? ==")
    n, bs = 1000, 256
    programs = triton.cdiv(n, bs)
    print(f"n={n}, BLOCK_SIZE={bs} -> {programs} programs, {programs * bs} lanes, "
          f"{programs * bs - n} of them masked off in the last program")


# --------------------------------------------------------------------------------------
# STAGE 4: speed. Decode attention is memory-bound, and so is this kernel:
# it does 1 add per 12 bytes moved, so the only question is how close we get to the
# GPU's memory bandwidth.
# --------------------------------------------------------------------------------------
def benchmark():
    print("\n== bandwidth ==")
    n = 2**26  # 64M floats = 256 MB per array, far bigger than any GPU cache
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    # Bytes that MUST move: read x (4n) + read y (4n) + write out (4n).
    nbytes = 3 * n * 4

    def report(name, fn):
        # do_bench warms up, runs repeatedly with CUDA events, and returns the median in ms.
        ms = cast(float, triton.testing.do_bench(fn))  # returns a float (median ms) by default
        print(f"{name:<22} {ms:7.3f} ms   {nbytes / (ms * 1e-3) / 1e9:7.1f} GB/s")

    report("torch  x + y", lambda: x + y)
    # Try several block sizes: too small = too many programs / little work each;
    # too large = few programs, more registers per program.
    for bs in (128, 256, 1024, 4096, 16384):
        report(f"triton BLOCK_SIZE={bs}", lambda bs=bs: add(x, y, block_size=bs))
    print(f"(GPU: {torch.cuda.get_device_name(0)}; compare with its peak bandwidth"
          " -- T4 is 320 GB/s, ~234 GB/s measured for a plain copy)")


# --------------------------------------------------------------------------------------
# STAGE 5: peek at what the compiler produced
# --------------------------------------------------------------------------------------
# Triton lowers your function through several IRs: TTIR (Triton IR) -> TTGIR (adds the
# GPU layout: how the block is spread over threads) -> LLIR (LLVM) -> PTX (NVIDIA's
# virtual assembly) -> cubin (machine code). The handle returned by the launch exposes
# them in `.asm`. PTX is where you can see the real loads/stores (`ld.global`,
# `st.global`) and whether they are vectorized (`.v4` = 16 bytes per instruction).
def show_ptx():
    print("\n== compiler output ==")
    x = torch.randn(4096, device="cuda")
    _, compiled = add(x, x, block_size=1024)
    print("available stages:", sorted(compiled.asm.keys()))
    ptx = compiled.asm["ptx"]
    mem_ops = [ln.strip() for ln in ptx.splitlines()
               if "ld.global" in ln or "st.global" in ln]
    print(f"{len(mem_ops)} global load/store instructions in the PTX; the first few:")
    for ln in mem_ops[:6]:
        print("   ", ln)
    print("num_warps =", compiled.metadata.num_warps,
          "(a warp = 32 threads; the block is spread over num_warps*32 threads)")


if __name__ == "__main__":
    check_correctness()
    demo_mask_matters()
    benchmark()
    show_ptx()
