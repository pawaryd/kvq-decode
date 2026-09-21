"""Smoke test: spin up a Modal GPU, print its identity, measure copy bandwidth.

    .venv/bin/modal run modal_scripts/gpu_smoke.py --gpu T4
    (or: .venv/bin/python -m modal run ... from outside the repo root)

Result is saved to results/gpu_smoke_<gpu>.json. Not a kernel benchmark: it only
checks the GPU name works and gives an achievable-bandwidth number to compare
with the roofline peaks in src/bpt/roofline.py.
"""
import json
from pathlib import Path

import modal

GPU_NAMES = ["T4", "A100-80GB", "H100", "B200"]

app = modal.App("bpt-gpu-smoke")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


def _smoke(size_mb: int = 1024, iters: int = 50, warmup: int = 10) -> dict:
    import subprocess

    import torch

    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    src = torch.empty(size_mb * 2**20 // 2, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    moved = 2 * src.numel() * src.element_size()  # read + write
    return {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "size_mb": size_mb,
        "iters": iters,
        "ms_per_copy": ms,
        "copy_GBps": moved / (ms * 1e-3) / 1e9,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
    }


# One function per GPU type; only the one called gets a container.
FUNCS = {
    g: app.function(gpu=g, image=image, timeout=300, max_containers=1,
                    scaledown_window=2, name=f"smoke_{g.replace('-', '_')}")(_smoke)
    for g in GPU_NAMES
}


@app.local_entrypoint()
def main(gpu: str = "T4"):
    result = FUNCS[gpu].remote()
    print(json.dumps(result, indent=2))
    out = Path(__file__).resolve().parent.parent / "results" / f"gpu_smoke_{gpu}.json"
    out.write_text(json.dumps({"gpu": gpu, **result}, indent=2) + "\n")
    print(f"saved {out}")
