"""Run a command (pytest by default) from this repo on a Modal GPU.

    cd /tmp && <repo>/.venv/bin/python -m modal run <repo>/modal_scripts/run_remote.py \
        --gpu T4 --cmd "python -m pytest -q -s tests/test_triton_fp16.py"
    ... --cmd "python bench/bench_triton_fp16.py" --save triton_fp16_T4

`--save NAME` writes the command's last stdout line (expected to be JSON) to
results/NAME.json.
"""
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU_NAMES = ["T4", "A100-80GB", "H100", "B200"]

app = modal.App("bpt-remote")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")  # triton compiles a small C launcher stub
    .pip_install("torch", "numpy", "pytest")
    .env({"PYTHONPATH": "/repo/src"})
    .add_local_dir(ROOT / "src", "/repo/src")
    .add_local_dir(ROOT / "tests", "/repo/tests")
    .add_local_dir(ROOT / "bench", "/repo/bench")
)


def _run(cmd: str) -> dict:
    r = subprocess.run(cmd, shell=True, cwd="/repo", capture_output=True, text=True)
    return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}


FUNCS = {
    g: app.function(gpu=g, image=image, timeout=900, max_containers=1,
                    scaledown_window=2, name=f"run_{g.replace('-', '_')}")(_run)
    for g in GPU_NAMES
}


@app.local_entrypoint()
def main(gpu: str = "T4", cmd: str = "python -m pytest -q -s tests/", save: str = ""):
    r = FUNCS[gpu].remote(cmd)
    print(r["stdout"])
    if r["stderr"].strip():
        print("--- stderr ---\n" + r["stderr"])
    print(f"exit code: {r['returncode']}")
    if save and r["returncode"] == 0:
        out = ROOT / "results" / f"{save}.json"
        out.write_text(r["stdout"].strip().splitlines()[-1] + "\n")
        print(f"saved {out}")
