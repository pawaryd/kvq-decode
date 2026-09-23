"""Plot achieved bandwidth vs KV splits for each batch size, per GPU and context length.

Reads results/triton_fp16_splitkv_{A100,T4}.json (from bench/bench_triton_fp16.py) and
writes docs/img/splitkv_scaling.png. Needs matplotlib (`pip install -e '.[plot]'`).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("A100-80GB", "triton_fp16_splitkv_A100.json"),
        ("Tesla T4", "triton_fp16_splitkv_T4.json")]
CTXS = [4096, 16384, 65536]
SPLITS = [1, 2, 4, 8, 16, 32, 64]
BATCH_STYLE = {1: ("#2a78d6", "o"), 8: ("#eb6834", "s"), 32: ("#1baf7a", "^")}  # palette slots 1-3

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def series(rows, B, ctx):
    by = {r["num_splits"]: r["pct_peak"] for r in rows if r["B"] == B and r["ctx"] == ctx}
    return [by[n] for n in SPLITS] if all(n in by for n in SPLITS) else None


def spread(ys, gap):
    """Nudge label y-positions apart so direct labels don't overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def main():
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.14,
                                          "top": 0.85, "bottom": 0.125, "left": 0.09, "right": 0.97})
    x = list(range(len(SPLITS)))
    for r, (gpu, fname) in enumerate(RUNS):
        d = json.load(open(ROOT / "results" / fname))
        ymax = 100 if r == 0 else 20  # T4 tops out ~15%: own scale so lines stay readable
        for c, ctx in enumerate(CTXS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-0.3, len(SPLITS) - 1 + 1.0)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0)
            ax.set_xticks(x, [str(n) for n in SPLITS])
            labels = []
            for B, (color, marker) in BATCH_STYLE.items():
                ys = series(d["rows"], B, ctx)
                if ys is None:
                    continue
                ax.plot(x, ys, color=color, lw=2, marker=marker, ms=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                labels.append((B, color, ys[-1]))
            raw = [y for _, _, y in labels]
            pos = spread(raw, ymax * 0.075)
            # Direct-label only when lines are separated; converged lines would need big
            # nudges that detach labels from their lines, so the legend/markers carry those.
            if max(abs(p - y) for p, y in zip(pos, raw)) > ymax * 0.02:
                labels = []
            for (B, color, _), py in zip(labels, pos):
                ax.text(len(SPLITS) - 1 + 0.22, py, f"B={B}", color=INK, va="center",
                        fontsize=9, fontweight="bold")
            ax.set_title(f"context {ctx // 1024}k", loc="left", fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f"{gpu}\n% of peak ({d['peak_GBps']:.0f} GB/s)", fontsize=10)
            if r == 1:
                ax.set_xlabel("KV splits  (1 = single-pass kernel)", fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=7, markeredgecolor=SURFACE,
                          label=f"batch {B}") for B, (c, m) in BATCH_STYLE.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.98, 0.985), labelcolor=INK)
    fig.text(0.02, 0.975, "Triton FP16 paged decode: bandwidth vs KV splits and batch size",
             fontsize=13, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.02, 0.938, "Llama-3-8B shape (32 q / 8 kv heads, D=128), one layer, page size 32. "
             "Note the different y-scales per GPU row.", fontsize=9, color=INK2, ha="left", va="top")
    fig.text(0.02, 0.012,
             "Single run per point. Batch 32 at 64k was not run (that config would not fit a T4). "
             "Converging lines are identified by the legend and marker shape.\n"
             "A100 batch-1 4k may be partly served from L2 (not flushed).",
             fontsize=8, color=INK2, ha="left", va="bottom")
    out = ROOT / "docs" / "img" / "splitkv_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
