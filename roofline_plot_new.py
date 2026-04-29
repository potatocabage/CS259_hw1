"""
roofline.py — reusable roofline model plotter
----------------------------------------------
Edit the constants in the two sections below, then run:

    python roofline.py

Requires: matplotlib, numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
# GPU SPECS  (edit for your card)
# ──────────────────────────────────────────────
PEAK_FLOPS_TFLOPS   = 13.8          # Titan V FP32 peak (TFLOP/s)
PEAK_BW_GBS         = 652.8         # Titan V memory bandwidth (GB/s)
GPU_LABEL           = "Titan V"

# ──────────────────────────────────────────────
# KERNEL CONFIGURATIONS
# Each entry:
#   label        — display name
#   color        — matplotlib color string
#   ai_achieved  — FLOPs / actual DRAM bytes  (from ncu)
#   ai_theoretical — FLOPs / minimum bytes     (algorithmic lower bound)
#
# To add a new kernel: copy one dict and fill in your numbers.
# ──────────────────────────────────────────────
KERNELS = [
    dict(
        label          = "Decode C=4096",
        color          = "#378ADD",
        ai_achieved    = 0.55,
        ai_theoretical = 0.512,
    ),
    dict(
        label          = "Decode C=65536",
        color          = "#1D9E75",
        ai_achieved    = 0.53,
        ai_theoretical = 0.524,
    ),
    dict(
        label          = "Prefill S=4096",
        color          = "#D85A30",
        ai_achieved    = 544.0,
        ai_theoretical = 512.0,
    ),
    dict(
        label          = "Prefill S=65536",
        color          = "#9F3FBD",
        ai_achieved    = 418.0,
        ai_theoretical = 8192.0,
    ),
]

# ──────────────────────────────────────────────
# PLOT SETTINGS  (rarely need changing)
# ──────────────────────────────────────────────
X_MIN   = 0.05      # FLOPs/byte  (left edge of x axis)
X_MAX   = 200_000   # FLOPs/byte  (right edge of x axis)
FIG_W   = 10        # figure width  (inches)
FIG_H   = 6         # figure height (inches)
OUT_FILE = "roofline.png"   # set to None to only show interactively

# ──────────────────────────────────────────────
# DERIVED CONSTANTS
# ──────────────────────────────────────────────
peak_flops = PEAK_FLOPS_TFLOPS * 1e12   # FLOPs/s
peak_bw    = PEAK_BW_GBS       * 1e9    # bytes/s
ridge      = peak_flops / peak_bw       # FLOPs/byte


def roofline_y(ai):
    """Performance ceiling at arithmetic intensity `ai` (FLOPs/s)."""
    return min(ai * peak_bw, peak_flops)


def make_plot():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # ── roofline curve ──────────────────────────────────────────────
    ai_range = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 1000)
    perf     = np.minimum(ai_range * peak_bw, peak_flops)
    ax.plot(ai_range, perf, color="black", linewidth=2, zorder=3,
            label=f"{GPU_LABEL} roofline")

    # ── ridge point vertical dashed line ───────────────────────────
    ax.axvline(ridge, color="black", linewidth=1,
               linestyle="--", alpha=0.4, zorder=2)
    ax.text(ridge * 1.08, peak_flops * 0.6,
        f"Ridge\n{ridge:.1f} FLOPs/B",
        fontsize=12, alpha=0.8, va="center")

    # ── memory-bound / compute-bound region labels ──────────────────
    ax.text(ridge * 1.02, peak_flops * 1.25,
        "compute-bound →", fontsize=12, alpha=0.75, ha="left", va="center")
    ax.text(ridge * 0.98, peak_flops * 1.25,
        "← memory-bound", fontsize=12, alpha=0.75, ha="right", va="center")

    # ── kernel points ───────────────────────────────────────────────
    for k in KERNELS:
        y_ach = roofline_y(k["ai_achieved"])
        y_thr = roofline_y(k["ai_achieved"])   # same y, x shifts

        # dashed connector between achieved and theoretical
        ax.plot(
            [k["ai_achieved"], k["ai_theoretical"]],
            [y_ach, y_thr],
            color=k["color"], linewidth=1, linestyle="--",
            alpha=0.55, zorder=4,
        )

        # achieved — filled circle
        ax.scatter(
            k["ai_achieved"], y_ach,
            color=k["color"], s=90, zorder=5,
            marker="o", label=f"{k['label']}  (achieved)",
        )

        # theoretical — open square
        ax.scatter(
            k["ai_theoretical"], y_thr,
            facecolors="none", edgecolors=k["color"],
            s=90, zorder=5, linewidths=1.8,
            marker="s", label=f"{k['label']}  (theoretical)",
        )

    # ── axes formatting ─────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(1e8, peak_flops * 5)

    ax.set_xlabel("Arithmetic intensity  (FLOPs / byte)", fontsize=14)
    ax.set_ylabel("Performance  (FLOPs / s)", fontsize=14)
    ax.set_title(f"Roofline model — {GPU_LABEL}", fontsize=16)

    # human-readable tick labels
    def fmt_flops(val, _):
        for unit, div in [("T", 1e12), ("G", 1e9), ("M", 1e6)]:
            if val >= div:
                return f"{val/div:.4g} {unit}FLOP/s"
        return f"{val:.4g}"

    def fmt_ai(val, _):
        if val >= 1000:
            return f"{val:.0f}"
        if val >= 1:
            return f"{val:.4g}"
        return f"{val:.3g}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_flops))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_ai))
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # ── legend ───────────────────────────────────────────────────────
    ax.legend(fontsize=12, loc="lower right", framealpha=0.85,
              ncol=2, handlelength=1.5)

    plt.tight_layout()

    if OUT_FILE:
        fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved → {OUT_FILE}")

    plt.show()


if __name__ == "__main__":
    make_plot()