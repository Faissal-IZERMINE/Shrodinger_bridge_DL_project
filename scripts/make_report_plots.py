"""Generate the publication-quality figures from the values in REPORT.pdf.

Produces:
  figures/w2_2d_tasks.png   - Bar chart of W2 across 5 2D tasks for the
                              best Lambda, OSC, and Combined configs vs Base.
  figures/w2_lambda_sweep.png - Line plot showing monotonic W2 improvement
                                as lambda snap ratio increases.
  figures/fid_mnist_stages.png - FID + MSD on constrained MNIST across
                                  pretrain/finetune stages.

All numbers are transcribed from REPORT.pdf Tables 1 and 2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Data (verbatim from REPORT.pdf Tables 1-2; 20 ODE steps, 1000 samples,
# averaged over 5 runs).
# ---------------------------------------------------------------------

TASKS = [
    "Blobs->Moons", "Circles->Blobs", "Swiss Roll->Moons",
    "S-Curve->Spiral", "Moons->Circle",
]

# W2 (lower is better).
W2 = {
    "Base":              [0.434, 0.529, 0.310, 0.294, 0.378],
    "Lambda r=0.25":     [0.282, 0.436, 0.225, 0.265, 0.274],
    "OSC r=0.75":        [0.415, 0.693, 0.295, 0.320, 0.355],
    "Combined (best)":   [0.285, 0.408, 0.234, 0.265, 0.277],
}

# Lambda-only sweep on W2 (Blobs->Moons used as illustrative task).
LAMBDA_SWEEP = {
    "Pretrain (no finetune)": ("0", 0.836),
    "Base finetune":          ("Base", 0.434),
    "lambda = 0.05":          (r"$\lambda=0.05$", 0.367),
    "lambda = 0.15":          (r"$\lambda=0.15$", 0.326),
    "lambda = 0.25":          (r"$\lambda=0.25$", 0.282),
}

# MNIST -> EMNIST (constrained, ~5.1M params).
MNIST_STAGES = ["Pretrained", "Finetune epoch 30", "Finetune epoch 60"]
MNIST_FID = [196.39, 170.98, 167.31]
MNIST_MSD = [0.2496, 0.1297, 0.1469]


def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":   "serif",
        "font.size":     10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })


def plot_w2_bars(out: Path) -> None:
    _apply_paper_style()
    methods = list(W2.keys())
    colors  = ["#888888", "#1f77b4", "#2ca02c", "#d62728"]
    x = np.arange(len(TASKS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=300)
    for i, (m, c) in enumerate(zip(methods, colors)):
        ax.bar(x + (i - 1.5) * width, W2[m], width, label=m, color=c,
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel(r"$W_2$ distance $(\downarrow)$")
    ax.set_title(r"Curriculum-Enhanced $\alpha$-DSBM vs baseline across 2D tasks")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_lambda_sweep(out: Path) -> None:
    _apply_paper_style()
    labels = [v[0] for v in LAMBDA_SWEEP.values()]
    values = [v[1] for v in LAMBDA_SWEEP.values()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)
    ax.plot(x, values, marker="o", linewidth=1.5, color="#1f77b4")
    for xi, vi in zip(x, values):
        ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$W_2$ on Blobs $\to$ Moons $(\downarrow)$")
    ax.set_title(r"Leap scheduler: larger $\lambda$ snap ratio $\Rightarrow$ lower $W_2$")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_mnist_stages(out: Path) -> None:
    _apply_paper_style()
    x = np.arange(len(MNIST_STAGES))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=300)

    ax1.plot(x, MNIST_FID, marker="o", color="#d62728", linewidth=1.8)
    for xi, v in zip(x, MNIST_FID):
        ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8.5)
    ax1.set_xticks(x); ax1.set_xticklabels(MNIST_STAGES, rotation=12, ha="right")
    ax1.set_ylabel(r"FID $(\downarrow)$")
    ax1.set_title("FID vs finetuning stage")
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    ax2.plot(x, MNIST_MSD, marker="o", color="#1f77b4", linewidth=1.8)
    for xi, v in zip(x, MNIST_MSD):
        ax2.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8.5)
    ax2.set_xticks(x); ax2.set_xticklabels(MNIST_STAGES, rotation=12, ha="right")
    ax2.set_ylabel(r"MSD $(\downarrow)$")
    ax2.set_title("MSD vs finetuning stage")
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Constrained MNIST translation (~5.1M params)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_w2_bars(out_dir / "w2_2d_tasks.png")
    plot_lambda_sweep(out_dir / "w2_lambda_sweep.png")
    plot_mnist_stages(out_dir / "fid_mnist_stages.png")


if __name__ == "__main__":
    main()
