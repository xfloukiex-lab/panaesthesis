"""Trend regeneration: history.jsonl -> TRENDS.md + trends.png.

Pure rendering of the monitor's running record. No math, no judgments —
the flags live in monitor.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREND_KEYS = [
    ("D_total", "D(layers)"),
    ("z_mean_strain", "z_mean strain"),
    ("z_mean_coast", "z_mean coast"),
    ("tau_slope_strain", "tau slope strain"),
    ("tau_slope_coast", "tau slope coast"),
    ("acc_clean", "acc clean"),
    ("acc_degraded", "acc degraded"),
]


def regenerate(rows, out_dir):
    """rows: list of history row dicts in checkpoint order. Writes TRENDS.md + trends.png."""
    out_dir = Path(out_dir)
    _write_md(rows, out_dir / "TRENDS.md")
    _write_png(rows, out_dir / "trends.png")


def _write_md(rows, path):
    L = []
    L.append("# Hodos monitor — trends\n")
    L.append("One row per checkpoint, in arrival order. Values are means over the "
             "checkpoint's stimulus replicates. Section caps in each portrait mark "
             "the INPUT regimes (planted); the braid is computed, never forced.\n")
    if not rows:
        L.append("_No checkpoints yet._\n")
    else:
        L.append("| # | checkpoint | arch | D | z strain | z coast | "
                 "tau slope strain | tau slope coast | acc clean | acc degraded | verification |")
        L.append("|---|------------|------|---|----------|---------|------------------|-----------------|-----------|--------------"
                 "|--------------|")
        for i, r in enumerate(rows):
            m = r["metrics_mean"]
            ver = r["verification_any_flagged"]
            L.append(
                f"| {i} | {r['checkpoint']} | {r['arch_name']} | "
                f"{m['D_total']:.4f} | {m['z_mean_strain']:.3f} | {m['z_mean_coast']:.3f} | "
                f"{m['tau_slope_strain']:.4f} | {m['tau_slope_coast']:.4f} | "
                f"{m['acc_clean']:.3f} | {m['acc_degraded']:.3f} | "
                f"{'FLAGGED' if ver else 'passed'} |"
            )
        L.append("")
    path.write_text("\n".join(L))


def _write_png(rows, path):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor("white")
    fig.suptitle("Hodos monitor — checkpoint trends", fontsize=14, weight="bold")
    xs = list(range(len(rows)))
    labels = [r["checkpoint"] for r in rows]

    def series(key):
        return [r["metrics_mean"][key] for r in rows]

    ax = axes[0]
    ax.plot(xs, series("D_total"), marker="o")
    ax.set_ylabel("D(layers)")
    ax.set_title("Diastema: how far apart are the layers", fontsize=11)

    ax = axes[1]
    ax.plot(xs, series("z_mean_strain"), marker="o", label="strain (degraded input)")
    ax.plot(xs, series("z_mean_coast"), marker="o", label="coast (clean input)")
    ax.set_ylabel("z_mean")
    ax.set_title("Symploke: does the braid light up under strain", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(xs, series("tau_slope_strain"), marker="o", label="strain")
    ax.plot(xs, series("tau_slope_coast"), marker="o", label="coast")
    ax.set_ylabel("tau slope / step")
    ax.set_title("Chronos: lived-time slope", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(xs, series("acc_clean"), marker="o", label="clean")
    ax.plot(xs, series("acc_degraded"), marker="o", label="degraded")
    ax.set_ylabel("accuracy")
    ax.set_title("Reader accuracy (measured, not tuned)", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel("checkpoint (arrival order)")

    for a in axes:
        a.set_xticks(xs)
        a.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)
