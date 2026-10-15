"""Turn results/results.jsonl into a markdown table and charts.

    python report.py
    -> results/summary.md, results/accuracy_vs_vram.png, results/cost_bars.png,
       results/accuracy_vs_params.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import base_parser, load_config  # noqa: E402

# Fixed hue per method (identity follows the method, never the rank).
METHOD_COLORS = {"head": "#2a78d6", "full": "#eb6834", "lora": "#1baf7a", "qlora": "#eda100"}
METHOD_LABELS = {"head": "Head only", "full": "Full FT", "lora": "LoRA", "qlora": "QLoRA (4-bit)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def legend(ax, methods: list[str]) -> None:
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHOD_COLORS
        if m in methods
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=INK, loc="best")


def scatter(rows, x_key, title, xlabel, path, logx=False) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for r in rows:
        if r.get(x_key) is None:
            continue
        ax.scatter(r[x_key], 100 * r["accuracy"], s=70, color=METHOD_COLORS[r["method"]],
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(r["run"], (r[x_key], 100 * r["accuracy"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color=INK)
    if logx:
        ax.set_xscale("log")
    style(ax, title, xlabel, "Test accuracy (%)")
    legend(ax, [r["method"] for r in rows])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bars(rows, path) -> None:
    metrics = [("train_minutes", "Training time (min)"), ("checkpoint_mb", "Saved checkpoint (MB)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    names = [r["run"] for r in rows]
    for ax, (key, label) in zip(axes, metrics):
        vals = [r[key] or 0 for r in rows]
        ax.barh(names, vals, color=[METHOD_COLORS[r["method"]] for r in rows], height=0.6,
                edgecolor="white", linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:g}", va="center", fontsize=8, color=INK)
        ax.invert_yaxis()
        style(ax, label, "", "")
        ax.grid(axis="y", visible=False)
    legend(axes[1], [r["method"] for r in rows])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(rows) -> str:
    cols = [
        ("run", "Run"), ("method", "Method"), ("r", "Rank"), ("accuracy", "Acc"),
        ("macro_f1", "Macro F1"), ("trainable_pct", "Trainable %"), ("peak_vram_gb", "Peak VRAM GB"),
        ("train_minutes", "Train min"), ("train_samples_per_s", "Samples/s"),
        ("infer_ms_per_sample", "Infer ms/sample"), ("checkpoint_mb", "Ckpt MB"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r[k]) for k, _ in cols) + " |")
    return "\n".join(out)


def main() -> None:
    args = base_parser(__doc__, "configs/study.yaml").parse_args()
    cfg = load_config(args.config, args.set)
    results_file = Path(cfg["train"]["results_file"])
    rows = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
    # last result wins if a run was repeated; keep config order
    by_run = {r["run"]: r for r in rows}
    order = [r["name"] for r in cfg["runs"]]
    rows = [by_run[n] for n in order if n in by_run] + [r for n, r in by_run.items() if n not in order]
    out = results_file.parent

    scatter(rows, "peak_vram_gb", "Accuracy vs. peak training memory", "Peak VRAM (GB)",
            out / "accuracy_vs_vram.png")
    scatter(rows, "trainable_params", "Accuracy vs. trainable parameters", "Trainable parameters (log)",
            out / "accuracy_vs_params.png", logx=True)
    bars(rows, out / "cost_bars.png")

    best = max(rows, key=lambda r: r["accuracy"])
    md = [
        f"# Fine-tuning method trade-offs: {rows[0]['model']} on {cfg['data']['dataset']}",
        "",
        f"GPU: {rows[0]['gpu']} | train examples: {rows[0]['n_train']} | test examples: {cfg['data']['n_test']}",
        "",
        table(rows),
        "",
        f"Best accuracy: **{best['run']}** ({100 * best['accuracy']:.2f}%).",
        "",
        "![accuracy vs vram](accuracy_vs_vram.png)",
        "![accuracy vs params](accuracy_vs_params.png)",
        "![cost](cost_bars.png)",
    ]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:6]))
    print(f"Charts + summary written to {out}/")


if __name__ == "__main__":
    main()
