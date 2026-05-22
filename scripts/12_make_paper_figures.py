#!/usr/bin/env python3
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


OUT = Path("outputs/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

TABLES = Path("outputs/paper_tables")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def autolabel_bars(ax, bars, fmt="{:.2f}", dy=0.01, fontsize=9):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def fig1_failure_rate():
    df = pd.read_csv(TABLES / "three_model_failure_distribution.csv")

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()

    bars = ax.bar(df["model"], df["failure_rate"])
    autolabel_bars(ax, bars, fmt="{:.1%}", dy=0.005)

    ax.set_ylabel("Judged failure rate")
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(df["failure_rate"]) * 1.25)
    ax.set_title("Failure rate across instruction-tuned LLMs")
    ax.grid(axis="y", alpha=0.25)

    savefig(OUT / "fig1_failure_rate_by_model.png")


def fig2_atlas_metrics():
    df = pd.read_csv(TABLES / "three_model_atlas_metrics.csv")

    metrics = [
        ("silhouette", "Silhouette"),
        ("nmi_failure_family", "NMI: family"),
        ("nmi_failure_type", "NMI: type"),
        ("nmi_failure_binary", "NMI: binary"),
    ]

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()

    for i, (col, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[col], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Atlas structure across models")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    savefig(OUT / "fig2_atlas_structure_metrics.png")


def fig3_failure_only_structure():
    df = pd.read_csv(TABLES / "three_model_validity_controls.csv")
    fo = df[df["target"] == "failure_only_atlas"].copy()

    if fo.empty:
        print("[WARN] No failure_only_atlas rows found.")
        return

    models = fo["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()

    bars1 = ax.bar(x - width, fo["nmi_failure_type"], width, label="NMI failure type")
    bars2 = ax.bar(x, fo["purity_failure_type"], width, label="Purity failure type")
    bars3 = ax.bar(x + width, fo["silhouette"], width, label="Silhouette")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Failure-only sub-atlas structure")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    savefig(OUT / "fig3_failure_only_subatlas.png")


def fig4_paired_controls_balanced_accuracy():
    df = pd.read_csv(TABLES / "three_model_best_paired_controls.csv")

    # Keep two separate panels as separate images to avoid clutter.
    for split, title_suffix, fname in [
        ("paired_random", "random matched pairs", "fig4a_paired_random_balanced_accuracy.png"),
        ("paired_group_source", "group-source matched pairs", "fig4b_paired_group_source_balanced_accuracy.png"),
    ]:
        s = df[df["split"] == split].copy()
        if s.empty:
            continue

        pivot = s.pivot(index="model", columns="best_method", values="balanced_accuracy")
        pivot = pivot[["prompt_tfidf", "activation_logreg"]]

        models = pivot.index.tolist()
        x = np.arange(len(models))
        width = 0.32

        plt.figure(figsize=(9, 5.5))
        ax = plt.gca()

        b1 = ax.bar(x - width / 2, pivot["prompt_tfidf"], width, label="Prompt TF-IDF")
        b2 = ax.bar(x + width / 2, pivot["activation_logreg"], width, label="Activation")

        autolabel_bars(ax, b1, fmt="{:.2f}", dy=0.01, fontsize=8)
        autolabel_bars(ax, b2, fmt="{:.2f}", dy=0.01, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.45, max(0.75, float(np.nanmax(pivot.values)) + 0.08))
        ax.set_title(f"Prompt-only vs activation classifier on {title_suffix}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        savefig(OUT / fname)


def fig5_pair_ranking_accuracy():
    df = pd.read_csv(TABLES / "three_model_best_paired_controls.csv")

    for split, title_suffix, fname in [
        ("paired_random", "random matched pairs", "fig5a_pair_ranking_random.png"),
        ("paired_group_source", "group-source matched pairs", "fig5b_pair_ranking_group_source.png"),
    ]:
        s = df[df["split"] == split].copy()
        if s.empty:
            continue

        pivot = s.pivot(index="model", columns="best_method", values="pair_ranking_accuracy")
        pivot = pivot[["prompt_tfidf", "activation_logreg"]]

        models = pivot.index.tolist()
        x = np.arange(len(models))
        width = 0.32

        plt.figure(figsize=(9, 5.5))
        ax = plt.gca()

        b1 = ax.bar(x - width / 2, pivot["prompt_tfidf"], width, label="Prompt TF-IDF")
        b2 = ax.bar(x + width / 2, pivot["activation_logreg"], width, label="Activation")

        ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.8)

        autolabel_bars(ax, b1, fmt="{:.2f}", dy=0.01, fontsize=8)
        autolabel_bars(ax, b2, fmt="{:.2f}", dy=0.01, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Pair-ranking accuracy")
        ax.set_ylim(0.40, max(0.80, float(np.nanmax(pivot.values)) + 0.08))
        ax.set_title(f"Failure/non-failure pair ranking on {title_suffix}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        savefig(OUT / fname)


def fig6_intervention_overall():
    path = TABLES / "three_model_intervention_comparison.csv"
    if not path.exists():
        print("[WARN] No intervention comparison table found.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("[WARN] Intervention comparison table is empty.")
        return

    # Currently Qwen only.
    row = df.iloc[0]

    labels = ["Targeted", "Generic"]
    failure_rates = [row["targeted_failure_rate"], row["generic_failure_rate"]]
    reductions = [row["targeted_reduction"], row["generic_reduction"]]

    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    bars = ax.bar(labels, failure_rates)
    autolabel_bars(ax, bars, fmt="{:.1%}", dy=0.01)

    ax.set_ylabel("Remaining failure rate")
    ax.set_ylim(0, max(failure_rates) * 1.35)
    ax.set_title("Targeted vs generic intervention, Qwen")
    ax.grid(axis="y", alpha=0.25)

    note = (
        f"Targeted fixed {int(row['targeted_fixed_count'])}, "
        f"generic fixed {int(row['generic_fixed_count'])}; "
        f"advantage {row['targeted_advantage_failure_rate']:.1%}"
    )
    ax.text(0.5, max(failure_rates) * 1.18, note, ha="center", fontsize=9)

    savefig(OUT / "fig6_targeted_vs_generic_overall.png")


def fig7_intervention_by_type():
    path = Path("outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_by_failure_type.csv")
    if not path.exists():
        print("[WARN] Missing targeted-vs-generic by type CSV.")
        return

    df = pd.read_csv(path)
    df = df.sort_values("targeted_advantage_failure_rate", ascending=True)

    y = np.arange(len(df))
    height = 0.36

    plt.figure(figsize=(10, 6.2))
    ax = plt.gca()

    ax.barh(y - height / 2, df["generic_failure_rate"], height, label="Generic")
    ax.barh(y + height / 2, df["targeted_failure_rate"], height, label="Targeted")

    ax.set_yticks(y)
    ax.set_yticklabels(df["failure_type"])
    ax.set_xlabel("Remaining failure rate")
    ax.set_xlim(0, max(df["generic_failure_rate"].max(), df["targeted_failure_rate"].max()) * 1.25)
    ax.set_title("Remaining failure rate by failure type, Qwen intervention")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    savefig(OUT / "fig7_intervention_by_failure_type.png")


def fig8_targeted_advantage_by_type():
    path = Path("outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_by_failure_type.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = df.sort_values("targeted_advantage_failure_rate", ascending=True)

    plt.figure(figsize=(10, 6.2))
    ax = plt.gca()

    bars = ax.barh(df["failure_type"], df["targeted_advantage_failure_rate"])
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Targeted advantage in failure-rate reduction")
    ax.set_title("Where targeted prompting beats generic prompting")
    ax.grid(axis="x", alpha=0.25)

    for b in bars:
        w = b.get_width()
        ax.text(
            w + (0.006 if w >= 0 else -0.006),
            b.get_y() + b.get_height() / 2,
            f"{w:.1%}",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=9,
        )

    savefig(OUT / "fig8_targeted_advantage_by_failure_type.png")


def fig9_umap_montage():
    # Optional montage from already-created atlas images.
    files = [
        ("Qwen", Path("outputs/atlas/qwen_7b_mvp/figures/umap_by_failure_type.png")),
        ("Mistral", Path("outputs/atlas/mistral_7b_mvp/figures/umap_by_failure_type.png")),
        ("Llama", Path("outputs/atlas/llama_8b_mvp/figures/umap_by_failure_type.png")),
    ]

    existing = [(name, p) for name, p in files if p.exists()]
    if len(existing) < 3:
        print("[WARN] Not all UMAP failure-type plots exist. Skipping montage.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, path) in zip(axes, existing):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name, fontsize=12)

    fig.suptitle("FailureAtlas UMAP views by judged failure type", fontsize=14)
    plt.tight_layout()

    out = OUT / "fig9_umap_failure_type_montage.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def write_manifest():
    rows = []

    captions = {
        "fig1_failure_rate_by_model.png": "Judged failure rates across Qwen2.5-7B, Mistral-7B, and Llama-3.1-8B on the same FailureAtlas prompt bank.",
        "fig2_atlas_structure_metrics.png": "Atlas quality and label-alignment metrics across models, showing strong failure-family structure but weaker binary-failure clustering.",
        "fig3_failure_only_subatlas.png": "Failure-only sub-atlas metrics, showing that judged failures have meaningful failure-type structure.",
        "fig4a_paired_random_balanced_accuracy.png": "Prompt-only versus activation classifier on random matched failure/non-failure pairs.",
        "fig4b_paired_group_source_balanced_accuracy.png": "Prompt-only versus activation classifier under group-source paired controls.",
        "fig5a_pair_ranking_random.png": "Pair-ranking accuracy for random matched pairs.",
        "fig5b_pair_ranking_group_source.png": "Pair-ranking accuracy for group-source matched pairs.",
        "fig6_targeted_vs_generic_overall.png": "Targeted versus generic intervention remaining failure rate on Qwen.",
        "fig7_intervention_by_failure_type.png": "Remaining failure rate by failure type for targeted and generic interventions.",
        "fig8_targeted_advantage_by_failure_type.png": "Failure-type-specific targeted prompting advantage over generic prompting.",
        "fig9_umap_failure_type_montage.png": "UMAP montage of failure-type geometry across models.",
    }

    for p in sorted(OUT.glob("*.png")):
        rows.append({
            "figure_file": str(p),
            "caption": captions.get(p.name, ""),
        })

    pd.DataFrame(rows).to_csv(OUT / "paper_figures_manifest.csv", index=False)
    print("Saved:", OUT / "paper_figures_manifest.csv")


def main():
    fig1_failure_rate()
    fig2_atlas_metrics()
    fig3_failure_only_structure()
    fig4_paired_controls_balanced_accuracy()
    fig5_pair_ranking_accuracy()
    fig6_intervention_overall()
    fig7_intervention_by_type()
    fig8_targeted_advantage_by_type()
    fig9_umap_montage()
    write_manifest()


if __name__ == "__main__":
    main()
