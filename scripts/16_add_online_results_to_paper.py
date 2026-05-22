#!/usr/bin/env python3
from pathlib import Path
import shutil
import pandas as pd
import json

SRC = Path("outputs/online_early_warning/qwen_7b_mvp/results")
FIG_SRC = SRC / "figures"
TABLE_OUT = Path("outputs/paper_tables")
FIG_OUT = Path("outputs/paper_figures")

TABLE_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC / "online_early_warning_results.csv")

cols = [
    "checkpoint_tokens",
    "split",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "average_precision",
    "f1_failure",
    "recall_failure",
    "pair_ranking_accuracy",
    "pair_mean_margin",
    "num_test_pairs",
]

df[cols].to_csv(TABLE_OUT / "qwen_online_early_warning_results.csv", index=False)

summary = json.load(open(SRC / "online_early_warning_summary.json"))
Path(TABLE_OUT / "qwen_online_early_warning_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

figures = [
    "online_balanced_accuracy_by_checkpoint.png",
    "online_roc_auc_by_checkpoint.png",
    "online_pair_ranking_by_checkpoint.png",
]

for f in figures:
    src = FIG_SRC / f
    if src.exists():
        dst = FIG_OUT / f"fig10_{f}"
        shutil.copy2(src, dst)
        print("Copied:", dst)

manifest_path = FIG_OUT / "paper_figures_manifest.csv"

if manifest_path.exists():
    manifest = pd.read_csv(manifest_path)
else:
    manifest = pd.DataFrame(columns=["figure_file", "caption"])

new_rows = [
    {
        "figure_file": str(FIG_OUT / "fig10_online_balanced_accuracy_by_checkpoint.png"),
        "caption": "Online early-warning balanced accuracy across generated-token checkpoints for Qwen matched pairs.",
    },
    {
        "figure_file": str(FIG_OUT / "fig10_online_roc_auc_by_checkpoint.png"),
        "caption": "Online early-warning ROC-AUC across generated-token checkpoints for Qwen matched pairs.",
    },
    {
        "figure_file": str(FIG_OUT / "fig10_online_pair_ranking_by_checkpoint.png"),
        "caption": "Online matched-pair ranking accuracy across generated-token checkpoints for Qwen.",
    },
]

manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
manifest = manifest.drop_duplicates(subset=["figure_file"], keep="last")
manifest.to_csv(manifest_path, index=False)

print("Saved:", TABLE_OUT / "qwen_online_early_warning_results.csv")
print("Saved:", TABLE_OUT / "qwen_online_early_warning_summary.json")
print("Updated:", manifest_path)

print("\n=== Online early-warning paper table ===")
print(df[cols].to_string(index=False))
