#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

OUT = Path("outputs/paper_tables")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "Qwen2.5-7B": "outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_summary.json",
    "Llama-3.1-8B": "outputs/interventions/llama_8b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_summary.json",
}

overall_rows = []
type_rows = []

for model, path in SOURCES.items():
    p = Path(path)
    if not p.exists():
        print("[WARN] Missing:", p)
        continue

    data = json.load(p.open())
    overall = data["overall"]

    overall_rows.append({
        "model": model,
        "n_compared": overall.get("n_compared"),
        "targeted_failure_rate": overall.get("targeted_failure_rate"),
        "generic_failure_rate": overall.get("generic_failure_rate"),
        "targeted_reduction": overall.get("targeted_relative_failure_reduction"),
        "generic_reduction": overall.get("generic_relative_failure_reduction"),
        "targeted_fixed_count": overall.get("targeted_fixed_count"),
        "generic_fixed_count": overall.get("generic_fixed_count"),
        "targeted_advantage_failure_rate": overall.get("targeted_advantage_in_failure_rate"),
        "targeted_advantage_fixed_count": overall.get("targeted_advantage_in_fixed_count"),
        "targeted_only_fixed": overall.get("targeted_only_fixed"),
        "generic_only_fixed": overall.get("generic_only_fixed"),
        "both_fixed": overall.get("both_fixed"),
        "both_failed": overall.get("both_failed"),
        "targeted_parse_errors": overall.get("targeted_parse_errors"),
        "generic_parse_errors": overall.get("generic_parse_errors"),
    })

    for r in data.get("by_failure_type", []):
        rr = dict(r)
        rr["model"] = model
        type_rows.append(rr)

overall_df = pd.DataFrame(overall_rows)
type_df = pd.DataFrame(type_rows)

overall_df.to_csv(OUT / "two_model_intervention_comparison.csv", index=False)
type_df.to_csv(OUT / "two_model_intervention_by_failure_type.csv", index=False)

print("Saved:", OUT / "two_model_intervention_comparison.csv")
print("Saved:", OUT / "two_model_intervention_by_failure_type.csv")

print("\n=== Two-model intervention comparison ===")
print(overall_df.to_string(index=False))

print("\n=== Two-model intervention by failure type ===")
cols = [
    "model",
    "failure_type",
    "n",
    "targeted_failure_rate",
    "generic_failure_rate",
    "targeted_advantage_failure_rate",
    "targeted_fixed_count",
    "generic_fixed_count",
]
available = [c for c in cols if c in type_df.columns]
print(type_df[available].sort_values(["model", "targeted_advantage_failure_rate"], ascending=[True, False]).to_string(index=False))
