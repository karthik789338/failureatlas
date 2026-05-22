#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

import pandas as pd

BASE = Path("outputs/interventions/qwen_7b_mvp")
TARGETED = BASE / "intervention_judged.jsonl"
GENERIC = BASE / "generic_baseline/generic_intervention_judged.jsonl"
OUT_DIR = BASE / "comparison_targeted_vs_generic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

targeted_rows = [json.loads(x) for x in TARGETED.open(encoding="utf-8") if x.strip()]
generic_rows = [json.loads(x) for x in GENERIC.open(encoding="utf-8") if x.strip()]

t = pd.DataFrame(targeted_rows)
g = pd.DataFrame(generic_rows)

t["prompt_id"] = t["prompt_id"].astype(str)
g["prompt_id"] = g["prompt_id"].astype(str)

target_cols = [
    "prompt_id",
    "failure_type",
    "failure_family",
    "source_dataset",
    "cluster",
    "original_failure_binary",
    "intervention_failure_binary",
    "intervention_failure_type",
    "intervention_severity",
    "intervention_judge_confidence",
    "intervention_judge_parse_error",
]

generic_cols = [
    "prompt_id",
    "generic_failure_binary",
    "generic_failure_type",
    "generic_severity",
    "generic_judge_confidence",
    "generic_judge_parse_error",
]

merged = t[target_cols].merge(g[generic_cols], on="prompt_id", how="inner")

merged["original_failure_binary"] = merged["original_failure_binary"].astype(int)
merged["intervention_failure_binary"] = pd.to_numeric(merged["intervention_failure_binary"], errors="coerce")
merged["generic_failure_binary"] = pd.to_numeric(merged["generic_failure_binary"], errors="coerce")

n = len(merged)

target_remaining = int((merged["intervention_failure_binary"] == 1).sum())
generic_remaining = int((merged["generic_failure_binary"] == 1).sum())

target_fixed = int((merged["intervention_failure_binary"] == 0).sum())
generic_fixed = int((merged["generic_failure_binary"] == 0).sum())

both_fixed = int(((merged["intervention_failure_binary"] == 0) & (merged["generic_failure_binary"] == 0)).sum())
target_only_fixed = int(((merged["intervention_failure_binary"] == 0) & (merged["generic_failure_binary"] == 1)).sum())
generic_only_fixed = int(((merged["intervention_failure_binary"] == 1) & (merged["generic_failure_binary"] == 0)).sum())
both_failed = int(((merged["intervention_failure_binary"] == 1) & (merged["generic_failure_binary"] == 1)).sum())

overall = {
    "n_compared": n,
    "targeted_failure_rate": target_remaining / n if n else None,
    "generic_failure_rate": generic_remaining / n if n else None,
    "targeted_relative_failure_reduction": target_fixed / n if n else None,
    "generic_relative_failure_reduction": generic_fixed / n if n else None,
    "targeted_fixed_count": target_fixed,
    "generic_fixed_count": generic_fixed,
    "both_fixed": both_fixed,
    "targeted_only_fixed": target_only_fixed,
    "generic_only_fixed": generic_only_fixed,
    "both_failed": both_failed,
    "targeted_advantage_in_fixed_count": target_fixed - generic_fixed,
    "targeted_advantage_in_failure_rate": (generic_remaining - target_remaining) / n if n else None,
    "targeted_parse_errors": int(merged["intervention_judge_parse_error"].fillna(False).astype(bool).sum()),
    "generic_parse_errors": int(merged["generic_judge_parse_error"].fillna(False).astype(bool).sum()),
}

by_type = []

for ft, x in merged.groupby("failure_type"):
    nn = len(x)
    tr = int((x["intervention_failure_binary"] == 1).sum())
    gr = int((x["generic_failure_binary"] == 1).sum())

    tf = int((x["intervention_failure_binary"] == 0).sum())
    gf = int((x["generic_failure_binary"] == 0).sum())

    by_type.append({
        "failure_type": ft,
        "n": nn,
        "targeted_failure_rate": tr / nn,
        "generic_failure_rate": gr / nn,
        "targeted_reduction": tf / nn,
        "generic_reduction": gf / nn,
        "targeted_fixed_count": tf,
        "generic_fixed_count": gf,
        "targeted_advantage_fixed_count": tf - gf,
        "targeted_advantage_failure_rate": (gr - tr) / nn,
        "both_fixed": int(((x["intervention_failure_binary"] == 0) & (x["generic_failure_binary"] == 0)).sum()),
        "targeted_only_fixed": int(((x["intervention_failure_binary"] == 0) & (x["generic_failure_binary"] == 1)).sum()),
        "generic_only_fixed": int(((x["intervention_failure_binary"] == 1) & (x["generic_failure_binary"] == 0)).sum()),
        "both_failed": int(((x["intervention_failure_binary"] == 1) & (x["generic_failure_binary"] == 1)).sum()),
        "top_targeted_failure_type": Counter(x["intervention_failure_type"].astype(str)).most_common(1)[0][0],
        "top_generic_failure_type": Counter(x["generic_failure_type"].astype(str)).most_common(1)[0][0],
    })

by_type_df = pd.DataFrame(by_type).sort_values("targeted_advantage_failure_rate", ascending=False)

by_cluster = []

for cluster, x in merged.groupby("cluster"):
    nn = len(x)
    tr = int((x["intervention_failure_binary"] == 1).sum())
    gr = int((x["generic_failure_binary"] == 1).sum())
    tf = int((x["intervention_failure_binary"] == 0).sum())
    gf = int((x["generic_failure_binary"] == 0).sum())

    by_cluster.append({
        "cluster": cluster,
        "n": nn,
        "targeted_failure_rate": tr / nn,
        "generic_failure_rate": gr / nn,
        "targeted_reduction": tf / nn,
        "generic_reduction": gf / nn,
        "targeted_advantage_failure_rate": (gr - tr) / nn,
        "top_failure_type": Counter(x["failure_type"].astype(str)).most_common(1)[0][0],
    })

by_cluster_df = pd.DataFrame(by_cluster).sort_values("targeted_advantage_failure_rate", ascending=False)

merged.to_csv(OUT_DIR / "targeted_vs_generic_row_comparison.csv", index=False)
by_type_df.to_csv(OUT_DIR / "targeted_vs_generic_by_failure_type.csv", index=False)
by_cluster_df.to_csv(OUT_DIR / "targeted_vs_generic_by_cluster.csv", index=False)

summary = {
    "overall": overall,
    "by_failure_type": by_type_df.to_dict(orient="records"),
    "by_cluster": by_cluster_df.to_dict(orient="records"),
}

(OUT_DIR / "targeted_vs_generic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 80)
print("Targeted vs Generic Intervention Comparison")
print("=" * 80)
print(json.dumps(overall, indent=2))

print("\nBy failure type:")
cols = [
    "failure_type",
    "n",
    "targeted_failure_rate",
    "generic_failure_rate",
    "targeted_reduction",
    "generic_reduction",
    "targeted_advantage_failure_rate",
    "targeted_fixed_count",
    "generic_fixed_count",
    "targeted_only_fixed",
    "generic_only_fixed",
]
print(by_type_df[cols].to_string(index=False))

print("\nBy cluster top 20:")
cols = [
    "cluster",
    "n",
    "targeted_failure_rate",
    "generic_failure_rate",
    "targeted_advantage_failure_rate",
    "top_failure_type",
]
print(by_cluster_df[cols].head(20).to_string(index=False))

print("\nSaved outputs to:", OUT_DIR)
