#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

INPUT = Path("outputs/interventions/qwen_7b_mvp/intervention_judged.jsonl")
OUT_DIR = Path("outputs/interventions/qwen_7b_mvp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = [json.loads(x) for x in INPUT.open(encoding="utf-8") if x.strip()]
df = pd.DataFrame(rows)

df["original_failure_binary"] = df["original_failure_binary"].astype(int)
df["intervention_failure_binary"] = pd.to_numeric(df["intervention_failure_binary"], errors="coerce")
df["intervention_judge_confidence"] = pd.to_numeric(df["intervention_judge_confidence"], errors="coerce")

total = len(df)
parse_errors = int(df["intervention_judge_parse_error"].fillna(False).astype(bool).sum())
missing = int(df["intervention_failure_binary"].isna().sum())

original_failures = int(df["original_failure_binary"].sum())
remaining_failures = int((df["intervention_failure_binary"] == 1).sum())
fixed = int(((df["original_failure_binary"] == 1) & (df["intervention_failure_binary"] == 0)).sum())

overall = {
    "total_rows": total,
    "parse_errors": parse_errors,
    "missing_intervention_failure_binary": missing,
    "original_failure_rate": float(original_failures / total) if total else None,
    "intervention_failure_rate": float(remaining_failures / total) if total else None,
    "absolute_failure_reduction": float((original_failures - remaining_failures) / total) if total else None,
    "relative_failure_reduction": float((original_failures - remaining_failures) / original_failures) if original_failures else None,
    "fixed_count": fixed,
    "remaining_failure_count": remaining_failures,
}

summary_rows = []

for ft, g in df.groupby("failure_type"):
    n = len(g)
    original = int(g["original_failure_binary"].sum())
    remaining = int((g["intervention_failure_binary"] == 1).sum())
    fixed_n = int(((g["original_failure_binary"] == 1) & (g["intervention_failure_binary"] == 0)).sum())

    summary_rows.append({
        "failure_type": ft,
        "n": n,
        "original_failure_rate": original / n if n else None,
        "intervention_failure_rate": remaining / n if n else None,
        "absolute_failure_reduction": (original - remaining) / n if n else None,
        "relative_failure_reduction": (original - remaining) / original if original else None,
        "fixed_count": fixed_n,
        "remaining_failure_count": remaining,
        "avg_judge_confidence": float(g["intervention_judge_confidence"].mean()),
        "top_intervention_failure_types": json.dumps(
            dict(Counter(g["intervention_failure_type"].astype(str)).most_common()),
            ensure_ascii=False,
        ),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("relative_failure_reduction", ascending=False)
summary_df.to_csv(OUT_DIR / "intervention_summary.csv", index=False)

# Cluster-level summary if cluster exists.
cluster_rows = []
if "cluster" in df.columns:
    for cluster, g in df.groupby("cluster"):
        n = len(g)
        original = int(g["original_failure_binary"].sum())
        remaining = int((g["intervention_failure_binary"] == 1).sum())
        cluster_rows.append({
            "cluster": cluster,
            "n": n,
            "original_failure_rate": original / n if n else None,
            "intervention_failure_rate": remaining / n if n else None,
            "relative_failure_reduction": (original - remaining) / original if original else None,
            "top_original_failure_type": Counter(g["failure_type"].astype(str)).most_common(1)[0][0],
            "top_intervention_failure_type": Counter(g["intervention_failure_type"].astype(str)).most_common(1)[0][0],
        })

cluster_df = pd.DataFrame(cluster_rows).sort_values("relative_failure_reduction", ascending=False)
cluster_df.to_csv(OUT_DIR / "intervention_cluster_summary.csv", index=False)

out_json = {
    "overall": overall,
    "by_failure_type": summary_df.to_dict(orient="records"),
    "by_cluster": cluster_df.to_dict(orient="records") if len(cluster_rows) else [],
}

(OUT_DIR / "intervention_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

# Safe row-level CSV, no prompt/response content.
safe_cols = [
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
    "intervention_judge_reason",
]
available = [c for c in safe_cols if c in df.columns]
df[available].to_csv(OUT_DIR / "intervention_judged_safe.csv", index=False)

print("=" * 80)
print("Phase 6B Intervention Summary")
print("=" * 80)
print(json.dumps(overall, indent=2))

print("\nBy failure type:")
print(summary_df.to_string(index=False))

print("\nBy cluster:")
print(cluster_df.head(30).to_string(index=False))

print("\nSaved:")
print(OUT_DIR / "intervention_summary.csv")
print(OUT_DIR / "intervention_cluster_summary.csv")
print(OUT_DIR / "intervention_summary.json")
print(OUT_DIR / "intervention_judged_safe.csv")
