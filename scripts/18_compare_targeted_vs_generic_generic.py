#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--targeted-judged", required=True)
    parser.add_argument("--generic-judged", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.DataFrame(load_jsonl(args.targeted_judged))
    g = pd.DataFrame(load_jsonl(args.generic_judged))

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

    target_cols = [c for c in target_cols if c in t.columns]
    generic_cols = [c for c in generic_cols if c in g.columns]

    merged = t[target_cols].merge(g[generic_cols], on="prompt_id", how="inner")

    merged["intervention_failure_binary"] = pd.to_numeric(
        merged["intervention_failure_binary"], errors="coerce"
    )
    merged["generic_failure_binary"] = pd.to_numeric(
        merged["generic_failure_binary"], errors="coerce"
    )

    n = len(merged)

    target_remaining = int((merged["intervention_failure_binary"] == 1).sum())
    generic_remaining = int((merged["generic_failure_binary"] == 1).sum())

    target_fixed = int((merged["intervention_failure_binary"] == 0).sum())
    generic_fixed = int((merged["generic_failure_binary"] == 0).sum())

    both_fixed = int(((merged["intervention_failure_binary"] == 0) & (merged["generic_failure_binary"] == 0)).sum())
    targeted_only_fixed = int(((merged["intervention_failure_binary"] == 0) & (merged["generic_failure_binary"] == 1)).sum())
    generic_only_fixed = int(((merged["intervention_failure_binary"] == 1) & (merged["generic_failure_binary"] == 0)).sum())
    both_failed = int(((merged["intervention_failure_binary"] == 1) & (merged["generic_failure_binary"] == 1)).sum())

    overall = {
        "model": args.model_label,
        "n_compared": n,
        "targeted_failure_rate": target_remaining / n if n else None,
        "generic_failure_rate": generic_remaining / n if n else None,
        "targeted_relative_failure_reduction": target_fixed / n if n else None,
        "generic_relative_failure_reduction": generic_fixed / n if n else None,
        "targeted_fixed_count": target_fixed,
        "generic_fixed_count": generic_fixed,
        "both_fixed": both_fixed,
        "targeted_only_fixed": targeted_only_fixed,
        "generic_only_fixed": generic_only_fixed,
        "both_failed": both_failed,
        "targeted_advantage_in_fixed_count": target_fixed - generic_fixed,
        "targeted_advantage_in_failure_rate": (generic_remaining - target_remaining) / n if n else None,
        "targeted_parse_errors": int(merged.get("intervention_judge_parse_error", pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()),
        "generic_parse_errors": int(merged.get("generic_judge_parse_error", pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()),
    }

    by_type = []

    for ft, x in merged.groupby("failure_type"):
        nn = len(x)
        tr = int((x["intervention_failure_binary"] == 1).sum())
        gr = int((x["generic_failure_binary"] == 1).sum())
        tf = int((x["intervention_failure_binary"] == 0).sum())
        gf = int((x["generic_failure_binary"] == 0).sum())

        by_type.append({
            "model": args.model_label,
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
            "targeted_only_fixed": int(((x["intervention_failure_binary"] == 0) & (x["generic_failure_binary"] == 1)).sum()),
            "generic_only_fixed": int(((x["intervention_failure_binary"] == 1) & (x["generic_failure_binary"] == 0)).sum()),
            "both_fixed": int(((x["intervention_failure_binary"] == 0) & (x["generic_failure_binary"] == 0)).sum()),
            "both_failed": int(((x["intervention_failure_binary"] == 1) & (x["generic_failure_binary"] == 1)).sum()),
            "top_targeted_failure_type": Counter(x["intervention_failure_type"].astype(str)).most_common(1)[0][0],
            "top_generic_failure_type": Counter(x["generic_failure_type"].astype(str)).most_common(1)[0][0],
        })

    by_type_df = pd.DataFrame(by_type).sort_values("targeted_advantage_failure_rate", ascending=False)

    merged.to_csv(out_dir / "targeted_vs_generic_row_comparison.csv", index=False)
    by_type_df.to_csv(out_dir / "targeted_vs_generic_by_failure_type.csv", index=False)

    summary = {
        "overall": overall,
        "by_failure_type": by_type_df.to_dict(orient="records"),
    }

    (out_dir / "targeted_vs_generic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"{args.model_label}: targeted vs generic intervention comparison")
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

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
