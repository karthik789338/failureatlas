#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("outputs/paper_tables")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Qwen2.5-7B": "outputs/judged/qwen_7b_mvp_judged_repaired.jsonl",
    "Mistral-7B": "outputs/judged/mistral_7b_mvp_judged_by_qwen.jsonl",
    "Llama-3.1-8B": "outputs/judged/llama_8b_mvp_judged_by_qwen.jsonl",
}

TARGETED_GENERIC = Path(
    "outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/"
    "targeted_vs_generic_row_comparison.csv"
)


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def bootstrap_ci(values, stat_fn=np.mean, n_boot=10000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    if len(values) == 0:
        return None, None, None

    stats = []
    n = len(values)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(values[idx]))

    stats = np.asarray(stats)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    point = float(stat_fn(values))
    return point, lo, hi


def exact_mcnemar_pvalue(b, c):
    # Two-sided exact McNemar test using binomial tail.
    # b = targeted fixed, generic failed
    # c = generic fixed, targeted failed
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    # exact two-sided p = 2 * P[X <= min(b,c)], X ~ Bin(n, 0.5)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return min(1.0, 2.0 * prob)


def main():
    rng_seed = 42

    # ------------------------------------------------------------------
    # 1. Model failure-rate CIs
    # ------------------------------------------------------------------
    model_rows = []
    by_model = {}

    for model, path in MODELS.items():
        rows = load_jsonl(path)
        df = pd.DataFrame([
            {
                "prompt_id": str(r["prompt_id"]),
                "failure_binary": int(r.get("failure_binary", 0)),
                "failure_type": str(r.get("failure_type", "none")),
                "failure_family": str(r.get("failure_family", "")),
            }
            for r in rows
        ])

        by_model[model] = df

        vals = df["failure_binary"].to_numpy()
        point, lo, hi = bootstrap_ci(vals, n_boot=10000, seed=rng_seed)

        model_rows.append({
            "model": model,
            "n": int(len(df)),
            "failure_count": int(vals.sum()),
            "failure_rate": point,
            "failure_rate_ci95_low": lo,
            "failure_rate_ci95_high": hi,
        })

    model_ci = pd.DataFrame(model_rows)
    model_ci.to_csv(OUT / "model_failure_rate_ci.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Pairwise model failure-rate difference CIs
    #    Same prompt bank, so use prompt-level paired differences.
    # ------------------------------------------------------------------
    pairwise_rows = []

    model_names = list(MODELS.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a = model_names[i]
            b = model_names[j]

            da = by_model[a][["prompt_id", "failure_binary"]].rename(columns={"failure_binary": "fail_a"})
            db = by_model[b][["prompt_id", "failure_binary"]].rename(columns={"failure_binary": "fail_b"})

            merged = da.merge(db, on="prompt_id", how="inner")
            diffs = merged["fail_a"].to_numpy() - merged["fail_b"].to_numpy()

            point, lo, hi = bootstrap_ci(diffs, n_boot=10000, seed=rng_seed)

            pairwise_rows.append({
                "model_a": a,
                "model_b": b,
                "n_matched_prompts": int(len(merged)),
                "failure_rate_a_minus_b": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "a_failure_rate": float(merged["fail_a"].mean()),
                "b_failure_rate": float(merged["fail_b"].mean()),
            })

    pairwise_ci = pd.DataFrame(pairwise_rows)
    pairwise_ci.to_csv(OUT / "model_pairwise_failure_diff_ci.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Qwen targeted-vs-generic intervention tests
    # ------------------------------------------------------------------
    intervention_results = []

    if TARGETED_GENERIC.exists():
        df = pd.read_csv(TARGETED_GENERIC)

        df["intervention_failure_binary"] = pd.to_numeric(
            df["intervention_failure_binary"], errors="coerce"
        )
        df["generic_failure_binary"] = pd.to_numeric(
            df["generic_failure_binary"], errors="coerce"
        )

        # Fixed = failure_binary == 0
        targeted_fixed = (df["intervention_failure_binary"] == 0).astype(int).to_numpy()
        generic_fixed = (df["generic_failure_binary"] == 0).astype(int).to_numpy()

        targeted_only_fixed = int(((targeted_fixed == 1) & (generic_fixed == 0)).sum())
        generic_only_fixed = int(((targeted_fixed == 0) & (generic_fixed == 1)).sum())

        p_mcnemar = exact_mcnemar_pvalue(targeted_only_fixed, generic_only_fixed)

        advantage = targeted_fixed - generic_fixed
        point, lo, hi = bootstrap_ci(advantage, n_boot=10000, seed=rng_seed)

        intervention_results.append({
            "comparison": "Qwen targeted vs generic",
            "n": int(len(df)),
            "targeted_fixed_count": int(targeted_fixed.sum()),
            "generic_fixed_count": int(generic_fixed.sum()),
            "targeted_fixed_rate": float(targeted_fixed.mean()),
            "generic_fixed_rate": float(generic_fixed.mean()),
            "targeted_advantage_fixed_rate": point,
            "targeted_advantage_ci95_low": lo,
            "targeted_advantage_ci95_high": hi,
            "targeted_only_fixed": targeted_only_fixed,
            "generic_only_fixed": generic_only_fixed,
            "mcnemar_exact_pvalue": p_mcnemar,
        })

        # By failure type.
        for ft, g in df.groupby("failure_type"):
            tf = (g["intervention_failure_binary"] == 0).astype(int).to_numpy()
            gf = (g["generic_failure_binary"] == 0).astype(int).to_numpy()
            adv = tf - gf
            point, lo, hi = bootstrap_ci(adv, n_boot=5000, seed=rng_seed)

            intervention_results.append({
                "comparison": f"Qwen targeted vs generic: {ft}",
                "n": int(len(g)),
                "targeted_fixed_count": int(tf.sum()),
                "generic_fixed_count": int(gf.sum()),
                "targeted_fixed_rate": float(tf.mean()),
                "generic_fixed_rate": float(gf.mean()),
                "targeted_advantage_fixed_rate": point,
                "targeted_advantage_ci95_low": lo,
                "targeted_advantage_ci95_high": hi,
                "targeted_only_fixed": int(((tf == 1) & (gf == 0)).sum()),
                "generic_only_fixed": int(((tf == 0) & (gf == 1)).sum()),
                "mcnemar_exact_pvalue": exact_mcnemar_pvalue(
                    int(((tf == 1) & (gf == 0)).sum()),
                    int(((tf == 0) & (gf == 1)).sum()),
                ),
            })

    intervention_df = pd.DataFrame(intervention_results)
    intervention_df.to_csv(OUT / "qwen_targeted_vs_generic_stat_tests.csv", index=False)

    summary = {
        "model_failure_rate_ci": str(OUT / "model_failure_rate_ci.csv"),
        "model_pairwise_failure_diff_ci": str(OUT / "model_pairwise_failure_diff_ci.csv"),
        "qwen_targeted_vs_generic_stat_tests": str(OUT / "qwen_targeted_vs_generic_stat_tests.csv"),
    }

    (OUT / "statistical_tests_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Model failure-rate CI")
    print("=" * 80)
    print(model_ci.to_string(index=False))

    print("\n" + "=" * 80)
    print("Pairwise model failure-rate difference CI")
    print("=" * 80)
    print(pairwise_ci.to_string(index=False))

    if not intervention_df.empty:
        print("\n" + "=" * 80)
        print("Qwen targeted-vs-generic statistical tests")
        print("=" * 80)
        print(intervention_df.to_string(index=False))

    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
