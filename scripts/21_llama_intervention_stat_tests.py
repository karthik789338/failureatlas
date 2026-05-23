#!/usr/bin/env python3
import math
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/paper_tables")
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "Qwen2.5-7B": "outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_row_comparison.csv",
    "Llama-3.1-8B": "outputs/interventions/llama_8b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_row_comparison.csv",
}

def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(values[idx].mean())
    return float(values.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))

def exact_mcnemar_pvalue(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * prob)

rows = []

for model, path in PATHS.items():
    p = Path(path)
    if not p.exists():
        print("[WARN] Missing:", p)
        continue

    df = pd.read_csv(p)

    df["intervention_failure_binary"] = pd.to_numeric(df["intervention_failure_binary"], errors="coerce")
    df["generic_failure_binary"] = pd.to_numeric(df["generic_failure_binary"], errors="coerce")

    for group_name, g in [("overall", df)] + [(ft, x) for ft, x in df.groupby("failure_type")]:
        targeted_fixed = (g["intervention_failure_binary"] == 0).astype(int).to_numpy()
        generic_fixed = (g["generic_failure_binary"] == 0).astype(int).to_numpy()

        adv = targeted_fixed - generic_fixed
        point, lo, hi = bootstrap_ci(adv, n_boot=10000 if group_name == "overall" else 5000)

        targeted_only = int(((targeted_fixed == 1) & (generic_fixed == 0)).sum())
        generic_only = int(((targeted_fixed == 0) & (generic_fixed == 1)).sum())

        rows.append({
            "model": model,
            "comparison_group": group_name,
            "n": int(len(g)),
            "targeted_fixed_count": int(targeted_fixed.sum()),
            "generic_fixed_count": int(generic_fixed.sum()),
            "targeted_fixed_rate": float(targeted_fixed.mean()),
            "generic_fixed_rate": float(generic_fixed.mean()),
            "targeted_advantage_fixed_rate": point,
            "targeted_advantage_ci95_low": lo,
            "targeted_advantage_ci95_high": hi,
            "targeted_only_fixed": targeted_only,
            "generic_only_fixed": generic_only,
            "mcnemar_exact_pvalue": exact_mcnemar_pvalue(targeted_only, generic_only),
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT / "two_model_targeted_vs_generic_stat_tests.csv", index=False)

print("Saved:", OUT / "two_model_targeted_vs_generic_stat_tests.csv")
print(df_out.to_string(index=False))
