#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

MODELS = {
    "Qwen2.5-7B": "outputs/atlas/qwen_7b_mvp/paired_only_atlas/paired_only_metrics.json",
    "Mistral-7B": "outputs/atlas/mistral_7b_mvp/paired_only_atlas/paired_only_metrics.json",
    "Llama-3.1-8B": "outputs/atlas/llama_8b_mvp/paired_only_atlas/paired_only_metrics.json",
}

OUT = Path("outputs/paper_tables")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for model, path in MODELS.items():
    p = Path(path)
    if not p.exists():
        print("[WARN] Missing:", p)
        continue

    m = json.load(p.open())

    rows.append({
        "model": model,
        "num_rows": m.get("num_rows"),
        "num_pairs": m.get("num_pairs"),
        "num_clusters_excluding_noise": m.get("num_clusters_excluding_noise"),
        "noise_rate": m.get("noise_rate"),
        "silhouette": m.get("silhouette_excluding_noise"),
        "davies_bouldin": m.get("davies_bouldin_excluding_noise"),
        "nmi_failure_binary": m.get("nmi_failure_binary"),
        "purity_failure_binary": m.get("purity_failure_binary"),
        "nmi_failure_type": m.get("nmi_failure_type"),
        "purity_failure_type": m.get("purity_failure_type"),
        "nmi_source_dataset": m.get("nmi_source_dataset"),
        "purity_source_dataset": m.get("purity_source_dataset"),
        "pair_different_cluster_rate": m.get("pair_different_cluster_rate"),
        "pair_eval_count_excluding_noise": m.get("pair_eval_count_excluding_noise"),
        "method": m.get("method"),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "three_model_paired_only_atlas_metrics.csv", index=False)

print("Saved:", OUT / "three_model_paired_only_atlas_metrics.csv")
print(df.to_string(index=False))
