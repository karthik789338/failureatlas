#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


MODELS = {
    "Qwen2.5-7B": {
        "judged": "outputs/judged/qwen_7b_mvp_judged_repaired.jsonl",
        "atlas": "outputs/atlas/qwen_7b_mvp/clustering_metrics.json",
        "validity": "outputs/atlas/qwen_7b_mvp/validity_controls",
        "paired": "outputs/meta_controller/qwen_7b_mvp/phase5b_paired",
        "intervention_compare": "outputs/interventions/qwen_7b_mvp/comparison_targeted_vs_generic/targeted_vs_generic_summary.json",
    },
    "Mistral-7B": {
        "judged": "outputs/judged/mistral_7b_mvp_judged_by_qwen.jsonl",
        "atlas": "outputs/atlas/mistral_7b_mvp/clustering_metrics.json",
        "validity": "outputs/atlas/mistral_7b_mvp/validity_controls",
        "paired": "outputs/meta_controller/mistral_7b_mvp/phase5b_paired",
        "intervention_compare": None,
    },
    "Llama-3.1-8B": {
        "judged": "outputs/judged/llama_8b_mvp_judged_by_qwen.jsonl",
        "atlas": "outputs/atlas/llama_8b_mvp/clustering_metrics.json",
        "validity": "outputs/atlas/llama_8b_mvp/validity_controls",
        "paired": "outputs/meta_controller/llama_8b_mvp/phase5b_paired",
        "intervention_compare": None,
    },
}

OUT = Path("outputs/paper_tables")
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def safe_json(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.load(p.open())


def safe_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


failure_rows = []
atlas_rows = []
validity_rows = []
paired_rows = []
intervention_rows = []

for model_name, paths in MODELS.items():
    judged = load_jsonl(paths["judged"])
    n = len(judged)
    fail_count = sum(1 for r in judged if int(r.get("failure_binary", 0)) == 1)

    by_type = Counter(r.get("failure_type") for r in judged)
    by_family = Counter(r.get("failure_family") for r in judged)

    failure_rows.append({
        "model": model_name,
        "n": n,
        "failure_count": fail_count,
        "failure_rate": fail_count / n if n else None,
        "nonfailure_count": n - fail_count,
        "top_failure_types": json.dumps(dict(by_type.most_common(10))),
        "family_counts": json.dumps(dict(by_family.most_common())),
    })

    atlas = safe_json(paths["atlas"])
    if atlas:
        atlas_rows.append({
            "model": model_name,
            "num_clusters_excluding_noise": atlas.get("num_clusters_excluding_noise"),
            "noise_rate": atlas.get("noise_rate"),
            "silhouette": atlas.get("silhouette_excluding_noise"),
            "davies_bouldin": atlas.get("davies_bouldin_excluding_noise"),
            "nmi_failure_binary": atlas.get("nmi_failure_binary"),
            "nmi_failure_family": atlas.get("nmi_failure_family"),
            "nmi_failure_type": atlas.get("nmi_failure_type"),
            "purity_failure_binary": atlas.get("purity_failure_binary"),
            "purity_failure_family": atlas.get("purity_failure_family"),
            "purity_failure_type": atlas.get("purity_failure_type"),
            "pca_variance_sum": atlas.get("pca_explained_variance_ratio_sum"),
        })

    validity_dir = Path(paths["validity"])
    clf = safe_csv(validity_dir / "classifier_predictability.csv")
    if clf is not None:
        for _, r in clf.iterrows():
            validity_rows.append({
                "model": model_name,
                "target": r.get("target"),
                "subset": r.get("subset"),
                "n": r.get("n"),
                "num_classes": r.get("num_classes"),
                "accuracy": r.get("accuracy"),
                "balanced_accuracy": r.get("balanced_accuracy"),
                "macro_f1": r.get("macro_f1"),
                "majority_accuracy": r.get("majority_accuracy"),
                "balanced_accuracy_gain_over_majority": r.get("balanced_accuracy_gain_over_majority"),
                "macro_f1_gain_over_majority": r.get("macro_f1_gain_over_majority"),
            })

    failure_only = safe_json(validity_dir / "failure_only/failure_only_metrics.json")
    if failure_only:
        validity_rows.append({
            "model": model_name,
            "target": "failure_only_atlas",
            "subset": "failures_only",
            "n": failure_only.get("n_failures"),
            "num_classes": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "majority_accuracy": None,
            "balanced_accuracy_gain_over_majority": None,
            "macro_f1_gain_over_majority": None,
            "nmi_failure_type": failure_only.get("nmi_failure_type"),
            "purity_failure_type": failure_only.get("purity_failure_type"),
            "nmi_failure_family": failure_only.get("nmi_failure_family"),
            "purity_failure_family": failure_only.get("purity_failure_family"),
            "silhouette": failure_only.get("silhouette_excluding_noise"),
        })

    paired = safe_csv(Path(paths["paired"]) / "paired_control_results.csv")
    pair_stats = safe_json(Path(paths["paired"]) / "pair_stats.json")
    if paired is not None:
        for _, r in paired.iterrows():
            paired_rows.append({
                "model_name": model_name,
                "method": r.get("model"),
                "split": r.get("split"),
                "stage": r.get("stage"),
                "n_train": r.get("n_train"),
                "n_test": r.get("n_test"),
                "num_train_pairs": r.get("num_train_pairs"),
                "num_test_pairs": r.get("num_test_pairs"),
                "accuracy": r.get("accuracy"),
                "balanced_accuracy": r.get("balanced_accuracy"),
                "macro_f1": r.get("macro_f1"),
                "roc_auc": r.get("roc_auc"),
                "average_precision": r.get("average_precision"),
                "f1_failure": r.get("f1_failure"),
                "recall_failure": r.get("recall_failure"),
                "pair_ranking_accuracy": r.get("pair_ranking_accuracy"),
                "pair_mean_margin": r.get("pair_mean_margin"),
                "total_pairs_available": pair_stats.get("num_pairs") if pair_stats else None,
            })

    intervention = safe_json(paths["intervention_compare"])
    if intervention:
        overall = intervention.get("overall", {})
        intervention_rows.append({
            "model": model_name,
            "n_compared": overall.get("n_compared"),
            "targeted_failure_rate": overall.get("targeted_failure_rate"),
            "generic_failure_rate": overall.get("generic_failure_rate"),
            "targeted_reduction": overall.get("targeted_relative_failure_reduction"),
            "generic_reduction": overall.get("generic_relative_failure_reduction"),
            "targeted_fixed_count": overall.get("targeted_fixed_count"),
            "generic_fixed_count": overall.get("generic_fixed_count"),
            "targeted_advantage_failure_rate": overall.get("targeted_advantage_in_failure_rate"),
            "targeted_advantage_fixed_count": overall.get("targeted_advantage_in_fixed_count"),
        })

pd.DataFrame(failure_rows).to_csv(OUT / "three_model_failure_distribution.csv", index=False)
pd.DataFrame(atlas_rows).to_csv(OUT / "three_model_atlas_metrics.csv", index=False)
pd.DataFrame(validity_rows).to_csv(OUT / "three_model_validity_controls.csv", index=False)
pd.DataFrame(paired_rows).to_csv(OUT / "three_model_paired_controls.csv", index=False)
pd.DataFrame(intervention_rows).to_csv(OUT / "three_model_intervention_comparison.csv", index=False)

# Compact best paired table.
paired_df = pd.DataFrame(paired_rows)
best_rows = []

if len(paired_df):
    for model in paired_df["model_name"].unique():
        sub = paired_df[paired_df["model_name"] == model]

        for split in ["paired_random", "paired_group_source"]:
            prompt = sub[(sub["method"] == "prompt_tfidf") & (sub["split"] == split)]
            act = sub[(sub["method"] == "activation_logreg") & (sub["split"] == split)]

            if len(prompt):
                pr = prompt.sort_values("balanced_accuracy", ascending=False).iloc[0]
                best_rows.append({
                    "model": model,
                    "split": split,
                    "best_method": "prompt_tfidf",
                    "stage": pr["stage"],
                    "balanced_accuracy": pr["balanced_accuracy"],
                    "roc_auc": pr["roc_auc"],
                    "pair_ranking_accuracy": pr["pair_ranking_accuracy"],
                })

            if len(act):
                ar = act.sort_values("balanced_accuracy", ascending=False).iloc[0]
                best_rows.append({
                    "model": model,
                    "split": split,
                    "best_method": "activation_logreg",
                    "stage": ar["stage"],
                    "balanced_accuracy": ar["balanced_accuracy"],
                    "roc_auc": ar["roc_auc"],
                    "pair_ranking_accuracy": ar["pair_ranking_accuracy"],
                })

best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUT / "three_model_best_paired_controls.csv", index=False)

summary = {
    "models": list(MODELS.keys()),
    "outputs": [
        "three_model_failure_distribution.csv",
        "three_model_atlas_metrics.csv",
        "three_model_validity_controls.csv",
        "three_model_paired_controls.csv",
        "three_model_best_paired_controls.csv",
        "three_model_intervention_comparison.csv",
    ],
}

(OUT / "three_model_summary_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Saved paper tables to:", OUT)
print("\n=== Failure distribution ===")
print(pd.DataFrame(failure_rows).to_string(index=False))

print("\n=== Atlas metrics ===")
print(pd.DataFrame(atlas_rows).to_string(index=False))

print("\n=== Best paired controls ===")
print(best_df.to_string(index=False))

if intervention_rows:
    print("\n=== Intervention comparison ===")
    print(pd.DataFrame(intervention_rows).to_string(index=False))
