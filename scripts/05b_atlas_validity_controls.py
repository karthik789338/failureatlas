#!/usr/bin/env python3
"""
Phase 4B: Validity controls for FailureAtlas.

Controls:
1. Within-family clustering
2. Failure-only clustering
3. Source/family/failure leakage classifiers

Inputs:
  outputs/features/qwen_7b_mvp/features.npy
  outputs/features/qwen_7b_mvp/metadata.csv

Outputs:
  outputs/atlas/qwen_7b_mvp/validity_controls/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(feature_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    X = np.load(feature_dir / "features.npy").astype("float32")
    meta = pd.read_csv(feature_dir / "metadata.csv")

    if len(meta) != X.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != metadata rows {len(meta)}")

    meta["failure_binary"] = meta["failure_binary"].astype(int)
    meta["failure_family"] = meta["failure_family"].astype(str)
    meta["failure_type"] = meta["failure_type"].astype(str)
    meta["source_dataset"] = meta["source_dataset"].astype(str)

    return X, meta


def purity_score(y_true: np.ndarray, y_pred: np.ndarray, ignore_noise: bool = False) -> float:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred)

    if ignore_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0.0

    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.sum() == 0:
            continue
        counts = Counter(y_true[mask])
        total += counts.most_common(1)[0][1]

    return float(total / len(y_true))


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    ent = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            ent -= p * np.log2(p)

    return float(ent)


def reduce_features(
    X: np.ndarray,
    pca_components: int,
    use_umap: bool,
    umap_components: int,
    umap_neighbors: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    n_pca = min(int(pca_components), Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    Xp = pca.fit_transform(Xs).astype("float32")

    info = {
        "pca_components": int(n_pca),
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "used_umap": False,
    }

    if use_umap and HAS_UMAP and Xp.shape[0] >= 20:
        reducer = umap.UMAP(
            n_components=int(umap_components),
            n_neighbors=min(int(umap_neighbors), max(2, Xp.shape[0] - 1)),
            min_dist=0.0,
            metric="euclidean",
            random_state=random_state,
        )
        Xu = reducer.fit_transform(Xp).astype("float32")
        info["used_umap"] = True
        info["umap_components"] = int(umap_components)
        info["umap_neighbors"] = min(int(umap_neighbors), max(2, Xp.shape[0] - 1))
        return Xu, info

    return Xp, info


def cluster_features(
    Xc: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = Xc.shape[0]

    if HAS_HDBSCAN and n >= max(30, min_cluster_size * 2):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(int(min_cluster_size), max(5, n // 2)),
            min_samples=min(int(min_samples), max(2, n // 4)),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(Xc)
        info = {
            "method": "hdbscan",
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
        }

        # If HDBSCAN collapses too much, fallback.
        non_noise = labels[labels != -1]
        if len(set(non_noise.tolist())) >= 2:
            return labels, info

    k = min(8, max(2, n // 80))
    if n < 160:
        k = min(4, max(2, n // 30))
    k = max(2, min(k, n - 1))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(Xc)
    info = {"method": "kmeans_fallback", "k": int(k)}
    return labels, info


def cluster_metrics(Xc: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    labels = np.asarray(labels)
    unique = sorted(set(labels.tolist()))
    valid = labels != -1

    out: Dict[str, Any] = {
        "n": int(len(labels)),
        "num_clusters_including_noise": int(len(unique)),
        "num_clusters_excluding_noise": int(len([x for x in unique if x != -1])),
        "noise_count": int((labels == -1).sum()),
        "noise_rate": float((labels == -1).sum() / len(labels)),
    }

    if valid.sum() > 3 and len(set(labels[valid].tolist())) > 1:
        try:
            out["silhouette_excluding_noise"] = float(silhouette_score(Xc[valid], labels[valid]))
        except Exception:
            out["silhouette_excluding_noise"] = None

        try:
            out["davies_bouldin_excluding_noise"] = float(davies_bouldin_score(Xc[valid], labels[valid]))
        except Exception:
            out["davies_bouldin_excluding_noise"] = None
    else:
        out["silhouette_excluding_noise"] = None
        out["davies_bouldin_excluding_noise"] = None

    for col in cols:
        if col not in meta.columns:
            continue

        y = meta[col].astype(str).to_numpy()
        out[f"ari_{col}"] = float(adjusted_rand_score(y, labels))
        out[f"nmi_{col}"] = float(normalized_mutual_info_score(y, labels))
        out[f"purity_{col}"] = float(purity_score(y, labels))
        out[f"purity_excluding_noise_{col}"] = float(purity_score(y, labels, ignore_noise=True))

    return out


def summarize_clusters(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = meta.copy()
    df["cluster"] = labels.astype(int)

    rows = []

    for cluster, g in df.groupby("cluster"):
        n = len(g)
        fail_count = int((g["failure_binary"].astype(int) == 1).sum())
        fail_rate = fail_count / n if n else 0.0

        ft_counts = Counter(g["failure_type"].astype(str))
        fam_counts = Counter(g["failure_family"].astype(str))
        src_counts = Counter(g["source_dataset"].astype(str))

        non_none = Counter(x for x in g["failure_type"].astype(str) if x not in {"none", "nan", "None"})

        rows.append({
            "cluster": int(cluster),
            "n": int(n),
            "failure_count": fail_count,
            "failure_rate": float(fail_rate),
            "top_failure_type": ft_counts.most_common(1)[0][0] if ft_counts else "",
            "top_non_none_failure_type": non_none.most_common(1)[0][0] if non_none else "none",
            "top_failure_family": fam_counts.most_common(1)[0][0] if fam_counts else "",
            "top_source_dataset": src_counts.most_common(1)[0][0] if src_counts else "",
            "failure_type_entropy": entropy_from_counts(ft_counts),
            "failure_family_entropy": entropy_from_counts(fam_counts),
            "failure_type_counts": json.dumps(dict(ft_counts), ensure_ascii=False),
            "failure_family_counts": json.dumps(dict(fam_counts), ensure_ascii=False),
            "source_dataset_counts": json.dumps(dict(src_counts), ensure_ascii=False),
        })

    return pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)

def run_within_family_controls(X: np.ndarray, meta: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    print("[INFO] Running within-family clustering controls")

    wf_dir = out_dir / "within_family"
    ensure_dir(wf_dir)

    wf_cfg = cfg["within_family"]
    random_state = int(cfg.get("pca", {}).get("random_state", 42))

    all_rows = []

    for family in sorted(meta["failure_family"].unique()):
        idx = np.where(meta["failure_family"].astype(str).to_numpy() == str(family))[0]
        n = len(idx)

        if n < int(wf_cfg.get("min_rows", 80)):
            print(f"[WARN] Skipping family={family}, n={n}")
            continue

        X_sub = X[idx]
        m_sub = meta.iloc[idx].reset_index(drop=True)

        Xc, reduce_info = reduce_features(
            X_sub,
            pca_components=int(wf_cfg.get("pca_components", 30)),
            use_umap=bool(wf_cfg.get("use_umap", True)),
            umap_components=int(wf_cfg.get("umap_components", 10)),
            umap_neighbors=int(wf_cfg.get("umap_neighbors", 20)),
            random_state=random_state,
        )

        labels, cluster_info = cluster_features(
            Xc,
            min_cluster_size=int(wf_cfg.get("hdbscan_min_cluster_size", 20)),
            min_samples=int(wf_cfg.get("hdbscan_min_samples", 5)),
            random_state=random_state,
        )

        metrics = cluster_metrics(
            Xc,
            labels,
            m_sub,
            cols=["failure_binary", "failure_type", "source_dataset"],
        )

        fail_rate = float(m_sub["failure_binary"].astype(int).mean())

        row = {
            "family": family,
            "n": int(n),
            "family_failure_rate": fail_rate,
            **reduce_info,
            **cluster_info,
            **metrics,
        }
        all_rows.append(row)

        assignments = m_sub.copy()
        assignments["cluster"] = labels.astype(int)
        assignments.to_csv(wf_dir / f"{family}_assignments.csv", index=False)

        summary = summarize_clusters(m_sub, labels)
        summary.to_csv(wf_dir / f"{family}_cluster_summary.csv", index=False)

        print(
            f"[INFO] family={family} n={n} "
            f"clusters={metrics['num_clusters_excluding_noise']} "
            f"noise={metrics['noise_rate']:.3f} "
            f"nmi_failure_binary={metrics.get('nmi_failure_binary'):.3f} "
            f"nmi_failure_type={metrics.get('nmi_failure_type'):.3f}"
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "within_family_metrics.csv", index=False)
    return df


def run_failure_only_atlas(X: np.ndarray, meta: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    print("[INFO] Running failure-only atlas")

    fo_dir = out_dir / "failure_only"
    ensure_dir(fo_dir)

    fo_cfg = cfg["failure_only"]
    random_state = int(cfg.get("pca", {}).get("random_state", 42))

    idx = np.where(meta["failure_binary"].astype(int).to_numpy() == 1)[0]
    X_fail = X[idx]
    m_fail = meta.iloc[idx].reset_index(drop=True)

    Xc, reduce_info = reduce_features(
        X_fail,
        pca_components=int(fo_cfg.get("pca_components", 50)),
        use_umap=bool(fo_cfg.get("use_umap", True)),
        umap_components=int(fo_cfg.get("umap_components", 10)),
        umap_neighbors=int(fo_cfg.get("umap_neighbors", 30)),
        random_state=random_state,
    )

    labels, cluster_info = cluster_features(
        Xc,
        min_cluster_size=int(fo_cfg.get("hdbscan_min_cluster_size", 25)),
        min_samples=int(fo_cfg.get("hdbscan_min_samples", 5)),
        random_state=random_state,
    )

    metrics = cluster_metrics(
        Xc,
        labels,
        m_fail,
        cols=["failure_type", "failure_family", "source_dataset"],
    )

    assignments = m_fail.copy()
    assignments["cluster"] = labels.astype(int)
    assignments.to_csv(fo_dir / "failure_only_assignments.csv", index=False)

    summary = summarize_clusters(m_fail, labels)
    summary.to_csv(fo_dir / "failure_only_cluster_summary.csv", index=False)

    out = {
        "subset": "failure_only",
        "n_failures": int(len(m_fail)),
        **reduce_info,
        **cluster_info,
        **metrics,
    }

    (fo_dir / "failure_only_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[INFO] Failure-only metrics:")
    print(json.dumps(out, indent=2))

    return out


def majority_baseline_metrics(y_test: np.ndarray) -> Dict[str, float]:
    counts = Counter(y_test)
    majority = counts.most_common(1)[0][0]
    pred = np.array([majority] * len(y_test))

    return {
        "majority_accuracy": float(accuracy_score(y_test, pred)),
        "majority_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "majority_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def run_one_classifier(
    X: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    clf_cfg = cfg["classifiers"]
    random_state = int(clf_cfg.get("random_state", 42))
    test_size = float(clf_cfg.get("test_size", 0.25))
    pca_components = int(clf_cfg.get("pca_components", 100))

    if target == "failure_type_failures_only":
        subset = meta["failure_binary"].astype(int) == 1
        X_use = X[subset.to_numpy()]
        y_raw = meta.loc[subset, "failure_type"].astype(str).to_numpy()
        actual_target = "failure_type"
        subset_name = "failures_only"
    else:
        X_use = X
        y_raw = meta[target].astype(str).to_numpy()
        actual_target = target
        subset_name = "full"

    counts = Counter(y_raw)
    classes = sorted(counts.keys())

    if len(classes) < 2:
        return {
            "target": target,
            "subset": subset_name,
            "n": int(len(y_raw)),
            "num_classes": int(len(classes)),
            "skipped": True,
            "reason": "Only one class.",
        }

    # Remove classes with fewer than 2 examples for stable splitting.
    keep_mask = np.array([counts[y] >= 2 for y in y_raw])
    X_use = X_use[keep_mask]
    y_raw = y_raw[keep_mask]
    counts = Counter(y_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    stratify = y if min(Counter(y).values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_use,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    n_pca = min(pca_components, X_train.shape[0] - 1, X_train.shape[1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    baseline = majority_baseline_metrics(y_test)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))

    # Some rare classes may be absent from the test split.
    # Passing explicit labels keeps target_names aligned with the full encoder.
    all_labels = np.arange(len(le.classes_))

    report = classification_report(
        y_test,
        pred,
        labels=all_labels,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    report_path = out_dir / f"classifier_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = {
        "target": target,
        "actual_target": actual_target,
        "subset": subset_name,
        "n": int(len(y_raw)),
        "num_classes": int(len(le.classes_)),
        "classes": json.dumps(list(le.classes_), ensure_ascii=False),
        "pca_components": int(n_pca),
        "test_size": test_size,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **baseline,
        "accuracy_gain_over_majority": acc - baseline["majority_accuracy"],
        "balanced_accuracy_gain_over_majority": bal_acc - baseline["majority_balanced_accuracy"],
        "macro_f1_gain_over_majority": macro_f1 - baseline["majority_macro_f1"],
        "skipped": False,
        "report_path": str(report_path),
    }

    return result


def run_leakage_classifiers(X: np.ndarray, meta: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    print("[INFO] Running leakage/predictability classifiers")

    clf_dir = out_dir / "classifiers"
    ensure_dir(clf_dir)

    targets = cfg["classifiers"].get("targets", [
        "source_dataset",
        "failure_family",
        "failure_binary",
        "failure_type",
        "failure_type_failures_only",
    ])

    rows = []

    for target in targets:
        print(f"[INFO] Classifier target={target}")
        result = run_one_classifier(X, meta, target, cfg, clf_dir)
        rows.append(result)

        if not result.get("skipped"):
            print(
                f"[INFO] target={target} acc={result['accuracy']:.3f} "
                f"bal_acc={result['balanced_accuracy']:.3f} "
                f"macro_f1={result['macro_f1']:.3f} "
                f"majority_acc={result['majority_accuracy']:.3f}"
            )
        else:
            print(f"[WARN] target={target} skipped: {result.get('reason')}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "classifier_predictability.csv", index=False)
    return df

def make_control_summary_text(
    within_df: pd.DataFrame,
    failure_only_metrics: Dict[str, Any],
    clf_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# FailureAtlas Phase 4B Validity Controls")
    lines.append("")
    lines.append("## 1. Within-family clustering")
    lines.append("")
    if within_df.empty:
        lines.append("No within-family results were produced.")
    else:
        cols = [
            "family",
            "n",
            "family_failure_rate",
            "method",
            "num_clusters_excluding_noise",
            "noise_rate",
            "nmi_failure_binary",
            "purity_failure_binary",
            "nmi_failure_type",
            "purity_failure_type",
        ]
        lines.append(within_df[cols].to_markdown(index=False))

    lines.append("")
    lines.append("## 2. Failure-only atlas")
    lines.append("")
    keep = [
        "n_failures",
        "method",
        "num_clusters_excluding_noise",
        "noise_rate",
        "nmi_failure_type",
        "purity_failure_type",
        "nmi_failure_family",
        "purity_failure_family",
        "silhouette_excluding_noise",
    ]
    for k in keep:
        if k in failure_only_metrics:
            lines.append(f"- {k}: {failure_only_metrics[k]}")

    lines.append("")
    lines.append("## 3. Predictability / leakage classifiers")
    lines.append("")
    if clf_df.empty:
        lines.append("No classifier results were produced.")
    else:
        cols = [
            "target",
            "subset",
            "n",
            "num_classes",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "majority_accuracy",
            "accuracy_gain_over_majority",
            "balanced_accuracy_gain_over_majority",
            "macro_f1_gain_over_majority",
        ]
        available = [c for c in cols if c in clf_df.columns]
        lines.append(clf_df[available].to_markdown(index=False))

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If source_dataset or failure_family predictability is much higher than failure_binary predictability, the atlas is strongly source/family structured.")
    lines.append("- If within-family NMI for failure_binary is meaningful, the atlas captures failure structure beyond source/family leakage.")
    lines.append("- If failure-only NMI for failure_type is meaningful, failed responses have internal substructure by failure type.")
    lines.append("- If failure_binary classifier balanced accuracy is high, activation features contain useful failure signal for Phase 5.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4b_validity_controls_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    feature_dir = Path(cfg["feature_dir"])
    control_dir = Path(cfg["control_dir"])
    ensure_dir(control_dir)

    print("[INFO] Phase 4B validity controls")
    print(f"[INFO] Feature dir: {feature_dir}")
    print(f"[INFO] Control dir: {control_dir}")
    print(f"[INFO] HAS_UMAP={HAS_UMAP}, HAS_HDBSCAN={HAS_HDBSCAN}")

    X, meta = load_inputs(feature_dir)
    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded metadata: {meta.shape}")

    basic_stats = {
        "num_rows": int(len(meta)),
        "feature_dim": int(X.shape[1]),
        "failure_binary_counts": meta["failure_binary"].value_counts().to_dict(),
        "failure_family_counts": meta["failure_family"].value_counts().to_dict(),
        "failure_type_counts": meta["failure_type"].value_counts().to_dict(),
        "source_dataset_counts": meta["source_dataset"].value_counts().to_dict(),
    }
    (control_dir / "basic_stats.json").write_text(json.dumps(basic_stats, indent=2), encoding="utf-8")

    within_df = run_within_family_controls(X, meta, cfg, control_dir)
    failure_only_metrics = run_failure_only_atlas(X, meta, cfg, control_dir)
    clf_df = run_leakage_classifiers(X, meta, cfg, control_dir)

    make_control_summary_text(
        within_df=within_df,
        failure_only_metrics=failure_only_metrics,
        clf_df=clf_df,
        out_path=control_dir / "validity_controls_summary.md",
    )

    print("[DONE] Phase 4B validity controls complete.")
    print(f"Summary: {control_dir / 'validity_controls_summary.md'}")
    print(f"Within-family metrics: {control_dir / 'within_family_metrics.csv'}")
    print(f"Failure-only metrics: {control_dir / 'failure_only/failure_only_metrics.json'}")
    print(f"Classifier predictability: {control_dir / 'classifier_predictability.csv'}")

    print("\nWithin-family metrics:")
    if not within_df.empty:
        cols = [
            "family",
            "n",
            "family_failure_rate",
            "method",
            "num_clusters_excluding_noise",
            "noise_rate",
            "nmi_failure_binary",
            "purity_failure_binary",
            "nmi_failure_type",
            "purity_failure_type",
        ]
        print(within_df[cols].to_string(index=False))
    else:
        print("No within-family metrics.")

    print("\nFailure-only metrics:")
    print(json.dumps(failure_only_metrics, indent=2))

    print("\nClassifier predictability:")
    if not clf_df.empty:
        cols = [
            "target",
            "subset",
            "n",
            "num_classes",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "majority_accuracy",
            "accuracy_gain_over_majority",
            "balanced_accuracy_gain_over_majority",
        ]
        available = [c for c in cols if c in clf_df.columns]
        print(clf_df[available].to_string(index=False))
    else:
        print("No classifier metrics.")


if __name__ == "__main__":
    main()
