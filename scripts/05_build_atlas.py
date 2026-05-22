#!/usr/bin/env python3
"""
Phase 4: Build FailureAtlas from activation trajectory features.

Input:
  outputs/features/qwen_7b_mvp/features.npy
  outputs/features/qwen_7b_mvp/metadata.csv

Output:
  outputs/atlas/qwen_7b_mvp/
    pca_features.npy
    umap_2d.csv
    cluster_assignments.csv
    cluster_summary.csv
    clustering_metrics.json
    clustering_comparison.csv
    figures/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
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
    X_path = feature_dir / "features.npy"
    meta_path = feature_dir / "metadata.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing {X_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    X = np.load(X_path).astype("float32")
    meta = pd.read_csv(meta_path)

    if len(meta) != X.shape[0]:
        raise ValueError(f"Feature rows {X.shape[0]} != metadata rows {len(meta)}")

    return X, meta


def safe_label_array(series: pd.Series) -> np.ndarray:
    return series.astype(str).fillna("missing").to_numpy()


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


def compute_cluster_metrics(
    X_cluster: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    label_columns: List[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))

    n_noise = int(np.sum(labels == -1))
    n_clusters_ex_noise = len([x for x in unique_labels if x != -1])

    metrics["num_rows"] = int(len(labels))
    metrics["num_clusters_including_noise"] = int(len(unique_labels))
    metrics["num_clusters_excluding_noise"] = int(n_clusters_ex_noise)
    metrics["noise_count"] = n_noise
    metrics["noise_rate"] = float(n_noise / len(labels))

    valid_mask = labels != -1
    if valid_mask.sum() > 2 and len(set(labels[valid_mask])) > 1:
        try:
            metrics["silhouette_excluding_noise"] = float(silhouette_score(X_cluster[valid_mask], labels[valid_mask]))
        except Exception as e:
            metrics["silhouette_excluding_noise"] = None
            metrics["silhouette_error"] = repr(e)

        try:
            metrics["davies_bouldin_excluding_noise"] = float(davies_bouldin_score(X_cluster[valid_mask], labels[valid_mask]))
        except Exception as e:
            metrics["davies_bouldin_excluding_noise"] = None
            metrics["davies_bouldin_error"] = repr(e)
    else:
        metrics["silhouette_excluding_noise"] = None
        metrics["davies_bouldin_excluding_noise"] = None

    for col in label_columns:
        if col not in meta.columns:
            continue

        y = safe_label_array(meta[col])

        metrics[f"ari_{col}"] = float(adjusted_rand_score(y, labels))
        metrics[f"nmi_{col}"] = float(normalized_mutual_info_score(y, labels))
        metrics[f"purity_{col}"] = float(purity_score(y, labels, ignore_noise=False))
        metrics[f"purity_excluding_noise_{col}"] = float(purity_score(y, labels, ignore_noise=True))

    return metrics


def run_kmeans_comparison(
    X_cluster: np.ndarray,
    meta: pd.DataFrame,
    k_values: List[int],
    random_state: int,
    n_init: int,
) -> pd.DataFrame:
    rows = []

    y_failure_type = safe_label_array(meta["failure_type"]) if "failure_type" in meta.columns else None
    y_failure_family = safe_label_array(meta["failure_family"]) if "failure_family" in meta.columns else None
    y_failure_binary = safe_label_array(meta["failure_binary"]) if "failure_binary" in meta.columns else None

    for k in k_values:
        km = KMeans(n_clusters=int(k), random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_cluster)

        row = {"method": "kmeans", "k": int(k), "num_clusters": int(k)}

        if len(set(labels)) > 1:
            try:
                row["silhouette"] = float(silhouette_score(X_cluster, labels))
            except Exception:
                row["silhouette"] = None

            try:
                row["davies_bouldin"] = float(davies_bouldin_score(X_cluster, labels))
            except Exception:
                row["davies_bouldin"] = None

        if y_failure_type is not None:
            row["nmi_failure_type"] = float(normalized_mutual_info_score(y_failure_type, labels))
            row["purity_failure_type"] = float(purity_score(y_failure_type, labels))

        if y_failure_family is not None:
            row["nmi_failure_family"] = float(normalized_mutual_info_score(y_failure_family, labels))
            row["purity_failure_family"] = float(purity_score(y_failure_family, labels))

        if y_failure_binary is not None:
            row["nmi_failure_binary"] = float(normalized_mutual_info_score(y_failure_binary, labels))
            row["purity_failure_binary"] = float(purity_score(y_failure_binary, labels))

        rows.append(row)

    return pd.DataFrame(rows)

def build_cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster_id, g in assignments.groupby("cluster"):
        n = len(g)

        failure_binary_counts = Counter(g["failure_binary"].astype(str))
        failure_type_counts = Counter(g["failure_type"].astype(str))
        family_counts = Counter(g["failure_family"].astype(str))
        source_counts = Counter(g["source_dataset"].astype(str))

        fail_count = int((g["failure_binary"].astype(int) == 1).sum())
        fail_rate = fail_count / n if n else 0.0

        non_none_failure_types = Counter(
            x for x in g["failure_type"].astype(str).tolist()
            if x not in {"none", "nan", "None"}
        )

        top_failure_type = failure_type_counts.most_common(1)[0][0] if failure_type_counts else "none"
        top_non_none_failure_type = (
            non_none_failure_types.most_common(1)[0][0]
            if non_none_failure_types
            else "none"
        )
        top_family = family_counts.most_common(1)[0][0] if family_counts else "none"
        top_source = source_counts.most_common(1)[0][0] if source_counts else "none"

        if str(cluster_id) == "-1":
            cluster_name = "noise_or_mixed"
        elif fail_rate < 0.10:
            cluster_name = "mostly_non_failure"
        elif top_non_none_failure_type != "none":
            cluster_name = f"latent_{top_non_none_failure_type}"
        else:
            cluster_name = f"latent_{top_family}"

        avg_conf = float(pd.to_numeric(g["judge_confidence"], errors="coerce").fillna(0).mean())

        row = {
            "cluster": int(cluster_id),
            "cluster_name": cluster_name,
            "n": int(n),
            "failure_count": int(fail_count),
            "failure_rate": float(fail_rate),
            "avg_judge_confidence": avg_conf,
            "top_failure_type": top_failure_type,
            "top_non_none_failure_type": top_non_none_failure_type,
            "top_failure_family": top_family,
            "top_source_dataset": top_source,
            "failure_type_counts": json.dumps(dict(failure_type_counts), ensure_ascii=False),
            "failure_family_counts": json.dumps(dict(family_counts), ensure_ascii=False),
            "failure_binary_counts": json.dumps(dict(failure_binary_counts), ensure_ascii=False),
            "failure_type_entropy": entropy_from_counts(failure_type_counts),
            "failure_family_entropy": entropy_from_counts(family_counts),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["cluster", "n"], ascending=[True, False])
    return summary


def make_scatter_plot(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    out_path: Path,
    max_categories: int = 15,
) -> None:
    plt.figure(figsize=(10, 8))

    values = df[color_col].astype(str)
    counts = values.value_counts()
    top_values = set(counts.head(max_categories).index)

    plot_values = values.apply(lambda x: x if x in top_values else "other")

    categories = list(plot_values.value_counts().index)
    cmap = plt.get_cmap("tab20")

    for i, cat in enumerate(categories):
        mask = plot_values == cat
        plt.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            s=8,
            alpha=0.75,
            label=str(cat),
            color=cmap(i % 20),
        )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, frameon=True, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_failure_rate_plot(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    s = summary.copy()
    s = s[s["cluster"] != -1]
    s = s.sort_values("failure_rate", ascending=False).head(25)

    if s.empty:
        return

    labels = [f"{int(c)}: {name}" for c, name in zip(s["cluster"], s["cluster_name"])]

    plt.figure(figsize=(12, 7))
    plt.bar(range(len(s)), s["failure_rate"])
    plt.xticks(range(len(s)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Failure rate")
    plt.title("Top clusters by failure rate")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_representative_examples(
    assignments: pd.DataFrame,
    original_rows_path: Path,
    out_path: Path,
    max_per_cluster: int = 8,
) -> None:
    # Includes prompt/response, so this file is for internal research use.
    if not original_rows_path.exists():
        return

    full_rows = []
    with original_rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                full_rows.append(json.loads(line))

    by_id = {str(r.get("prompt_id")): r for r in full_rows}

    rows = []

    for cluster_id, g in assignments.groupby("cluster"):
        # Prefer failures first, then high judge confidence.
        gg = g.copy()
        gg["failure_binary_int"] = gg["failure_binary"].astype(int)
        gg["judge_confidence_float"] = pd.to_numeric(gg["judge_confidence"], errors="coerce").fillna(0)

        gg = gg.sort_values(
            ["failure_binary_int", "judge_confidence_float"],
            ascending=[False, False],
        )

        for _, r in gg.head(max_per_cluster).iterrows():
            pid = str(r["prompt_id"])
            src = by_id.get(pid, {})

            rows.append({
                "cluster": int(cluster_id),
                "prompt_id": pid,
                "failure_family": r.get("failure_family"),
                "failure_type": r.get("failure_type"),
                "failure_binary": int(r.get("failure_binary")),
                "severity": r.get("severity"),
                "judge_confidence": r.get("judge_confidence"),
                "judge_reason": src.get("judge_reason", ""),
                "prompt": src.get("prompt", ""),
                "response": src.get("response", ""),
                "source_dataset": r.get("source_dataset"),
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)


def choose_main_clustering(
    X_cluster: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    hcfg = cfg.get("hdbscan", {})
    if bool(hcfg.get("enabled", True)) and HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(hcfg.get("min_cluster_size", 40)),
            min_samples=int(hcfg.get("min_samples", 10)),
            metric=hcfg.get("metric", "euclidean"),
            cluster_selection_method=hcfg.get("cluster_selection_method", "eom"),
        )
        labels = clusterer.fit_predict(X_cluster)
        info = {
            "method": "hdbscan",
            "min_cluster_size": int(hcfg.get("min_cluster_size", 40)),
            "min_samples": int(hcfg.get("min_samples", 10)),
        }
        return labels, "hdbscan", info

    # Fallback KMeans k=12.
    print("[WARN] HDBSCAN unavailable or disabled. Falling back to KMeans k=12.")
    km = KMeans(n_clusters=12, random_state=42, n_init=20)
    labels = km.fit_predict(X_cluster)
    info = {"method": "kmeans_fallback", "k": 12}
    return labels, "kmeans_fallback", info

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_atlas_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    feature_dir = Path(cfg["feature_dir"])
    atlas_dir = Path(cfg["atlas_dir"])
    figures_dir = atlas_dir / "figures"

    ensure_dir(atlas_dir)
    ensure_dir(figures_dir)

    print("[INFO] Phase 4 atlas construction")
    print(f"[INFO] Feature dir: {feature_dir}")
    print(f"[INFO] Atlas dir: {atlas_dir}")

    X, meta = load_inputs(feature_dir)
    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded metadata: {meta.shape}")

    if bool(cfg.get("preprocess", {}).get("standardize", True)):
        print("[INFO] Standardizing features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype("float32")
    else:
        X_scaled = X.astype("float32")

    pca_cfg = cfg.get("pca", {})
    n_components = min(int(pca_cfg.get("n_components", 50)), X_scaled.shape[0] - 1, X_scaled.shape[1])

    print(f"[INFO] Running PCA n_components={n_components}")
    pca = PCA(n_components=n_components, random_state=int(pca_cfg.get("random_state", 42)))
    X_pca = pca.fit_transform(X_scaled).astype("float32")

    np.save(atlas_dir / "pca_features.npy", X_pca)

    pca_info = {
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "explained_variance_ratio_first_10": [float(x) for x in pca.explained_variance_ratio_[:10]],
    }
    (atlas_dir / "pca_info.json").write_text(json.dumps(pca_info, indent=2), encoding="utf-8")
    print("[INFO] PCA explained variance sum:", pca_info["explained_variance_ratio_sum"])

    # Cluster space.
    X_cluster = X_pca
    cluster_space_name = "pca"

    cu_cfg = cfg.get("cluster_umap", {})
    if bool(cu_cfg.get("enabled", True)) and HAS_UMAP:
        print("[INFO] Running UMAP for clustering space")
        reducer_cluster = umap.UMAP(
            n_components=int(cu_cfg.get("n_components", 10)),
            n_neighbors=int(cu_cfg.get("n_neighbors", 30)),
            min_dist=float(cu_cfg.get("min_dist", 0.0)),
            metric=cu_cfg.get("metric", "euclidean"),
            random_state=int(cu_cfg.get("random_state", 42)),
        )
        X_cluster = reducer_cluster.fit_transform(X_pca).astype("float32")
        cluster_space_name = "umap10"
        np.save(atlas_dir / "umap_cluster_features.npy", X_cluster)
    else:
        if not HAS_UMAP:
            print("[WARN] umap-learn not installed. Using PCA space for clustering.")

    # 2D visualization.
    u_cfg = cfg.get("umap", {})
    if bool(u_cfg.get("enabled", True)) and HAS_UMAP:
        print("[INFO] Running UMAP 2D for visualization")
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=int(u_cfg.get("n_neighbors", 30)),
            min_dist=float(u_cfg.get("min_dist", 0.10)),
            metric=u_cfg.get("metric", "euclidean"),
            random_state=int(u_cfg.get("random_state", 42)),
        )
        X_2d = reducer_2d.fit_transform(X_pca).astype("float32")
        viz_method = "umap"
    else:
        print("[WARN] UMAP unavailable. Using first two PCA dimensions for visualization.")
        X_2d = X_pca[:, :2].astype("float32")
        viz_method = "pca2"

    # Main clustering.
    labels, main_method, main_info = choose_main_clustering(X_cluster, cfg)

    assignments = meta.copy()
    assignments["cluster"] = labels.astype(int)
    assignments["umap_x"] = X_2d[:, 0]
    assignments["umap_y"] = X_2d[:, 1]
    assignments["viz_method"] = viz_method
    assignments["cluster_space"] = cluster_space_name

    assignments_path = atlas_dir / "cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    umap_df = assignments[[
        "prompt_id",
        "umap_x",
        "umap_y",
        "cluster",
        "failure_binary",
        "failure_family",
        "failure_type",
        "source_dataset",
        "severity",
        "judge_confidence",
    ]]
    umap_df.to_csv(atlas_dir / "umap_2d.csv", index=False)

    summary = build_cluster_summary(assignments)
    summary_path = atlas_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    label_columns = cfg.get("analysis", {}).get("label_columns", ["failure_binary", "failure_family", "failure_type"])
    metrics = compute_cluster_metrics(X_cluster, labels, assignments, label_columns)
    metrics["main_method"] = main_method
    metrics["main_method_info"] = main_info
    metrics["cluster_space"] = cluster_space_name
    metrics["viz_method"] = viz_method
    metrics["pca_explained_variance_ratio_sum"] = pca_info["explained_variance_ratio_sum"]

    (atlas_dir / "clustering_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # KMeans comparison.
    kcfg = cfg.get("kmeans", {})
    k_values = [int(k) for k in kcfg.get("k_values", [6, 8, 10, 12, 15, 20])]
    print("[INFO] Running KMeans comparison:", k_values)
    comp = run_kmeans_comparison(
        X_cluster=X_cluster,
        meta=assignments,
        k_values=k_values,
        random_state=int(kcfg.get("random_state", 42)),
        n_init=int(kcfg.get("n_init", 20)),
    )

    # Add main HDBSCAN row.
    main_row = {
        "method": main_method,
        "k": None,
        "num_clusters": metrics.get("num_clusters_excluding_noise"),
        "noise_rate": metrics.get("noise_rate"),
        "silhouette": metrics.get("silhouette_excluding_noise"),
        "davies_bouldin": metrics.get("davies_bouldin_excluding_noise"),
        "nmi_failure_type": metrics.get("nmi_failure_type"),
        "purity_failure_type": metrics.get("purity_failure_type"),
        "nmi_failure_family": metrics.get("nmi_failure_family"),
        "purity_failure_family": metrics.get("purity_failure_family"),
        "nmi_failure_binary": metrics.get("nmi_failure_binary"),
        "purity_failure_binary": metrics.get("purity_failure_binary"),
    }
    comp = pd.concat([pd.DataFrame([main_row]), comp], ignore_index=True)
    comp.to_csv(atlas_dir / "clustering_comparison.csv", index=False)

    # Figures.
    print("[INFO] Saving figures")
    make_scatter_plot(assignments, "cluster", "FailureAtlas UMAP by cluster", figures_dir / "umap_by_cluster.png")
    make_scatter_plot(assignments, "failure_type", "FailureAtlas UMAP by judged failure type", figures_dir / "umap_by_failure_type.png")
    make_scatter_plot(assignments, "failure_family", "FailureAtlas UMAP by original family", figures_dir / "umap_by_failure_family.png")
    make_scatter_plot(assignments, "failure_binary", "FailureAtlas UMAP by failure binary", figures_dir / "umap_by_failure_binary.png")
    make_failure_rate_plot(summary, figures_dir / "cluster_failure_rates.png")

    judged_path = Path(cfg.get("judged_path", "outputs/judged/qwen_7b_mvp_judged_repaired.jsonl"))
    save_representative_examples(
        assignments=assignments,
        original_rows_path=judged_path,
        out_path=atlas_dir / "cluster_representative_examples_internal.csv",
        max_per_cluster=8,
    )

    print("[DONE] Phase 4 atlas construction complete.")
    print(f"Assignments: {assignments_path}")
    print(f"Cluster summary: {summary_path}")
    print(f"Metrics: {atlas_dir / 'clustering_metrics.json'}")
    print(f"Comparison: {atlas_dir / 'clustering_comparison.csv'}")
    print(f"Figures: {figures_dir}")

    print("\nMain metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nTop cluster summary:")
    if len(summary):
        cols = [
            "cluster",
            "cluster_name",
            "n",
            "failure_count",
            "failure_rate",
            "top_non_none_failure_type",
            "top_failure_family",
        ]
        print(summary[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
