#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def purity_score(y_true, y_pred, ignore_noise=False):
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

    return total / len(y_true)


def entropy_counts(values):
    c = Counter(values)
    total = sum(c.values())
    ent = 0.0
    for v in c.values():
        p = v / total
        if p > 0:
            ent -= p * np.log2(p)
    return float(ent)


def compute_metrics(Xc, labels, df):
    labels = np.asarray(labels)
    valid = labels != -1

    out = {
        "num_rows": int(len(df)),
        "num_pairs": int(df["pair_id"].nunique()) if "pair_id" in df.columns else None,
        "num_clusters_including_noise": int(len(set(labels.tolist()))),
        "num_clusters_excluding_noise": int(len([x for x in set(labels.tolist()) if x != -1])),
        "noise_count": int((labels == -1).sum()),
        "noise_rate": float((labels == -1).sum() / len(labels)),
    }

    if valid.sum() > 3 and len(set(labels[valid].tolist())) > 1:
        out["silhouette_excluding_noise"] = float(silhouette_score(Xc[valid], labels[valid]))
        out["davies_bouldin_excluding_noise"] = float(davies_bouldin_score(Xc[valid], labels[valid]))
    else:
        out["silhouette_excluding_noise"] = None
        out["davies_bouldin_excluding_noise"] = None

    for col in ["failure_binary", "failure_type", "failure_family", "source_dataset", "match_level"]:
        if col not in df.columns:
            continue
        y = df[col].astype(str).to_numpy()
        out[f"ari_{col}"] = float(adjusted_rand_score(y, labels))
        out[f"nmi_{col}"] = float(normalized_mutual_info_score(y, labels))
        out[f"purity_{col}"] = float(purity_score(y, labels))
        out[f"purity_excluding_noise_{col}"] = float(purity_score(y, labels, ignore_noise=True))

    # Pair separation: in matched pairs, do failure and nonfailure land in different clusters?
    pair_total = 0
    pair_diff = 0
    pair_same = 0
    pair_noise = 0

    tmp = df.copy()
    tmp["cluster"] = labels.astype(int)

    for _, g in tmp.groupby("pair_id"):
        if len(g) != 2:
            continue

        clusters = g["cluster"].tolist()
        if -1 in clusters:
            pair_noise += 1
            continue

        pair_total += 1
        if clusters[0] != clusters[1]:
            pair_diff += 1
        else:
            pair_same += 1

    out["pair_eval_count_excluding_noise"] = int(pair_total)
    out["pair_same_cluster_count"] = int(pair_same)
    out["pair_different_cluster_count"] = int(pair_diff)
    out["pair_different_cluster_rate"] = float(pair_diff / pair_total) if pair_total else None
    out["pair_noise_count"] = int(pair_noise)

    return out


def cluster_summary(df):
    rows = []

    for cluster, g in df.groupby("cluster"):
        n = len(g)
        fail_count = int((g["failure_binary"].astype(int) == 1).sum())
        fail_rate = fail_count / n if n else 0.0

        ft = Counter(g["failure_type"].astype(str))
        fam = Counter(g["failure_family"].astype(str))
        src = Counter(g["source_dataset"].astype(str))

        rows.append({
            "cluster": int(cluster),
            "n": int(n),
            "failure_count": fail_count,
            "failure_rate": fail_rate,
            "top_failure_type": ft.most_common(1)[0][0],
            "top_failure_family": fam.most_common(1)[0][0],
            "top_source_dataset": src.most_common(1)[0][0],
            "failure_type_entropy": entropy_counts(g["failure_type"].astype(str)),
            "failure_family_entropy": entropy_counts(g["failure_family"].astype(str)),
            "source_dataset_entropy": entropy_counts(g["source_dataset"].astype(str)),
            "failure_type_counts": json.dumps(dict(ft), ensure_ascii=False),
            "failure_family_counts": json.dumps(dict(fam), ensure_ascii=False),
            "source_dataset_counts": json.dumps(dict(src), ensure_ascii=False),
        })

    return pd.DataFrame(rows).sort_values("cluster")


def scatter(df, color_col, title, out_path):
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    vals = df[color_col].astype(str)
    cats = vals.value_counts().index.tolist()
    cmap = plt.get_cmap("tab20")

    for i, cat in enumerate(cats):
        mask = vals == cat
        ax.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            s=10,
            alpha=0.75,
            label=str(cat),
            color=cmap(i % 20),
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, markerscale=2, frameon=True)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--pair-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-components", type=int, default=10)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=20)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    pair_rows_path = Path(args.pair_rows)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    X = np.load(feature_dir / "features.npy").astype("float32")
    meta = pd.read_csv(feature_dir / "metadata.csv")
    pair_rows = pd.read_csv(pair_rows_path)

    meta["prompt_id"] = meta["prompt_id"].astype(str)
    pair_rows["prompt_id"] = pair_rows["prompt_id"].astype(str)

    id_to_idx = {pid: i for i, pid in enumerate(meta["prompt_id"].tolist())}

    missing = [pid for pid in pair_rows["prompt_id"].tolist() if pid not in id_to_idx]
    if missing:
        raise ValueError(f"Missing {len(missing)} prompt_ids from feature metadata. First few: {missing[:5]}")

    idx = np.array([id_to_idx[pid] for pid in pair_rows["prompt_id"].tolist()], dtype=np.int64)
    X_sub = X[idx]

    df = pair_rows.copy()
    df["model_label"] = args.model_label

    # Standardize + PCA.
    Xs = StandardScaler().fit_transform(X_sub).astype("float32")
    n_pca = min(args.pca_components, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_pca, random_state=args.random_state)
    Xp = pca.fit_transform(Xs).astype("float32")

    # UMAP cluster space.
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=args.umap_components,
            n_neighbors=min(30, max(2, Xp.shape[0] - 1)),
            min_dist=0.0,
            metric="euclidean",
            random_state=args.random_state,
        )
        Xc = reducer.fit_transform(Xp).astype("float32")

        reducer2 = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(2, Xp.shape[0] - 1)),
            min_dist=0.1,
            metric="euclidean",
            random_state=args.random_state,
        )
        X2 = reducer2.fit_transform(Xp).astype("float32")
        cluster_space = "umap10"
        viz_method = "umap2"
    else:
        Xc = Xp
        X2 = Xp[:, :2]
        cluster_space = "pca"
        viz_method = "pca2"

    # HDBSCAN with fallback.
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(Xc)
        non_noise = labels[labels != -1]
        if len(set(non_noise.tolist())) < 2:
            k = min(8, max(2, Xc.shape[0] // 100))
            labels = KMeans(n_clusters=k, random_state=args.random_state, n_init=20).fit_predict(Xc)
            method = f"kmeans_fallback_k{k}"
        else:
            method = "hdbscan"
    else:
        k = min(8, max(2, Xc.shape[0] // 100))
        labels = KMeans(n_clusters=k, random_state=args.random_state, n_init=20).fit_predict(Xc)
        method = f"kmeans_fallback_k{k}"

    df["cluster"] = labels.astype(int)
    df["umap_x"] = X2[:, 0]
    df["umap_y"] = X2[:, 1]

    metrics = compute_metrics(Xc, labels, df)
    metrics["model_label"] = args.model_label
    metrics["method"] = method
    metrics["cluster_space"] = cluster_space
    metrics["viz_method"] = viz_method
    metrics["pca_components"] = int(n_pca)
    metrics["pca_explained_variance_sum"] = float(pca.explained_variance_ratio_.sum())

    summary = cluster_summary(df)

    df.to_csv(out_dir / "paired_only_cluster_assignments.csv", index=False)
    summary.to_csv(out_dir / "paired_only_cluster_summary.csv", index=False)
    np.save(out_dir / "paired_only_pca_features.npy", Xp)
    np.save(out_dir / "paired_only_cluster_features.npy", Xc)

    (out_dir / "paired_only_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scatter(df, "cluster", f"{args.model_label}: paired-only atlas by cluster", fig_dir / "paired_only_umap_by_cluster.png")
    scatter(df, "failure_binary", f"{args.model_label}: paired-only atlas by failure label", fig_dir / "paired_only_umap_by_failure_binary.png")
    scatter(df, "failure_type", f"{args.model_label}: paired-only atlas by failure type", fig_dir / "paired_only_umap_by_failure_type.png")
    scatter(df, "source_dataset", f"{args.model_label}: paired-only atlas by source dataset", fig_dir / "paired_only_umap_by_source.png")

    print("=" * 80)
    print(f"Paired-only atlas: {args.model_label}")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))

    print("\nCluster summary:")
    cols = [
        "cluster",
        "n",
        "failure_count",
        "failure_rate",
        "top_failure_type",
        "top_failure_family",
        "top_source_dataset",
        "failure_type_entropy",
    ]
    print(summary[cols].head(30).to_string(index=False))
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
