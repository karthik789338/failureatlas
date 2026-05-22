#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pair_split_random(meta: pd.DataFrame, test_size: float, random_state: int):
    pair_ids = sorted(meta["pair_id"].unique())

    train_pairs, test_pairs = train_test_split(
        pair_ids,
        test_size=test_size,
        random_state=random_state,
    )

    train_pairs = set(train_pairs)
    test_pairs = set(test_pairs)

    train_mask = meta["pair_id"].isin(train_pairs).to_numpy()
    test_mask = meta["pair_id"].isin(test_pairs).to_numpy()

    return train_mask, test_mask


def pair_split_group_source(meta: pd.DataFrame, test_size: float, random_state: int):
    pair_meta = meta.drop_duplicates("pair_id").copy()

    pair_ids = pair_meta["pair_id"].to_numpy()
    groups = pair_meta["source_dataset"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_pair_idx, test_pair_idx = next(splitter.split(pair_ids, groups=groups))

    train_pairs = set(pair_ids[train_pair_idx])
    test_pairs = set(pair_ids[test_pair_idx])

    train_mask = meta["pair_id"].isin(train_pairs).to_numpy()
    test_mask = meta["pair_id"].isin(test_pairs).to_numpy()

    heldout_sources = sorted(set(groups[test_pair_idx]))

    return train_mask, test_mask, heldout_sources


def make_pipeline(n_train: int, n_features: int, pca_components: int, random_state: int):
    n_pca = min(int(pca_components), max(1, n_train - 1), n_features)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def get_positive_probability(pipe, X):
    probs = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return probs[:, pos_idx]


def binary_metrics(y_true, pred, prob):
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }

    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    out["precision_nonfailure"] = float(p[0])
    out["recall_nonfailure"] = float(r[0])
    out["f1_nonfailure"] = float(f[0])
    out["precision_failure"] = float(p[1])
    out["recall_failure"] = float(r[1])
    out["f1_failure"] = float(f[1])

    if len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out


def pair_ranking_accuracy(test_meta: pd.DataFrame, prob):
    df = test_meta.copy()
    df["prob_failure"] = prob

    total = 0
    correct = 0
    ties = 0
    margins = []

    for _, g in df.groupby("pair_id"):
        if len(g) != 2:
            continue

        fail = g[g["failure_binary"] == 1]
        nonfail = g[g["failure_binary"] == 0]

        if len(fail) != 1 or len(nonfail) != 1:
            continue

        pf = float(fail["prob_failure"].iloc[0])
        pn = float(nonfail["prob_failure"].iloc[0])
        margin = pf - pn

        margins.append(margin)
        total += 1

        if pf > pn:
            correct += 1
        elif pf == pn:
            ties += 1

    return {
        "pair_ranking_accuracy": float(correct / total) if total else None,
        "pair_tie_rate": float(ties / total) if total else None,
        "pair_mean_margin": float(np.mean(margins)) if margins else None,
        "num_eval_pairs": int(total),
    }


def run_one_split(X, meta, checkpoint, split_name, train_mask, test_mask, pca_components, random_state, heldout_sources=None):
    y = meta["failure_binary"].astype(int).to_numpy()

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    test_meta = meta.loc[test_mask].copy()

    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return {
            "checkpoint_tokens": checkpoint,
            "split": split_name,
            "skipped": True,
            "reason": "Train or test has only one class.",
            "heldout_sources": json.dumps(heldout_sources or []),
        }

    pipe = make_pipeline(
        n_train=len(y_train),
        n_features=X_train.shape[1],
        pca_components=pca_components,
        random_state=random_state,
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = get_positive_probability(pipe, X_test)

    out = {
        "checkpoint_tokens": int(checkpoint),
        "split": split_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "num_train_pairs": int(meta.loc[train_mask, "pair_id"].nunique()),
        "num_test_pairs": int(meta.loc[test_mask, "pair_id"].nunique()),
        "feature_dim": int(X_train.shape[1]),
        "pca_components": int(pipe.named_steps["pca"].n_components_),
        "heldout_sources": json.dumps(heldout_sources or []),
        "skipped": False,
    }

    out.update(binary_metrics(y_test, pred, prob))
    out.update(pair_ranking_accuracy(test_meta, prob))

    return out


def plot_metric(results: pd.DataFrame, metric: str, out_path: Path, title: str, ylabel: str):
    plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()

    for split in ["paired_random", "paired_group_source"]:
        s = results[(results["split"] == split) & (~results["skipped"].fillna(False))]
        if s.empty:
            continue

        s = s.sort_values("checkpoint_tokens")
        ax.plot(
            s["checkpoint_tokens"],
            s[metric],
            marker="o",
            linewidth=2,
            label=split.replace("paired_", "").replace("_", " "),
        )

    ax.set_xlabel("Generated tokens observed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(results["checkpoint_tokens"].unique()))
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/online_early_warning/qwen_7b_mvp")
    parser.add_argument("--output-dir", default="outputs/online_early_warning/qwen_7b_mvp/results")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(fig_dir)

    X_all = np.load(input_dir / "online_features.npy").astype("float32")
    meta_all = pd.read_csv(input_dir / "online_metadata.csv")

    meta_all["pair_id"] = meta_all["pair_id"].astype(str)
    meta_all["source_dataset"] = meta_all["source_dataset"].astype(str)
    meta_all["failure_binary"] = meta_all["failure_binary"].astype(int)
    meta_all["checkpoint_tokens"] = meta_all["checkpoint_tokens"].astype(int)

    if len(meta_all) != X_all.shape[0]:
        raise ValueError(f"Feature rows {X_all.shape[0]} != metadata rows {len(meta_all)}")

    rows = []

    for checkpoint in sorted(meta_all["checkpoint_tokens"].unique()):
        ckpt_mask = meta_all["checkpoint_tokens"].to_numpy() == checkpoint
        X = X_all[ckpt_mask]
        meta = meta_all.loc[ckpt_mask].reset_index(drop=True)

        print(f"[INFO] Checkpoint {checkpoint}: X={X.shape}, rows={len(meta)}, pairs={meta['pair_id'].nunique()}")

        # Random paired split.
        train_mask, test_mask = pair_split_random(meta, args.test_size, args.random_state)
        rows.append(run_one_split(
            X=X,
            meta=meta,
            checkpoint=checkpoint,
            split_name="paired_random",
            train_mask=train_mask,
            test_mask=test_mask,
            pca_components=args.pca_components,
            random_state=args.random_state,
        ))

        # Group-source paired split.
        train_mask, test_mask, heldout_sources = pair_split_group_source(meta, args.test_size, args.random_state)
        rows.append(run_one_split(
            X=X,
            meta=meta,
            checkpoint=checkpoint,
            split_name="paired_group_source",
            train_mask=train_mask,
            test_mask=test_mask,
            pca_components=args.pca_components,
            random_state=args.random_state,
            heldout_sources=heldout_sources,
        ))

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "online_early_warning_results.csv", index=False)

    summary = {
        "input_dir": str(input_dir),
        "num_feature_rows": int(X_all.shape[0]),
        "feature_dim": int(X_all.shape[1]),
        "num_pairs": int(meta_all["pair_id"].nunique()),
        "checkpoints": sorted([int(x) for x in meta_all["checkpoint_tokens"].unique()]),
        "best_random_balanced_accuracy": None,
        "best_group_source_balanced_accuracy": None,
        "best_random_pair_ranking": None,
        "best_group_source_pair_ranking": None,
    }

    valid = results[~results["skipped"].fillna(False)].copy()

    for split, key_ba, key_pr in [
        ("paired_random", "best_random_balanced_accuracy", "best_random_pair_ranking"),
        ("paired_group_source", "best_group_source_balanced_accuracy", "best_group_source_pair_ranking"),
    ]:
        s = valid[valid["split"] == split]
        if not s.empty:
            summary[key_ba] = s.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()
            summary[key_pr] = s.sort_values("pair_ranking_accuracy", ascending=False).iloc[0].to_dict()

    (output_dir / "online_early_warning_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_metric(
        results,
        "balanced_accuracy",
        fig_dir / "online_balanced_accuracy_by_checkpoint.png",
        "Online early-warning balanced accuracy",
        "Balanced accuracy",
    )

    plot_metric(
        results,
        "roc_auc",
        fig_dir / "online_roc_auc_by_checkpoint.png",
        "Online early-warning ROC-AUC",
        "ROC-AUC",
    )

    plot_metric(
        results,
        "pair_ranking_accuracy",
        fig_dir / "online_pair_ranking_by_checkpoint.png",
        "Online matched-pair ranking accuracy",
        "Pair-ranking accuracy",
    )

    print("\n[DONE] Online early-warning training complete.")
    print("Saved:", output_dir / "online_early_warning_results.csv")
    print("Saved:", output_dir / "online_early_warning_summary.json")

    cols = [
        "checkpoint_tokens",
        "split",
        "n_train",
        "n_test",
        "num_train_pairs",
        "num_test_pairs",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "f1_failure",
        "recall_failure",
        "pair_ranking_accuracy",
        "pair_mean_margin",
    ]

    available = [c for c in cols if c in results.columns]
    print("\n=== Online early-warning results ===")
    print(results[available].to_string(index=False))


if __name__ == "__main__":
    main()
