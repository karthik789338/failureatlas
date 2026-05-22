#!/usr/bin/env python3
"""
Phase 5: Train FailureAtlas meta-controller.

Inputs:
  outputs/features/qwen_7b_mvp/features.npy
  outputs/features/qwen_7b_mvp/metadata.csv

Outputs:
  outputs/meta_controller/qwen_7b_mvp/
    binary_failure_results.csv
    failure_type_results.csv
    within_family_binary_results.csv
    leave_one_family_out_results.csv
    phase5_summary.json

This script trains controlled classifiers:
- random stratified split
- group-by-source split
- within-family split
- leave-one-family-out split
- failure-only type classifier

It also evaluates different feature stages:
- prompt_final
- response_25
- response_50
- response_75
- response_final
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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


def get_stage_indices(stage: str, layout: Dict[str, Any]) -> np.ndarray:
    """
    Feature layout from Phase 3:
    - projected vectors: num_layers * num_positions * projection_dim
      order: layer outer, position inner, projection dim inner
    - delta projected vectors: num_layers * projection_dim
      response_final - prompt_final, only available at final stage
    - norm features: num_layers * num_positions

    For early stages, include projected vectors and norms up to that position.
    For response_final, include all projected vectors, all deltas, and all norms.
    """

    num_layers = int(layout["num_layers"])
    num_positions = int(layout["num_positions"])
    proj_dim = int(layout["projection_dim"])

    pos_map = layout["positions"]
    stage_pos = int(pos_map[stage])

    projected_dim = num_layers * num_positions * proj_dim
    delta_dim = num_layers * proj_dim
    norm_start = projected_dim + delta_dim

    idx = []

    # Projected vectors up to current stage.
    for layer in range(num_layers):
        for pos in range(stage_pos + 1):
            start = (layer * num_positions + pos) * proj_dim
            end = start + proj_dim
            idx.extend(range(start, end))

    # Include delta features only at final stage because they require response_final.
    if stage == "response_final":
        idx.extend(range(projected_dim, projected_dim + delta_dim))

    # Norms up to current stage.
    for layer in range(num_layers):
        for pos in range(stage_pos + 1):
            norm_idx = norm_start + (layer * num_positions + pos)
            idx.append(norm_idx)

    return np.array(idx, dtype=np.int64)


def make_pipeline(n_train: int, n_features: int, cfg: Dict[str, Any]) -> Pipeline:
    requested_pca = int(cfg["models"].get("pca_components", 100))
    n_pca = min(requested_pca, max(1, n_train - 1), n_features)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=int(cfg["splits"].get("random_state", 42)))),
        ("clf", LogisticRegression(
            max_iter=int(cfg["models"].get("logistic_max_iter", 3000)),
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def majority_baseline(y_test: np.ndarray) -> Dict[str, float]:
    values, counts = np.unique(y_test, return_counts=True)
    majority = values[np.argmax(counts)]
    pred = np.full_like(y_test, majority)

    return {
        "majority_accuracy": float(accuracy_score(y_test, pred)),
        "majority_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "majority_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def binary_metrics(y_test: np.ndarray, pred: np.ndarray, prob: np.ndarray | None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

    p, r, f, _ = precision_recall_fscore_support(
        y_test,
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

    if prob is not None and len(np.unique(y_test)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_test, prob))
        out["average_precision"] = float(average_precision_score(y_test, prob))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out


def multiclass_metrics(y_test: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }


def predict_proba_positive(pipe: Pipeline, X_test: np.ndarray) -> np.ndarray | None:
    if not hasattr(pipe.named_steps["clf"], "predict_proba"):
        return None

    probs = pipe.predict_proba(X_test)

    classes = pipe.named_steps["clf"].classes_
    if len(classes) != 2:
        return None

    pos_idx = list(classes).index(1) if 1 in classes else 1
    return probs[:, pos_idx]

def run_binary_random_split(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    stage_indices: np.ndarray,
) -> Dict[str, Any]:
    y = meta["failure_binary"].astype(int).to_numpy()
    Xs = X[:, stage_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        Xs,
        y,
        test_size=float(cfg["splits"].get("test_size", 0.25)),
        random_state=int(cfg["splits"].get("random_state", 42)),
        stratify=y,
    )

    pipe = make_pipeline(len(y_train), X_train.shape[1], cfg)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = predict_proba_positive(pipe, X_test)

    out = {
        "task": "binary_failure",
        "split": "random_stratified",
        "stage": stage,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        **majority_baseline(y_test),
        **binary_metrics(y_test, pred, prob),
    }

    out["balanced_accuracy_gain_over_majority"] = out["balanced_accuracy"] - out["majority_balanced_accuracy"]
    out["macro_f1_gain_over_majority"] = out["macro_f1"] - out["majority_macro_f1"]
    return out


def run_binary_group_source_split(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    stage_indices: np.ndarray,
) -> Dict[str, Any]:
    y = meta["failure_binary"].astype(int).to_numpy()
    groups = meta["source_dataset"].astype(str).to_numpy()
    Xs = X[:, stage_indices]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(cfg["splits"].get("test_size", 0.25)),
        random_state=int(cfg["splits"].get("random_state", 42)),
    )

    train_idx, test_idx = next(splitter.split(Xs, y, groups=groups))

    X_train, X_test = Xs[train_idx], Xs[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    heldout_sources = sorted(set(groups[test_idx]))

    # If test has only one class, metrics like ROC are not useful but balanced accuracy still computes poorly.
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "task": "binary_failure",
            "split": "group_source",
            "stage": stage,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_dim": int(X_train.shape[1]),
            "heldout_groups": json.dumps(heldout_sources),
            "skipped": True,
            "reason": "Train or test split has only one class.",
        }

    pipe = make_pipeline(len(y_train), X_train.shape[1], cfg)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = predict_proba_positive(pipe, X_test)

    out = {
        "task": "binary_failure",
        "split": "group_source",
        "stage": stage,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "heldout_groups": json.dumps(heldout_sources),
        "skipped": False,
        **majority_baseline(y_test),
        **binary_metrics(y_test, pred, prob),
    }

    out["balanced_accuracy_gain_over_majority"] = out["balanced_accuracy"] - out["majority_balanced_accuracy"]
    out["macro_f1_gain_over_majority"] = out["macro_f1"] - out["majority_macro_f1"]
    return out


def run_within_family_binary(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    stage_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    Xs = X[:, stage_indices]

    for family in sorted(meta["failure_family"].unique()):
        idx = np.where(meta["failure_family"].astype(str).to_numpy() == str(family))[0]
        m = meta.iloc[idx]
        y = m["failure_binary"].astype(int).to_numpy()

        if len(idx) < 80 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
            rows.append({
                "task": "within_family_binary",
                "family": family,
                "stage": stage,
                "n": int(len(idx)),
                "skipped": True,
                "reason": "Too few rows or one class.",
            })
            continue

        X_sub = Xs[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub,
            y,
            test_size=float(cfg["splits"].get("test_size", 0.25)),
            random_state=int(cfg["splits"].get("random_state", 42)),
            stratify=y,
        )

        pipe = make_pipeline(len(y_train), X_train.shape[1], cfg)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = predict_proba_positive(pipe, X_test)

        out = {
            "task": "within_family_binary",
            "family": family,
            "stage": stage,
            "n": int(len(idx)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_dim": int(X_train.shape[1]),
            "skipped": False,
            **majority_baseline(y_test),
            **binary_metrics(y_test, pred, prob),
        }
        out["balanced_accuracy_gain_over_majority"] = out["balanced_accuracy"] - out["majority_balanced_accuracy"]
        out["macro_f1_gain_over_majority"] = out["macro_f1"] - out["majority_macro_f1"]
        rows.append(out)

    return pd.DataFrame(rows)


def run_leave_one_family_out_binary(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    stage_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    Xs = X[:, stage_indices]
    y = meta["failure_binary"].astype(int).to_numpy()
    groups = meta["failure_family"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(Xs, y, groups=groups):
        family = sorted(set(groups[test_idx]))[0]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            rows.append({
                "task": "leave_one_family_out_binary",
                "heldout_family": family,
                "stage": stage,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "skipped": True,
                "reason": "Train or test split has only one class.",
            })
            continue

        X_train, X_test = Xs[train_idx], Xs[test_idx]

        pipe = make_pipeline(len(y_train), X_train.shape[1], cfg)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = predict_proba_positive(pipe, X_test)

        out = {
            "task": "leave_one_family_out_binary",
            "heldout_family": family,
            "stage": stage,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_dim": int(X_train.shape[1]),
            "skipped": False,
            **majority_baseline(y_test),
            **binary_metrics(y_test, pred, prob),
        }
        out["balanced_accuracy_gain_over_majority"] = out["balanced_accuracy"] - out["majority_balanced_accuracy"]
        out["macro_f1_gain_over_majority"] = out["macro_f1"] - out["majority_macro_f1"]
        rows.append(out)

    return pd.DataFrame(rows)

def run_failure_type_failures_only(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    stage_indices: np.ndarray,
) -> Dict[str, Any]:
    mask = meta["failure_binary"].astype(int).to_numpy() == 1
    m = meta[mask].reset_index(drop=True)
    Xs = X[mask][:, stage_indices]

    y_raw = m["failure_type"].astype(str).to_numpy()

    # Drop ultra-rare labels if any.
    counts = pd.Series(y_raw).value_counts()
    keep_labels = set(counts[counts >= 3].index)
    keep = np.array([y in keep_labels for y in y_raw])

    Xs = Xs[keep]
    y_raw = y_raw[keep]
    m = m.iloc[np.where(keep)[0]].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        return {
            "task": "failure_type_failures_only",
            "stage": stage,
            "skipped": True,
            "reason": "Only one failure type.",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        Xs,
        y,
        test_size=float(cfg["splits"].get("test_size", 0.25)),
        random_state=int(cfg["splits"].get("random_state", 42)),
        stratify=y,
    )

    pipe = make_pipeline(len(y_train), X_train.shape[1], cfg)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    out = {
        "task": "failure_type_failures_only",
        "split": "random_stratified_failures_only",
        "stage": stage,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "num_classes": int(len(le.classes_)),
        "classes": json.dumps(list(le.classes_), ensure_ascii=False),
        "feature_dim": int(X_train.shape[1]),
        **majority_baseline(y_test),
        **multiclass_metrics(y_test, pred),
    }

    out["balanced_accuracy_gain_over_majority"] = out["balanced_accuracy"] - out["majority_balanced_accuracy"]
    out["macro_f1_gain_over_majority"] = out["macro_f1"] - out["majority_macro_f1"]

    report = classification_report(
        y_test,
        pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    out["classification_report"] = report

    return out


def summarize_best(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        gg = g.copy()
        gg = gg[pd.to_numeric(gg[metric], errors="coerce").notna()]
        if gg.empty:
            continue
        best = gg.sort_values(metric, ascending=False).iloc[0].to_dict()
        rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_meta_controller_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    feature_dir = Path(cfg["feature_dir"])
    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)

    print("[INFO] Phase 5 meta-controller training")
    print(f"[INFO] Feature dir: {feature_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    X, meta = load_inputs(feature_dir)
    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded metadata: {meta.shape}")

    stages = cfg.get("stages", [
        "prompt_final",
        "response_25",
        "response_50",
        "response_75",
        "response_final",
    ])

    binary_rows = []
    within_family_frames = []
    logo_frames = []
    failure_type_rows = []

    for stage in stages:
        print(f"\n[INFO] Stage: {stage}")
        stage_indices = get_stage_indices(stage, cfg["feature_layout"])
        print(f"[INFO] Stage feature dim: {len(stage_indices)}")

        # Binary random split.
        print("[INFO] Running binary random split")
        binary_rows.append(run_binary_random_split(X, meta, cfg, stage, stage_indices))

        # Binary group-source split.
        print("[INFO] Running binary group-source split")
        binary_rows.append(run_binary_group_source_split(X, meta, cfg, stage, stage_indices))

        # Within-family binary.
        print("[INFO] Running within-family binary controls")
        within_df = run_within_family_binary(X, meta, cfg, stage, stage_indices)
        within_family_frames.append(within_df)

        # Leave-one-family-out binary.
        print("[INFO] Running leave-one-family-out binary controls")
        logo_df = run_leave_one_family_out_binary(X, meta, cfg, stage, stage_indices)
        logo_frames.append(logo_df)

        # Failure-only type classifier.
        print("[INFO] Running failure-type classifier on failures only")
        failure_type_rows.append(run_failure_type_failures_only(X, meta, cfg, stage, stage_indices))

    binary_df = pd.DataFrame(binary_rows)
    within_family_df = pd.concat(within_family_frames, ignore_index=True)
    logo_df = pd.concat(logo_frames, ignore_index=True)
    failure_type_df = pd.DataFrame([
        {k: v for k, v in row.items() if k != "classification_report"}
        for row in failure_type_rows
    ])

    binary_df.to_csv(output_dir / "binary_failure_results.csv", index=False)
    within_family_df.to_csv(output_dir / "within_family_binary_results.csv", index=False)
    logo_df.to_csv(output_dir / "leave_one_family_out_results.csv", index=False)
    failure_type_df.to_csv(output_dir / "failure_type_failures_only_results.csv", index=False)

    # Save full reports separately.
    reports = {
        row.get("stage", f"stage_{i}"): row.get("classification_report")
        for i, row in enumerate(failure_type_rows)
        if row.get("classification_report") is not None
    }
    (output_dir / "failure_type_classification_reports.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    summary = {
        "num_rows": int(len(meta)),
        "feature_dim_original": int(X.shape[1]),
        "stages": stages,
        "binary_best_random_by_balanced_accuracy": summarize_best(
            binary_df[binary_df["split"] == "random_stratified"],
            "balanced_accuracy",
            ["task"],
        ).to_dict(orient="records"),
        "binary_best_group_source_by_balanced_accuracy": summarize_best(
            binary_df[binary_df["split"] == "group_source"],
            "balanced_accuracy",
            ["task"],
        ).to_dict(orient="records"),
        "failure_type_best_by_balanced_accuracy": summarize_best(
            failure_type_df,
            "balanced_accuracy",
            ["task"],
        ).to_dict(orient="records"),
    }

    (output_dir / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE] Phase 5 meta-controller training complete.")
    print(f"Saved to: {output_dir}")

    print("\n=== Binary failure results ===")
    cols = [
        "task",
        "split",
        "stage",
        "n_train",
        "n_test",
        "feature_dim",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "majority_accuracy",
        "balanced_accuracy_gain_over_majority",
        "f1_failure",
        "recall_failure",
    ]
    available = [c for c in cols if c in binary_df.columns]
    print(binary_df[available].to_string(index=False))

    print("\n=== Failure type among failures only ===")
    cols2 = [
        "task",
        "stage",
        "n_train",
        "n_test",
        "num_classes",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "majority_accuracy",
        "balanced_accuracy_gain_over_majority",
    ]
    available2 = [c for c in cols2 if c in failure_type_df.columns]
    print(failure_type_df[available2].to_string(index=False))

    print("\n=== Within-family binary: response_final summary ===")
    wf_final = within_family_df[within_family_df["stage"] == "response_final"].copy()
    cols3 = [
        "family",
        "stage",
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "majority_accuracy",
        "balanced_accuracy_gain_over_majority",
    ]
    available3 = [c for c in cols3 if c in wf_final.columns]
    print(wf_final[available3].to_string(index=False))

    print("\n=== Leave-one-family-out binary: response_final summary ===")
    logo_final = logo_df[logo_df["stage"] == "response_final"].copy()
    cols4 = [
        "heldout_family",
        "stage",
        "n_test",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "majority_accuracy",
        "balanced_accuracy_gain_over_majority",
    ]
    available4 = [c for c in cols4 if c in logo_final.columns]
    print(logo_final[available4].to_string(index=False))


if __name__ == "__main__":
    main()
