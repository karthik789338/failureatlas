#!/usr/bin/env python3
"""
Phase 5B: Prompt-controlled paired evaluation.

Purpose:
- Build matched failure/non-failure pairs within similar source/family/category groups.
- Compare activation-based classifiers against prompt-only TF-IDF classifiers.
- Report pair-level ranking accuracy.

Inputs:
  outputs/features/qwen_7b_mvp/features.npy
  outputs/features/qwen_7b_mvp/metadata.csv
  outputs/judged/qwen_7b_mvp_judged_repaired.jsonl

Outputs:
  outputs/meta_controller/qwen_7b_mvp/phase5b_paired/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_inputs(feature_dir: Path, judged_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    X = np.load(feature_dir / "features.npy").astype("float32")
    meta = pd.read_csv(feature_dir / "metadata.csv")

    judged_rows = load_jsonl(judged_path)
    judged_by_id = {str(r.get("prompt_id")): r for r in judged_rows}

    if len(meta) != X.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != metadata rows {len(meta)}")

    prompts = []
    responses = []
    categories = []
    eval_types = []
    expected_behaviors = []
    judge_reasons = []

    for _, row in meta.iterrows():
        pid = str(row["prompt_id"])
        src = judged_by_id.get(pid, {})

        prompts.append(clean_text(src.get("prompt", "")))
        responses.append(clean_text(src.get("response", "")))
        categories.append(clean_text(src.get("category", "")))
        eval_types.append(clean_text(src.get("eval_type", "")))
        expected_behaviors.append(clean_text(src.get("expected_behavior", "")))
        judge_reasons.append(clean_text(src.get("judge_reason", "")))

    meta = meta.copy()
    meta["prompt"] = prompts
    meta["response"] = responses
    meta["category"] = categories
    meta["eval_type"] = eval_types
    meta["expected_behavior"] = expected_behaviors
    meta["judge_reason"] = judge_reasons

    meta["failure_binary"] = meta["failure_binary"].astype(int)
    meta["failure_family"] = meta["failure_family"].astype(str)
    meta["failure_type"] = meta["failure_type"].astype(str)
    meta["source_dataset"] = meta["source_dataset"].astype(str)
    meta["risk_level"] = meta["risk_level"].astype(str)

    return X, meta


def strict_group_key(row: pd.Series) -> str:
    return "||".join([
        str(row.get("source_dataset", "")),
        str(row.get("failure_family", "")),
        str(row.get("category", "")),
        str(row.get("eval_type", "")),
        str(row.get("risk_level", "")),
    ])


def fallback_group_key(row: pd.Series) -> str:
    return "||".join([
        str(row.get("source_dataset", "")),
        str(row.get("failure_family", "")),
    ])


def make_pairs_for_group(
    group_df: pd.DataFrame,
    min_similarity: float,
    random_state: int,
    max_pairs_per_group: int | None = None,
) -> List[Dict[str, Any]]:
    failures = group_df[group_df["failure_binary"] == 1].copy()
    nonfails = group_df[group_df["failure_binary"] == 0].copy()

    if len(failures) == 0 or len(nonfails) == 0:
        return []

    rng = random.Random(random_state)

    # Keep deterministic order but shuffle failures to avoid source-order artifacts.
    failure_indices = failures.index.tolist()
    rng.shuffle(failure_indices)

    nonfail_indices = nonfails.index.tolist()
    used_nonfail = set()

    texts = group_df["prompt"].fillna("").astype(str).tolist()
    local_indices = group_df.index.tolist()
    pos_by_global = {idx: i for i, idx in enumerate(local_indices)}

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
    )

    try:
        T = vectorizer.fit_transform(texts)
    except Exception:
        return []

    pairs = []

    for fidx in failure_indices:
        candidate_nonfails = [idx for idx in nonfail_indices if idx not in used_nonfail]
        if not candidate_nonfails:
            break

        f_pos = pos_by_global[fidx]
        c_pos = [pos_by_global[idx] for idx in candidate_nonfails]

        sims = cosine_similarity(T[f_pos], T[c_pos]).flatten()
        best_j = int(np.argmax(sims))
        best_sim = float(sims[best_j])
        best_nidx = candidate_nonfails[best_j]

        if best_sim < min_similarity:
            continue

        used_nonfail.add(best_nidx)

        frow = group_df.loc[fidx]
        nrow = group_df.loc[best_nidx]

        pairs.append({
            "pair_id": f"pair_{len(pairs):06d}",
            "failure_prompt_id": str(frow["prompt_id"]),
            "nonfailure_prompt_id": str(nrow["prompt_id"]),
            "source_dataset": str(frow["source_dataset"]),
            "failure_family": str(frow["failure_family"]),
            "category": str(frow.get("category", "")),
            "eval_type": str(frow.get("eval_type", "")),
            "risk_level": str(frow.get("risk_level", "")),
            "failure_type": str(frow["failure_type"]),
            "prompt_similarity": best_sim,
            "group_size": int(len(group_df)),
            "num_failures_in_group": int(len(failures)),
            "num_nonfailures_in_group": int(len(nonfails)),
        })

        if max_pairs_per_group is not None and len(pairs) >= max_pairs_per_group:
            break

    return pairs


def build_matched_pairs(meta: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    pairing_cfg = cfg["pairing"]
    min_sim = float(pairing_cfg.get("min_prompt_similarity", 0.05))
    random_state = int(pairing_cfg.get("random_state", 42))
    max_pairs = pairing_cfg.get("max_pairs_per_group", None)
    if max_pairs is not None:
        max_pairs = int(max_pairs)

    all_pairs = []

    # First attempt: strict grouping.
    meta = meta.copy()
    meta["strict_group"] = meta.apply(strict_group_key, axis=1)
    meta["fallback_group"] = meta.apply(fallback_group_key, axis=1)

    used_failure_ids = set()
    used_nonfailure_ids = set()

    for group, g in meta.groupby("strict_group"):
        pairs = make_pairs_for_group(
            g,
            min_similarity=min_sim,
            random_state=random_state,
            max_pairs_per_group=max_pairs,
        )

        for p in pairs:
            if p["failure_prompt_id"] in used_failure_ids or p["nonfailure_prompt_id"] in used_nonfailure_ids:
                continue
            p["match_level"] = "strict"
            p["group_key"] = group
            used_failure_ids.add(p["failure_prompt_id"])
            used_nonfailure_ids.add(p["nonfailure_prompt_id"])
            all_pairs.append(p)

    # Fallback grouping catches failures that had no strict match.
    if bool(pairing_cfg.get("allow_fallback_group", True)):
        remaining = meta[
            ~meta["prompt_id"].astype(str).isin(used_failure_ids | used_nonfailure_ids)
        ].copy()

        for group, g in remaining.groupby("fallback_group"):
            pairs = make_pairs_for_group(
                g,
                min_similarity=min_sim,
                random_state=random_state + 1,
                max_pairs_per_group=max_pairs,
            )

            for p in pairs:
                if p["failure_prompt_id"] in used_failure_ids or p["nonfailure_prompt_id"] in used_nonfailure_ids:
                    continue
                p["match_level"] = "fallback_source_family"
                p["group_key"] = group
                used_failure_ids.add(p["failure_prompt_id"])
                used_nonfailure_ids.add(p["nonfailure_prompt_id"])
                all_pairs.append(p)

    # Reassign pair ids globally.
    for i, p in enumerate(all_pairs):
        p["pair_id"] = f"pair_{i:06d}"

    return pd.DataFrame(all_pairs)

def expand_pairs_to_rows(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    by_id = meta.set_index("prompt_id", drop=False)

    rows = []

    for _, p in pairs.iterrows():
        for label_name, pid in [
            ("failure", p["failure_prompt_id"]),
            ("nonfailure", p["nonfailure_prompt_id"]),
        ]:
            r = by_id.loc[pid]

            rows.append({
                "pair_id": p["pair_id"],
                "prompt_id": pid,
                "pair_label": label_name,
                "failure_binary": int(r["failure_binary"]),
                "failure_type": str(r["failure_type"]),
                "failure_family": str(r["failure_family"]),
                "source_dataset": str(r["source_dataset"]),
                "category": str(r.get("category", "")),
                "eval_type": str(r.get("eval_type", "")),
                "risk_level": str(r.get("risk_level", "")),
                "match_level": str(p["match_level"]),
                "prompt_similarity": float(p["prompt_similarity"]),
                "prompt": str(r.get("prompt", "")),
                "judge_confidence": float(r.get("judge_confidence", 0.0)),
            })

    return pd.DataFrame(rows)


def get_stage_indices(stage: str, layout: Dict[str, Any]) -> np.ndarray:
    num_layers = int(layout["num_layers"])
    num_positions = int(layout["num_positions"])
    proj_dim = int(layout["projection_dim"])

    pos_map = layout["positions"]
    stage_pos = int(pos_map[stage])

    projected_dim = num_layers * num_positions * proj_dim
    delta_dim = num_layers * proj_dim
    norm_start = projected_dim + delta_dim

    idx = []

    for layer in range(num_layers):
        for pos in range(stage_pos + 1):
            start = (layer * num_positions + pos) * proj_dim
            end = start + proj_dim
            idx.extend(range(start, end))

    if stage == "response_final":
        idx.extend(range(projected_dim, projected_dim + delta_dim))

    for layer in range(num_layers):
        for pos in range(stage_pos + 1):
            norm_idx = norm_start + (layer * num_positions + pos)
            idx.append(norm_idx)

    return np.array(idx, dtype=np.int64)


def make_activation_pipeline(n_train: int, n_features: int, cfg: Dict[str, Any]) -> Pipeline:
    requested = int(cfg["models"].get("pca_components", 100))
    n_pca = min(requested, max(1, n_train - 1), n_features)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=int(cfg["splits"].get("random_state", 42)))),
        ("clf", LogisticRegression(
            max_iter=int(cfg["models"].get("logistic_max_iter", 3000)),
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def make_prompt_pipeline(cfg: Dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=int(cfg["models"].get("tfidf_max_features", 12000)),
            ngram_range=(
                int(cfg["models"].get("tfidf_ngram_min", 1)),
                int(cfg["models"].get("tfidf_ngram_max", 2)),
            ),
        )),
        ("clf", LogisticRegression(
            max_iter=int(cfg["models"].get("logistic_max_iter", 3000)),
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def binary_metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray | None) -> Dict[str, float]:
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

    if prob is not None and len(set(y_true.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out


def pair_ranking_accuracy(test_rows: pd.DataFrame, prob_failure: np.ndarray) -> Dict[str, float]:
    df = test_rows.copy()
    df["prob_failure"] = prob_failure

    correct = 0
    total = 0
    ties = 0

    margins = []

    for pair_id, g in df.groupby("pair_id"):
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

        if pf > pn:
            correct += 1
        elif pf == pn:
            ties += 1

        total += 1

    if total == 0:
        return {
            "pair_ranking_accuracy": None,
            "pair_tie_rate": None,
            "pair_mean_margin": None,
            "num_eval_pairs": 0,
        }

    return {
        "pair_ranking_accuracy": float(correct / total),
        "pair_tie_rate": float(ties / total),
        "pair_mean_margin": float(np.mean(margins)) if margins else None,
        "num_eval_pairs": int(total),
    }


def split_pairs_random(pair_rows: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pair_ids = sorted(pair_rows["pair_id"].unique())

    train_pairs, test_pairs = train_test_split(
        pair_ids,
        test_size=float(cfg["splits"].get("test_size", 0.25)),
        random_state=int(cfg["splits"].get("random_state", 42)),
    )

    train_pairs = set(train_pairs)
    test_pairs = set(test_pairs)

    train_idx = pair_rows.index[pair_rows["pair_id"].isin(train_pairs)].to_numpy()
    test_idx = pair_rows.index[pair_rows["pair_id"].isin(test_pairs)].to_numpy()

    return train_idx, test_idx


def split_pairs_group_source(pair_rows: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pair_meta = pair_rows.drop_duplicates("pair_id").copy()
    pair_ids = pair_meta["pair_id"].to_numpy()
    groups = pair_meta["source_dataset"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(cfg["splits"].get("test_size", 0.25)),
        random_state=int(cfg["splits"].get("random_state", 42)),
    )

    train_pair_idx, test_pair_idx = next(splitter.split(pair_ids, groups=groups))

    train_pairs = set(pair_ids[train_pair_idx])
    test_pairs = set(pair_ids[test_pair_idx])

    train_idx = pair_rows.index[pair_rows["pair_id"].isin(train_pairs)].to_numpy()
    test_idx = pair_rows.index[pair_rows["pair_id"].isin(test_pairs)].to_numpy()

    return train_idx, test_idx


def get_positive_proba(pipe: Pipeline, X_test: Any) -> np.ndarray:
    probs = pipe.predict_proba(X_test)
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return probs[:, pos_idx]

def run_prompt_only_eval(pair_rows: pd.DataFrame, cfg: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    if split_name == "paired_random":
        train_idx, test_idx = split_pairs_random(pair_rows, cfg)
    elif split_name == "paired_group_source":
        train_idx, test_idx = split_pairs_group_source(pair_rows, cfg)
    else:
        raise ValueError(split_name)

    train = pair_rows.loc[train_idx].copy()
    test = pair_rows.loc[test_idx].copy()

    y_train = train["failure_binary"].astype(int).to_numpy()
    y_test = test["failure_binary"].astype(int).to_numpy()

    pipe = make_prompt_pipeline(cfg)
    pipe.fit(train["prompt"].astype(str).tolist(), y_train)

    pred = pipe.predict(test["prompt"].astype(str).tolist())
    prob = get_positive_proba(pipe, test["prompt"].astype(str).tolist())

    out = {
        "model": "prompt_tfidf",
        "split": split_name,
        "stage": "prompt_text",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "num_train_pairs": int(train["pair_id"].nunique()),
        "num_test_pairs": int(test["pair_id"].nunique()),
        **binary_metrics(y_test, pred, prob),
        **pair_ranking_accuracy(test, prob),
    }

    return out


def run_activation_eval(
    X: np.ndarray,
    meta: pd.DataFrame,
    pair_rows: pd.DataFrame,
    cfg: Dict[str, Any],
    stage: str,
    split_name: str,
) -> Dict[str, Any]:
    id_to_pos = {str(pid): i for i, pid in enumerate(meta["prompt_id"].astype(str).tolist())}
    row_positions = np.array([id_to_pos[str(pid)] for pid in pair_rows["prompt_id"].astype(str)], dtype=np.int64)

    stage_idx = get_stage_indices(stage, cfg["feature_layout"])
    X_stage = X[row_positions][:, stage_idx]

    if split_name == "paired_random":
        train_idx, test_idx = split_pairs_random(pair_rows, cfg)
    elif split_name == "paired_group_source":
        train_idx, test_idx = split_pairs_group_source(pair_rows, cfg)
    else:
        raise ValueError(split_name)

    y = pair_rows["failure_binary"].astype(int).to_numpy()

    X_train, X_test = X_stage[train_idx], X_stage[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    test_rows = pair_rows.loc[test_idx].copy()

    pipe = make_activation_pipeline(len(y_train), X_train.shape[1], cfg)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = get_positive_proba(pipe, X_test)

    out = {
        "model": "activation_logreg",
        "split": split_name,
        "stage": stage,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "num_train_pairs": int(pair_rows.loc[train_idx, "pair_id"].nunique()),
        "num_test_pairs": int(pair_rows.loc[test_idx, "pair_id"].nunique()),
        "feature_dim": int(X_train.shape[1]),
        **binary_metrics(y_test, pred, prob),
        **pair_ranking_accuracy(test_rows, prob),
    }

    return out


def summarize_pairs(pairs: pd.DataFrame, pair_rows: pd.DataFrame) -> Dict[str, Any]:
    return {
        "num_pairs": int(len(pairs)),
        "num_rows": int(len(pair_rows)),
        "match_level_counts": pairs["match_level"].value_counts().to_dict(),
        "source_dataset_counts": pairs["source_dataset"].value_counts().to_dict(),
        "failure_family_counts": pairs["failure_family"].value_counts().to_dict(),
        "failure_type_counts": pairs["failure_type"].value_counts().to_dict(),
        "prompt_similarity_mean": float(pairs["prompt_similarity"].mean()) if len(pairs) else None,
        "prompt_similarity_median": float(pairs["prompt_similarity"].median()) if len(pairs) else None,
        "prompt_similarity_min": float(pairs["prompt_similarity"].min()) if len(pairs) else None,
        "prompt_similarity_max": float(pairs["prompt_similarity"].max()) if len(pairs) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5b_paired_control_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    feature_dir = Path(cfg["feature_dir"])
    judged_path = Path(cfg["judged_path"])
    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)

    print("[INFO] Phase 5B prompt-controlled paired evaluation")
    print(f"[INFO] Feature dir: {feature_dir}")
    print(f"[INFO] Judged path: {judged_path}")
    print(f"[INFO] Output dir: {output_dir}")

    X, meta = load_inputs(feature_dir, judged_path)
    print(f"[INFO] Loaded X: {X.shape}")
    print(f"[INFO] Loaded metadata: {meta.shape}")

    print("[INFO] Building matched pairs")
    pairs = build_matched_pairs(meta, cfg)
    if pairs.empty:
        raise SystemExit("No matched pairs found. Lower min_prompt_similarity or enable fallback grouping.")

    pair_rows = expand_pairs_to_rows(pairs, meta)

    # Safe outputs: no prompt/response text in pair index.
    safe_pairs = pairs.copy()
    safe_pairs.to_csv(output_dir / "matched_pairs.csv", index=False)

    safe_pair_rows = pair_rows.drop(columns=["prompt"], errors="ignore")
    safe_pair_rows.to_csv(output_dir / "matched_pair_rows_safe.csv", index=False)

    # Internal file includes prompt text for debugging; do not share publicly.
    pair_rows.to_csv(output_dir / "matched_pair_rows_internal.csv", index=False)

    pair_stats = summarize_pairs(pairs, pair_rows)
    (output_dir / "pair_stats.json").write_text(json.dumps(pair_stats, indent=2), encoding="utf-8")

    print("[INFO] Pair stats:")
    print(json.dumps(pair_stats, indent=2))

    results = []

    # Prompt-only baselines.
    for split_name in ["paired_random", "paired_group_source"]:
        print(f"[INFO] Running prompt-only baseline split={split_name}")
        try:
            results.append(run_prompt_only_eval(pair_rows, cfg, split_name))
        except Exception as e:
            print(f"[WARN] Prompt-only eval failed for {split_name}: {e}")

    # Activation stage models.
    for split_name in ["paired_random", "paired_group_source"]:
        for stage in cfg.get("stages", ["prompt_final", "response_final"]):
            print(f"[INFO] Running activation eval split={split_name}, stage={stage}")
            try:
                results.append(run_activation_eval(X, meta, pair_rows, cfg, stage, split_name))
            except Exception as e:
                print(f"[WARN] Activation eval failed for split={split_name}, stage={stage}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "paired_control_results.csv", index=False)

    summary = {
        "pair_stats": pair_stats,
        "best_activation_random_balanced_accuracy": None,
        "best_activation_group_source_balanced_accuracy": None,
        "best_prompt_random_balanced_accuracy": None,
        "best_prompt_group_source_balanced_accuracy": None,
    }

    if not results_df.empty:
        act_random = results_df[(results_df["model"] == "activation_logreg") & (results_df["split"] == "paired_random")]
        act_group = results_df[(results_df["model"] == "activation_logreg") & (results_df["split"] == "paired_group_source")]
        prompt_random = results_df[(results_df["model"] == "prompt_tfidf") & (results_df["split"] == "paired_random")]
        prompt_group = results_df[(results_df["model"] == "prompt_tfidf") & (results_df["split"] == "paired_group_source")]

        if len(act_random):
            summary["best_activation_random_balanced_accuracy"] = act_random.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()
        if len(act_group):
            summary["best_activation_group_source_balanced_accuracy"] = act_group.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()
        if len(prompt_random):
            summary["best_prompt_random_balanced_accuracy"] = prompt_random.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()
        if len(prompt_group):
            summary["best_prompt_group_source_balanced_accuracy"] = prompt_group.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict()

    (output_dir / "phase5b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE] Phase 5B complete.")
    print(f"Results: {output_dir / 'paired_control_results.csv'}")
    print(f"Summary: {output_dir / 'phase5b_summary.json'}")

    print("\n=== Paired control results ===")
    cols = [
        "model",
        "split",
        "stage",
        "n_train",
        "n_test",
        "num_train_pairs",
        "num_test_pairs",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "f1_failure",
        "recall_failure",
        "pair_ranking_accuracy",
        "pair_mean_margin",
    ]
    available = [c for c in cols if c in results_df.columns]
    print(results_df[available].to_string(index=False))


if __name__ == "__main__":
    main()
