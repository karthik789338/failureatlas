#!/usr/bin/env python3
"""
Phase 3: Extract compact activation trajectory features.

Input:
  outputs/judged/qwen_7b_mvp_judged_repaired.jsonl

Output:
  outputs/features/qwen_7b_mvp/shards/activation_shard_XXXXX.npz

Design:
- batch size 1
- selected layers only
- selected positions only
- random projection per hidden vector
- resumable through shard prompt_ids
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_for_csv(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace('"', "'")
    return text


def scan_done_ids(feature_dir: Path) -> set:
    done = set()
    shard_dir = feature_dir / "shards"
    if not shard_dir.exists():
        return done

    for path in sorted(shard_dir.glob("activation_shard_*.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            for pid in data["prompt_ids"]:
                done.add(str(pid))
        except Exception:
            continue

    return done


def next_shard_index(feature_dir: Path) -> int:
    shard_dir = feature_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(shard_dir.glob("activation_shard_*.npz"))
    if not existing:
        return 0

    max_idx = -1
    for p in existing:
        m = re.search(r"activation_shard_(\d+)\.npz$", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    return max_idx + 1


def load_model_and_tokenizer(model_cfg: Dict[str, Any]):
    model_name = model_cfg["name"]
    dtype_cfg = model_cfg.get("torch_dtype", "auto")

    if dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif dtype_cfg == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    model.eval()
    return model, tokenizer


def build_chat_texts(tokenizer, system_prompt: str, user_prompt: str, assistant_response: str) -> Tuple[str, str]:
    prefix_messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": str(user_prompt).strip()},
    ]

    full_messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": str(user_prompt).strip()},
        {"role": "assistant", "content": str(assistant_response).strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prefix_text, full_text

    prefix_text = (
        f"System: {system_prompt.strip()}\n\n"
        f"User: {str(user_prompt).strip()}\n\n"
        "Assistant:"
    )
    full_text = prefix_text + " " + str(assistant_response).strip()
    return prefix_text, full_text


def get_selected_layer_indices(model) -> List[int]:
    # hidden_states has embedding layer at index 0 and final transformer layer at index num_hidden_layers.
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    candidates = [
        0,
        max(1, n_layers // 4),
        max(1, n_layers // 2),
        max(1, (3 * n_layers) // 4),
        n_layers,
    ]
    return sorted(set(candidates))


def get_positions(prompt_len: int, seq_len: int) -> List[int]:
    last_idx = max(0, seq_len - 1)

    prompt_final = max(0, min(prompt_len - 1, last_idx))

    resp_start = min(max(prompt_len, 0), last_idx)
    resp_end = last_idx

    if resp_end <= resp_start:
        # Fallback for very long prompts where response was truncated.
        resp_start = max(0, int(seq_len * 0.75))
        resp_end = last_idx

    def frac_pos(frac: float) -> int:
        if resp_end <= resp_start:
            return resp_end
        return int(round(resp_start + frac * (resp_end - resp_start)))

    return [
        prompt_final,
        frac_pos(0.25),
        frac_pos(0.50),
        frac_pos(0.75),
        resp_end,
    ]

def make_projection(hidden_dim: int, proj_dim: int, seed: int, device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(hidden_dim)
    matrix = rng.normal(0.0, scale, size=(hidden_dim, proj_dim)).astype("float32")
    return torch.tensor(matrix, dtype=torch.float32, device=device)


@torch.inference_mode()
def extract_one_feature(
    model,
    tokenizer,
    row: Dict[str, Any],
    system_prompt: str,
    max_length: int,
    projection: torch.Tensor | None,
    projection_dim: int,
    projection_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any], torch.Tensor]:
    prefix_text, full_text = build_chat_texts(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=row.get("prompt", ""),
        assistant_response=row.get("response", ""),
    )

    prefix_ids = tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    seq_len = int(input_ids.shape[-1])
    prompt_len = int(prefix_ids.shape[-1])
    prompt_len = min(prompt_len, seq_len)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    layer_indices = get_selected_layer_indices(model)
    positions = get_positions(prompt_len=prompt_len, seq_len=seq_len)

    selected_vecs = []

    for layer_idx in layer_indices:
        h = hidden_states[layer_idx][0]  # [seq, hidden]
        for pos in positions:
            selected_vecs.append(h[pos].float())

    vecs = torch.stack(selected_vecs, dim=0)  # [num_layer_pos, hidden_dim]
    hidden_dim = int(vecs.shape[-1])

    if projection is None:
        projection = make_projection(
            hidden_dim=hidden_dim,
            proj_dim=projection_dim,
            seed=projection_seed,
            device=vecs.device,
        )

    norms = torch.linalg.vector_norm(vecs, dim=1)
    vecs_normed = vecs / norms.clamp(min=1e-6).unsqueeze(1)

    projected = vecs_normed @ projection  # [num_layer_pos, projection_dim]

    # Add response shift features: response_final minus prompt_final for each selected layer.
    delta_vecs = []
    num_positions = len(positions)

    for li in range(len(layer_indices)):
        base_idx = li * num_positions
        prompt_vec = vecs[base_idx + 0]
        final_vec = vecs[base_idx + 4]
        delta = final_vec - prompt_vec
        delta_norm = torch.linalg.vector_norm(delta).clamp(min=1e-6)
        delta_vecs.append(delta / delta_norm)

    delta_vecs = torch.stack(delta_vecs, dim=0)
    delta_projected = delta_vecs @ projection

    norm_features = (norms / math.sqrt(hidden_dim)).float()

    feature = torch.cat(
        [
            projected.flatten(),
            delta_projected.flatten(),
            norm_features.flatten(),
        ],
        dim=0,
    )

    feature_np = feature.detach().cpu().numpy().astype("float32")

    meta = {
        "prompt_id": row.get("prompt_id"),
        "failure_family": row.get("failure_family"),
        "failure_binary": row.get("failure_binary"),
        "failure_type": row.get("failure_type"),
        "severity": row.get("severity"),
        "judge_confidence": row.get("judge_confidence"),
        "source_dataset": row.get("source_dataset"),
        "risk_level": row.get("risk_level"),
        "eval_type": row.get("eval_type"),
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "selected_layers": ",".join(str(x) for x in layer_indices),
        "selected_positions": ",".join(str(x) for x in positions),
        "feature_dim": int(feature_np.shape[0]),
    }

    return feature_np, meta, projection


def save_shard(
    feature_dir: Path,
    shard_index: int,
    features: List[np.ndarray],
    metas: List[Dict[str, Any]],
) -> None:
    shard_dir = feature_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    X = np.stack(features, axis=0).astype("float32")

    prompt_ids = np.array([str(m["prompt_id"]) for m in metas], dtype=str)
    failure_family = np.array([str(m["failure_family"]) for m in metas], dtype=str)
    failure_type = np.array([str(m["failure_type"]) for m in metas], dtype=str)
    source_dataset = np.array([str(m["source_dataset"]) for m in metas], dtype=str)
    severity = np.array([str(m["severity"]) for m in metas], dtype=str)
    risk_level = np.array([str(m["risk_level"]) for m in metas], dtype=str)

    failure_binary = np.array([int(m["failure_binary"]) for m in metas], dtype=np.int64)
    judge_confidence = np.array([float(m["judge_confidence"]) for m in metas], dtype=np.float32)
    seq_len = np.array([int(m["seq_len"]) for m in metas], dtype=np.int64)
    prompt_len = np.array([int(m["prompt_len"]) for m in metas], dtype=np.int64)

    shard_path = shard_dir / f"activation_shard_{shard_index:05d}.npz"

    np.savez_compressed(
        shard_path,
        features=X,
        prompt_ids=prompt_ids,
        failure_family=failure_family,
        failure_type=failure_type,
        failure_binary=failure_binary,
        source_dataset=source_dataset,
        severity=severity,
        risk_level=risk_level,
        judge_confidence=judge_confidence,
        seq_len=seq_len,
        prompt_len=prompt_len,
    )

    meta_path = shard_dir / f"activation_shard_{shard_index:05d}.metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved shard {shard_index:05d}: {X.shape} -> {shard_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_activation_qwen.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    input_path = Path(cfg["input_path"])
    feature_dir = Path(cfg["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    rows = load_jsonl(input_path)

    limit = cfg.get("runtime", {}).get("limit", None)
    if limit is not None:
        rows = rows[: int(limit)]

    done_ids = scan_done_ids(feature_dir)
    remaining = [r for r in rows if str(r.get("prompt_id")) not in done_ids]

    print("[INFO] Phase 3 activation extraction")
    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Already extracted: {len(done_ids)}")
    print(f"[INFO] Remaining: {len(remaining)}")
    print(f"[INFO] Feature dir: {feature_dir}")

    if not remaining:
        print("[DONE] Nothing to extract.")
        return

    model, tokenizer = load_model_and_tokenizer(cfg["model"])

    system_prompt = cfg.get("system_prompt", "")
    max_length = int(cfg.get("extraction", {}).get("max_length", 4096))
    projection_dim = int(cfg.get("extraction", {}).get("projection_dim_per_vector", 64))

    shard_size = int(cfg.get("runtime", {}).get("shard_size", 100))
    log_every = int(cfg.get("runtime", {}).get("log_every", 25))

    shard_index = next_shard_index(feature_dir)

    projection = None
    shard_features: List[np.ndarray] = []
    shard_metas: List[Dict[str, Any]] = []

    start_all = time.time()
    errors = 0

    feature_config = {
        "input_path": str(input_path),
        "model_name": cfg["model"]["name"],
        "max_length": max_length,
        "projection_dim_per_vector": projection_dim,
        "projection_seed": seed,
        "feature_design": "selected layer-position normalized random projections + response shift projections + norm features",
    }

    (feature_dir / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2),
        encoding="utf-8",
    )

    for i, row in enumerate(tqdm(remaining, desc="Extracting activations"), start=1):
        prompt_id = row.get("prompt_id")

        try:
            feature, meta, projection = extract_one_feature(
                model=model,
                tokenizer=tokenizer,
                row=row,
                system_prompt=system_prompt,
                max_length=max_length,
                projection=projection,
                projection_dim=projection_dim,
                projection_seed=seed,
            )

            shard_features.append(feature)
            shard_metas.append(meta)

        except RuntimeError as e:
            errors += 1
            print(f"[WARN] Runtime error prompt_id={prompt_id}: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        except Exception as e:
            errors += 1
            print(f"[WARN] Error prompt_id={prompt_id}: {e}")
            continue

        if i % log_every == 0:
            print(
                f"[INFO] {i}/{len(remaining)} extracted | "
                f"current_shard_items={len(shard_features)} | errors={errors}"
            )

        if len(shard_features) >= shard_size:
            save_shard(feature_dir, shard_index, shard_features, shard_metas)
            shard_index += 1
            shard_features = []
            shard_metas = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if shard_features:
        save_shard(feature_dir, shard_index, shard_features, shard_metas)

    total_elapsed = time.time() - start_all

    stats = {
        "input_rows": len(rows),
        "already_done_at_start": len(done_ids),
        "attempted_remaining": len(remaining),
        "errors": errors,
        "runtime_minutes": total_elapsed / 60.0,
    }

    (feature_dir / "extraction_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] Activation extraction complete in {total_elapsed / 60:.2f} minutes.")
    print(f"[DONE] Errors: {errors}")
    print(f"[DONE] Feature dir: {feature_dir}")


if __name__ == "__main__":
    main()
