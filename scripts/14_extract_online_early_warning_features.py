#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_generation_cfg(cfg):
    gen = dict(cfg)
    if not gen.get("do_sample", False):
        gen.pop("temperature", None)
        gen.pop("top_p", None)
        gen.pop("top_k", None)
    return gen


def load_model_and_tokenizer(model_cfg):
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


def build_prompt_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": str(user_prompt).strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {system_prompt.strip()}\n\nUser: {str(user_prompt).strip()}\n\nAssistant:"


def selected_layer_indices(model) -> List[int]:
    n = int(getattr(model.config, "num_hidden_layers"))
    return sorted(set([0, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n]))


def make_projection(hidden_dim: int, proj_dim: int, seed: int, device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(hidden_dim)
    matrix = rng.normal(0.0, scale, size=(hidden_dim, proj_dim)).astype("float32")
    return torch.tensor(matrix, dtype=torch.float32, device=device)


@torch.inference_mode()
def extract_online_feature(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
    projection: torch.Tensor | None,
    proj_dim: int,
    seed: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hs = outputs.hidden_states
    layers = selected_layer_indices(model)

    seq_len = int(input_ids.shape[-1])
    prompt_pos = max(0, min(prompt_len - 1, seq_len - 1))
    current_pos = seq_len - 1

    vecs = []
    for li in layers:
        h = hs[li][0]
        vecs.append(h[prompt_pos].float())
        vecs.append(h[current_pos].float())

    vecs = torch.stack(vecs, dim=0)
    hidden_dim = int(vecs.shape[-1])

    if projection is None:
        projection = make_projection(hidden_dim, proj_dim, seed, vecs.device)

    norms = torch.linalg.vector_norm(vecs, dim=1)
    vecs_normed = vecs / norms.clamp(min=1e-6).unsqueeze(1)
    projected = vecs_normed @ projection

    deltas = []
    for i in range(0, len(vecs), 2):
        delta = vecs[i + 1] - vecs[i]
        delta = delta / torch.linalg.vector_norm(delta).clamp(min=1e-6)
        deltas.append(delta)

    deltas = torch.stack(deltas, dim=0)
    delta_projected = deltas @ projection

    norm_features = (norms / math.sqrt(hidden_dim)).float()

    feat = torch.cat(
        [
            projected.flatten(),
            delta_projected.flatten(),
            norm_features.flatten(),
        ],
        dim=0,
    )

    return feat.detach().cpu().numpy().astype("float32"), projection


@torch.inference_mode()
def generate_sequence(model, tokenizer, prompt_ids: torch.Tensor, gen_cfg: Dict[str, Any]) -> torch.Tensor:
    prompt_ids = prompt_ids.to(model.device)

    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids, device=model.device),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_cfg,
    )

    return out.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase7d_qwen_online_features.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg["runtime"].get("seed", 42)))

    judged_path = Path(cfg["judged_path"])
    pair_rows_path = Path(cfg["pair_rows_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "online_features.npy"
    metadata_path = output_dir / "online_metadata.csv"
    stats_path = output_dir / "online_feature_stats.json"

    judged = load_jsonl(judged_path)
    by_id = {str(r["prompt_id"]): r for r in judged}

    pair_rows = pd.read_csv(pair_rows_path)
    pair_rows["prompt_id"] = pair_rows["prompt_id"].astype(str)

    limit = cfg["runtime"].get("limit", None)
    if limit is not None:
        pair_rows = pair_rows.head(int(limit))

    print("[INFO] Phase 7D online feature extraction")
    print(f"[INFO] Pair rows: {len(pair_rows)}")
    print(f"[INFO] Output dir: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    gen_cfg = normalize_generation_cfg(cfg["generation"])

    checkpoints = [int(x) for x in cfg["checkpoints"]]
    max_input_length = int(cfg["extraction"].get("max_input_length", 4096))
    proj_dim = int(cfg["extraction"].get("projection_dim", 64))
    proj_seed = int(cfg["extraction"].get("seed", 42))
    system_prompt = cfg["system_prompt"]

    projection = None
    features = []
    meta_rows = []

    log_every = int(cfg["runtime"].get("log_every", 25))

    for i, row in enumerate(tqdm(pair_rows.itertuples(index=False), total=len(pair_rows)), start=1):
        pid = str(row.prompt_id)
        src = by_id.get(pid)
        if src is None:
            print(f"[WARN] Missing judged row for prompt_id={pid}")
            continue

        prompt_text = build_prompt_text(tokenizer, system_prompt, src.get("prompt", ""))

        toks = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )

        prompt_ids = toks["input_ids"]
        prompt_len = int(prompt_ids.shape[-1])

        try:
            full_ids = generate_sequence(model, tokenizer, prompt_ids, gen_cfg)
        except RuntimeError as e:
            print(f"[WARN] Generation failed prompt_id={pid}: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        total_len = int(full_ids.shape[-1])
        generated_len = max(0, total_len - prompt_len)

        for ckpt in checkpoints:
            use_gen = min(ckpt, generated_len)
            prefix_len = prompt_len + use_gen
            prefix_len = max(1, min(prefix_len, total_len))

            prefix_ids = full_ids[:, :prefix_len]

            try:
                feat, projection = extract_online_feature(
                    model=model,
                    input_ids=prefix_ids,
                    prompt_len=prompt_len,
                    projection=projection,
                    proj_dim=proj_dim,
                    seed=proj_seed,
                )
            except RuntimeError as e:
                print(f"[WARN] Feature extraction failed prompt_id={pid}, ckpt={ckpt}: {e}")
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            features.append(feat)
            meta_rows.append({
                "prompt_id": pid,
                "pair_id": getattr(row, "pair_id"),
                "pair_label": getattr(row, "pair_label"),
                "checkpoint_tokens": ckpt,
                "actual_generated_tokens_used": use_gen,
                "generated_len_total": generated_len,
                "failure_binary": int(getattr(row, "failure_binary")),
                "failure_type": str(getattr(row, "failure_type")),
                "failure_family": str(getattr(row, "failure_family")),
                "source_dataset": str(getattr(row, "source_dataset")),
                "match_level": str(getattr(row, "match_level")),
                "prompt_similarity": float(getattr(row, "prompt_similarity")),
                "seq_len_used": int(prefix_len),
            })

        if i % log_every == 0:
            print(f"[INFO] {i}/{len(pair_rows)} prompts processed | feature rows={len(features)}")

        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    X = np.stack(features, axis=0).astype("float32")
    meta = pd.DataFrame(meta_rows)

    np.save(feature_path, X)
    meta.to_csv(metadata_path, index=False)

    stats = {
        "num_prompt_rows": int(len(pair_rows)),
        "num_feature_rows": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "checkpoints": checkpoints,
        "feature_mean": float(X.mean()),
        "feature_std": float(X.std()),
        "failure_binary_counts": meta["failure_binary"].value_counts().to_dict(),
    }

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[DONE] Online feature extraction complete")
    print("Features:", X.shape)
    print("Metadata:", meta.shape)
    print("Saved:", feature_path)
    print("Saved:", metadata_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
