#!/usr/bin/env python3
"""
Phase 2A: Generate model outputs for FailureAtlas-LLM.

Reads:
  data/prompts/failure_prompts_mvp_v2.jsonl

Writes:
  outputs/generations/qwen_7b_mvp_outputs.jsonl

Features:
- resume support
- no unsafe prompt/response printing
- deterministic decoding by default
- JSONL output after every record
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


HIGH_RISK_FAMILIES = {"unsafe_compliance"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("prompt_id")
                if pid:
                    done.add(pid)
            except Exception:
                continue

    return done


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"System: {system_prompt.strip()}\n\n"
        f"User: {user_prompt.strip()}\n\n"
        "Assistant:"
    )


def normalize_generation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gen = dict(cfg)

    # Transformers expects sampling fields only when sampling.
    # Some model repos also define default temperature/top_p/top_k in generation_config,
    # so we remove them from both the explicit kwargs and the model config later.
    if not gen.get("do_sample", False):
        gen.pop("temperature", None)
        gen.pop("top_p", None)
        gen.pop("top_k", None)

    return gen


def load_model_and_tokenizer(model_cfg: Dict[str, Any]):
    model_name = model_cfg["name"]
    torch_dtype_cfg = model_cfg.get("torch_dtype", "auto")

    if torch_dtype_cfg == "auto":
        torch_dtype = "auto"
    elif torch_dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_cfg == "bfloat16":
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

    # Keep deterministic greedy decoding clean when do_sample=False.
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    generation_cfg: Dict[str, Any],
) -> str:
    full_prompt = build_chat_prompt(tokenizer, system_prompt, user_prompt)

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_cfg,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_len:]

    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_generation_qwen.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])

    set_seed(int(cfg.get("runtime", {}).get("seed", 42)))

    prompts = load_jsonl(input_path)
    done_ids = load_done_ids(output_path)

    limit = cfg.get("runtime", {}).get("limit", None)
    if limit is not None:
        prompts = prompts[: int(limit)]

    remaining = [r for r in prompts if r.get("prompt_id") not in done_ids]

    print("[INFO] Phase 2A generation")
    print(f"[INFO] Input prompts: {len(prompts)}")
    print(f"[INFO] Already done: {len(done_ids)}")
    print(f"[INFO] Remaining: {len(remaining)}")
    print(f"[INFO] Output: {output_path}")

    if not remaining:
        print("[DONE] Nothing to generate.")
        return

    model, tokenizer = load_model_and_tokenizer(cfg["model"])

    generation_cfg = normalize_generation_cfg(cfg["generation"])
    system_prompt = cfg.get("system_prompt", "")
    save_every = int(cfg.get("runtime", {}).get("save_every", 25))
    log_every = int(cfg.get("runtime", {}).get("log_every", 25))

    start_all = time.time()

    for i, row in enumerate(tqdm(remaining, desc="Generating"), start=1):
        prompt_id = row["prompt_id"]
        family = row.get("failure_family", "")
        source = row.get("source_dataset", "")

        start = time.time()
        error = None
        response = ""

        try:
            response = generate_one(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=row["prompt"],
                generation_cfg=generation_cfg,
            )
        except RuntimeError as e:
            error = repr(e)
            if "out of memory" in str(e).lower():
                print("[WARN] CUDA OOM. Clearing cache and continuing.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[WARN] Runtime error on prompt_id={prompt_id}: {e}")
        except Exception as e:
            error = repr(e)
            print(f"[WARN] Error on prompt_id={prompt_id}: {e}")

        elapsed = time.time() - start

        out = {
            "prompt_id": prompt_id,
            "failure_family": family,
            "source_dataset": source,
            "source_split": row.get("source_split"),
            "source_index": row.get("source_index"),
            "category": row.get("category"),
            "risk_level": row.get("risk_level"),
            "eval_type": row.get("eval_type"),
            "prompt": row.get("prompt"),
            "expected_behavior": row.get("expected_behavior"),
            "gold_answer": row.get("gold_answer"),
            "metadata": row.get("metadata", {}),
            "model_name": cfg["model"]["name"],
            "generation_config": cfg["generation"],
            "response": response,
            "runtime_seconds": elapsed,
            "error": error,
        }

        append_jsonl(output_path, out)

        if i % log_every == 0:
            # Do not print unsafe prompt/response content.
            if family in HIGH_RISK_FAMILIES:
                print(
                    f"[INFO] {i}/{len(remaining)} done | family={family} | "
                    f"source={source} | last_runtime={elapsed:.2f}s | content_hidden=True"
                )
            else:
                preview = response[:80].replace("\n", " ")
                print(
                    f"[INFO] {i}/{len(remaining)} done | family={family} | "
                    f"source={source} | last_runtime={elapsed:.2f}s | response_preview={preview!r}"
                )

        if i % save_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - start_all
    print(f"[DONE] Generation complete in {total_elapsed / 60:.2f} minutes.")
    print(f"[DONE] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
