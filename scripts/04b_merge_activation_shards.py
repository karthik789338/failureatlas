#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_DIR = Path("outputs/features/qwen_7b_mvp")
SHARD_DIR = FEATURE_DIR / "shards"

shards = sorted(SHARD_DIR.glob("activation_shard_*.npz"))
if not shards:
    raise SystemExit(f"No shards found in {SHARD_DIR}")

features = []
meta_rows = []

for shard in shards:
    d = np.load(shard, allow_pickle=False)
    X = d["features"]
    features.append(X)

    n = X.shape[0]
    for i in range(n):
        meta_rows.append({
            "prompt_id": str(d["prompt_ids"][i]),
            "failure_family": str(d["failure_family"][i]),
            "failure_type": str(d["failure_type"][i]),
            "failure_binary": int(d["failure_binary"][i]),
            "source_dataset": str(d["source_dataset"][i]),
            "severity": str(d["severity"][i]),
            "risk_level": str(d["risk_level"][i]),
            "judge_confidence": float(d["judge_confidence"][i]),
            "seq_len": int(d["seq_len"][i]),
            "prompt_len": int(d["prompt_len"][i]),
            "shard": shard.name,
        })

X_all = np.concatenate(features, axis=0).astype("float32")
meta = pd.DataFrame(meta_rows)

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

np.save(FEATURE_DIR / "features.npy", X_all)
meta.to_csv(FEATURE_DIR / "metadata.csv", index=False)

stats = {
    "num_shards": len(shards),
    "num_rows": int(X_all.shape[0]),
    "feature_dim": int(X_all.shape[1]),
    "failure_binary_counts": meta["failure_binary"].value_counts().to_dict(),
    "failure_family_counts": meta["failure_family"].value_counts().to_dict(),
    "failure_type_counts": meta["failure_type"].value_counts().to_dict(),
}

(FEATURE_DIR / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

print("Merged features:", X_all.shape)
print("Metadata rows:", len(meta))
print("Saved:", FEATURE_DIR / "features.npy")
print("Saved:", FEATURE_DIR / "metadata.csv")
print("Saved:", FEATURE_DIR / "feature_stats.json")
print(json.dumps(stats, indent=2))
