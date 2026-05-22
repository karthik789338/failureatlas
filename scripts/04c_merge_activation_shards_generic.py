#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    shard_dir = feature_dir / "shards"

    shards = sorted(shard_dir.glob("activation_shard_*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {shard_dir}")

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

    np.save(feature_dir / "features.npy", X_all)
    meta.to_csv(feature_dir / "metadata.csv", index=False)

    stats = {
        "num_shards": len(shards),
        "num_rows": int(X_all.shape[0]),
        "feature_dim": int(X_all.shape[1]),
        "failure_binary_counts": meta["failure_binary"].value_counts().to_dict(),
        "failure_family_counts": meta["failure_family"].value_counts().to_dict(),
        "failure_type_counts": meta["failure_type"].value_counts().to_dict(),
    }

    (feature_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Merged features:", X_all.shape)
    print("Metadata rows:", len(meta))
    print("Saved:", feature_dir / "features.npy")
    print("Saved:", feature_dir / "metadata.csv")
    print("Saved:", feature_dir / "feature_stats.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
