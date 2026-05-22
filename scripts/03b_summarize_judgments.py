#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict

INPUT = Path("outputs/judged/qwen_7b_mvp_judged.jsonl")
OUT_DIR = Path("outputs/judged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = [json.loads(x) for x in INPUT.open(encoding="utf-8") if x.strip()]

print("=" * 80)
print("FailureAtlas Phase 2B Judgment Summary")
print("=" * 80)
print("Total rows:", len(rows))
print("Parse errors:", sum(1 for r in rows if r.get("judge_parse_error")))
print("Judge runtime errors:", sum(1 for r in rows if r.get("judge_error")))
print("Missing failure_binary:", sum(1 for r in rows if r.get("failure_binary") is None))

print("\nBy original failure family:")
for k, v in Counter(r.get("failure_family") for r in rows).most_common():
    print(f"  {k}: {v}")

print("\nBy judged failure_binary:")
for k, v in Counter(r.get("failure_binary") for r in rows).most_common():
    print(f"  {k}: {v}")

print("\nBy judged failure_type:")
for k, v in Counter(r.get("failure_type") for r in rows).most_common():
    print(f"  {k}: {v}")

print("\nBy severity:")
for k, v in Counter(r.get("severity") for r in rows).most_common():
    print(f"  {k}: {v}")

print("\nFailure rate by original family:")
family_counts = defaultdict(int)
family_fails = defaultdict(int)
family_conf = defaultdict(list)

for r in rows:
    fam = r.get("failure_family")
    family_counts[fam] += 1
    if r.get("failure_binary") == 1:
        family_fails[fam] += 1
    if isinstance(r.get("judge_confidence"), (int, float)):
        family_conf[fam].append(float(r.get("judge_confidence")))

for fam in sorted(family_counts):
    n = family_counts[fam]
    f = family_fails[fam]
    confs = family_conf[fam]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"  {fam}: {f}/{n} = {f/n:.3f}, avg_conf={avg_conf:.3f}")

print("\nConfusion-style table: original_family -> judged_failure_type")
matrix = defaultdict(Counter)
for r in rows:
    matrix[r.get("failure_family")][r.get("failure_type")] += 1

for fam in sorted(matrix):
    print(f"\n[{fam}]")
    for typ, cnt in matrix[fam].most_common():
        print(f"  {typ}: {cnt}")

# Save compact CSV without prompt/response to avoid exposing unsafe content.
csv_path = OUT_DIR / "qwen_7b_mvp_judged_summary_safe.csv"
with csv_path.open("w", encoding="utf-8") as f:
    f.write("prompt_id,original_family,source_dataset,risk_level,failure_binary,failure_type,severity,judge_confidence,judge_parse_error,judge_error,judge_reason\n")
    for r in rows:
        reason = str(r.get("judge_reason", "")).replace('"', "'").replace("\n", " ")
        f.write(
            f'{r.get("prompt_id")},'
            f'{r.get("failure_family")},'
            f'{r.get("source_dataset")},'
            f'{r.get("risk_level")},'
            f'{r.get("failure_binary")},'
            f'{r.get("failure_type")},'
            f'{r.get("severity")},'
            f'{r.get("judge_confidence")},'
            f'{r.get("judge_parse_error")},'
            f'{bool(r.get("judge_error"))},'
            f'"{reason}"\n'
        )

print("\nSaved safe summary CSV:", csv_path)
