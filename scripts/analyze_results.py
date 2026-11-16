"""Validate public episode evidence and reproduce aggregate / paired results.

Standard library only. Run from any directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ("pi05_base", "800", "1000", "1250", "1500", "1750", "2000")
EPISODE_FIELDS = ("checkpoint", "split", "task_id", "initial_state_id", "success", "policy_steps")


def load_rows(root=ROOT):
    with (root / "results/episodes.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
            raise ValueError("Unexpected episode columns")
        rows = list(reader)
    for row in rows:
        for key in EPISODE_FIELDS[2:]:
            row[key] = int(row[key])
    return rows


def validate_rows(rows, protocol):
    seen = set()
    groups = defaultdict(list)
    states = {x["task_index"]: x for x in protocol["spatial_states"]}
    for row in rows:
        checkpoint, split = row["checkpoint"], row["split"]
        key = (checkpoint, split, row["task_id"], row["initial_state_id"])
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}")
        seen.add(key)
        if checkpoint not in CHECKPOINTS or split not in ("development", "held-out"):
            raise ValueError(f"Unknown checkpoint or split: {key}")
        if row["success"] not in (0, 1):
            raise ValueError(f"Success must be binary: {key}")
        if not 1 <= row["policy_steps"] <= protocol["max_policy_steps"]:
            raise ValueError(f"Invalid policy-step count: {key}")
        if row["task_id"] not in states:
            raise ValueError(f"Unknown task: {key}")
        field = "dev" if split == "development" else "formal"
        if row["initial_state_id"] not in states[row["task_id"]][field]:
            raise ValueError(f"Initial state outside the frozen split: {key}")
        groups[(checkpoint, split)].append(row)
    for task in states.values():
        if set(task["dev"]) & set(task["formal"]):
            raise ValueError("Development and held-out states overlap")
        if len(task["dev"]) != 10 or len(task["formal"]) != 40:
            raise ValueError("Protocol state counts differ")
        if set(task["dev"]) | set(task["formal"]) != set(range(50)):
            raise ValueError("Protocol must partition all 50 initial states")
    for group, records in groups.items():
        field = "dev" if group[1] == "development" else "formal"
        expected = {(task, initial) for task, value in states.items() for initial in value[field]}
        actual = {(row["task_id"], row["initial_state_id"]) for row in records}
        if actual != expected:
            raise ValueError(f"Incomplete split: {group}: {len(actual)}/{len(expected)}")
    return groups


def per_task(rows):
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        value = counts[(row["checkpoint"], row["split"], row["task_id"])]
        value[0] += row["success"]
        value[1] += 1
    return dict(counts)


def validate_repository(root=ROOT):
    protocol = json.loads((root / "configs/evaluation.json").read_text())
    metadata = json.loads((root / "results/provenance.json").read_text())
    rows = load_rows(root)
    groups = validate_rows(rows, protocol)
    expected_groups = {(x["checkpoint"], x["split"]) for x in metadata["groups"]}
    if set(groups) != expected_groups or len(rows) != metadata["published_episodes"]:
        raise ValueError("Published coverage differs from the provenance manifest")
    digest = hashlib.sha256((root / "results/episodes.csv").read_bytes()).hexdigest()
    if digest != metadata["episodes_csv_sha256"]:
        raise ValueError("Episode CSV checksum differs from the provenance manifest")
    with (root / "results/per_task.csv").open(newline="") as stream:
        aggregate_rows = list(csv.DictReader(stream))
    aggregate = {}
    for row in aggregate_rows:
        key = (row["checkpoint"], row["split"], int(row["task_id"]))
        if key in aggregate:
            raise ValueError("Duplicate per-task aggregate")
        aggregate[key] = [int(row["successes"]), int(row["episodes"])]
    if aggregate != per_task(rows):
        raise ValueError("Per-task counts do not match the episode evidence")
    media = json.loads((root / "assets/demos/manifest.json").read_text())
    index = {(r["checkpoint"], r["split"], r["task_id"], r["initial_state_id"]): r for r in rows}
    for clip in media["clips"]:
        key = (str(clip["checkpoint"]), clip["split"], clip["task_id"], clip["initial_state_id"])
        row = index[key]
        if row["success"] != int(clip["success"]) or row["policy_steps"] != clip["policy_steps"]:
            raise ValueError(f"Demo label differs from episode outcome: {key}")
        source = root / clip["mp4"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != clip["video_sha256"]:
            raise ValueError(f"Demo checksum mismatch: {source.name}")
    return rows, groups


def paired_counts(rows, left, right, split):
    maps = {}
    for checkpoint in (left, right):
        selected = [r for r in rows if r["checkpoint"] == checkpoint and r["split"] == split]
        if not selected:
            raise ValueError(f"No completed {split} split published for checkpoint {checkpoint}")
        maps[checkpoint] = {(r["task_id"], r["initial_state_id"]): r["success"] for r in selected}
    if set(maps[left]) != set(maps[right]):
        raise ValueError("Paired comparison requires identical task/initial-state keys")
    tally = Counter((maps[left][key], maps[right][key]) for key in maps[left])
    total = len(maps[left])
    return {
        "split": split, "left": left, "right": right, "episodes": total,
        "both_success": tally[1, 1], "both_failure": tally[0, 0],
        "left_only": tally[1, 0], "right_only": tally[0, 1],
        "right_minus_left_percentage_points": 100 * (tally[0, 1] - tally[1, 0]) / total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--split", choices=("development", "held-out"), default="development")
    args = parser.parse_args()
    try:
        rows, groups = validate_repository()
        if args.pair:
            print(json.dumps(paired_counts(rows, *args.pair, args.split), indent=2))
        else:
            print(f"Verified {len(rows):,} episodes in {len(groups)} complete splits.")
            print(f"{'Checkpoint':<15}{'Split':<16}{'Success':>10}{'Rate':>10}")
            for (checkpoint, split), records in groups.items():
                wins = sum(r["success"] for r in records)
                print(f"{checkpoint:<15}{split:<16}{wins:>5}/{len(records):<4}{wins / len(records):>9.1%}")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Validation failed: {error}\n")


if __name__ == "__main__":
    main()
