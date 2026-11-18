"""Validate public episode evidence and reproduce aggregate / paired results.

Standard library only. Run from any directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ("pi05_base", "800", "1000", "1250", "1500", "1750", "2000")
EPISODE_FIELDS = ("checkpoint", "split", "task_id", "initial_state_id", "success", "policy_steps")
ORIGINAL_SEED = 20261104
NEW_SEEDS = (20261118, 20261119)
STRICT_OBSERVATION_EXCLUSIONS = ((1, 34),)


def load_rows(root=ROOT, *, extended=False):
    name = "inference_seeds.csv" if extended else "episodes.csv"
    fields = (*EPISODE_FIELDS, "inference_seed") if extended else EPISODE_FIELDS
    with (root / "results" / name).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"Unexpected episode columns in {name}")
        rows = list(reader)
    for row in rows:
        for key in fields[2:]:
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
        if not 1 <= row["policy_steps"] <= protocol["max_policy_steps"] or (
            not row["success"] and row["policy_steps"] != protocol["max_policy_steps"]
        ):
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
    validate_observation_exclusions(metadata)
    rows = load_rows(root)
    groups = validate_rows(rows, protocol)
    expected_groups = {(x["checkpoint"], x["split"]) for x in metadata["groups"]}
    required = {(checkpoint, split) for checkpoint in CHECKPOINTS for split in ("development", "held-out")}
    if set(groups) != expected_groups or set(groups) != required or len(rows) != metadata["published_episodes"]:
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
    seeds = validate_inference_seeds(root, protocol, metadata)
    validate_comparisons(root, rows, seeds, metadata)
    return rows, groups


def validate_observation_exclusions(metadata):
    expected = [list(key) for key in STRICT_OBSERVATION_EXCLUSIONS]
    if metadata.get("strict_observation_exclusions") != expected:
        raise ValueError(f"Strict observation exclusions must match the audit: {expected}")


def validate_inference_seeds(root, protocol, metadata):
    expected = {(str(step), "held-out", seed) for step in (1250, 2000) for seed in NEW_SEEDS}
    rows = load_rows(root, extended=True)
    seen = {(r["checkpoint"], r["split"], r["inference_seed"]) for r in rows}
    manifest = {(g["checkpoint"], g["split"], g["inference_seed"]) for g in metadata["inference_seed_groups"]}
    if seen != expected or manifest != expected or len(rows) != metadata["inference_seed_episodes"]:
        raise ValueError("Inference seed coverage differs from the provenance manifest")
    digest = hashlib.sha256((root / "results/inference_seeds.csv").read_bytes()).hexdigest()
    if digest != metadata["inference_seeds_csv_sha256"]:
        raise ValueError("Inference seed CSV checksum differs from the provenance manifest")
    for seed in NEW_SEEDS:
        validate_rows([r for r in rows if r["inference_seed"] == seed], protocol)
    return rows


def validate_comparisons(root, rows, seeds, metadata):
    by_case = {
        (r["checkpoint"], r["task_id"], r["initial_state_id"], r.get("inference_seed", ORIGINAL_SEED)): r
        for r in rows + seeds if r["split"] == "held-out"
    }
    media = json.loads((root / "assets/failures/manifest.json").read_text())
    case_keys = {(c["task_id"], c["initial_state_id"]) for c in media["cases"]}
    if case_keys != {(4, 0), (4, 45), (8, 12)} or len(media["cases"]) != len(case_keys):
        raise ValueError("Unexpected set of failure comparisons")
    excluded = {tuple(key) for key in metadata["strict_observation_exclusions"]}
    for case in media["cases"]:
        task, initial = case["task_id"], case["initial_state_id"]
        if (task, initial) in excluded or by_case["2000", task, initial, ORIGINAL_SEED]["success"]:
            raise ValueError("Comparison is outside the matched original failures")
        for variant in case["variants"]:
            row = by_case[str(variant["checkpoint"]), task, initial, variant["inference_seed"]]
            if (row["success"], row["policy_steps"]) != (variant["success"], variant["policy_steps"]):
                raise ValueError("Failure comparison label differs from episode outcome")
        for asset in (case["video"], case["frames"]):
            if hashlib.sha256((root / asset["path"]).read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Failure comparison checksum mismatch: {asset['path']}")


def task_differences(left, right, excluded=()):
    if left.keys() != right.keys():
        raise ValueError("Bootstrap requires identical task/initial-state keys")
    changes = defaultdict(list)
    for task, initial in sorted(left):
        key = (task, initial)
        if key not in excluded:
            changes[task].append(right[key] - left[key])
    return list(changes.values())


def paired_interval(differences, seed):
    rng = random.Random(seed)
    draws = sorted(
        100 * sum(sum(rng.choice(group) for _ in group) / len(group) for group in differences)
        / len(differences)
        for _ in range(10_000)
    )
    return [round(draws[index], 6) for index in (249, 9749)]


def held_out_report(rows, seed_rows, excluded):
    outcomes = defaultdict(dict)
    for row in rows + seed_rows:
        if row["split"] == "held-out":
            outcomes[row["checkpoint"], row.get("inference_seed", ORIGINAL_SEED)][
                row["task_id"], row["initial_state_id"]
            ] = row["success"]
    a0 = outcomes["pi05_base", ORIGINAL_SEED]
    original = outcomes["2000", ORIGINAL_SEED]
    strict = task_differences(a0, original, excluded)
    base_comparison = dict(
        paired_counts(rows, "pi05_base", "2000", "held-out", exclude=excluded),
        task_mean_delta_pp=round(100 * sum(sum(group) / len(group) for group in strict) / len(strict), 6),
        paired_CI95_pp=paired_interval(strict, 20261118),
    )
    checks = []
    for seed in (ORIGINAL_SEED, *NEW_SEEDS):
        left, right = outcomes["2000", seed], outcomes["1250", seed]
        tally = Counter((left[key], right[key]) for key in left)
        pair = dict(
            seed=seed, successes_1250=sum(right.values()), successes_2000=sum(left.values()),
            both_success=tally[1, 1], only_1250=tally[0, 1], only_2000=tally[1, 0],
            both_failure=tally[0, 0],
            delta_1250_minus_2000_pp=round(100 * (sum(right.values()) - sum(left.values())) / len(left), 6),
            paired_CI95_pp=paired_interval(task_differences(left, right), 20261117),
        )
        if seed in NEW_SEEDS:
            pair["2000_failure_to_success"] = sum(not value and left[key] for key, value in original.items())
            pair["2000_success_to_failure"] = sum(value and not left[key] for key, value in original.items())
        checks.append(pair)
    new_mean = {
        step: {key: sum(outcomes[step, seed][key] for seed in NEW_SEEDS) / len(NEW_SEEDS) for key in original}
        for step in ("1250", "2000")
    }
    seed_deltas_by_task = [
        {
            "task_id": task,
            "success_delta_by_seed": {
                str(seed): sum(
                    outcomes["1250", seed][key] - outcomes["2000", seed][key]
                    for key in original if key[0] == task
                )
                for seed in NEW_SEEDS
            },
        }
        for task in sorted({key[0] for key in original})
    ]
    strict_1750 = task_differences(outcomes["1750", ORIGINAL_SEED], original, excluded)
    failures = [{
        "task_id": task, "initial_state_id": initial,
        "successful_new_seeds": [seed for seed in NEW_SEEDS if outcomes["2000", seed][task, initial]],
    } for (task, initial), success in sorted(original.items()) if not success]
    return {
        "strict_a0_to_2000": base_comparison,
        "a0_to_2000_all_states": dict(paired_counts(rows, "pi05_base", "2000", "held-out"),
                                     paired_CI95_pp=paired_interval(
                                         task_differences(a0, original), 20261118)),
        "1750_to_2000": dict(paired_counts(rows, "1750", "2000", "held-out"),
                             paired_CI95_pp=paired_interval(task_differences(
                                 outcomes["1750", ORIGINAL_SEED], original), ORIGINAL_SEED)),
        "strict_1750_to_2000": dict(
            paired_counts(rows, "1750", "2000", "held-out", exclude=excluded),
            task_mean_delta_pp=round(100 * sum(sum(group) / len(group) for group in strict_1750)
                                     / len(strict_1750), 6),
            paired_CI95_pp=paired_interval(strict_1750, ORIGINAL_SEED),
        ),
        "1250_vs_2000_by_inference_seed": checks,
        "mean_new_seed_1250_minus_2000_pp": sum(c["delta_1250_minus_2000_pp"] for c in checks[1:]) / len(NEW_SEEDS),
        "mean_new_seed_paired_CI95_pp": paired_interval(task_differences(new_mean["2000"], new_mean["1250"]), 20261117),
        "new_seed_success_deltas_by_task": seed_deltas_by_task,
        "original_2000_failures": failures,
    }


def paired_counts(rows, left, right, split, *, exclude=()):
    maps = {}
    for checkpoint in (left, right):
        selected = [r for r in rows if r["checkpoint"] == checkpoint and r["split"] == split]
        if not selected:
            raise ValueError(f"No completed {split} split published for checkpoint {checkpoint}")
        maps[checkpoint] = {(r["task_id"], r["initial_state_id"]): r["success"] for r in selected}
    if set(maps[left]) != set(maps[right]):
        raise ValueError("Paired comparison requires identical task/initial-state keys")
    keys = maps[left].keys() - set(exclude)
    tally = Counter((maps[left][key], maps[right][key]) for key in keys)
    total = len(keys)
    return {
        "split": split, "left": left, "right": right, "episodes": total,
        "both_success": tally[1, 1], "both_failure": tally[0, 0],
        "left_only": tally[1, 0], "right_only": tally[0, 1],
        "right_minus_left_percentage_points": 100 * (tally[0, 1] - tally[1, 0]) / total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pair", nargs=2, metavar=("LEFT", "RIGHT"))
    selection.add_argument("--report", action="store_true", help="Reproduce the held-out comparisons and failure list")
    parser.add_argument("--split", choices=("development", "held-out"), default="development")
    args = parser.parse_args()
    try:
        rows, groups = validate_repository()
        if args.pair:
            print(json.dumps(paired_counts(rows, *args.pair, args.split), indent=2))
        elif args.report:
            metadata = json.loads((ROOT / "results/provenance.json").read_text())
            excluded = {tuple(key) for key in metadata["strict_observation_exclusions"]}
            print(json.dumps(held_out_report(rows, load_rows(extended=True), excluded), indent=2))
        else:
            additional = len(load_rows(extended=True))
            print(f"Verified {len(rows):,} original-seed episodes in {len(groups)} complete groups and {additional:,} additional-seed episodes.")
            print(f"{'Checkpoint':<15}{'Split':<16}{'Success':>10}{'Rate':>10}")
            for (checkpoint, split), records in groups.items():
                wins = sum(r["success"] for r in records)
                print(f"{checkpoint:<15}{split:<16}{wins:>5}/{len(records):<4}{wins / len(records):>9.2%}")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Validation failed: {error}\n")


if __name__ == "__main__":
    main()
