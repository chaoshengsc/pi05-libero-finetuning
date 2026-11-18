"""Evidence guards: malformed data must not become a published score."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_results import held_out_report, load_rows, paired_counts, validate_repository, validate_rows
from unittest.mock import patch


class EvidenceGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows()
        cls.protocol = json.loads((ROOT / "configs/evaluation.json").read_text())

    def test_duplicate_episode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_rows(self.rows + [self.rows[0]], self.protocol)

    def test_partial_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_rows(self.rows[1:], self.protocol)

    def test_wrong_split_initial_state_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["initial_state_id"] = self.protocol["spatial_states"][0]["formal"][0]
        with self.assertRaisesRegex(ValueError, "outside the frozen split"):
            validate_rows(rows, self.protocol)

    def test_unpublished_held_out_pair_cannot_be_reported(self):
        rows = [r for r in self.rows if (r["checkpoint"], r["split"]) != ("1750", "held-out")]
        with self.assertRaisesRegex(ValueError, "No completed"):
            paired_counts(rows, "1750", "2000", "held-out")

    def test_matching_counts_cannot_hide_different_episode_keys(self):
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if (r["checkpoint"], r["split"]) == ("2000", "development"))
        row["initial_state_id"] = 999
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_counts(rows, "1750", "2000", "development")

    def test_strict_pairing_rejects_a_changed_exclusion(self):
        read_text = Path.read_text
        for wrong in ([], [[99, 99]], [[0, 2]]):
            with self.subTest(exclusions=wrong):
                def substituted(path, *args, **kwargs):
                    content = read_text(path, *args, **kwargs)
                    if path == ROOT / "results/provenance.json":
                        metadata = json.loads(content)
                        metadata["strict_observation_exclusions"] = wrong
                        return json.dumps(metadata)
                    return content

                with patch.object(Path, "read_text", substituted):
                    with self.assertRaisesRegex(ValueError, "Strict observation exclusions"):
                        validate_repository()

    def test_report_reproduces_the_audited_comparisons(self):
        metadata = json.loads((ROOT / "results/provenance.json").read_text())
        exclusions = {tuple(key) for key in metadata["strict_observation_exclusions"]}
        report = held_out_report(self.rows, load_rows(extended=True), exclusions)
        strict = report["strict_a0_to_2000"]
        self.assertEqual((strict["episodes"], strict["right_only"], strict["both_failure"]), (399, 381, 18))
        self.assertEqual(strict["task_mean_delta_pp"], 95.5)
        self.assertEqual(strict["paired_CI95_pp"], [93.5, 97.25])
        checkpoint_pair = report["strict_1750_to_2000"]
        self.assertEqual((checkpoint_pair["episodes"], checkpoint_pair["both_success"],
                          checkpoint_pair["left_only"], checkpoint_pair["right_only"]), (399, 363, 15, 18))
        self.assertEqual(checkpoint_pair["task_mean_delta_pp"], 0.75)
        tasks = report["new_seed_success_deltas_by_task"]
        subtotal = sum(sum(row["success_delta_by_seed"].values()) for row in tasks if row["task_id"] in (4, 8))
        total = sum(sum(row["success_delta_by_seed"].values()) for row in tasks)
        self.assertEqual((subtotal, total - subtotal), (12, -2))


if __name__ == "__main__":
    unittest.main()
