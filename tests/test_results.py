"""Evidence guards: malformed data must not become a published score."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_results import load_rows, validate_rows, paired_counts


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


if __name__ == "__main__":
    unittest.main()
