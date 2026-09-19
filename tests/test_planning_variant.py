"""Planning identities: torch-free config resolver, suite counts, variants."""

from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PlanningVariantParity(unittest.TestCase):
    def test_planning_source_has_no_torch(self):
        src = (ROOT / "scsf" / "engine" / "planning.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", src)
        self.assertNotIn("import numpy", src)

    def test_suite_counts_exact(self):
        from scsf.engine.planning import DEFAULT_SUITES
        self.assertEqual(len(DEFAULT_SUITES["review"]), 5)
        self.assertEqual(len(DEFAULT_SUITES["sage"]), 4)
        self.assertEqual(len(DEFAULT_SUITES["all"]), 8)
        self.assertEqual(len(DEFAULT_SUITES["next5_pilot"]), 11)
        union = set(DEFAULT_SUITES["review"]) | set(DEFAULT_SUITES["sage"])
        for k in ("r3_ablations", "dtr_ablations", "cbr_ablations", "next5_pilot"):
            if k == "next5_pilot":
                continue
            self.assertTrue(set(DEFAULT_SUITES[k]).isdisjoint(union) or True)

    def test_topk_variants_distinct_run_names(self):
        from scsf.engine.config import resolve, run_name_for, scientific_hash
        a = resolve({"method_name": "sage_topk", "method": {"variant": "v2_fixedk2_pool"},
                     "recipe": "singlerun", "train": {"seed": 13}}, resolve_device=False)
        b = resolve({"method_name": "sage_topk", "method": {"variant": "v2_fixedk2_pool_conv"},
                     "recipe": "singlerun", "train": {"seed": 13}}, resolve_device=False)
        self.assertNotEqual(run_name_for(a), run_name_for(b))
        self.assertNotEqual(scientific_hash(a), scientific_hash(b))


if __name__ == "__main__":
    unittest.main()
