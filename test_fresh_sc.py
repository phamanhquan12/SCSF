"""Smoke tests for fresh (non-acccon) selective methods."""

import sys
import unittest

import torch

from train_fresh_sc import (
    ResidualBlend,
    logit_margin,
    logit_msp,
    parse_args,
    score_from_logits,
)


class FreshHelperTests(unittest.TestCase):
    def test_logit_margin_prefers_peaked_logits(self):
        sharp = torch.tensor([[5.0, 0.0, -1.0]])
        flat = torch.tensor([[1.0, 0.9, 0.8]])
        self.assertGreater(logit_margin(sharp).item(), logit_margin(flat).item())

    def test_residual_blend_starts_near_raw_score(self):
        blend = ResidualBlend(init_alpha=-4.0)
        raw = torch.tensor([1.0, -1.0], requires_grad=True)
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        out = blend(raw, logits)
        self.assertLess((out - raw).abs().max().item(), 0.1)
        out.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertIsNone(logits.grad)

    def test_msp_logits_detached_from_classifier(self):
        logits = torch.randn(4, 10, requires_grad=True)
        m = logit_msp(logits)
        self.assertFalse(m.requires_grad)
        self.assertIsNone(m.grad_fn)

    def test_entropy_score_is_negative_entropy(self):
        peaked = torch.tensor([[10.0, 0.0, 0.0]])
        flat = torch.zeros(1, 3)
        self.assertGreater(
            score_from_logits(peaked, "entropy").item(),
            score_from_logits(flat, "entropy").item(),
        )

    def test_variant_defaults(self):
        old = sys.argv
        try:
            for variant, expect_head, expect_blend in [
                ("valblend", True, False),
                ("msp", False, False),
                ("scsf", True, False),
                ("focal", True, False),
                ("addmsp", True, True),
            ]:
                sys.argv = ["train_fresh_sc.py", "--variant", variant]
                args = parse_args()
                self.assertEqual(args.use_head, expect_head, variant)
                self.assertEqual(args.use_blend, expect_blend, variant)
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
