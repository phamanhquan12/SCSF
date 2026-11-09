"""Focused unit tests for the differentiable CBR-SCSF components."""

import unittest

import torch

from train_cbr_scsf import (
    ProjectedCoverageDual,
    cbr_selective_losses,
    class_coverages,
    implicit_soft_threshold,
    soft_coverage_masks,
)


class SoftCoverageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.scores = torch.tensor(
            [-2.0, -0.7, -0.1, 0.3, 0.9, 1.4, 2.2],
            dtype=torch.double,
        )

    def test_soft_coverage_matches_target(self):
        coverages = [0.2, 0.55, 0.9]
        masks, _ = soft_coverage_masks(
            self.scores, coverages, temperature=0.3, iterations=80
        )
        observed = masks.mean(dim=0)
        torch.testing.assert_close(
            observed,
            torch.tensor(coverages, dtype=torch.double),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_acceptance_masks_are_nested(self):
        masks, thresholds = soft_coverage_masks(
            self.scores, [0.25, 0.5, 0.8], temperature=0.2, iterations=80
        )
        self.assertTrue(torch.all(masks[:, 0] <= masks[:, 1] + 1e-12))
        self.assertTrue(torch.all(masks[:, 1] <= masks[:, 2] + 1e-12))
        self.assertTrue(thresholds[0] >= thresholds[1])
        self.assertTrue(thresholds[1] >= thresholds[2])

    def test_threshold_uses_implicit_gradient(self):
        scores = self.scores.clone().requires_grad_(True)
        temperature = 0.4
        threshold = implicit_soft_threshold(
            scores, coverage=0.6, temperature=temperature, iterations=80
        )
        threshold.backward()

        with torch.no_grad():
            acceptance = torch.sigmoid((scores - threshold) / temperature)
            slope = acceptance * (1.0 - acceptance)
            expected = slope / slope.sum()
        torch.testing.assert_close(scores.grad, expected, rtol=1e-8, atol=1e-10)
        # Translation equivariance requires dh/d(sum of equal score shifts)=1.
        self.assertAlmostEqual(scores.grad.sum().item(), 1.0, places=10)


class ClassConstraintTests(unittest.TestCase):
    def test_absent_class_is_ignored(self):
        scores = torch.tensor([-1.0, -0.2, 0.4, 1.1], requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor(
            [
                [2.0, 0.1, -0.5],
                [0.2, 1.0, -0.1],
                [0.1, 1.7, -0.2],
                [0.3, 0.8, 0.5],
            ],
            requires_grad=True,
        )
        masks, _ = soft_coverage_masks(scores, [0.5, 0.8], 0.25, 80)
        observed, present = class_coverages(masks, targets, num_classes=3)
        self.assertEqual(present.tolist(), [True, True, False])
        self.assertTrue(torch.equal(observed[2], torch.zeros_like(observed[2])))

        losses = cbr_selective_losses(
            logits,
            masks,
            targets,
            [0.5, 0.8],
            confusion_temperature=0.1,
        )
        total = losses["micro"] + losses["confusion"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isnan(losses["pair_risks"][:, 2]).all())

    def test_dual_update_projects_and_skips_absent_classes(self):
        dual = ProjectedCoverageDual(
            num_classes=3,
            coverages=[0.5, 0.9],
            floor_ratio=0.8,
            learning_rate=1.0,
            ema_decay=0.0,
            max_value=0.3,
            device=torch.device("cpu"),
        )
        observed = torch.tensor(
            [
                [0.0, 0.0],  # positive violations, clipped at dual max
                [1.0, 1.0],  # negative violations, projected to zero
                [0.0, 0.0],  # absent: must not update
            ]
        )
        present = torch.tensor([True, True, False])
        dual.update(observed, present)
        torch.testing.assert_close(dual.values[0], torch.tensor([0.3, 0.3]))
        torch.testing.assert_close(dual.values[1], torch.zeros(2))
        torch.testing.assert_close(dual.values[2], torch.zeros(2))

        absent_before = dual.values[0].clone()
        dual.update(torch.ones_like(observed), torch.tensor([False, True, False]))
        torch.testing.assert_close(dual.values[0], absent_before)
        self.assertTrue(torch.all(dual.values >= 0.0))


if __name__ == "__main__":
    unittest.main()
