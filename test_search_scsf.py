"""Tests for CIFAR-100 search-variant losses."""

import unittest

import torch

from train_cbr_scsf import cbr_selective_losses, coarse_group_matrix, soft_coverage_masks
from train_search_scsf import FeatureQueue, ProjectionHead, confidence_supcon, parse_args


class QueueAndContrastiveTests(unittest.TestCase):
    def test_queue_wraps_and_keeps_latest(self):
        queue = FeatureQueue(dim=4, size=5, device=torch.device("cpu"))
        first = torch.eye(4)
        queue.enqueue(first, torch.arange(4), torch.ones(4))
        self.assertEqual(queue.filled, 4)
        extra = torch.ones(3, 4)
        queue.enqueue(extra, torch.tensor([4, 5, 6]), torch.full((3,), 0.5))
        self.assertEqual(queue.filled, 5)
        self.assertEqual(queue.y.tolist(), [5, 6, 2, 3, 4])

    def test_supcon_is_finite_and_uses_same_class_positives(self):
        torch.manual_seed(0)
        projector = ProjectionHead(in_dim=8, hidden_dim=16, out_dim=8)
        features = torch.randn(6, 8, requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        confidence = torch.tensor([0.9, 0.2, 0.8, 0.3, 0.7, 0.4])
        queue = FeatureQueue(dim=8, size=10, device=features.device)
        query = projector(features)
        loss = confidence_supcon(query, targets, confidence, queue, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())


class CoarseCBRTests(unittest.TestCase):
    def test_coarse_confusion_uses_twenty_groups(self):
        torch.manual_seed(1)
        scores = torch.linspace(-1.0, 2.0, 12, requires_grad=True)
        targets = torch.tensor([0, 4, 1, 32, 54, 70, 2, 8, 41, 69, 3, 42])
        logits = torch.randn(12, 100, requires_grad=True)
        masks, _ = soft_coverage_masks(scores, [0.5, 0.8], 0.25, 40)
        group_matrix = coarse_group_matrix(100, scores.device, logits.dtype)
        losses = cbr_selective_losses(
            logits,
            masks,
            targets,
            [0.5, 0.8],
            confusion_temperature=0.1,
            group_matrix=group_matrix,
        )
        self.assertEqual(tuple(losses["pair_risks"].shape), (2, 20, 20))
        total = losses["micro"] + losses["confusion"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class VariantFlagTests(unittest.TestCase):
    def test_scsf_variant_disables_extra_losses(self):
        import sys

        old = sys.argv
        sys.argv = ["train_search_scsf.py", "--variant", "scsf"]
        try:
            args = parse_args()
        finally:
            sys.argv = old
        self.assertEqual(args.rank_weight, 0.0)
        self.assertEqual(args.csc_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
