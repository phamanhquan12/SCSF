import unittest

import torch
import torch.nn as nn

from train_r3_scsf import (
    RawScoreCalibrator,
    empirical_aurc_influence,
    final_classifier_probe,
    pairwise_rank_loss,
    repair_loss,
    virtual_repairability,
)


class R3LossTests(unittest.TestCase):
    def test_aurc_influence_is_larger_at_higher_confidence_rank(self):
        scores = torch.tensor([3.0, 1.0, -2.0])
        weights = empirical_aurc_influence(scores)
        expected = torch.tensor(
            [(1.0 + 0.5 + 1.0 / 3.0) / 3.0, (0.5 + 1.0 / 3.0) / 3.0, 1.0 / 9.0]
        )
        torch.testing.assert_close(weights, expected)
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())
        self.assertFalse(weights.requires_grad)

    def test_rank_loss_pushes_wrong_score_down_and_correct_score_up(self):
        scores = torch.tensor([2.0, -1.0], requires_grad=True)
        errors = torch.tensor([True, False])
        rho = torch.tensor([0.0, 0.0])
        loss = pairwise_rank_loss(scores, errors, rho)
        loss.backward()
        self.assertGreater(scores.grad[0].item(), 0.0)
        self.assertLess(scores.grad[1].item(), 0.0)

        badly_ordered = pairwise_rank_loss(
            torch.tensor([2.0, -1.0]), errors, rho
        )
        correctly_ordered = pairwise_rank_loss(
            torch.tensor([-1.0, 2.0]), errors, rho
        )
        self.assertGreater(badly_ordered.item(), correctly_ordered.item())

    def test_higher_rho_reduces_rank_pressure(self):
        scores_low = torch.tensor([1.0, 0.0], requires_grad=True)
        scores_high = scores_low.detach().clone().requires_grad_(True)
        errors = torch.tensor([True, False])
        low_rho_loss = pairwise_rank_loss(
            scores_low, errors, torch.tensor([0.0, 0.0]), rank_epsilon=0.1
        )
        high_rho_loss = pairwise_rank_loss(
            scores_high, errors, torch.tensor([1.0, 0.0]), rank_epsilon=0.1
        )
        self.assertGreater(low_rho_loss.item(), high_rho_loss.item())

    def test_higher_rho_increases_repair_pressure(self):
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        targets = torch.tensor([0, 0])
        scores = torch.tensor([2.0, -1.0])
        errors = torch.tensor([True, False])
        low = repair_loss(
            logits, targets, scores, errors, torch.tensor([0.1, 0.0])
        )
        high = repair_loss(
            logits, targets, scores, errors, torch.tensor([0.9, 0.0])
        )
        self.assertGreater(high.item(), low.item())

    def test_virtual_rho_is_bounded_and_tracks_helpful_update(self):
        targets = torch.tensor([0, 0])
        logits_u = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        logits_v = logits_u.clone()
        aligned_u = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        aligned_v = aligned_u.clone()
        rho = virtual_repairability(
            logits_u,
            logits_v,
            aligned_u,
            aligned_v,
            targets,
            virtual_lr=0.5,
        )
        self.assertTrue(torch.all(rho >= 0.0))
        self.assertTrue(torch.all(rho <= 1.0))
        self.assertTrue(torch.all(rho > 0.0))

        zero_step = virtual_repairability(
            logits_u,
            logits_v,
            aligned_u,
            aligned_v,
            targets,
            virtual_lr=0.0,
        )
        torch.testing.assert_close(zero_step, torch.zeros_like(zero_step))
        self.assertFalse(rho.requires_grad)


class R3GradientRoutingTests(unittest.TestCase):
    def test_confidence_losses_route_through_features_not_logits(self):
        calibrator = RawScoreCalibrator(2, 2, 2)
        calibrator.eval()
        pool4 = torch.randn(3, 2, requires_grad=True)
        pool5 = torch.randn(3, 2, requires_grad=True)
        logits = torch.randn(3, 2, requires_grad=True)
        scores = calibrator(pool4, pool5, logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores, torch.tensor([1.0, 0.0, 1.0])
        )
        loss.backward()
        self.assertIsNotNone(pool4.grad)
        self.assertIsNotNone(pool5.grad)
        self.assertGreater(pool4.grad.abs().sum().item(), 0.0)
        self.assertIsNone(logits.grad)

    def test_repair_routes_to_classifier_not_score(self):
        logits = torch.tensor(
            [[2.0, 0.0], [2.0, 0.0]], requires_grad=True
        )
        targets = torch.tensor([1, 0])
        scores = torch.tensor([2.0, -1.0], requires_grad=True)
        errors = torch.tensor([True, False])
        rho = torch.tensor([0.8, 0.2], requires_grad=True)
        loss = repair_loss(logits, targets, scores, errors, rho)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        self.assertIsNone(scores.grad)
        self.assertIsNone(rho.grad)


class TinyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(nn.Linear(2, 2))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = TinyBaseModel()


class R3ProbeStateTests(unittest.TestCase):
    def test_virtual_probe_preserves_bn_and_module_modes(self):
        backbone = TinyBackbone()
        backbone.train()
        backbone.base_model.features[2].eval()
        modes_before = {
            module: module.training for module in backbone.modules()
        }
        bn = backbone.base_model.features[1]
        mean_before = bn.running_mean.clone()
        variance_before = bn.running_var.clone()

        logits, hidden = final_classifier_probe(
            backbone, torch.randn(4, 2)
        )

        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(hidden.shape), (4, 2))
        for module, training in modes_before.items():
            self.assertEqual(module.training, training)
        torch.testing.assert_close(bn.running_mean, mean_before)
        torch.testing.assert_close(bn.running_var, variance_before)


if __name__ == "__main__":
    unittest.main()
