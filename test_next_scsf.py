"""Tests for next-wave RC losses and variant locks."""

import sys
import unittest

import torch
import torch.nn.functional as F

from train_cbr_scsf import implicit_soft_threshold, soft_coverage_masks
from train_next_scsf import (
    DeepSupervisionBranch,
    DeeplySupervisedScore,
    accepted_cross_entropy,
    apply_variant_defaults,
    core_boundary_query_weight,
    hard_pair_rank_loss,
    parse_args,
    rc_contrastive_weights,
    soft_micro_risk,
    soft_selective_risk_loss,
    uses_official_10k,
    weighted_supcon,
)
from train_cbr_scsf import RawConfidenceHead
from dyn_scsf import depth_disagreement
from train_search_scsf import FeatureQueue, ProjectionHead


class SoftRcLossTests(unittest.TestCase):
    def test_micro_risk_is_coverage_weighted_soft_error(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 5, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        scores = torch.linspace(-1.0, 2.0, 8, requires_grad=True)
        masks, _ = soft_coverage_masks(scores, [0.5, 0.8], 0.25, 40)
        micro, errors = soft_micro_risk(logits, masks, targets)
        expected = (masks * errors.unsqueeze(1)).sum(0) / masks.sum(0)
        self.assertTrue(torch.isfinite(micro))
        torch.testing.assert_close(micro, expected.mean())
        (micro).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_accepted_ce_ignores_fully_rejected_samples(self):
        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 0, 1])
        masks = torch.tensor([[1.0], [0.0], [1.0]])
        loss = accepted_cross_entropy(logits, masks, targets)
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        expected = (per_sample[0] + per_sample[2]) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class ContrastiveWeightTests(unittest.TestCase):
    def test_accept_mode_uses_mean_acceptance_for_both_sides(self):
        masks = torch.tensor([[0.2, 0.4], [0.8, 1.0], [0.0, 0.1]])
        logits = torch.zeros(3, 2)
        targets = torch.tensor([0, 1, 0])
        query_w, key_w = rc_contrastive_weights(masks, logits, targets, "accept")
        torch.testing.assert_close(query_w, masks.mean(dim=1))
        torch.testing.assert_close(key_w, query_w)

    def test_boundary_query_is_zero_at_ends_and_peaks_near_half(self):
        w = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        q = core_boundary_query_weight(w, beta=1.0)
        torch.testing.assert_close(q[0], torch.tensor(0.0))
        torch.testing.assert_close(q[-1], torch.tensor(1.0))
        torch.testing.assert_close(q[2], torch.tensor(1.5))
        torch.testing.assert_close(q[3], torch.tensor(1.5))
        self.assertGreater(q[2].item(), q[1].item())
        self.assertGreater(q[1].item(), q[0].item())

    def test_accept_bound_keeps_keys_as_plain_acceptance(self):
        masks = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        logits = torch.zeros(3, 2)
        targets = torch.tensor([0, 1, 0])
        query_w, key_w = rc_contrastive_weights(
            masks, logits, targets, "accept_bound", boundary_beta=1.0
        )
        accept = masks.mean(dim=1)
        torch.testing.assert_close(key_w, accept)
        torch.testing.assert_close(query_w, core_boundary_query_weight(accept, 1.0))

    def test_accept_floor_lifts_queries_and_keeps_keys(self):
        masks = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        logits = torch.zeros(3, 2)
        targets = torch.tensor([0, 1, 0])
        query_w, key_w = rc_contrastive_weights(
            masks, logits, targets, "accept_floor", query_floor=0.20
        )
        accept = masks.mean(dim=1)
        torch.testing.assert_close(key_w, accept)
        torch.testing.assert_close(query_w, 0.20 + 0.80 * accept)
        self.assertGreater(query_w[1].item(), query_w[0].item())
        self.assertGreater(query_w[2].item(), query_w[1].item())

    def test_leftover_queries_and_accepted_correct_keys(self):
        masks = torch.ones(3, 2)
        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 0])
        query_w, key_w = rc_contrastive_weights(masks, logits, targets, "leftover")
        p_y = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        torch.testing.assert_close(query_w, 1.0 - p_y)
        torch.testing.assert_close(key_w, p_y)
        self.assertLess(query_w[0].item(), query_w[2].item())
        self.assertGreater(key_w[0].item(), key_w[2].item())

    def test_hard_pair_rank_pushes_confident_errors_down(self):
        scores = torch.tensor([3.0, 2.5, 0.1, -1.0], requires_grad=True)
        correctness = torch.tensor([True, False, True, False])
        loss = hard_pair_rank_loss(
            scores, correctness, margin=0.2, k_wrong=1, k_correct=1
        )
        # Only pair: wrong@2.5 vs correct@3.0 → softplus(2.5-3.0+0.2)
        expected = F.softplus(torch.tensor(2.5 - 3.0 + 0.2))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(scores.grad[1].item(), 0.0)  # descent lowers wrong
        self.assertLess(scores.grad[0].item(), 0.0)  # descent raises correct

    def test_hard_pair_rank_zero_when_batch_all_correct(self):
        scores = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
        correctness = torch.tensor([True, True, True])
        loss = hard_pair_rank_loss(scores, correctness)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_weighted_supcon_is_finite_and_asymmetric(self):
        torch.manual_seed(1)
        projector = ProjectionHead(in_dim=8, hidden_dim=16, out_dim=8)
        features = torch.randn(6, 8, requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        query_w = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.0])
        key_w = torch.tensor([0.2, 0.9, 0.1, 0.8, 0.3, 0.6])
        queue = FeatureQueue(dim=8, size=10, device=features.device)
        query = projector(features)
        loss = weighted_supcon(query, targets, query_w, key_w, queue, 0.2)
        self.assertTrue(torch.isfinite(loss))
        flipped = weighted_supcon(query, targets, key_w, query_w, queue, 0.2)
        self.assertGreater((loss - flipped).abs().item(), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_query_weight_scales_the_batch_mean(self):
        torch.manual_seed(2)
        query = F.normalize(torch.tensor([
            [1.0, 0.0],
            [0.6, 0.8],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]), dim=1)
        targets = torch.tensor([0, 0, 1, 1])
        key_w = torch.ones(4)
        queue = FeatureQueue(dim=2, size=4, device=query.device)
        loss_first = weighted_supcon(
            query, targets, torch.tensor([1.0, 1e-8, 1e-8, 1e-8]), key_w, queue, 0.2
        )
        loss_second = weighted_supcon(
            query, targets, torch.tensor([1e-8, 1.0, 1e-8, 1e-8]), key_w, queue, 0.2
        )
        self.assertGreater((loss_first - loss_second).abs().item(), 1e-5)
        mixed = weighted_supcon(
            query, targets, torch.tensor([0.75, 0.25, 0.0, 0.0]), key_w, queue, 0.2
        )
        expected = 0.75 * loss_first + 0.25 * loss_second
        torch.testing.assert_close(mixed, expected, rtol=1e-5, atol=1e-6)


class TailRiskTests(unittest.TestCase):
    def test_tail_loss_uses_live_scores_and_detached_errors(self):
        torch.manual_seed(0)
        scores = torch.linspace(-1.5, 2.0, 8, requires_grad=True)
        wrong = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
        wrong.requires_grad_(True)
        loss = soft_selective_risk_loss(
            scores, wrong.gt(0.5), (0.70, 0.90), temperature=0.25
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertIsNone(wrong.grad)

    def test_tail_threshold_is_computed_from_detached_scores(self):
        scores = torch.linspace(-1.0, 2.0, 6, requires_grad=True)
        wrong = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
        loss = soft_selective_risk_loss(scores, wrong, (0.80,), temperature=0.2)
        threshold = implicit_soft_threshold(scores.detach(), 0.80, 0.2, 60)
        accept = torch.sigmoid((scores - threshold) / 0.2)
        expected = accept.mul(wrong.to(scores.dtype)).sum() / accept.sum().clamp_min(
            1e-12
        )
        torch.testing.assert_close(loss, expected)


class VariantFlagTests(unittest.TestCase):
    def _parse(self, variant):
        old = sys.argv
        sys.argv = ["train_next_scsf.py", "--variant", variant]
        try:
            return parse_args()
        finally:
            sys.argv = old

    def test_micro_hi_drops_low_coverage_and_feature_terms(self):
        args = self._parse("micro_hi")
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.con_weight, 0.0)
        self.assertGreater(args.micro_weight, 0.0)

    def test_acccon_bound_uses_boundary_query_mode(self):
        args = self._parse("acccon_bound")
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.con_mode, "accept_bound")
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])
        self.assertEqual(args.tail_weight, 0.0)

    def test_acccon_floor_uses_floored_query_weights(self):
        args = self._parse("acccon_floor")
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.con_mode, "accept_floor")
        self.assertEqual(args.query_floor, 0.20)
        self.assertEqual(args.tail_weight, 0.0)
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])

    def test_acccon_tail_keeps_acccon_and_adds_tail_loss(self):
        args = self._parse("acccon_tail")
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.con_mode, "accept")
        self.assertAlmostEqual(args.tail_weight, 0.10)
        self.assertEqual(args.tail_coverages, [0.7, 0.8, 0.9, 0.95])
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])

    def test_acccon_disables_micro_and_uses_accept_weights(self):
        args = self._parse("acccon")
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.con_mode, "accept")
        self.assertFalse(args.use_ds_score)
        self.assertEqual(args.aux_ce_weight, 0.0)
        self.assertTrue(uses_official_10k("acccon"))

    def test_acccon_ds_uses_fused_score_and_query_weighted_acccon(self):
        args = self._parse("acccon_ds")
        self.assertTrue(args.use_ds_score)
        self.assertAlmostEqual(args.aux_ce_weight, 0.3)
        self.assertAlmostEqual(args.aux_bce_weight, 0.3)
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.tail_weight, 0.0)
        self.assertEqual(args.con_mode, "accept")
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])
        self.assertTrue(uses_official_10k("acccon_ds"))

    def test_dss_acccon_keeps_scsf_and_enables_live_dsn(self):
        args = self._parse("dss_acccon")
        self.assertTrue(args.use_ds_branch)
        self.assertFalse(args.use_ds_score)
        self.assertTrue(args.append_disagreement)
        self.assertEqual(args.extra_dim, 1)
        self.assertAlmostEqual(args.aux_ce_weight, 0.3)
        self.assertEqual(args.aux_bce_weight, 0.0)
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.tail_weight, 0.0)
        self.assertEqual(args.con_mode, "accept")
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])
        self.assertTrue(uses_official_10k("dss_acccon"))

    def test_rank_acccon_keeps_qw_acccon_and_enables_hard_rank(self):
        args = self._parse("rank_acccon")
        self.assertAlmostEqual(args.rank_weight, 0.05)
        self.assertAlmostEqual(args.rank_margin, 0.2)
        self.assertEqual(args.rank_k_wrong, 16)
        self.assertEqual(args.rank_k_correct, 4)
        self.assertEqual(args.tail_weight, 0.0)
        self.assertEqual(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertFalse(args.use_ds_score)
        self.assertFalse(args.use_ds_branch)
        self.assertEqual(args.con_mode, "accept")
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.coverages, [0.8, 0.9, 0.95])
        self.assertTrue(uses_official_10k("rank_acccon"))

    def test_msp_acccon_appends_msp_and_keeps_qw_acccon(self):
        args = self._parse("msp_acccon")
        self.assertTrue(args.append_msp)
        self.assertEqual(args.extra_dim, 1)
        self.assertEqual(args.rank_weight, 0.0)
        self.assertEqual(args.tail_weight, 0.0)
        self.assertEqual(args.micro_weight, 0.0)
        self.assertFalse(args.use_ds_score)
        self.assertFalse(args.use_ds_branch)
        self.assertEqual(args.con_mode, "accept")
        self.assertGreater(args.con_weight, 0.0)
        self.assertTrue(uses_official_10k("msp_acccon"))

    def test_micro_leftcon_keeps_micro_and_leftover_mode(self):
        args = self._parse("micro_leftcon")
        self.assertGreater(args.micro_weight, 0.0)
        self.assertEqual(args.acceptce_weight, 0.0)
        self.assertEqual(args.con_mode, "leftover")
        self.assertGreater(args.con_weight, 0.0)

    def test_apply_variant_defaults_rejects_unknown_names(self):
        class Args:
            variant = "rank_csc"

        with self.assertRaises(ValueError):
            apply_variant_defaults(Args())


class DeeplySupervisedScoreTests(unittest.TestCase):
    def test_fusion_weights_start_deeper_heavier(self):
        head = DeeplySupervisedScore(num_classes=10)
        weights = head.fusion_weights()
        torch.testing.assert_close(weights, torch.tensor([0.2, 0.3, 0.5]))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    def test_fused_score_is_convex_combination(self):
        head = DeeplySupervisedScore(num_classes=10)
        head.eval()
        spatial3 = torch.randn(3, 256, 4, 4)
        spatial4 = torch.randn(3, 512, 2, 2)
        spatial5 = torch.randn(3, 512, 1, 1)
        logits = torch.randn(3, 10)
        fused, aux = head(spatial3, spatial4, spatial5, logits)
        weights = aux["weights"]
        expected = (
            weights[0] * aux["score3"]
            + weights[1] * aux["score4"]
            + weights[2] * aux["score5"]
        )
        torch.testing.assert_close(fused, expected)

    def test_live_spatial_features_receive_gradients(self):
        head = DeeplySupervisedScore(num_classes=10)
        spatial3 = torch.randn(2, 256, 4, 4, requires_grad=True)
        spatial4 = torch.randn(2, 512, 2, 2, requires_grad=True)
        spatial5 = torch.randn(2, 512, 1, 1, requires_grad=True)
        logits = torch.randn(2, 10, requires_grad=True)
        fused, aux = head(spatial3, spatial4, spatial5, logits)
        (fused.sum() + aux["logits3"].sum()).backward()
        self.assertTrue(torch.isfinite(spatial3.grad).all())
        self.assertTrue(torch.isfinite(spatial4.grad).all())
        self.assertTrue(torch.isfinite(spatial5.grad).all())
        self.assertIsNone(logits.grad)

    def test_deepest_score_reads_final_logits_not_aux(self):
        head = DeeplySupervisedScore(num_classes=10)
        head.eval()
        spatial3 = torch.randn(2, 256, 4, 4)
        spatial4 = torch.randn(2, 512, 2, 2)
        spatial5 = torch.randn(2, 512, 1, 1)
        logits = torch.randn(2, 10)
        fused_a, aux_a = head(spatial3, spatial4, spatial5, logits)
        fused_b, aux_b = head(spatial3, spatial4, spatial5, logits + 5.0)
        self.assertGreater((aux_a["score5"] - aux_b["score5"]).abs().max().item(), 1e-5)
        torch.testing.assert_close(aux_a["score3"], aux_b["score3"])
        torch.testing.assert_close(aux_a["score4"], aux_b["score4"])
        self.assertGreater((fused_a - fused_b).abs().max().item(), 1e-6)


class DeepSupervisionBranchTests(unittest.TestCase):
    def test_aux_ce_grads_reach_spatial_features(self):
        branch = DeepSupervisionBranch(num_classes=10)
        spatial3 = torch.randn(2, 256, 4, 4, requires_grad=True)
        spatial4 = torch.randn(2, 512, 2, 2, requires_grad=True)
        spatial5 = torch.randn(2, 512, 1, 1, requires_grad=True)
        targets = torch.tensor([0, 1])
        out = branch(spatial3, spatial4, spatial5)
        aux_ce = (
            F.cross_entropy(out["logits3"], targets)
            + F.cross_entropy(out["logits4"], targets)
            + F.cross_entropy(out["logits5"], targets)
        ) / 3.0
        aux_ce.backward()
        self.assertTrue(torch.isfinite(spatial3.grad).all())
        self.assertTrue(torch.isfinite(spatial4.grad).all())
        self.assertTrue(torch.isfinite(spatial5.grad).all())

    def test_disagreement_appended_to_score_is_detached(self):
        branch = DeepSupervisionBranch(num_classes=10)
        head = RawConfidenceHead(512 * 4, 512, 10, extra_dim=1)
        spatial3 = torch.randn(2, 256, 4, 4, requires_grad=True)
        spatial4 = torch.randn(2, 512, 2, 2, requires_grad=True)
        spatial5 = torch.randn(2, 512, 1, 1, requires_grad=True)
        pool4 = F.adaptive_avg_pool2d(spatial4, 2).flatten(1)
        pool5 = F.adaptive_avg_pool2d(spatial5, 1).flatten(1)
        logits = torch.randn(2, 10)
        out = branch(spatial3, spatial4, spatial5)
        disagreement = depth_disagreement(
            out["logits3"], out["logits4"], out["logits5"], logits
        )
        scores = head(pool4, pool5, logits, extra=disagreement.detach().unsqueeze(1))
        scores.sum().backward()
        self.assertIsNone(spatial3.grad)
        self.assertTrue(torch.isfinite(spatial4.grad).all())
        self.assertTrue(torch.isfinite(spatial5.grad).all())


if __name__ == "__main__":
    unittest.main()
