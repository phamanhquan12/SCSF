"""Unit tests for training-dynamics variants (recipe §§7–15)."""

import sys
import unittest

import torch
import torch.nn.functional as F

from dyn_scsf import (
    IndexedDataset,
    TrainingDynamicsTracker,
    apply_cartography_weights,
    apply_el2n_weights,
    binary_auroc,
    depth_disagreement,
    enrichment_metrics,
    make_confidence_target,
    make_query_weight,
    rank_band_analysis,
    variant_defaults,
)
from train_next_scsf import apply_variant_defaults, parse_args


class FakeMapDataset:
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), index % 3


class TrackerTests(unittest.TestCase):
    def _commit_epoch(self, tracker, ids, logits, labels):
        tracker.observe_batch(ids, logits, labels)
        before = tracker.temporal_correctness(ids).clone()
        tracker.commit_epoch()
        return before

    def test_ids_are_dataset_indices(self):
        wrapped = IndexedDataset(FakeMapDataset(5))
        image, label, sample_id = wrapped[3]
        self.assertEqual(int(sample_id), 3)
        self.assertEqual(int(label), 0)
        self.assertEqual(int(image.item()), 3)

    def test_current_epoch_does_not_leak_into_target(self):
        tracker = TrainingDynamicsTracker(n_samples=4, window=3)
        ids = torch.arange(4)
        logits = torch.tensor(
            [
                [4.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 4.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        before = self._commit_epoch(tracker, ids, logits, labels)
        torch.testing.assert_close(before, torch.zeros(4))
        torch.testing.assert_close(tracker.temporal_correctness(ids), torch.ones(4))

    def test_temporal_mean_is_previous_epochs_only(self):
        tracker = TrainingDynamicsTracker(n_samples=2, window=4)
        ids = torch.tensor([0, 1])
        labels = torch.tensor([0, 1])
        correct = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        wrong = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        for logits in (correct, correct, wrong, correct):
            tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        # C C W C -> 0.75
        torch.testing.assert_close(
            tracker.temporal_correctness(ids), torch.tensor([0.75, 0.75])
        )

    def test_window_rolls_and_drops_old_epochs(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=3)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        correct = torch.tensor([[4.0, 0.0]])
        wrong = torch.tensor([[0.0, 4.0]])
        for logits in (wrong, correct, correct, correct):
            tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        torch.testing.assert_close(tracker.temporal_correctness(ids), torch.tensor([1.0]))

    def test_forgetting_and_learning_counts(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=5)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        correct = torch.tensor([[4.0, 0.0]])
        wrong = torch.tensor([[0.0, 4.0]])
        for logits in (correct, wrong, correct, wrong):
            tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        torch.testing.assert_close(tracker.forgetting_count(ids), torch.tensor([2.0]))
        torch.testing.assert_close(tracker.learning_count(ids), torch.tensor([1.0]))

    def test_variability_is_zero_for_constant_ptrue(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=4)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        logits = torch.tensor([[8.0, 0.0]])
        for _ in range(4):
            tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        self.assertLess(tracker.variability(ids).item(), 1e-5)
        self.assertGreater(tracker.mean_confidence(ids).item(), 0.99)

    def test_ambiguity_uses_95th_percentile_ref(self):
        tracker = TrainingDynamicsTracker(n_samples=20, window=6)
        labels = torch.zeros(1, dtype=torch.long)
        for epoch in range(6):
            for sample in range(20):
                ids = torch.tensor([sample])
                # sample 19 oscillates; others are stable and correct
                if sample == 19 and epoch % 2 == 1:
                    logits = torch.tensor([[0.0, 6.0]])
                else:
                    logits = torch.tensor([[6.0, 0.0]])
                tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        amb = tracker.normalized_ambiguity(torch.arange(20))
        self.assertLess(amb[:19].max().item(), 0.2)
        self.assertGreater(amb[19].item(), 0.9)


class TargetAndWeightTests(unittest.TestCase):
    def test_hard_target_ignores_tracker(self):
        tracker = TrainingDynamicsTracker(n_samples=2, window=2)
        ids = torch.tensor([0, 1])
        hard = torch.tensor([1.0, 0.0])
        out = make_confidence_target(hard, ids, tracker, "hard")
        torch.testing.assert_close(out, hard)

    def test_temporal_falls_back_to_hard_without_history(self):
        tracker = TrainingDynamicsTracker(n_samples=2, window=2)
        ids = torch.tensor([0, 1])
        hard = torch.tensor([1.0, 0.0])
        out = make_confidence_target(hard, ids, tracker, "temporal")
        torch.testing.assert_close(out, hard)

    def test_hybrid_is_convex_combination(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=2)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        tracker.observe_batch(ids, torch.tensor([[4.0, 0.0]]), labels)
        tracker.commit_epoch()
        hard = torch.tensor([0.0])
        out = make_confidence_target(hard, ids, tracker, "hybrid", alpha=0.25)
        torch.testing.assert_close(out, torch.tensor([0.75]))

    def test_forget_target_downweights_unstable_examples(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=4)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        for logits in (
            torch.tensor([[4.0, 0.0]]),
            torch.tensor([[0.0, 4.0]]),
        ):
            tracker.observe_batch(ids, logits, labels)
            tracker.commit_epoch()
        hard = torch.tensor([1.0])
        out = make_confidence_target(hard, ids, tracker, "forget", forget_gamma=1.0)
        # temporal=0.5, N_forget=1, S=e^{-1}
        torch.testing.assert_close(out, torch.tensor([0.5 * torch.exp(torch.tensor(-1.0))]))

    def test_margin_target_is_sigmoid_of_mean_margin(self):
        tracker = TrainingDynamicsTracker(n_samples=1, window=2)
        ids = torch.tensor([0])
        labels = torch.tensor([0])
        tracker.observe_batch(ids, torch.tensor([[2.0, 0.0]]), labels)
        tracker.commit_epoch()
        out = make_confidence_target(
            torch.tensor([0.0]), ids, tracker, "margin", margin_temperature=1.0
        )
        torch.testing.assert_close(out, torch.sigmoid(torch.tensor([2.0])))

    def test_query_weight_modes(self):
        w = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(make_query_weight(w, "acceptance"), w)
        torch.testing.assert_close(make_query_weight(w, "uniform"), torch.ones_like(w))
        torch.testing.assert_close(
            make_query_weight(w, "floor", floor=0.2),
            torch.tensor([0.2, 0.6, 1.0]),
        )
        boundary = make_query_weight(w, "boundary", beta=0.1)
        torch.testing.assert_close(boundary[0], torch.tensor(0.0))
        torch.testing.assert_close(boundary[2], torch.tensor(1.0))
        self.assertGreater(boundary[1].item(), 0.5)
        carto = make_query_weight(
            w, "cartography", beta=0.2, ambiguity=torch.tensor([1.0, 1.0, 0.0])
        )
        torch.testing.assert_close(carto, torch.tensor([0.0, 0.6, 1.0]))

    def test_cartography_keys_are_stability_weighted(self):
        tracker = TrainingDynamicsTracker(n_samples=2, window=4)
        labels = torch.tensor([0])
        # id 0 stable correct; id 1 oscillates
        for epoch in range(4):
            tracker.observe_batch(
                torch.tensor([0]), torch.tensor([[5.0, 0.0]]), labels
            )
            logits1 = (
                torch.tensor([[5.0, 0.0]])
                if epoch % 2 == 0
                else torch.tensor([[0.0, 5.0]])
            )
            tracker.observe_batch(torch.tensor([1]), logits1, labels)
            tracker.commit_epoch()
        accept = torch.tensor([0.8, 0.8])
        query, key = apply_cartography_weights(
            accept, tracker, torch.tensor([0, 1]), beta=0.2, gamma=4.0, device="cpu"
        )
        self.assertGreater(key[0].item(), key[1].item())
        self.assertGreater(query[1].item(), accept[1].item())

    def test_el2n_boosts_hard_queries_and_shrinks_keys(self):
        tracker = TrainingDynamicsTracker(n_samples=2, window=2)
        easy = torch.tensor([[8.0, 0.0]])
        hard = torch.tensor([[0.2, 0.0]])
        labels = torch.tensor([0])
        tracker.observe_el2n(torch.tensor([0]), easy, labels)
        tracker.observe_el2n(torch.tensor([1]), hard, labels)
        accept = torch.ones(2)
        query, key = apply_el2n_weights(
            accept, tracker, torch.tensor([0, 1]), beta=1.0, device="cpu"
        )
        self.assertLess(query[0].item(), query[1].item())
        self.assertGreater(key[0].item(), key[1].item())


class MetricAndProbeTests(unittest.TestCase):
    def test_auroc_perfect_and_inverted(self):
        scores = torch.tensor([0.9, 0.8, 0.1, 0.0])
        correct = torch.tensor([True, True, False, False])
        self.assertAlmostEqual(binary_auroc(scores, correct), 1.0)
        self.assertAlmostEqual(binary_auroc(-scores, correct), 0.0)

    def test_bands_and_enrichment_put_errors_in_the_tail(self):
        scores = torch.arange(100, 0, -1).float()
        correct = torch.ones(100, dtype=torch.bool)
        correct[-10:] = False
        bands = rank_band_analysis(scores, correct)
        self.assertEqual(bands["90-95"]["n_errors"], 5)
        self.assertEqual(bands["95-100"]["n_errors"], 5)
        self.assertEqual(bands["0-10"]["n_errors"], 0)
        enrich = enrichment_metrics(scores, correct)
        self.assertEqual(enrich["bottom10_capture"], 1.0)
        self.assertEqual(enrich["top10_purity"], 1.0)

    def test_depth_disagreement_counts_probe_votes(self):
        final = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
        p3 = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
        p4 = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
        p5 = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
        d = depth_disagreement(p3, p4, p5, final)
        torch.testing.assert_close(d, torch.tensor([2.0 / 3.0, 0.0]))


class VariantIsolationTests(unittest.TestCase):
    def _parse(self, variant):
        old = sys.argv
        sys.argv = ["train_next_scsf.py", "--variant", variant]
        try:
            return parse_args()
        finally:
            sys.argv = old

    def test_temporal_target_does_not_enable_acccon_or_probes(self):
        args = self._parse("temporal_target")
        self.assertEqual(args.confidence_target, "temporal")
        self.assertTrue(args.record_dynamics)
        self.assertEqual(args.con_weight, 0.0)
        self.assertFalse(args.use_depth_probes)
        self.assertEqual(args.dyn_window, 20)

    def test_temporal_hybrid_uses_half_hard(self):
        args = self._parse("temporal_hybrid")
        self.assertEqual(args.confidence_target, "hybrid")
        self.assertEqual(args.target_alpha, 0.5)
        self.assertEqual(args.con_weight, 0.0)

    def test_carto_acccon_keeps_hard_target(self):
        args = self._parse("carto_acccon")
        self.assertEqual(args.confidence_target, "hard")
        self.assertGreater(args.con_weight, 0.0)
        self.assertEqual(args.query_weight_mode, "cartography")
        self.assertEqual(args.ambiguity_beta, 0.10)
        self.assertEqual(args.stability_gamma, 2.0)
        self.assertFalse(args.append_disagreement)

    def test_el2n_acccon_records_early_prior_only(self):
        args = self._parse("el2n_acccon")
        self.assertTrue(args.record_el2n)
        self.assertFalse(args.record_dynamics)
        self.assertEqual(args.query_weight_mode, "el2n")
        self.assertEqual(args.confidence_target, "hard")

    def test_depth_diag_does_not_change_selector_inputs(self):
        args = self._parse("depth_diag")
        self.assertTrue(args.use_depth_probes)
        self.assertFalse(args.append_disagreement)
        self.assertEqual(args.extra_dim, 0)
        self.assertEqual(args.con_weight, 0.0)
        self.assertEqual(args.confidence_target, "hard")

    def test_temporal_depth_combines_only_section_15(self):
        args = self._parse("temporal_depth")
        self.assertEqual(args.confidence_target, "temporal")
        self.assertTrue(args.append_disagreement)
        self.assertEqual(args.extra_dim, 1)
        self.assertEqual(args.con_weight, 0.0)

    def test_acccon_defaults_are_unchanged(self):
        args = self._parse("acccon")
        self.assertEqual(args.confidence_target, "hard")
        self.assertFalse(args.record_dynamics)
        self.assertFalse(args.use_depth_probes)
        self.assertEqual(args.extra_dim, 0)
        self.assertEqual(args.con_mode, "accept")

    def test_unknown_variant_still_rejected(self):
        class Args:
            variant = "rank_csc"

        with self.assertRaises(ValueError):
            apply_variant_defaults(Args())

    def test_registry_covers_sections_8_to_15(self):
        names = set(variant_defaults())
        self.assertTrue(
            {
                "temporal_target",
                "temporal_hybrid",
                "forget_target",
                "margin_target",
                "carto_acccon",
                "el2n_acccon",
                "depth_diag",
                "depth_score",
                "temporal_depth",
            }.issubset(names)
        )


if __name__ == "__main__":
    unittest.main()
