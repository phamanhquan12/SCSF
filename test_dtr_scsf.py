import unittest

import torch

from train_dtr_scsf import (
    TransitionCalibrator,
    conditional_repair_loss,
    rc_rank_weights,
    transition_targets,
)


class DTRComponentsTest(unittest.TestCase):
    def test_transition_state_encoding(self):
        targets = torch.tensor([0, 0, 0, 0])
        probe_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        final_logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])

        states, _, _ = transition_targets(probe_logits, final_logits, targets)

        self.assertEqual(states.tolist(), [3, 1, 2, 0])

    def test_confidence_marginalizes_final_correct_states(self):
        transition_logits = torch.log(
            torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        )

        confidence = TransitionCalibrator.confidence(transition_logits)

        self.assertAlmostEqual(confidence.item(), 0.6, places=6)

    def test_high_rank_examples_receive_larger_rc_weight(self):
        confidence = torch.tensor([0.9, 0.7, 0.2])
        selected = torch.tensor([True, True, True])

        weights = rc_rank_weights(confidence, selected)

        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)

    def test_repair_loss_updates_student_not_teacher(self):
        targets = torch.tensor([0, 1])
        probe_u = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0]], requires_grad=True
        )
        probe_v = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        final_u = torch.tensor(
            [[0.0, 3.0], [3.0, 0.0]], requires_grad=True
        )
        confidence = torch.tensor([0.9, 0.2], requires_grad=True)

        loss, selected = conditional_repair_loss(
            probe_u,
            probe_v,
            final_u,
            confidence,
            targets,
            temperature=2.0,
            max_rc_weight=5.0,
        )
        loss.backward()

        self.assertEqual(selected.tolist(), [True, True])
        self.assertIsNotNone(final_u.grad)
        self.assertIsNone(probe_u.grad)
        self.assertIsNone(confidence.grad)


if __name__ == "__main__":
    unittest.main()
