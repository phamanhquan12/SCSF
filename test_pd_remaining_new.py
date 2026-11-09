"""Numerical PD ablation checks using tiny deterministic tensors; no downloads."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from scsf.methods import PDSCSFMethod, pd_aux_coefficient
from scsf.dsspmi_campaign import matrix, VARIANTS, FULL, resolved_args


torch.set_num_threads(1)


class TinyBackbone(nn.Module):
    _is_resnet = True
    _stage_dims = {"early": 4, "mid": 6, "late": 8}
    feature_dim = 14

    def __init__(self, classes=3):
        super().__init__()
        self.layer2 = nn.Conv2d(3, 4, 1)
        self.layer3 = nn.Conv2d(4, 6, 1)
        self.layer4 = nn.Conv2d(6, 8, 1)
        self.classifier = nn.Linear(8, classes)

    def forward_with_maps(self, x):
        early = self.layer2(x)
        mid = self.layer3(early)
        late = self.layer4(mid)
        pooled = [F.adaptive_avg_pool2d(t, 1).flatten(1) for t in (mid, late)]
        return self.classifier(pooled[1]), torch.cat(pooled, 1), {"early": early, "mid": mid, "late": late}

    def forward(self, x, return_features=False):
        logits, features, _ = self.forward_with_maps(x)
        return (logits, features) if return_features else logits


def args():
    from run_experiment import parse_args
    with patch.object(sys, "argv", ["test", "--method", "pd_scsf", "--pretrain", "20", "--epochs", "100"]):
        return parse_args()


def model(**kwargs):
    return PDSCSFMethod(TinyBackbone(), 3, hidden_dim=4, head_dim=4, init_seed=17, **kwargs)


class PDTests(unittest.TestCase):
    def test_matrix_and_command_protocol(self):
        self.assertEqual(len(matrix()), 87)
        self.assertEqual(sum(d in {"covid_qu_ex", "chest_ct_scan"} and s == 0 for _, d, s in matrix()), 18)
        self.assertEqual(len({v.name for v in VARIANTS}), 9)
        self.assertNotIn(FULL, {v.name for v in VARIANTS})
        options = SimpleNamespace(results_root=Path('/tmp/new-results'), run_id='tests',
                                  data_dir=Path('data'), gpu=None, run_root=Path('/tmp/new-results/tests'))
        for v in VARIANTS:
            effective = resolved_args(options, (v, 'ham10000', 0))
            self.assertEqual(effective.variant, v.name)
            self.assertEqual((effective.batch_size, effective.eval_batch_size, effective.workers), (64, 128, 4))
            self.assertTrue(effective.pretrained)
            self.assertTrue(effective.defer_test_evaluation)
            self.assertFalse(effective.amp)
            self.assertEqual(effective.pd_target_aggregation, v.aggregation)
            self.assertEqual(effective.pd_aux_gradient_schedule, v.schedule)

    def test_target_values_and_endpoints(self):
        m = model()
        logits = torch.tensor([[2., 0., -1.], [-1., 2., 0.]], requires_grad=True)
        targets = torch.tensor([0, 1])
        probs = logits.softmax(1)
        m._depth_tcp = torch.tensor([.4, .6])
        output = SimpleNamespace(train_logits=logits)
        p = probs.gather(1, targets[:, None]).squeeze(1)
        torch.testing.assert_close(m._current_tcp(output, targets, probs), p.sqrt() * m._depth_tcp.sqrt())
        m.target_aggregation = 'arithmetic'
        torch.testing.assert_close(m._current_tcp(output, targets, probs), .5*p + .5*m._depth_tcp)
        for aggregation in ('geometric', 'arithmetic'):
            m.target_aggregation = aggregation
            for beta, expected in ((0., p), (1., m._depth_tcp)):
                m.beta = beta
                torch.testing.assert_close(m._current_tcp(output, targets, probs), expected)

    def test_weight_only_changes_target_not_depth_ce(self):
        torch.manual_seed(3)
        original = model()
        weighted = copy.deepcopy(original)
        weighted.depth_weight_layer2 = .25
        original.eval(); weighted.eval()
        x, y = torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1])
        out_a, out_b = original(x), weighted(x)
        original.training_loss(out_a, y, None, 30, args())
        weighted.training_loss(out_b, y, None, 30, args())
        q = [z.softmax(1).gather(1, y[:, None]).squeeze(1) for z in
             weighted._depth_logits(out_b.aux_outputs['pooled'], .1)]
        torch.testing.assert_close(weighted._depth_tcp, .25*q[0] + .75*q[1])
        self.assertEqual(original.last_loss_terms['depth_ce'], weighted.last_loss_terms['depth_ce'])
        self.assertFalse(weighted._depth_tcp.requires_grad)

    def test_schedule_and_head_gradient(self):
        a = args()
        epochs = [1, 20, 21, 22, 24, 25, 26, 100]
        for schedule in ('legacy', 'step', 'early'):
            m = model(aux_gradient_schedule=schedule)
            for epoch in epochs:
                scale = pd_aux_coefficient(epoch, a, schedule)
                expected = (.1 if schedule == 'early' or (schedule == 'step' and epoch > 20)
                            else min(.1, max(0., (epoch - 20)*.02)) if schedule == 'legacy' else 0.)
                self.assertAlmostEqual(scale, expected)
                pooled = {k: torch.randn(4, m.backbone._stage_dims[k], requires_grad=True) for k in m.depth_stages}
                z = m._depth_logits(pooled, scale)
                loss = sum(F.cross_entropy(t, torch.tensor([0, 1, 2, 0])) for t in z)
                grads = torch.autograd.grad(loss, [*pooled.values(), m.heads['early'].class_weight])
                self.assertGreater(float(grads[-1].abs().sum()), 0)
                if scale == 0:
                    self.assertEqual(float(grads[0].abs().sum()), 0)
                else:
                    self.assertGreater(float(grads[0].abs().sum()), 0)
            m.train()
            output = m(torch.randn(4, 3, 4, 4))
            m.training_loss(output, torch.tensor([0, 1, 2, 0]), None, 1, a)
            self.assertEqual(m.last_loss_terms['weighted_confidence_loss'], 0)

    def test_single_head_and_validation(self):
        m = model(depths='layer2')
        self.assertEqual(list(m.heads), ['early'])
        with self.assertRaises(ValueError):
            model(depths='layer2', depth_weight_layer2=.25)
        for invalid in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                model(depth_weight_layer2=invalid)
        for invalid in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                model(beta=invalid)

    def test_primary_only_skips_heads_and_preserves_scores(self):
        m = model().eval()
        x = torch.randn(4, 3, 4, 4)
        normal = m(x)
        m.emit_aux_confidence = False
        with patch.object(m, '_depth_logits', side_effect=AssertionError('depth head executed')):
            primary = m(x)
        torch.testing.assert_close(primary.eval_logits, normal.eval_logits, rtol=0, atol=0)
        torch.testing.assert_close(primary.confidence, normal.confidence, rtol=0, atol=0)
        self.assertNotIn('aux_confidence', primary.aux_outputs)

    def test_default_parity_with_original_source_and_updates(self):
        source = subprocess.check_output(['git', 'show', 'e9366ae:scsf/methods.py'], text=True)
        baseline = types.ModuleType('scsf._original_pd_for_tests')
        baseline.__package__ = 'scsf'
        sys.modules[baseline.__name__] = baseline
        exec(compile(source, '<original methods>', 'exec'), baseline.__dict__)
        torch.manual_seed(12)
        old = baseline.PDSCSFMethod(TinyBackbone(), 3, hidden_dim=4, head_dim=4, init_seed=17)
        torch.manual_seed(12)
        new = model()
        new.load_state_dict(old.state_dict(), strict=True)
        optimizers = []
        for m in (old, new):
            optimizers.append((torch.optim.SGD([p for n,p in m.named_parameters() if not n.startswith('calibrator.')], lr=.01),
                               torch.optim.Adam(m.calibrator.parameters(), lr=.001)))
        x, y = torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1])
        for epoch in (21, 22, 25):
            losses = []
            for m, opts in zip((old, new), optimizers):
                for opt in opts: opt.zero_grad(set_to_none=True)
                torch.manual_seed(99)
                output = m(x)
                loss = m.training_loss(output, y, None, epoch, args())
                losses.append(loss.detach())
                loss.backward()
                for opt in opts: opt.step()
            torch.testing.assert_close(*losses, rtol=0, atol=0)
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

    def test_real_resnet50_class_shapes_and_backward(self):
        from scsf.models import build_backbone
        for classes in (2, 4, 5, 7):
            backbone = build_backbone('resnet50', classes, 32, pretrained=False, feature_spec='mid+late+logits')
            m = PDSCSFMethod(backbone, classes, hidden_dim=4, head_dim=4, depths='layer2')
            x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, classes-1])
            output = m(x)
            self.assertEqual(output.eval_logits.shape, (2, classes))
            loss = m.training_loss(output, y, None, 21, args())
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertIsNotNone(m.heads['early'].class_weight.grad)


if __name__ == '__main__':
    unittest.main()
