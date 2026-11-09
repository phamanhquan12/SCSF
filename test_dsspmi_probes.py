"""Probe isolation and fixed classifier tests without real medical images."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent / 'analysis'))
from train_pd_frozen_probes import fit, scores, stratified_folds, inputs, load_features
from pd_common import state_sha256, require_test_release
from test_pd_remaining_new import TinyBackbone
from scsf.dsspmi_contract import atomic_json, file_sha256, digest


class ProbeTests(unittest.TestCase):
    def test_minimum_errors_and_stratification(self):
        self.assertIsNone(stratified_folds(torch.tensor([1.]*20+[0.]*14)))
        labels=torch.tensor([1.]*15+[0.]*15)
        folds=stratified_folds(labels)
        for held in folds:
            self.assertEqual(int(labels[held].sum()),5)
            self.assertEqual(len(held),10)
        self.assertEqual(len(set(torch.cat(folds).tolist())),30)

    def test_fit_standardization_and_frozen_backbone(self):
        torch.manual_seed(8)
        frozen=TinyBackbone()
        before=state_sha256(frozen)
        x=torch.randn(32,4); x[:,3]=8
        correct=(x[:,0]>0).float()
        model,mean,std=fit(x,correct,0,.001)
        torch.testing.assert_close(mean,x.mean(0))
        self.assertEqual(float(std[3]),1)
        self.assertEqual(before,state_sha256(frozen))
        self.assertTrue(torch.isfinite(torch.from_numpy(scores(model,x,mean,std,'cpu'))).all())
        self.assertEqual(sum(p.numel() for p in model.parameters()),4*64+64+64+1)

    def test_test_release_required_and_chunk_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.assertRaises(ValueError): require_test_release(None)
            payload={'logits':torch.tensor([[0.,1.],[2.,0.]]),'features':torch.randn(2,5),
                     'target':torch.tensor([1,0]),'prediction':torch.tensor([1,0]),
                     'correct':torch.ones(2),'sample_ids':['val/a/1','val/b/2']}
            torch.save(payload,folder/'part.pt')
            atomic_json(folder/'manifest.json',{'n':2,'sample_ids_sha256':digest(payload['sample_ids']),
                        'chunks':[{'file':'part.pt','sha256':file_sha256(folder/'part.pt')}]})
            values,_=load_features(folder)
            self.assertEqual(inputs(values,'logits_only').shape,(2,2))
            self.assertEqual(inputs(values,'features_only').shape,(2,5))
            self.assertEqual(inputs(values,'combined').shape,(2,7))
            (folder/'part.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load_features(folder)

    def test_wrong_python_defaults_fail_production_protocol(self):
        from run_experiment import parse_args
        from scsf.dsspmi_campaign import validate_protocol
        with patch.object(sys,'argv',['test','--method','pd_scsf','--arch','resnet50']):
            options=parse_args()
        with self.assertRaises(ValueError):validate_protocol(options)


if __name__=='__main__':unittest.main()
