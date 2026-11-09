from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from scsf.dsspmi_contract import audit_cache, atomic_json, digest, file_sha256, require_contract
from test_pd_remaining_new import TinyBackbone
from scsf.methods import PDSCSFMethod


class ContractTests(unittest.TestCase):
    def make_cache(self, root):
        cache = root / 'preprocessed/ham10000_224'
        rows = []
        for split in ('train', 'val', 'test'):
            for label in ('a', 'b'):
                directory = cache / split / label
                directory.mkdir(parents=True)
                path = directory / 'sample.png'
                path.write_bytes(f'{split}_{label}'.encode())
                rows.append(dict(split=split, class_name=label, source_path=f'raw/{split}/{label}.png', cached_path=str(path)))
        with (cache/'split_manifest.csv').open('w') as handle:
            writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        atomic_json(cache/'preprocess_manifest.json', dict(split_seed=42, input_size=224, cache_format_version=3,
                                                         mean=[0,0,0], std=[1,1,1]))
        return cache

    def test_cache_integrity_seed_extras_and_label_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=self.make_cache(root)
            original=audit_cache(root, 'ham10000')
            self.assertEqual(original['sha256'], audit_cache(root, 'ham10000')['sha256'])
            with self.assertRaises(ValueError): audit_cache(root, 'ham10000', split_seed=99)
            extra=cache/'train/a/extra.png'; extra.write_bytes(b'extra')
            with self.assertRaises(ValueError): audit_cache(root,'ham10000')
            extra.unlink()
            (cache/'test/b/sample.png').unlink(); (cache/'test/b').rmdir()
            with self.assertRaises(ValueError): audit_cache(root, 'ham10000')

    def test_source_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=self.make_cache(root)
            path=cache/'split_manifest.csv'
            with path.open() as f: rows=list(csv.DictReader(f))
            rows[-1]['source_path']=rows[0]['source_path']
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError): audit_cache(root,'ham10000')

    def test_config_mutations_rejected(self):
        from scsf.dsspmi_contract import run_contract
        base=SimpleNamespace(dataset='ham10000', pd_beta=.5, pd_target_aggregation='geometric',
                             pd_depth_weight_layer2=.5, pd_aux_gradient_schedule='legacy')
        original=run_contract(base, {'sha256':'original'})
        for key,value in [('pd_beta',.75), ('pd_target_aggregation','arithmetic'),
                          ('pd_depth_weight_layer2',.25), ('pd_aux_gradient_schedule','early')]:
            changed=copy.deepcopy(base); setattr(changed,key,value)
            with self.assertRaises(ValueError): require_contract(original, run_contract(changed, {'sha256':'original'}))
        with self.assertRaises(ValueError): require_contract(original,run_contract(base,{'sha256':'changed'}))

    def test_reference_compatibility_requires_original_evidence(self):
        from scsf.dsspmi_campaign import reference_check, VARIANTS, FULL
        from scsf.dsspmi_contract import normalized_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = (VARIANTS[1], 'ham10000', 0)
            original = {'dataset': 'ham10000', 'method': 'pd_scsf', 'seed': 0,
                        'pd_target_aggregation': 'geometric', 'batch_size': 64}
            wanted = normalized_config({**original, 'pd_target_aggregation': 'arithmetic'})
            expected = {'effective_config': wanted, 'data_sha256': 'historical-data', 'sha256': 'contract'}
            checkpoint = root/'last.pt'
            torch.save({'args': original, 'epoch': 100, 'checkpoint_type': 'last'}, checkpoint)
            provenance = root/'provenance.json'
            atomic_json(provenance, {'checkpoint_sha256': file_sha256(checkpoint), 'data_sha256': 'historical-data'})
            references = root/'references.json'
            atomic_json(references, {'references': [{'dataset': 'ham10000', 'seed': 0, 'method': 'pd_scsf',
                                                    'variant': FULL, 'checkpoint': str(checkpoint), 'provenance': str(provenance)}]})
            options = SimpleNamespace(reference_config=references)
            with patch('scsf.dsspmi_campaign.expected_contract', return_value=expected):
                self.assertEqual(reference_check(options, job), expected)
                atomic_json(provenance, {'checkpoint_sha256': file_sha256(checkpoint), 'data_sha256': 'wrong-split'})
                with self.assertRaisesRegex(ValueError, 'UNVERIFIED_SPLIT'):
                    reference_check(options, job)
                original['batch_size'] = 128
                torch.save({'args': original, 'epoch':100, 'checkpoint_type':'last'}, checkpoint)
                atomic_json(provenance, {'checkpoint_sha256': file_sha256(checkpoint), 'data_sha256':'historical-data'})
                with self.assertRaisesRegex(ValueError, 'batch_size'):
                    reference_check(options, job)

    def test_simple_launcher_requires_no_historic_reference_setup(self):
        from scsf.dsspmi_campaign import run_jobs, VARIANTS
        with tempfile.TemporaryDirectory() as tmp:
            options = SimpleNamespace(simple=True, run_root=Path(tmp), results_root=Path(tmp),
                                      run_id='simple', data_dir=Path('data'), gpu=None)
            job = (VARIANTS[0], 'ham10000', 0)
            with patch('scsf.dsspmi_campaign.clean_source', side_effect=AssertionError('clean checkout required')), \
                 patch('scsf.dsspmi_campaign.reference_check', side_effect=AssertionError('old references required')), \
                 patch('scsf.dsspmi_campaign.expected_contract', return_value={'sha256':'new'}), \
                 patch('scsf.dsspmi_campaign.freeze_matrix', return_value={}), \
                 patch('scsf.dsspmi_campaign.subprocess.run') as launch:
                run_jobs(options, [job])
            command = launch.call_args.args[0]
            self.assertIn('run_medical_suite.sh', command[1])
            self.assertIn('--defer-test-evaluation', command)
            self.assertIn('--strict-run-contract', command)
            with (Path(tmp)/'tables/comparison_contract.csv').open() as f:
                self.assertEqual(next(csv.DictReader(f))['status'], 'new_run_verified')

    def test_deferred_training_and_eval_only_are_separate(self):
        import run_experiment as runner
        torch.manual_seed(1)
        x=torch.randn(4,3,4,4); y=torch.tensor([0,1,2,0])
        train=DataLoader(TensorDataset(x,y,torch.arange(4)),batch_size=4)
        val=DataLoader(TensorDataset(x,y),batch_size=4)
        class ForbiddenTest:
            def __iter__(self): raise AssertionError('test accessed before release')
        loaders={'train':train,'val':val,'test':ForbiddenTest(),'test_full':ForbiddenTest()}
        data={'sha256':'fixed-data','identity':{}}
        spec=SimpleNamespace(input_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            base=['run_experiment.py','--method','pd_scsf','--dataset','ham10000','--arch','resnet50',
                  '--epochs','2','--pretrain','1','--batch-size','4','--input-size','4','--hidden-dim','4',
                  '--save-dir',tmp+'/checkpoints','--metrics-dir',tmp+'/metrics','--output-layout','paper',
                  '--variant','tiny','--strict-run-contract']
            def make(*a, **kw): return PDSCSFMethod(TinyBackbone(),3,hidden_dim=4,head_dim=4)
            with patch('scsf.dsspmi_contract.audit_cache',return_value=data), patch.object(runner,'build_loaders',return_value=(loaders,spec,3)), patch.object(runner,'build_method',side_effect=make):
                with patch.object(sys,'argv',base+['--defer-test-evaluation']): runner.main()
                folder=Path(tmp)/'checkpoints/ham10000/resnet50/pd_scsf/tiny/42'
                marker=json.loads((folder/'trained_complete.json').read_text())
                sha=file_sha256(folder/'last.pt')
                self.assertFalse((folder/'complete.json').exists())
                self.assertFalse(list((Path(tmp)/'metrics').glob('**/last/summary.csv')))
                with patch.object(sys,'argv',base+['--defer-test-evaluation']), patch.object(runner,'train_one_epoch',side_effect=AssertionError('retrained')):
                    runner.main()
                release=Path(tmp)/'release.json'
                atomic_json(release,{'checkpoints':{marker['contract']['sha256']:sha}})
                loaders['test_full']=val
                with patch.object(sys,'argv',base+['--evaluate-last-only','--test-release-lock',str(release)]), patch.object(runner,'train_one_epoch',side_effect=AssertionError('training called')):
                    runner.main()
                self.assertEqual(sha,file_sha256(folder/'last.pt'))
                self.assertTrue((folder/'complete.json').exists())
                self.assertTrue(list((Path(tmp)/'metrics').glob('**/last/summary.csv')))
                changed=base+['--evaluate-last-only','--test-release-lock',str(release),'--pd-beta','.75']
                with patch.object(sys,'argv',changed), self.assertRaises(ValueError): runner.main()


if __name__=='__main__': unittest.main()
