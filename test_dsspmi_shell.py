"""Exercise the real medical wrapper's argument handling without training or downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scsf.dsspmi_campaign import VARIANTS, suite_flags, validate_protocol


class ShellCompatibilityTests(unittest.TestCase):
    def test_all_variants_through_friend_wrapper_with_and_without_gpu(self):
        import run_paper_medical_suite as suite
        import run_experiment as experiment
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Logging process substitution is blocked by macOS's local sandbox.
            # All interpreter discovery, argument parsing and child dispatch remain original.
            shell = (root / 'run_medical_suite.sh').read_text()
            shell = shell.replace('exec > >(tee -a "$LAUNCH_LOG") 2>&1', ': # test capture handles logging')
            (directory / 'run_medical_suite.sh').write_text(shell)
            output = directory / 'captured.jsonl'
            recorder = ('import json, os, sys\n'
                        f'with open({str(output)!r}, "a") as f:\n'
                        '    f.write(json.dumps({"argv": sys.argv, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")\n')
            (directory / 'run_paper_medical_suite.py').write_text(recorder)
            options = SimpleNamespace(results_root=directory/'results', run_id='check', data_dir=directory/'data',
                                      run_root=directory/'results/check', gpu=None)
            environment = dict(os.environ, VIRTUAL_ENV=sys.prefix,
                               KAGGLE_CONFIG_DIR=str(directory/'no_credentials'), CUDA_VISIBLE_DEVICES='7')
            for gpu in (None, '0'):
                options.gpu = gpu
                for variant in VARIANTS:
                    command = ['bash', str(directory/'run_medical_suite.sh'),
                               *suite_flags(options, (variant, variant.datasets[0], 0))]
                    result = subprocess.run(command, env=environment, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            captured = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(captured), 18)
            for index, item in enumerate(captured):
                with patch.object(sys, 'argv', item['argv']):
                    parsed = suite.parse_args()
                variant, flags = suite.method_variants(parsed, 'pd_scsf')[0]
                self.assertEqual(variant, VARIANTS[index % 9].name)
                command, _ = suite.train_command(parsed, directory/'results/check', parsed.datasets[0],
                                                 'pd_scsf', variant, flags, parsed.seeds[0])
                with patch.object(sys, 'argv', command[1:]):
                    effective = experiment.parse_args()
                validate_protocol(effective)
                self.assertEqual(effective.gpu, None if index < 9 else '0')
                self.assertEqual(item['visible'], '7')
                self.assertTrue(effective.defer_test_evaluation)
                self.assertFalse(effective.download)


if __name__ == '__main__':
    unittest.main()
