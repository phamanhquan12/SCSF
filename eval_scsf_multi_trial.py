"""
Multi-trial evaluation for SCSF model.
Evaluates with different shuffle seeds to get mean ± std results.
"""
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import argparse
import os
import sys
import numpy as np

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.cifar import vgg
import torch.nn as nn

# Coverage points (same as baseline)
COVERAGE_POINTS = [100, 99, 98, 97, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]

# =============================================================================
# VGG16_bn with Feature Extraction (same as train_scsf.py)
# =============================================================================

class VGG16BN_FeatureExtractor(nn.Module):
    """Wrapper around original VGG16_bn to expose intermediate features."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = vgg.vgg16_bn(num_classes=num_classes)
        self._find_pool_indices()
    
    def _find_pool_indices(self):
        """Find indices of MaxPool2d layers in features."""
        self.pool_indices = []
        for i, layer in enumerate(self.base_model.features):
            if isinstance(layer, nn.MaxPool2d):
                self.pool_indices.append(i)
    
    def forward(self, x, return_features=False):
        if not return_features:
            return self.base_model(x)
        
        # Run through features, capturing intermediate outputs
        feat_pool4 = None
        feat_pool5 = None
        
        for i, layer in enumerate(self.base_model.features):
            x = layer(x)
            # Capture after pool4 (4th MaxPool, index 3 in pool_indices)
            if len(self.pool_indices) >= 4 and i == self.pool_indices[3]:
                feat_pool4 = x
            # Capture after pool5 (5th MaxPool, index 4 in pool_indices)  
            if len(self.pool_indices) >= 5 and i == self.pool_indices[4]:
                feat_pool5 = x
        
        # Classifier
        x = x.view(x.size(0), -1)
        logits = self.base_model.classifier(x)
        
        return logits, feat_pool4, feat_pool5

class MetaCalibrator(torch.nn.Module):
    """Meta-Calibrator for confidence estimation"""
    def __init__(self, pool4_dim=512*4, pool5_dim=512*1, logit_dim=10, hidden_dim=256):
        super().__init__()
        
        input_dim = pool4_dim + pool5_dim + logit_dim
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, feat_pool4, feat_pool5, logits):
        flat_pool4 = feat_pool4.view(feat_pool4.size(0), -1)
        flat_pool5 = feat_pool5.view(feat_pool5.size(0), -1)
        combined = torch.cat([flat_pool4, flat_pool5, logits.detach()], dim=1)
        confidence = self.network(combined).squeeze(-1)
        return confidence

def load_model(checkpoint_path, device):
    """Load SCSF model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = VGG16BN_FeatureExtractor(num_classes=10)
    model = model.to(device)
    
    state_dict = checkpoint['model_state_dict']
    if not any(k.startswith('base_model.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict['base_model.' + k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    meta_cal = MetaCalibrator(pool4_dim=512*4, pool5_dim=512*1, logit_dim=10, hidden_dim=256)
    meta_cal = meta_cal.to(device)
    meta_cal.load_state_dict(checkpoint['meta_cal_state_dict'])
    meta_cal.eval()
    
    return model, meta_cal

def collect_results_on_full_testset(model, meta_cal, testloader, device):
    """Collect confidence scores and correctness on FULL test set"""
    confidence_scores = []
    correctness_flags = []
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, pool4_feat, pool5_feat = model.forward(inputs, return_features=True)
            confidence = meta_cal(pool4_feat, pool5_feat, logits)
            _, predictions = torch.max(logits, 1)
            correct = predictions.eq(targets)
            
            confidence_scores.extend(confidence.cpu().tolist())
            correctness_flags.extend(correct.cpu().tolist())
    
    return confidence_scores, correctness_flags

def shuffle_lists(list1, list2, seed):
    """Shuffle two lists with the same seed"""
    random.seed(seed)
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    list1_shuffled, list2_shuffled = zip(*combined)
    return list(list1_shuffled), list(list2_shuffled)

def sort_by_confidence_to_find_results(confidence, correct, coverage_points):
    """Sort by confidence and compute error rate at each coverage point"""
    results = []
    correct_sorted = [corr for _, corr in sorted(zip(confidence, correct), key=lambda pair: -pair[0])]
    
    for coverage in coverage_points:
        passed = round(coverage / 100.0 * len(correct_sorted))
        if passed == 0:
            passed = 1
        passed_correct = correct_sorted[:passed]
        passed_acc = sum([int(corr) for corr in passed_correct]) / passed
        results.append((passed / len(correct_sorted), passed_acc))
    
    return results

def compute_aurc(coverage_dict):
    """Compute AURC from coverage vs error dict"""
    coverages = sorted(coverage_dict.keys(), reverse=True)
    aurc = 0
    for i in range(len(coverages) - 1):
        c1, c2 = coverages[i], coverages[i+1]
        e1, e2 = coverage_dict[c1]/100, coverage_dict[c2]/100
        width = (c1 - c2) / 100
        aurc += (e1 + e2) / 2 * width
    return aurc

def evaluate_single_trial(model, meta_cal, testloader, device, shuffle_seed, dataset='cifar10'):
    """Evaluate with a single shuffle seed"""
    
    # Collect results on full test set
    confidence_scores, correctness_flags = collect_results_on_full_testset(model, meta_cal, testloader, device)
    
    # Shuffle with given seed
    confidence_shuffled, correct_shuffled = shuffle_lists(confidence_scores, correctness_flags, seed=shuffle_seed)
    
    # Split into val + test (different sizes per dataset)
    # Paper: "The validation set sizes for SVHN, CIFAR-10 and Cats vs. Dogs 
    # are respectively 5000, 2000 and 2000."
    if dataset == 'svhn':
        val_size = 5000
    else:  # cifar10, catsdogs
        val_size = 2000
    
    confidence_test = confidence_shuffled[val_size:]
    correct_test = correct_shuffled[val_size:]
    
    # Evaluate on test split
    results_test = sort_by_confidence_to_find_results(confidence_test, correct_test, COVERAGE_POINTS)
    
    # Build results dict
    scsf_results = {}
    for idx, cov in enumerate(COVERAGE_POINTS):
        test_err = (1 - results_test[idx][1]) * 100
        scsf_results[cov] = test_err
    
    # Compute AURC
    scsf_aurc = compute_aurc(scsf_results)
    
    return scsf_results, scsf_aurc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SCSF checkpoint')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'])
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 42, 123], 
                        help='Shuffle seeds for multiple trials')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ==========================================================================
    # Load model
    # ==========================================================================
    print(f'\n{"="*80}')
    print(f'MULTI-TRIAL EVALUATION: {len(args.seeds)} trials')
    print(f'{"="*80}')
    print(f'\nLoading checkpoint: {args.checkpoint}')
    model, meta_cal = load_model(args.checkpoint, device)
    
    # ==========================================================================
    # Load FULL test set
    # ==========================================================================
    print(f'\nLoading {args.dataset} FULL test set...')
    
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)
        
    elif args.dataset == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        svhn_root = os.path.join(args.data_dir, 'svhn')
        testset = datasets.SVHN(root=svhn_root, split='test', download=False, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)
    print(f'Full test set size: {len(testset)} samples')
    
    # Validation split size (for paper compliance)
    val_size = 5000 if args.dataset == 'svhn' else 2000
    print(f'Evaluation split: val={val_size}, test={len(testset) - val_size}')
    
    # ==========================================================================
    # Run multiple trials
    # ==========================================================================
    all_results = []
    all_aurcs = []
    
    for trial_idx, seed in enumerate(args.seeds, 1):
        print(f'\n{"="*80}')
        print(f'Trial {trial_idx}/{len(args.seeds)}: Shuffle seed = {seed}')
        print(f'{"="*80}')
        
        results, aurc = evaluate_single_trial(model, meta_cal, testloader, device, seed, args.dataset)
        all_results.append(results)
        all_aurcs.append(aurc)
        
        print(f'AURC: {aurc:.6f}')
    
    # ==========================================================================
    # Compute statistics
    # ==========================================================================
    print(f'\n{"="*80}')
    print('MULTI-TRIAL STATISTICS')
    print(f'{"="*80}\n')
    
    # AURC statistics
    mean_aurc = np.mean(all_aurcs)
    std_aurc = np.std(all_aurcs, ddof=1)  # Sample std
    best_aurc = min(all_aurcs)
    best_trial = all_aurcs.index(best_aurc) + 1
    
    print(f'AURC Results ({len(args.seeds)} trials):')
    print(f'  Mean:   {mean_aurc:.6f}')
    print(f'  Std:    {std_aurc:.6f}')
    print(f'  Best:   {best_aurc:.6f} (Trial {best_trial}, seed={args.seeds[best_trial-1]})')
    print(f'  Range:  [{min(all_aurcs):.6f}, {max(all_aurcs):.6f}]')
    
    # Per-coverage statistics
    print(f'\n{"="*80}')
    print('COVERAGE VS ERROR: Mean ± Std')
    print(f'{"="*80}\n')
    
    # Baseline for comparison
    baseline_test = {
        100: 6.037, 99: 5.442, 98: 4.974, 97: 4.497, 95: 3.645,
        90: 2.069, 85: 1.103, 80: 0.734, 75: 0.467, 70: 0.339,
        60: 0.229, 50: 0.200, 40: 0.156, 30: 0.208, 20: 0.250, 10: 0.250
    }
    baseline_aurc = compute_aurc(baseline_test)
    
    print(f'Coverage  | SCSF Mean±Std    | Baseline     | SCSF vs Baseline')
    print('-'*75)
    
    best_results = all_results[best_trial - 1]
    for cov in COVERAGE_POINTS:
        errors = [r[cov] for r in all_results]
        mean_err = np.mean(errors)
        std_err = np.std(errors, ddof=1)
        baseline_err = baseline_test[cov]
        diff = mean_err - baseline_err
        
        print(f'{cov:>7}%  | {mean_err:>6.3f}±{std_err:>5.3f}%  | {baseline_err:>10.3f}%  | {diff:>+9.3f}%')
    
    # ==========================================================================
    # Final summary
    # ==========================================================================
    print(f'\n{"="*80}')
    print('FINAL SUMMARY')
    print(f'{"="*80}\n')
    
    improvement = (1 - mean_aurc / baseline_aurc) * 100
    best_improvement = (1 - best_aurc / baseline_aurc) * 100
    
    print(f'Baseline AURC:      {baseline_aurc:.6f}')
    print(f'SCSF AURC (mean):   {mean_aurc:.6f} ± {std_aurc:.6f}')
    print(f'SCSF AURC (best):   {best_aurc:.6f}')
    print(f'\nImprovement (mean): {improvement:+.2f}%')
    print(f'Improvement (best): {best_improvement:+.2f}%')
    
    if mean_aurc < baseline_aurc:
        print(f'\n✓ SCSF BEATS baseline!')
        print(f'  Mean: {improvement:.1f}% improvement')
        print(f'  Best: {best_improvement:.1f}% improvement')
    else:
        print(f'\n✗ SCSF is worse than baseline')
    
    # ==========================================================================
    # Save best trial results
    # ==========================================================================
    csv_path = os.path.join(os.path.dirname(args.checkpoint), 'coverage_vs_err_best.csv')
    with open(csv_path, 'w') as f:
        f.write(f'# Best trial: seed={args.seeds[best_trial-1]}, AURC={best_aurc:.6f}\n')
        f.write(f'# Mean AURC: {mean_aurc:.6f} ± {std_aurc:.6f}\n')
        for cov in COVERAGE_POINTS:
            err = best_results[cov]
            f.write(f'test{cov:.0f},{cov:.2f},{err:.3f}\n')
    
    print(f'\n✓ Best trial results saved to: {csv_path}')
    
    return mean_aurc, std_aurc, best_aurc

if __name__ == '__main__':
    main()
