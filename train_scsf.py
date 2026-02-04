#!/usr/bin/env python
"""
SCSF: Self-Calibrated Selective Framework
==========================================

A Meta-Learning approach for Selective Classification that replaces
the Gambler's Loss with a learned confidence scorer.

Key Innovations:
  1. NO C+1 reservation neuron - pure C=10 classification
  2. Meta-Calibrator predicts True Class Probability (TCP)
  3. RL Controller tunes decision threshold τ
  4. Confidence = Meta-Calibrator output (not softmax max)

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  VGG16_bn Backbone (C=10 outputs, NO reservation)          │
  │  ├─ Pool4 features (512×2×2) ──┐                           │
  │  ├─ Pool5 features (512×1×1) ──┼──→ Meta-Calibrator        │
  │  └─ Logits (10-dim) ───────────┘         ↓                 │
  │                                    ĉ(x) ∈ [0,1]            │
  │                                          ↓                 │
  │                              Reject if ĉ(x) < τ            │
  └─────────────────────────────────────────────────────────────┘

Training Protocol:
  - Phase 1 (Epoch 1-100): Cross-Entropy warmup (backbone only)
  - Phase 2 (Epoch 101-300): Joint training (CE + Meta-Loss) + RL threshold tuning

Evaluation:
  - Rank by ĉ(x) from Meta-Calibrator
  - Compute Error @ Coverage and AURC
"""

import argparse
import os
import sys
import random
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Add parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import Bar, Logger, AverageMeter, mkdir_p


# =============================================================================
# Calibration Loss Functions
# =============================================================================

def focal_mse_loss(confidence, target, gamma=2.0):
    """
    Focal MSE Loss: Focuses more on hard-to-calibrate samples.
    
    Loss = |confidence - target|^gamma * MSE(confidence, target)
    
    Samples with large calibration error get higher weight.
    """
    error = torch.abs(confidence - target)
    focal_weight = error.pow(gamma)
    mse = (confidence - target).pow(2)
    return (focal_weight * mse).mean()


def margin_calibration_loss(confidence, correctness, margin=0.3):
    """
    Margin-based Calibration Loss: Penalizes overconfidence on wrong predictions.
    
    For CORRECT predictions: confidence should be HIGH (> 1 - margin)
    For WRONG predictions: confidence should be LOW (< margin)
    
    Uses hinge-style loss to enforce separation.
    """
    # Correctness is 1 for correct, 0 for wrong
    correct_mask = correctness > 0.5
    wrong_mask = ~correct_mask
    
    loss = 0.0
    n_samples = 0
    
    if correct_mask.sum() > 0:
        # Correct: penalize if confidence < (1 - margin)
        correct_loss = F.relu((1 - margin) - confidence[correct_mask]).pow(2)
        loss = loss + correct_loss.sum()
        n_samples += correct_mask.sum()
    
    if wrong_mask.sum() > 0:
        # Wrong: penalize if confidence > margin
        wrong_loss = F.relu(confidence[wrong_mask] - margin).pow(2)
        loss = loss + wrong_loss.sum()
        n_samples += wrong_mask.sum()
    
    return loss / (n_samples + 1e-8)


def ece_soft_loss(confidence, correctness, n_bins=15):
    """
    Soft differentiable ECE (Expected Calibration Error) approximation.
    
    Uses soft binning with Gaussian kernels instead of hard bins.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = 1.0 / n_bins
    
    # Soft assignment to bins using Gaussian kernel
    sigma = bin_width / 2
    weights = torch.exp(-0.5 * ((confidence.unsqueeze(1) - bin_centers) / sigma).pow(2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
    
    # Weighted average confidence and accuracy per bin
    weighted_conf = (weights * confidence.unsqueeze(1)).sum(dim=0)
    weighted_acc = (weights * correctness.unsqueeze(1)).sum(dim=0)
    bin_counts = weights.sum(dim=0)
    
    # ECE = sum over bins of |avg_conf - avg_acc| * bin_proportion
    avg_conf = weighted_conf / (bin_counts + 1e-8)
    avg_acc = weighted_acc / (bin_counts + 1e-8)
    
    ece = (torch.abs(avg_conf - avg_acc) * bin_counts).sum() / (bin_counts.sum() + 1e-8)
    
    return ece


def smoothed_tcp_target(logits, targets, label_smoothing=0.1):
    """
    Label-smoothed TCP target to prevent overconfidence.
    
    Instead of using raw softmax(logits)[target], applies label smoothing:
    smoothed_tcp = (1 - label_smoothing) * tcp + label_smoothing * (1 / num_classes)
    
    This prevents the target from being too close to 1.0.
    """
    probs = F.softmax(logits, dim=1)
    tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    num_classes = logits.size(1)
    smoothed = (1 - label_smoothing) * tcp + label_smoothing * (1.0 / num_classes)
    return smoothed


def avuc_loss(confidence, correctness, penalty_weight=2.0):
    """
    Accuracy vs Uncertainty Calibration Loss.
    
    Heavily penalizes:
    - High confidence + Wrong prediction (overconfident errors)
    - Low confidence + Correct prediction (underconfident correct)
    
    This directly targets the problematic cases in the calibration curve.
    """
    # Overconfident wrong: confidence > 0.5 but wrong
    wrong_mask = correctness < 0.5
    overconf_wrong = F.relu(confidence - 0.5) * wrong_mask.float()
    
    # Underconfident correct: confidence < 0.5 but correct  
    correct_mask = correctness > 0.5
    underconf_correct = F.relu(0.5 - confidence) * correct_mask.float()
    
    # Weighted penalty for critical errors
    loss = penalty_weight * overconf_wrong.pow(2).mean() + underconf_correct.pow(2).mean()
    
    return loss


def compute_spatial_variance(feat):
    """
    Compute per-channel spatial variance as uncertainty signal.
    
    Hypothesis: When a network is uncertain, different spatial regions
    of the feature map "disagree" about the classification.
    High spatial variance = spatial disagreement = uncertainty.
    
    Args:
        feat: (B, C, H, W) feature map from conv layer
    
    Returns:
        variance: (B, C) per-channel spatial variance
    """
    # feat: (B, C, H, W)
    # Compute mean across spatial dimensions
    mean = feat.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    
    # Compute variance across spatial dimensions
    variance = ((feat - mean) ** 2).mean(dim=(2, 3))  # (B, C)
    
    return variance


# Constants
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
COVERAGE_POINTS = [100, 99, 98, 97, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]


# =============================================================================
# VGG16_bn with Feature Extraction (C=10, NO reservation)
# =============================================================================

from models.cifar import vgg

class VGG16BN_FeatureExtractor(nn.Module):
    """
    Wrapper around original VGG16_bn to expose intermediate features.
    Uses EXACTLY C outputs (no C+1 reservation neuron).
    Supports both 32x32 (CIFAR/SVHN) and 64x64 (Cats vs Dogs) inputs.
    """
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        
        # Use original VGG16_bn from Deep Gamblers
        self.base_model = vgg.vgg16_bn(num_classes=num_classes, input_size=input_size)
        self.input_size = input_size
        
        # Find MaxPool indices
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


# =============================================================================
# Meta-Calibrator Module
# =============================================================================

class MetaCalibrator(nn.Module):
    """
    Meta-Calibrator: Predicts True Class Probability (TCP).
    
    Input: Concatenated features from:
      - Pool4 (512×2×2 = 2048-dim)
      - Pool5 (512×1×1 = 512-dim)  
      - Logits (10-dim)
    
    Output: Confidence score ĉ(x) ∈ [0, 1]
    
    Target: TCP = p(y*|x) = softmax(logits)[y*]
    Loss: MSE(ĉ(x), TCP)
    
    Modes:
        - Default (detach): Post-hoc calibration, no gradients to backbone
        - End-to-end: Full gradient flow for joint feature-calibration optimization
    """
    def __init__(self, pool4_dim=512*4, pool5_dim=512*1, logit_dim=10, hidden_dim=256, 
                 logits_only=False, spatial_var=False, end_to_end=False):
        super().__init__()
        
        self.logits_only = logits_only
        self.spatial_var = spatial_var
        self.end_to_end = end_to_end
        
        if logits_only:
            input_dim = logit_dim  # Only logits (10)
        elif spatial_var:
            # Spatial variance: 512 (from pool4 channels) + 10 (logits) = 522
            input_dim = 512 + logit_dim
        else:
            input_dim = pool4_dim + pool5_dim + logit_dim  # 2048 + 512 + 10 = 2570
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        ) #-> conv net, attention block
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, feat_pool4, feat_pool5, logits):
        """
        Args:
            feat_pool4: (B, 512, 2, 2) or None if logits_only
            feat_pool5: (B, 512, 1, 1) or None if logits_only
            logits: (B, 10)
        Returns:
            confidence: (B,) in [0, 1]
        
        Notes:
            - Default mode: detach() prevents gradients from flowing to backbone
            - End-to-end mode: No detach, full gradient flow for joint optimization
        """
        if self.logits_only:
            # Use only logits
            if self.end_to_end:
                combined = logits  # End-to-end: allow gradients
            else:
                combined = logits.detach()  # Post-hoc: block gradients
        elif self.spatial_var:
            # Compute spatial variance from pool4 features
            # pool4 shape: (B, 512, 2, 2) for 32x32 input
            spatial_variance = compute_spatial_variance(feat_pool4)  # (B, 512)
            
            if self.end_to_end:
                combined = torch.cat([spatial_variance, logits], dim=1)  # (B, 522)
            else:
                combined = torch.cat([spatial_variance.detach(), logits.detach()], dim=1)
        else:
            # Flatten features
            flat_pool4 = feat_pool4.view(feat_pool4.size(0), -1)  # (B, 2048)
            flat_pool5 = feat_pool5.view(feat_pool5.size(0), -1)  # (B, 512)
            
            if self.end_to_end:
                # End-to-end: allow gradients to flow back to backbone
                combined = torch.cat([flat_pool4, flat_pool5, logits], dim=1)  # (B, 2570)
            else:
                # Post-hoc: detach all inputs to prevent backbone modification
                combined = torch.cat([flat_pool4.detach(), flat_pool5.detach(), logits.detach()], dim=1)
        
        # Predict confidence
        confidence = self.network(combined).squeeze(-1)  # (B,)
        
        return confidence


# =============================================================================
# PPO Agent for Threshold Tuning
# =============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim, action_dim=2, hidden_dim=64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: outputs mean for actions (threshold τ, meta_loss_weight)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        shared = self.shared(state)
        action_mean = torch.sigmoid(self.actor_mean(shared))  # Actions in [0, 1]
        action_std = torch.exp(self.actor_log_std.clamp(-2, 0))  # Small std
        value = self.critic(shared)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, _ = self.forward(state)
        if deterministic:
            return action_mean, None
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)  # Clamp to valid range
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """PPO Agent for tuning threshold τ and meta_loss_weight."""
    
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, device='cuda'):
        self.gamma = gamma
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.device = device
        
        # Action dim = 2: [threshold τ, meta_loss_weight]
        self.actor_critic = ActorCritic(state_dim, action_dim=2).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor_critic.get_action(state_tensor)
        return action.cpu().numpy().squeeze(), log_prob.item() if log_prob is not None else 0.0
    
    def store_transition(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def update(self, n_epochs=4):
        if len(self.states) < 2:
            return {'policy_loss': 0, 'value_loss': 0}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(n_epochs):
            log_probs, values, entropy = self.actor_critic.evaluate(states, actions)
            
            # Advantages
            advantages = returns - values.detach()
            
            # Policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        
        return {
            'policy_loss': total_policy_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs
        }


# =============================================================================
# Dataset Loading
# =============================================================================

def get_dataset(args):
    """
    Load dataset with proper splits:
    - Train: training samples
    - Val: 2,000 samples (for RL reward)
    - Test: remaining samples (for final evaluation)
    
    Supported: cifar10, svhn, catsdogs
    """
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    if args.dataset == 'cifar10':
        # CIFAR-10 normalization
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
        
        # Split test set: 20% val (2000) + 80% test (8000)
        # Training eval uses FULL test set (10000)
        # RL uses val (2000), Final eval uses test (8000)
        torch.manual_seed(args.seed)
        valset, testset_final = random_split(testset, [2000, 8000])
        print(f'CIFAR-10: Train={len(trainset)}, Full_Test={len(testset)}, Val={len(valset)}, Final_Test={len(testset_final)}')
        
    elif args.dataset == 'svhn':
        # SVHN normalization - MUST match baseline (0.5, 0.5, 0.5)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # SVHN data is in data/svhn/ folder
        svhn_root = os.path.join(data_root, 'svhn')
        trainset = datasets.SVHN(root=svhn_root, split='train', download=False, transform=transform_train)
        testset = datasets.SVHN(root=svhn_root, split='test', download=False, transform=transform_test)
        num_classes = 10
        
        # Split test set: val=5000 (paper setting), test=21032
        # Paper: "The validation set sizes for SVHN, CIFAR-10 and Cats vs. Dogs 
        # are respectively 5000, 2000 and 2000."
        torch.manual_seed(args.seed)
        test_size = len(testset)  # 26032
        val_size = 5000  # Paper setting for SVHN
        test_final_size = test_size - val_size  # 21032
        valset, testset_final = random_split(testset, [val_size, test_final_size])
        print(f'SVHN: Train={len(trainset)}, Full_Test={test_size}, Val={val_size}, Final_Test={test_final_size}')
        
    elif args.dataset == 'catsdogs':
        # Cats vs Dogs normalization (ImageNet stats) - MUST match baseline
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Baseline uses 64x64 images with RandomCrop(64, padding=6)
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # Load Cats vs Dogs using ImageFolder (baseline format)
        # Dataset should be in: data/cats_dogs/train and data/cats_dogs/test
        cats_dogs_root = os.path.join(data_root, 'cats_dogs')
        train_path = os.path.join(cats_dogs_root, 'train')
        test_path = os.path.join(cats_dogs_root, 'test')
        
        assert os.path.exists(train_path), f"Train folder not found: {train_path}. Please organize dataset as: data/cats_dogs/train/"
        assert os.path.exists(test_path), f"Test folder not found: {test_path}. Please organize dataset as: data/cats_dogs/test/"
        
        # Load datasets
        trainset_raw = datasets.ImageFolder(train_path)
        testset_raw = datasets.ImageFolder(test_path)
        
        # Resize to 64x64 using center crop (matching baseline)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import dataset_utils
        
        trainset = dataset_utils.resized_dataset(trainset_raw, transform_train, resize=64)
        testset = dataset_utils.resized_dataset(testset_raw, transform_test, resize=64)
        num_classes = 2
        
        # Split test set: val=2000 (paper setting), test=remaining
        # Paper: "The validation set sizes for SVHN, CIFAR-10 and Cats vs. Dogs 
        # are respectively 5000, 2000 and 2000."
        torch.manual_seed(args.seed)
        test_size = len(testset)
        val_size = 2000  # Paper setting for Cats vs Dogs
        test_final_size = test_size - val_size
        valset, testset_final = random_split(testset, [val_size, test_final_size])
        print(f'Cats vs Dogs: Train={len(trainset)}, Full_Test={test_size}, Val={val_size}, Final_Test={test_final_size}')
        
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=200, shuffle=False, 
                           num_workers=args.workers, pin_memory=True)
    testloader = DataLoader(testset_final, batch_size=200, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)
    
    # Full test set for training-time evaluation
    testloader_full = DataLoader(testset, batch_size=200, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    
    return trainloader, valloader, testloader, testloader_full, num_classes


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_selective_risk(model, meta_cal, loader, coverage_points, device):
    """
    Calculate Error Rate at each coverage point using Meta-Calibrator confidence.
    
    Confidence = ĉ(x) from Meta-Calibrator
    
    Returns:
        coverage_errors: dict mapping coverage → error rate
        global_acc: overall accuracy
        global_aurc: Area Under Risk-Coverage curve
        mean_confidence: average Meta-Calibrator output
        mean_tcp: average True Class Probability
    """
    model.eval()
    meta_cal.eval()
    
    all_scores = []  # (confidence, correct, tcp)
    total_correct = 0
    total_samples = 0
    total_confidence = 0
    total_tcp = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward with features
            logits, feat_pool4, feat_pool5 = model(inputs, return_features=True)
            confidence = meta_cal(feat_pool4, feat_pool5, logits)
            
            # Compute softmax and TCP
            probs = F.softmax(logits, dim=1)
            tcp = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # True class prob
            
            # Predictions
            _, preds = logits.max(dim=1)
            correct = preds.eq(targets)
            
            for i in range(len(targets)):
                all_scores.append((
                    confidence[i].item(),
                    correct[i].item(),
                    tcp[i].item()
                ))
            
            total_correct += correct.sum().item()
            total_samples += len(targets)
            total_confidence += confidence.sum().item()
            total_tcp += tcp.sum().item()
    
    # Sort by confidence (descending)
    all_scores_sorted = sorted(all_scores, key=lambda x: x[0], reverse=True)
    n = len(all_scores_sorted)
    
    # Compute error at each coverage point
    coverage_errors = {}
    for cov in coverage_points:
        k = max(1, int(n * cov / 100))
        top_k = all_scores_sorted[:k]
        n_errors = sum(1 for _, correct, _ in top_k if not correct)
        error_rate = n_errors / k * 100
        coverage_errors[cov] = error_rate
    
    # Compute AURC
    aurc = 0.0
    sorted_risks = []
    for i in range(1, n + 1):
        n_errors = sum(1 for _, correct, _ in all_scores_sorted[:i] if not correct)
        risk = n_errors / i
        sorted_risks.append(risk)
    
    for i in range(len(sorted_risks) - 1):
        aurc += (sorted_risks[i] + sorted_risks[i + 1]) / 2 * (1 / n)
    
    global_acc = total_correct / total_samples * 100
    mean_confidence = total_confidence / total_samples
    mean_tcp = total_tcp / total_samples
    
    return coverage_errors, global_acc, aurc, mean_confidence, mean_tcp


def evaluate_with_threshold(model, meta_cal, loader, threshold, device):
    """Evaluate with a specific rejection threshold."""
    model.eval()
    meta_cal.eval()
    
    accepted = 0
    accepted_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, feat_pool4, feat_pool5 = model(inputs, return_features=True)
            confidence = meta_cal(feat_pool4, feat_pool5, logits)
            
            _, preds = logits.max(dim=1)
            correct = preds.eq(targets)
            
            # Accept if confidence >= threshold
            accept_mask = confidence >= threshold
            accepted += accept_mask.sum().item()
            accepted_correct += (correct & accept_mask).sum().item()
            total += len(targets)
    
    coverage = accepted / total * 100
    accuracy = accepted_correct / accepted * 100 if accepted > 0 else 0
    error = 100 - accuracy
    
    return coverage, error, accuracy


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch_warmup(model, trainloader, optimizer, device):
    """Phase 1: Train backbone with standard Cross-Entropy."""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    bar = Bar('Warmup', max=len(trainloader))
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward (no features needed for warmup)
        logits = model(inputs, return_features=False)
        
        # Standard Cross-Entropy
        loss = F.cross_entropy(logits, targets)
        
        # Accuracy
        _, preds = logits.max(dim=1)
        acc = preds.eq(targets).float().mean().item() * 100
        
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bar.suffix = '({batch}/{size}) | Loss: {loss:.4f} | Acc: {acc:.2f}'.format(
            batch=batch_idx+1, size=len(trainloader), loss=losses.avg, acc=accuracies.avg)
        bar.next()
    
    bar.finish()
    
    return losses.avg, accuracies.avg


def compute_gradient_norms(model):
    """Compute gradient norm statistics for backbone model.
    
    Returns:
        dict: Contains total_norm, max_layer_norm, and layer_norms
    """
    total_norm = 0.0
    layer_norms = {}
    max_layer_norm = 0.0
    max_layer_name = ''
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            layer_norms[name] = param_norm
            if param_norm > max_layer_norm:
                max_layer_norm = param_norm
                max_layer_name = name
    
    total_norm = total_norm ** 0.5
    return {
        'total_norm': total_norm,
        'max_layer_norm': max_layer_norm,
        'max_layer_name': max_layer_name,
        'layer_norms': layer_norms
    }


def train_epoch_joint(model, meta_cal, trainloader, optimizer, meta_optimizer, 
                      meta_loss_weight, device, use_brier=False, calib_args=None, track_grads=False):
    """
    Phase 2: Joint training with CE + Meta-Loss.
    
    Meta-Loss options:
      - TCP (default): MSE(ĉ(x), TCP) where TCP = p(y*|x) = softmax(logits)[target]
      - Brier Score: MSE(ĉ(x), correctness) where correctness = 1 if pred==target else 0
    
    Calibration Enhancement Options:
      - focal: Focal MSE loss (focus on hard samples)
      - margin: Margin-based separation loss
      - ece_reg: Soft ECE regularization
      - label_smooth: Label smoothing for TCP target
      - avuc: AVUC loss (penalize overconfident wrong predictions)
    
    Args:
        track_grads: If True, compute and return gradient norm statistics
    
    Returns:
        Tuple of (train_loss, ce_loss, meta_loss, accuracy, confidence, [grad_stats])
    """
    if calib_args is None:
        calib_args = {}
    
    model.train()
    meta_cal.train()
    
    ce_losses = AverageMeter()
    meta_losses = AverageMeter()
    total_losses = AverageMeter()
    accuracies = AverageMeter()
    confidences = AverageMeter()
    
    bar = Bar('Joint', max=len(trainloader))
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward with features
        logits, feat_pool4, feat_pool5 = model(inputs, return_features=True)
        confidence = meta_cal(feat_pool4, feat_pool5, logits)
        
        # Cross-Entropy loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # Compute correctness for calibration losses
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            correctness = (preds == targets).float()  # (B,)
        
        # Meta target computation
        if use_brier:
            # Brier Score: target is binary correctness
            meta_target = correctness.clone()
        else:
            # TCP with optional label smoothing
            if calib_args.get('label_smooth', 0) > 0:
                meta_target = smoothed_tcp_target(logits, targets, calib_args['label_smooth'])
            else:
                probs = F.softmax(logits, dim=1)
                meta_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Primary Meta-Loss: MSE or Focal MSE
        if calib_args.get('focal', 0) > 0:
            meta_loss = focal_mse_loss(confidence, meta_target.detach(), gamma=calib_args['focal'])
        else:
            meta_loss = F.mse_loss(confidence, meta_target.detach())
        
        # Additional calibration losses
        calib_loss = 0.0
        
        if calib_args.get('margin', 0) > 0:
            calib_loss = calib_loss + calib_args['margin'] * margin_calibration_loss(
                confidence, correctness, margin=0.3)
        
        if calib_args.get('ece_reg', 0) > 0:
            calib_loss = calib_loss + calib_args['ece_reg'] * ece_soft_loss(
                confidence, correctness, n_bins=15)
        
        if calib_args.get('avuc', 0) > 0:
            calib_loss = calib_loss + calib_args['avuc'] * avuc_loss(
                confidence, correctness, penalty_weight=2.0)
        
        # Total meta-related loss
        total_meta_loss = meta_loss + calib_loss
        
        # Total loss
        total_loss = ce_loss + meta_loss_weight * total_meta_loss
        
        # Accuracy
        _, preds = logits.max(dim=1)
        acc = preds.eq(targets).float().mean().item() * 100
        
        ce_losses.update(ce_loss.item(), inputs.size(0))
        meta_losses.update(meta_loss.item(), inputs.size(0))
        total_losses.update(total_loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        confidences.update(confidence.mean().item(), inputs.size(0))
        
        # Backward
        optimizer.zero_grad()
        meta_optimizer.zero_grad()
        total_loss.backward()
        
        # Track gradient norms if requested (for end-to-end stability monitoring)
        if track_grads and batch_idx == len(trainloader) - 1:
            grad_stats = compute_gradient_norms(model)
        
        optimizer.step()
        meta_optimizer.step()
        
        bar.suffix = '({batch}/{size}) | CE: {ce:.4f} | Meta: {meta:.4f} | Acc: {acc:.2f}'.format(
            batch=batch_idx+1, size=len(trainloader), ce=ce_losses.avg, meta=meta_losses.avg, acc=accuracies.avg)
        bar.next()
    
    bar.finish()
    
    if track_grads:
        return total_losses.avg, ce_losses.avg, meta_losses.avg, accuracies.avg, confidences.avg, grad_stats
    return total_losses.avg, ce_losses.avg, meta_losses.avg, accuracies.avg, confidences.avg


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SCSF: Self-Calibrated Selective Framework')
    
    # Dataset
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    
    # Training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--pretrain', default=100, type=int, help='CE warmup epochs')
    
    # Optimizer (matching baseline exactly)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    
    # RL
    parser.add_argument('--rl-freq', default=2, type=int, help='RL update frequency')
    parser.add_argument('--rl-lr', default=3e-4, type=float)
    parser.add_argument('--init-threshold', default=0.5, type=float)
    parser.add_argument('--init-meta-weight', default=1.0, type=float)
    parser.add_argument('--no-rl', action='store_true', help='Disable RL, use fixed threshold and meta_weight')
    
    # Meta-Loss Type
    parser.add_argument('--brier', action='store_true', 
                        help='Use Brier Score (confidence vs correctness) instead of MSE on TCP')
    parser.add_argument('--logits-only', action='store_true',
                        help='Meta-Calibrator uses only logits as input (no pooled features)')
    parser.add_argument('--spatial-var', action='store_true',
                        help='Use spatial variance of features as uncertainty signal (replaces pooled features)')
    parser.add_argument('--end-to-end', action='store_true',
                        help='Enable full end-to-end training (no detach, gradients flow to backbone)')
    
    # Calibration Enhancement Options
    parser.add_argument('--focal', type=float, default=0.0, metavar='GAMMA',
                        help='Use Focal MSE loss with gamma (default: 0 = disabled, try 2.0)')
    parser.add_argument('--margin', type=float, default=0.0, metavar='M',
                        help='Add margin calibration loss with weight M (default: 0 = disabled, try 0.5)')
    parser.add_argument('--ece-reg', type=float, default=0.0, metavar='W',
                        help='Add soft ECE regularization with weight W (default: 0 = disabled, try 1.0)')
    parser.add_argument('--label-smooth', type=float, default=0.0, metavar='LS',
                        help='Label smoothing for TCP target (default: 0 = disabled, try 0.1)')
    parser.add_argument('--avuc', type=float, default=0.0, metavar='W',
                        help='Add AVUC loss (penalize overconf wrong) with weight W (default: 0, try 1.0)')
    
    # Meta-Calibrator
    parser.add_argument('--meta-lr', default=1e-3, type=float)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--eval-only', action='store_true',
                        help='Load pretrained weights and evaluate only (no training)')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset
    trainloader, valloader, testloader, testloader_full, num_classes = get_dataset(args)
    
    # Determine input size based on dataset
    input_size = 64 if args.dataset == 'catsdogs' else 32
    
    # Model: VGG16_bn with C outputs (NO reservation!)
    model = VGG16BN_FeatureExtractor(num_classes=num_classes, input_size=input_size).to(device)
    
    # Meta-Calibrator dimensions depend on input size
    # For 32x32: pool4=512*4, pool5=512*1
    # For 64x64: pool4=512*16, pool5=512*4
    if input_size == 32:
        pool4_dim = 512 * 4  # 512×2×2
        pool5_dim = 512 * 1  # 512×1×1
    else:  # 64x64
        pool4_dim = 512 * 16  # 512×4×4
        pool5_dim = 512 * 4   # 512×2×2
    
    # Meta-Calibrator
    meta_cal = MetaCalibrator(
        pool4_dim=pool4_dim, 
        pool5_dim=pool5_dim, 
        logit_dim=num_classes,
        hidden_dim=256,
        logits_only=args.logits_only,
        spatial_var=args.spatial_var,
        end_to_end=args.end_to_end
    ).to(device)
    
    # Optimizers (matching baseline exactly)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    meta_optimizer = optim.Adam(meta_cal.parameters(), lr=args.meta_lr)
    
    # LR Scheduler (matching baseline)
    milestones = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    
    # RL Controller
    # State: [global_aurc, val_acc, mean_confidence, epoch_progress, class_acc_vector (10)]
    state_dim = 4 + num_classes
    ppo_agent = PPOAgent(state_dim=state_dim, lr=args.rl_lr, device=device)
    
    # Current RL-controlled parameters
    current_threshold = args.init_threshold
    current_meta_weight = args.init_meta_weight
    
    # Logging - include flags in directory name
    suffix = ''
    if args.logits_only:
        suffix += '_logitsonly'
    if args.spatial_var:
        suffix += '_spatialvar'
    if args.brier:
        suffix += '_brier'
    if args.focal > 0:
        suffix += f'_focal{args.focal}'
    if args.margin > 0:
        suffix += f'_margin{args.margin}'
    if args.ece_reg > 0:
        suffix += f'_ece{args.ece_reg}'
    if args.label_smooth > 0:
        suffix += f'_ls{args.label_smooth}'
    if args.avuc > 0:
        suffix += f'_avuc{args.avuc}'
    if args.end_to_end:
        suffix += '_e2e'
    if args.no_rl:
        suffix += '_norl'
    if args.seed != 42:
        suffix += f'_seed{args.seed}'
    save_dir = f'./save/{args.dataset}/vgg16_bn_scsf{suffix}'
    os.makedirs(save_dir, exist_ok=True)
    
    logger = Logger(os.path.join(save_dir, 'log.txt'))
    if args.end_to_end:
        logger.set_names([
            'Epoch', 'LR', 'Train_Loss', 'CE_Loss', 'Meta_Loss', 
            'Test_Err', 'AURC', 'Mean_Conf', 'Threshold', 'Meta_Weight',
            'Grad_Norm', 'Max_Layer_Grad'
        ])
    else:
        logger.set_names([
            'Epoch', 'LR', 'Train_Loss', 'CE_Loss', 'Meta_Loss', 
            'Test_Err', 'AURC', 'Mean_Conf', 'Threshold', 'Meta_Weight'
        ])
    
    # Previous state for RL
    prev_state = None
    prev_action = None
    prev_aurc = None
    
    # ==========================================================================
    # Evaluation-Only Mode
    # ==========================================================================
    if args.eval_only:
        print(f'\n{"="*80}')
        print('EVALUATION MODE: Loading pretrained weights')
        print(f'{"="*80}')
        
        checkpoint_path = os.path.join(save_dir, '300.pth')
        if not os.path.exists(checkpoint_path):
            print(f'Error: Checkpoint not found at {checkpoint_path}')
            sys.exit(1)
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        meta_cal.load_state_dict(checkpoint['meta_cal_state_dict'])
        print(f'Loaded checkpoint: {checkpoint_path}')
        print(f'Save directory: {save_dir}')
        print(f'{"="*80}\n')
        
        # Run evaluation
        model.eval()
        meta_cal.eval()
        
        coverage_points = torch.linspace(0.0, 1.0, 500)
        _, _, test_aurc, test_conf, _ = evaluate_selective_risk(
            model, meta_cal, testloader, coverage_points, device
        )
        
        # Calculate test error at full coverage
        test_err = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _, _ = model(inputs, return_features=True)
                _, preds = logits.max(dim=1)
                test_err += (~preds.eq(targets)).sum().item()
                total += targets.size(0)
        test_err = (test_err / total) * 100
        
        print(f'\n{"="*80}')
        print(f'Test Error: {test_err:.2f}%')
        print(f'Test AURC:  {test_aurc:.6f}')
        print(f'Mean Conf:  {test_conf:.3f}')
        print(f'{"="*80}')
        
        sys.exit(0)
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    print(f'\n{"="*80}')
    print('SCSF: Self-Calibrated Selective Framework')
    print(f'{"="*80}')
    print(f'Model: VGG16_bn with C={num_classes} outputs (NO reservation neuron)')
    if args.logits_only:
        meta_input = 'Logits only (10 dims)'
    elif args.spatial_var:
        meta_input = 'Spatial Variance (512) + Logits (10) = 522 dims'
    else:
        meta_input = 'Pool4 (2048) + Pool5 (512) + Logits (10) = 2570 dims'
    print(f'Meta-Calibrator Input: {meta_input}')
    meta_loss_type = 'Brier Score (correctness)' if args.brier else 'TCP (true class probability)'
    print(f'Meta-Loss Type: {meta_loss_type}')
    
    # Print training mode status
    if args.end_to_end:
        print(f'Training Mode: END-TO-END (full gradient flow to backbone)')
    else:
        print(f'Training Mode: Training-aware (backbone evolves, meta adapts w/ decoupled gradients)')
    
    # Print calibration enhancements
    calib_opts = []
    if args.focal > 0:
        calib_opts.append(f'Focal(gamma={args.focal})')
    if args.margin > 0:
        calib_opts.append(f'Margin(w={args.margin})')
    if args.ece_reg > 0:
        calib_opts.append(f'ECE-Reg(w={args.ece_reg})')
    if args.label_smooth > 0:
        calib_opts.append(f'LabelSmooth({args.label_smooth})')
    if args.avuc > 0:
        calib_opts.append(f'AVUC(w={args.avuc})')
    if calib_opts:
        print(f'Calibration Enhancements: {" + ".join(calib_opts)}')
    else:
        print('Calibration Enhancements: None (baseline MSE)')
    
    print(f'Phase 1 (Epoch 1-{args.pretrain}): Cross-Entropy Warmup')
    if args.no_rl:
        print(f'Phase 2 (Epoch {args.pretrain+1}-{args.epochs}): Joint Training (CE + Meta-Loss) [RL DISABLED]')
        print(f'Fixed Meta-Loss Weight (λ): {current_meta_weight:.2f}')
    else:
        print(f'Phase 2 (Epoch {args.pretrain+1}-{args.epochs}): Joint Training (CE + Meta-Loss) + RL')
        print(f'RL Update Frequency: Every {args.rl_freq} epochs')
        print(f'Initial Threshold: {current_threshold:.2f}')
        print(f'Initial Meta-Loss Weight: {current_meta_weight:.2f}')
    print(f'Save Directory: {save_dir}')
    print(f'{"="*80}\n')
    
    best_aurc = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # ======================================================================
        # Phase 1: CE Warmup (Epoch 1-100)
        # ======================================================================
        if epoch <= args.pretrain:
            train_loss, train_acc = train_epoch_warmup(model, trainloader, optimizer, device)
            ce_loss = train_loss
            meta_loss = 0.0
            mean_conf = 0.0
            grad_stats = None  # No gradient tracking during warmup
            
        # ======================================================================
        # Phase 2: Joint Training (Epoch 101-300)
        # ======================================================================
        else:
            # Calibration arguments
            calib_args = {
                'focal': args.focal,
                'margin': args.margin,
                'ece_reg': args.ece_reg,
                'label_smooth': args.label_smooth,
                'avuc': args.avuc,
            }
            
            # Track gradients for end-to-end mode
            if args.end_to_end:
                result = train_epoch_joint(
                    model, meta_cal, trainloader, optimizer, meta_optimizer,
                    current_meta_weight, device, use_brier=args.brier, calib_args=calib_args,
                    track_grads=True
                )
                train_loss, ce_loss, meta_loss, train_acc, mean_conf, grad_stats = result
            else:
                train_loss, ce_loss, meta_loss, train_acc, mean_conf = train_epoch_joint(
                    model, meta_cal, trainloader, optimizer, meta_optimizer,
                    current_meta_weight, device, use_brier=args.brier, calib_args=calib_args
                )
                grad_stats = None
        
        # Step scheduler
        scheduler.step()
        
        # Evaluate on full test set
        coverage_errors, test_acc, global_aurc, test_mean_conf, test_mean_tcp = \
            evaluate_selective_risk(model, meta_cal, testloader_full, COVERAGE_POINTS, device)
        
        test_err = 100 - test_acc
        
        # Track best
        if global_aurc < best_aurc:
            best_aurc = global_aurc
        
        # ======================================================================
        # RL Controller Update (after pretraining, every rl_freq epochs)
        # ======================================================================
        
        if not args.no_rl and epoch > args.pretrain and epoch % args.rl_freq == 0:
            
            # Evaluate on validation set
            val_coverage, val_acc, val_aurc, val_conf, val_tcp = \
                evaluate_selective_risk(model, meta_cal, valloader, COVERAGE_POINTS, device)
            
            # Compute per-class accuracy
            class_acc = compute_class_accuracy(model, valloader, num_classes, device)
            
            # Build state vector
            epoch_progress = (epoch - args.pretrain) / (args.epochs - args.pretrain)
            current_state = np.concatenate([
                [val_aurc * 10, val_acc / 100, val_conf, epoch_progress],
                class_acc / 100
            ])
            
            # Compute reward (AURC improvement)
            rl_reward = 0.0
            if prev_aurc is not None:
                rl_reward = 100 * (prev_aurc - val_aurc)  # Reward for decreasing AURC
            
            # Store transition
            if prev_state is not None and prev_action is not None:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(prev_state).unsqueeze(0).to(device)
                    _, _, value = ppo_agent.actor_critic(state_tensor)
                    value = value.item()
                
                ppo_agent.store_transition(prev_state, prev_action, rl_reward, value, 0.0)
            
            # Select action: [threshold, meta_loss_weight]
            action, log_prob = ppo_agent.select_action(current_state)
            
            # Apply action
            current_threshold = float(action[0])
            current_meta_weight = float(action[1]) * 2.0  # Scale to [0, 2]
            
            # Update PPO
            if len(ppo_agent.states) >= 4:
                ppo_stats = ppo_agent.update()
                print(f'  PPO Update: Policy Loss={ppo_stats["policy_loss"]:.4f}')
            
            # Store for next step
            prev_state = current_state
            prev_action = action
            prev_aurc = val_aurc
            
            print(f'  RL: τ={current_threshold:.3f}, meta_w={current_meta_weight:.3f}, reward={rl_reward:+.4f}')
        
        # Initialize prev_aurc at end of pretraining
        if not args.no_rl and epoch == args.pretrain:
            _, _, prev_aurc, _, _ = evaluate_selective_risk(
                model, meta_cal, valloader, COVERAGE_POINTS, device
            )
        
        # Log
        phase = '[Warmup]' if epoch <= args.pretrain else '[Joint]'
        
        if args.end_to_end and grad_stats is not None:
            print(f'{phase} Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | '
                  f'Err: {test_err:.2f}% | AURC: {global_aurc:.4f} | '
                  f'Conf: {test_mean_conf:.4f} | GradNorm: {grad_stats["total_norm"]:.4f}')
            logger.append([
                epoch, current_lr, train_loss, ce_loss, meta_loss,
                test_err, global_aurc, test_mean_conf, current_threshold, current_meta_weight,
                grad_stats['total_norm'], grad_stats['max_layer_norm']
            ])
        else:
            print(f'{phase} Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | '
                  f'Err: {test_err:.2f}% | AURC: {global_aurc:.4f} | '
                  f'Conf: {test_mean_conf:.4f}')
            if args.end_to_end:
                logger.append([
                    epoch, current_lr, train_loss, ce_loss, meta_loss,
                    test_err, global_aurc, test_mean_conf, current_threshold, current_meta_weight,
                    0.0, 0.0
                ])
            else:
                logger.append([
                    epoch, current_lr, train_loss, ce_loss, meta_loss,
                    test_err, global_aurc, test_mean_conf, current_threshold, current_meta_weight
                ])
        
        # Print coverage table every 20 epochs
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f'\n  {"Coverage":<10} | {"Error Rate":<12}')
            print(f'  {"-"*25}')
            for cov in COVERAGE_POINTS:
                print(f'  {cov:>8}% | {coverage_errors[cov]:>10.3f}%')
            print()
    
    # ==========================================================================
    # Final Evaluation on Test Set (80% of full test set)
    # ==========================================================================
    
    print(f'\n{"="*80}')
    print('FINAL EVALUATION ON TEST SET (80% of full test set)')
    print(f'{"="*80}')
    
    final_coverage, final_acc, final_aurc, final_conf, final_tcp = \
        evaluate_selective_risk(model, meta_cal, testloader, COVERAGE_POINTS, device)
    
    print(f'\nTotal Accuracy: {final_acc:.2f}%')
    print(f'Global AURC: {final_aurc:.4f}')
    print(f'Mean Meta-Calibrator Confidence: {final_conf:.4f}')
    print(f'Mean True Class Probability: {final_tcp:.4f}')
    print(f'Final Threshold τ: {current_threshold:.4f}')
    print(f'Final Meta-Loss Weight: {current_meta_weight:.4f}')
    
    print(f'\n{"Coverage":<10} | {"Error Rate":<12}')
    print(f'{"-"*25}')
    for cov in COVERAGE_POINTS:
        print(f'{cov:>8}% | {final_coverage[cov]:>10.3f}%')
    
    # Evaluate with learned threshold
    thr_coverage, thr_error, thr_acc = evaluate_with_threshold(
        model, meta_cal, testloader, current_threshold, device
    )
    print(f'\n{"="*80}')
    print(f'THRESHOLD-BASED REJECTION (τ = {current_threshold:.3f})')
    print(f'{"="*80}')
    print(f'Coverage: {thr_coverage:.2f}%')
    print(f'Error Rate: {thr_error:.2f}%')
    print(f'Accuracy: {thr_acc:.2f}%')
    
    # ==========================================================================
    # Save Model
    # ==========================================================================
    
    save_path = os.path.join(save_dir, '300.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'meta_cal_state_dict': meta_cal.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'meta_optimizer_state_dict': meta_optimizer.state_dict(),
        'ppo_state_dict': ppo_agent.actor_critic.state_dict(),
        'threshold': current_threshold,
        'meta_loss_weight': current_meta_weight,
        'final_aurc': final_aurc,
        'final_coverage': final_coverage,
        'args': args,
    }, save_path)
    print(f'\nSaved to: {save_path}')
    
    # Save coverage CSV
    csv_path = os.path.join(save_dir, 'coverage_vs_err.csv')
    with open(csv_path, 'w') as f:
        f.write('Coverage,Error\n')
        for cov in COVERAGE_POINTS:
            f.write(f'{cov},{final_coverage[cov]:.4f}\n')
    print(f'Saved coverage data to: {csv_path}')
    
    logger.close()
    
    print(f'\n{"="*80}')
    print('SCSF TRAINING COMPLETE!')
    print(f'{"="*80}\n')


def compute_class_accuracy(model, loader, num_classes, device):
    """Compute per-class accuracy."""
    model.eval()
    
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs, return_features=False)
            _, preds = logits.max(dim=1)
            
            for i in range(len(targets)):
                c = targets[i].item()
                class_total[c] += 1
                if preds[i] == targets[i]:
                    class_correct[c] += 1
    
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c] * 100
    
    return class_acc


if __name__ == '__main__':
    main()
