# Verification Report: DualAug Implementation & CCL-SC Comparison Fairness

Code references: [`train_fresh_sc.py`](file:///home/viet2005/workspace/Research/SCSF/train_fresh_sc.py) · [`train_cbr_scsf.py`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py) · [`train_scsf.py`](file:///home/viet2005/workspace/Research/SCSF/train_scsf.py) · [`eval_cbr_multi_trial.py`](file:///home/viet2005/workspace/Research/SCSF/eval_cbr_multi_trial.py)

---

## 1. Training Hyperparameter Parity

Both `dualaug` (and all its ablations) and `CBR-SCSF` (CCL-SC) share **exactly the same training configuration**, since all methods inherit from the same `get_dataset()`, optimizer, and scheduler setup.

| Hyperparameter | DualAug | CCL-SC | Match? |
| :--- | :---: | :---: | :---: |
| Backbone | `VGG16BN_FeatureExtractor` | `VGG16BN_FeatureExtractor` | ✅ |
| Confidence head | `RawConfidenceHead(2048, 512, C)` | `RawConfidenceHead(2048, 512, C)` | ✅ |
| Backbone optimizer | SGD | SGD | ✅ |
| Learning rate | 0.1 | 0.1 | ✅ |
| Momentum | 0.9 | 0.9 | ✅ |
| Weight decay | 5e-4 | 5e-4 | ✅ |
| Head optimizer | Adam, lr=1e-3 | Adam, lr=1e-3 | ✅ |
| LR schedule | `MultiStepLR(range(25,300,25), γ=0.5)` | `MultiStepLR(range(25,300,25), γ=0.5)` | ✅ |
| Total epochs | 300 | 300 | ✅ |
| CE-only pretrain | 100 epochs | 100 epochs | ✅ |
| BCE ramp-in | 20 epochs | 20 epochs | ✅ |
| Batch size | 128 | 128 | ✅ |
| Grad clip norm | 5.0 | 5.0 | ✅ |

---

## 2. Data Augmentation Parity

Both methods call the **same `get_dataset()` function** from `train_scsf.py`. The training transforms are identical:

```python
# CIFAR-10 and CIFAR-100 — used by both methods
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

The **horizontal flip** used by dualaug during its BCE step is computed **in-loop** on the already-loaded batch (`inputs.flip(-1)`), not from a separate DataLoader. It is an algorithmic component of the method, not an additional data augmentation advantage.

> [!NOTE]
> `inputs.flip(-1)` is a deterministic tensor op applied inside the loss function, not a stochastic DataLoader augmentation. Every variant sees the same random crops/flips from the loader; dualaug additionally uses the flipped view of that same batch for computing the agreement BCE term.

---

## 3. Model Architecture Parity

### 3.1 Backbone

Both use `VGG16BN_FeatureExtractor` ([`train_scsf.py:L375`](file:///home/viet2005/workspace/Research/SCSF/train_scsf.py#L375-L449)), which wraps `vgg16_bn` and exposes two pooled feature vectors:

- **pool4**: after 4th MaxPool → `AdaptiveAvgPool2d(2,2)` → flattened → **2048-dim**
- **pool5**: after 5th MaxPool → 1×1 spatial → flattened → **512-dim**

```python
backbone = VGG16BN_FeatureExtractor(num_classes=num_classes, input_size=32)
```

Identical constructor call in both trainers. ✅

### 3.2 Confidence Head

Both use `RawConfidenceHead` ([`train_cbr_scsf.py:L51`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py#L51-L91)):

```
Input: [pool4(2048) | pool5(512) | logits.detach()(C)]   → 2560+C dim
Body:   Linear → ReLU → Dropout(0.3)  ×4 layers
Output: Linear(128 → 1) — one unbounded confidence logit
```

Critical detail: **logits are always `.detach()`ed** before entering the head ([`train_cbr_scsf.py:L83`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py#L83)):

```python
parts = [pool4, pool5, logits.detach()]   # logits: read-only to the head
```

This applies to **all methods** — the head never sends gradient into the backbone through logits. The `dualaug` vs `dualaug_detach` difference is whether **pool4/pool5** are detached, not logits.

---

## 4. AURC Computation Parity

### DualAug — `metrics_from_scores()` [`train_fresh_sc.py:L201`](file:///home/viet2005/workspace/Research/SCSF/train_fresh_sc.py#L201)

```python
order = torch.argsort(scores, descending=True, stable=True)
sorted_errors = (~correctness[order]).float()
prefix = sorted_errors.cumsum(0) / torch.arange(1, len(scores)+1, dtype=torch.float32)
aurc = float(prefix.mean())
```

### CBR-SCSF — `evaluate()` [`train_cbr_scsf.py:L462`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py#L462)

```python
prefix_risk = sorted_errors.cumsum(0) / torch.arange(1, len(scores)+1, dtype=torch.float32)
"aurc": prefix_risk.mean().item()
```

Both implement the **identical prefix-AURC formula**. ✅

---

## 5. Gradient Flow into Backbone

The key architectural difference between `dualaug` and `dualaug_detach` also applies when comparing to CBR:

| Method | pool4/pool5 gradient from BCE into backbone? |
| :--- | :---: |
| `dualaug` | ✅ Yes — deep supervision active |
| `dualaug_detach` | ❌ No — `pool4.detach(), pool5.detach()` before head |
| `dualce_scsf`, `scsf` | ✅ Yes |
| **CBR-SCSF** | ✅ Yes — pool4/pool5 not detached in `train_epoch` |

**CBR uses the same deep supervision as `dualaug`** (not `dualaug_detach`). Since `dualaug_detach` outperforms `dualaug` on CIFAR-100 (AURC 0.0719 vs 0.0743), CBR is also subject to this same gradient-interference penalty. This is an informational finding — no fairness issue, just a potential improvement for CBR.

---

## 6. Checkpoint Selection Policy

Both methods save `last.pth` each epoch and `best.pth` on val AURC improvement. Both **report results from `last.pth` (epoch 300)**:

```python
# train_fresh_sc.py:L777  and  train_cbr_scsf.py:L693
checkpoint = torch.load(last_path, map_location=device, weights_only=False)
```

Results JSON: `"eval_checkpoint": "last"`. ✅

---

## 7. ⚠️ Fairness Issue: Test Set Mismatch

> [!CAUTION]
> **DualAug and CBR-SCSF are evaluated on different test sets.** This is the one genuine fairness concern identified.

### How `get_dataset()` splits the data

```python
# train_scsf.py:L731-732, L754-755
torch.manual_seed(args.seed)
valset, testset_final = random_split(testset, [2000, 8000])
# ...
return trainloader, valloader, testloader, testloader_full, num_classes
#                                ↑ 8k          ↑ full 10k
```

| Loader | Size | Purpose |
| :--- | :---: | :--- |
| `val_loader` | 2,000 | Val AURC monitoring during training |
| `test_loader` | 8,000 | CBR-SCSF final evaluation |
| `test_loader_full` | 10,000 | DualAug final evaluation |

### How each method uses these loaders

**DualAug** ([`train_fresh_sc.py:L647`](file:///home/viet2005/workspace/Research/SCSF/train_fresh_sc.py#L647), [`L800`](file:///home/viet2005/workspace/Research/SCSF/train_fresh_sc.py#L800)):

```python
train_loader, val_loader, test_loader, test_loader_full, num_classes = get_dataset(args)
# ...
test = evaluate_fresh(..., test_loader_full, ...)   # FULL 10k
results["eval_test_set"] = "official_10k"
```

**CBR-SCSF** ([`train_cbr_scsf.py:L601`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py#L601), [`L706`](file:///home/viet2005/workspace/Research/SCSF/train_cbr_scsf.py#L706)):

```python
train_loader, val_loader, test_loader, _, num_classes = get_dataset(args)
#                                               ↑ testloader_full discarded
# ...
test = evaluate(..., test_loader, ...)   # Only 8k split
```

### Consequences

1. **Different N**: DualAug AURC is computed over 10,000 samples; CBR over 8,000. The smaller N gives CBR ~12% higher variance in its AURC estimate.
2. **Val samples in DualAug test**: The 2,000 val samples used for monitoring during training are included in DualAug's 10k test. Since `last.pth` (not `best.pth`) is used for reporting, direct selection bias is minimal — but these 2k samples were observed during training.
3. **`eval_cbr_multi_trial.py` already has the fix**: it loads the **full official 10k** independently of the training split:
   ```python
   testset = dataset_cls(root=data_root, train=False, download=False, transform=transform)
   loader = DataLoader(testset, batch_size=200, ...)   # full 10k ✅
   # → use "full_test.prefix_aurc" for comparison
   ```

---

## 8. Summary & Action Items

### ✅ Verified correct and fair

| Check | Status |
| :--- | :---: |
| Same backbone (VGG16BN) | ✅ |
| Same head architecture (RawConfidenceHead) | ✅ |
| Same train data augmentation | ✅ |
| Same backbone optimizer (SGD lr=0.1, wd=5e-4, m=0.9) | ✅ |
| Same head optimizer (Adam lr=1e-3) | ✅ |
| Same LR schedule (MultiStepLR every 25 epochs, γ=0.5) | ✅ |
| Same training length (300 ep, 100 pretrain, 20 ramp) | ✅ |
| Same batch size (128) | ✅ |
| Same gradient clipping (norm=5.0) | ✅ |
| Logits always detached inside confidence head | ✅ |
| Same AURC formula (prefix, stable sort) | ✅ |
| Same checkpoint policy (last.pth, epoch 300) | ✅ |
| dualaug `inputs.flip(-1)` is in-loop, not a DataLoader augmentation | ✅ |

### ⚠️ Issues to fix before final comparison

| Issue | Severity | Fix |
| :--- | :---: | :--- |
| CBR reports 8k AURC; DualAug reports 10k AURC | **HIGH** | Re-evaluate CBR with `eval_cbr_multi_trial.py`, use `full_test.prefix_aurc` |
| Val 2k is part of DualAug's 10k test | **LOW** | Acceptable if using `last.pth`; note it in paper |
| CBR uses deep supervision (same as `dualaug`, not `dualaug_detach`) | **Informational** | Mention as a potential improvement for CBR |

### Recommended fix command

```bash
# Run on the VastAI instance for each CBR checkpoint:
python eval_cbr_multi_trial.py \
    --checkpoint save/<cbr_run>/last.pth \
    -d cifar100 \
    --output save/<cbr_run>/multi_trial_10k.json

# The comparable number is:
# → payload["full_test"]["prefix_aurc"]   (full 10k, same as dualaug)
# NOT the shuffle_split_8k numbers
```
