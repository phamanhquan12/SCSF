# Báo cáo thí nghiệm SCSF / acccon

So sánh với CCL-SC (Wu et al., ICML 2024). Số liệu dưới đây lấy từ checkpoint đã chạy; không trộn split 8k với official 10k.

## 1. Mục tiêu

Xuất phát từ SCSF (một head tin cậy học từ pool4/pool5 + logits stopgrad, điểm \(s\) không bị chặn, BCE-with-logits trên tính đúng/sai). Thêm một hạng contrastive có trọng số acceptance (acccon) để chỉnh feature pool5, **không** thay score lúc test. Câu hỏi: acccon có thắng CCL-SC trên CIFAR-10/100 và CelebA không?

acccon **không** phải CCL-SC đổi tên:

| | CCL-SC | acccon |
|---|---|---|
| Điểm test | softmax response (SR) | head SCSF \(s\) |
| Positive | dự đoán đúng class \(y\) | cùng nhãn thật |
| Queue | 2 queue MoCo + momentum encoder | 1 FIFO online |
| Trọng số | SR trên anchor | acceptance mềm \(w_i,w_j\) (query-weighted) |

Sửa quan trọng: **query-weighted** SupCon (không rút gọn \(w_i\) trong tỉ số từng query). Bản “cancelled \(w_i\)” cũ kém hơn ở đuôi 95%.

## 2. Protocol dùng để so với paper

- **CIFAR-10/100:** VGG16-BN, SGD 0.1, 300 epoch, warmup CE 100, ramp 20, batch 128, queue 3000, \(\eta=0.5\), \(\tau=0.1\), coverage mask \(\{0.80,0.90,0.95\}\). Đánh giá **last.pth**, **official test 10k**.
- **CelebA (Attractive):** ResNet-18, Adam \(1\mathrm{e}{-5}\) backbone / \(1\mathrm{e}{-3}\) head, 50 epoch, \(E_s=1\), queue 300, **best val accuracy** (đúng CCL-SC), official test \(n=19962\).
- **ImageNet:** chưa chạy (overlay ~100 GB, ImageNet ~140 GB).

Metric chính: accuracy, AURC, selective risk @100/95/90/85/80 (và @10 để kiểm tra core). AUROC = P(\(s_{\mathrm{đúng}} > s_{\mathrm{sai}}\)).

---

## 3. Kết quả chính vs CCL-SC

### 3.1 CIFAR-100 (dataset khó nhất, đáng tin nhất)

Official 10k, last.pth. CCL-SC là mean±std **5 seed**. acccon: seed 42 (máy local, query-weighted) + seeds 0–3 (VAST).

| Method | Acc | AURC | AUROC | @95 | @90 | @80 | @10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CCL-SC (5 seed)** | **73.45** | — | — | **23.54±0.15** | **20.97±0.20** | **16.07±0.15** | 0.36±0.08 |
| acccon seed 42 | 73.54 | 0.0763 | 0.8717 | 23.54 | 21.19 | 16.29 | 0.20 |
| acccon seed 0 | 72.82 | 0.0789 | 0.8751 | 24.36 | 21.62 | 16.56 | 0.50 |
| acccon seed 1 | 73.43 | 0.0767 | 0.8730 | 23.77 | 21.29 | 16.22 | 0.30 |
| acccon seed 2 | 73.12 | 0.0770 | 0.8753 | 24.09 | 21.47 | 16.81 | 0.30 |
| acccon seed 3 | 73.11 | 0.0776 | 0.8740 | 24.28 | 21.68 | 16.63 | 0.50 |
| **acccon 0–3** | 73.12±0.25 | 0.0776 | 0.8744 | 24.13±0.26 | 21.51±0.17 | 16.56±0.25 | 0.40 |
| **acccon +42 (5 run)** | 73.20±0.29 | 0.0773 | 0.8738 | 24.01 | 21.45 | 16.50 | 0.36 |
| SAT+EM+SR (paper) | — | — | — | 24.14±0.12 | 21.52±0.22 | 16.71±0.18 | — |

**Kết luận CIFAR-100:** seed 42 gần CCL-SC ở @95 nhưng kém ở @90/@80. Trung bình 5 run, acccon **không thắng** CCL-SC (lệch khoảng +0.5 risk ở 80–95%). Nằm cùng vùng **SAT+EM+SR**, không phải SOTA của paper.

Bản acccon cũ (cancelled \(w_i\)), official 10k seed 42: 73.59 / AURC 0.0768 / @95=23.77 / @90=21.17 / @80=16.00. Query-weighted tốt hơn ở 95% và core (@10=0.20 vs 0.60); không được gọi là thắng nếu chỉ nhìn @80.

### 3.2 CelebA (5 seed, protocol paper)

Best val-acc, official test. Model overfit mạnh (train ~99%); checkpoint chọn epoch 4–6. `last.pth` không dùng được (seed 42 last: 76.98% / AURC 0.114).

| Seed | Ep | Acc | AURC | AUROC | @95 | @90 | @80 | @10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5 | 81.37 | 0.0709 | 0.775 | 17.48 | 16.04 | 13.08 | 0.50 |
| 0 | 6 | 80.23 | 0.0743 | 0.779 | 18.56 | 17.17 | 14.20 | 0.60 |
| 1 | 4 | 80.50 | 0.0726 | 0.782 | 18.23 | 16.83 | 13.56 | 0.40 |
| 2 | 6 | 80.77 | 0.0695 | 0.787 | 17.76 | 16.26 | 13.28 | 0.35 |
| 3 | 4 | 80.82 | 0.0714 | 0.781 | 18.11 | 16.80 | 13.64 | 0.20 |
| **acccon mean** | | **80.74±0.43** | **0.0718** | **0.781** | **18.03±0.42** | **16.62±0.46** | **13.55±0.42** | **0.41±0.15** |
| **CCL-SC** | | ~81.29 | — | — | **17.00±0.09** | **15.47±0.14** | **12.51±0.21** | **0.25±0.08** |

Chênh vs CCL-SC: @100 +0.55, @95 +1.03, @90 +1.15, @80 +1.04, @10 +0.16. **Thua rõ trên mọi coverage trùng.** Core top-10% bẩn hơn ~2 lần.

### 3.3 CIFAR-10 (bão hòa, 1 seed official 10k)

Paper: CCL-SC 5 seed, @100=5.97±0.11 (acc ~94.03), @95=3.56±0.06, @90=2.01±0.07, @80=0.69±0.08.

| Method | Acc | AURC | AUROC | @95 | @90 | @80 |
|---|---:|---:|---:|---:|---:|---:|
| CCL-SC (5 seed) | 94.03 | — | — | **3.56** | **2.01** | 0.69 |
| acccon official (VAST, seed 42) | **94.21** | 0.00593 | 0.935 | 3.60 | 2.28 | **0.58** |
| acccon_ds (local, seed 42) | 94.03 | 0.00586 | **0.941** | 3.57 | **1.91** | 0.68 |

CIFAR-10 phân biệt yếu (ít lỗi). acccon seed 42: acc cao hơn, @80 tốt hơn, @90 kém hơn. **Không đủ 5 seed** để kết luận thống kê. Paper cũng nói CIFAR-10 không phải dataset chính.

---

## 4. Ablation trên acccon (CIFAR-100, seed 42, official 10k)

Mục tiêu: cải thiện vai 90/95 mà không làm bẩn core. Baseline thắng: **query-weighted acccon**.

| Variant | Ý tưởng | Acc | AURC | @95 | @90 | @80 | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| **acccon (qw)** | query \(w_i\), key \(w_j\) | **73.54** | **0.0763** | **23.54** | 21.19 | 16.29 | **baseline tốt nhất** |
| acccon old | huỷ \(w_i\) trong ratio | 73.59 | 0.0768 | 23.77 | 21.17 | **16.00** | @80 đẹp hơn, 95% và core xấu hơn |
| acccon_floor α=0.20 | nâng query thấp | 73.16 | 0.0782 | 24.03 | 21.64 | 16.74 | thua |
| acccon_bound β=0.10 | query core+boundary | 73.06 | 0.0783 | 24.12 | 21.42 | 16.65 | thua |
| acccon_bound β=1.0 | (8k / run sớm) | 72.87 | 0.0773 | 24.30 | 21.67 | 16.53 | thua |
| acccon_tail μ=0.10 | soft 0/1 risk | 73.35 | 0.1399 | 24.64 | 22.74 | 19.10 | **sụp cuối epoch** |
| carto_acccon | ambiguity query | 72.98 | 0.0790 | 24.25 | 21.80 | 16.89 | thua |
| el2n_acccon | EL2N prior | 73.02 | 0.0779 | 24.26 | 21.76 | 16.53 | thua |
| temporal_target | BCE trên rolling correctness | 72.60 | 0.1144 | 25.91 | 24.26 | 20.76 | **không thay hard BCE** |
| temporal_hybrid | 0.5 hard + 0.5 temporal | 73.38 | 0.0803 | 24.37 | 22.04 | 17.01 | thua |
| forget_target | × exp(−γ N_forget) | 72.97 | 0.0958 | 25.07 | 23.17 | 19.22 | thua |
| margin_target | sigmoid(margin) | 72.70 | 0.1018 | 25.67 | 24.00 | 20.01 | thua |
| depth_diag | probe detach (chỉ phân tích) | 73.03 | 0.0770 | 24.23 | 21.48 | 16.35 | probe không giúp \(s\) |
| depth_score | nối disagreement vào head | 73.53 | 0.0767 | 23.76 | **21.10** | 16.22 | gần acccon nhất; vẫn kém CCL-SC @90 |
| temporal_depth | temporal + disagreement | 72.87 | 0.1184 | 25.56 | 23.89 | 20.64 | thua nặng |

Quy tắc đã xác nhận: **không** thay hard correctness BCE bằng teacher temporal. Không gọi thắng @80 nếu top-10% core bẩn hơn.

### Kiến trúc acccon_ds (theo DS-SCSF, không copy RL / sigmoid)

Mini-clf pool3/4/5, head \(s_m\) không bị chặn, \(s=\sum \beta_m s_m\), contrastive vẫn trên fused \(s\), projector GAP pool5. Aux CE 0.3 từ epoch 1; aux BCE sau pretrain.

| Dataset | Acc | AURC | AUROC | @95 | @90 | @80 | Fusion \(\beta\) |
|---|---:|---:|---:|---:|---:|---:|---|
| CIFAR-100 | 73.17 | 0.0769 | **0.8771** | 23.97 | 21.27 | 16.32 | (0.11, 0.09, **0.80**) — gần như chỉ \(s_5\) |
| CIFAR-10 | 94.03 | 0.00586 | 0.941 | 3.57 | 1.91 | 0.68 | (0.09, 0.37, 0.54) — dùng được 3 tầng |

Trên CIFAR-100: **AUROC tốt nhất** (+0.005 vs acccon) nhưng acc/AURC/@95 kém — detector tốt hơn, selective classifier kém hơn (nhiều lỗi hơn để xếp hạng). Trên CIFAR-10: acc/@90 hơi tốt vs acccon local, AURC/AUROC/@80 vẫn nghiêng về acccon thuần.

### 4.1 DSS-ACCon (deep supervision, hướng mới)

**Chẩn đoán:** `acccon_ds` thay SCSF bằng fused \(s_3,s_4,s_5\) → mất accuracy và \(\beta\) collapse về \(s_5\). `depth_score` gần CCL-SC ở @90 nhất nhưng probe **detach** nên không train VGG.

**Phát minh DSS-ACCon** (`--variant dss_acccon`): giữ keyword *deep supervision* nhưng tách hai vai trò:

1. **DSN live:** mini-classifier tại spatial pool3/4/5, aux CE \(w=0.3\) từ epoch 1 → gradient vào backbone (cải accuracy / representation).
2. **Selector thắng cuộc:** vẫn RawConfidenceHead SCSF trên GAP pool4/pool5 + stopgrad logits (không fused early scores).
3. **Depth consistency:** sau pretrain, append \(D=\mathrm{disagreement}(\hat y_3,\hat y_4,\hat y_5,\hat y)\) **detached** vào head (`extra_dim=1`) — tín hiệu selective từ deep supervision.
4. **Acccon qw:** không đổi (mask từ \(s\), projector GAP pool5).

Không copy: learnable \(\beta\) fusion score, sigmoid confidence, RL.

**Protocol:** CIFAR-100, last.pth, official 10k; dirs `save/next_dss_acccon_cifar100_seed{0,42}` (laptop + VAST).

**Success bar:** không tệ hơn qw acccon seed 42 ở @90/@95; acc ≳ 73.3%; mục tiêu thu hẹp ~0.2–0.5 risk vs CCL-SC ở 90/80 mà không bẩn @10.

| Method | Acc | AURC | AUROC | @95 | @90 | @80 | @10 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CCL-SC (5 seed) | 73.45 | — | — | **23.54** | **20.97** | **16.07** | 0.36 | paper |
| acccon qw seed 42 | 73.54 | 0.0763 | 0.8717 | 23.54 | 21.19 | 16.29 | 0.20 | done |
| depth_score | 73.53 | 0.0767 | 0.8711 | 23.76 | 21.10 | 16.22 | 0.40 | done |
| acccon_ds | 73.17 | 0.0769 | 0.8771 | 23.97 | 21.27 | 16.32 | 0.40 | done |
| DSS-ACCon seed 0 | 73.47 | 0.0764 | 0.8725 | 23.85 | 21.33 | 16.18 | 0.10 | done |
| DSS-ACCon seed 42 | 73.28 | 0.0765 | 0.8748 | 23.98 | 21.40 | 16.44 | 0.30 | done |
| **DSS-ACCon (2 seed)** | **73.38±0.13** | **0.0764** | **0.8737** | **23.92±0.09** | **21.37±0.05** | **16.31±0.19** | **0.20±0.14** | **done — bar not cleared** |

**Kết quả:** acc ổn (~73.4%); @90/@95 **không tốt hơn** qw acccon seed 42 (23.54/21.19) cũng không gần CCL-SC hơn. AUROC hơi cao hơn qw nhưng AURC gần như ngang. Success bar **không clear** → không thêm seed; chẩn đoán tiếp theo: tách aux CE vs \(D\) nếu cần.

### 4.2 HardRank-ACCon + MSP-ACCon (hướng tiếp theo)

**Chẩn đoán probe (offline, seed 42 qw checkpoint, official 10k):** điểm test \(s+\mathrm{MSP}\) (z-score cộng) đã **vượt CCL-SC ở @90** (20.70 vs 20.97) và giữ @10=0.20 — tín hiệu SR/MSP còn thiếu trong SCSF thuần. Soft tail / reweight đã fail; pairwise rank (recipe §5) chưa thử cùng qw acccon.

| Method | Ý tưởng | Acc | AURC | @95 | @90 | @80 | @10 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HardRank-ACCon seed42 | hard-mined pairwise rank + qw | 72.78 | 0.0813 | 24.66 | 22.11 | 16.89 | 0.60 | **fail** |
| MSP-ACCon seed0 | MSP append + qw | 73.01 | 0.0800 | 24.20 | 21.52 | 16.81 | 0.20 | done |
| MSP-ACCon seed42 | MSP append + qw | 73.11 | 0.0789 | 23.92 | 21.40 | 16.55 | 0.10 | done |
| **MSP-ACCon (2 seed)** | | **73.06** | **0.0795** | **24.06** | **21.46** | **16.68** | **0.15** | **fail bar** |

**HardRank:** thua rõ vs qw — không thêm seed.
**MSP-ACCon (học):** append MSP vào head **làm xấu** vs qw (kém cả probe offline \(s+\mathrm{MSP}\) trên ckpt qw thuần). Học kèm MSP phá ranking; không thêm seed. Hướng còn mở: **eval hybrid cố định** trên qw (không retrain), chọn \(\lambda\) trên val.

### 4.3 Fresh methods (không dựa acccon)

Tách khỏi acceptance-weighted SupCon. Ba biến thể trong `train_fresh_sc.py`:

| Method | Ý tưởng | Acc | AURC | @95 | @90 | @80 | @10 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| AddMSP-SC seed42 | \(s+\mathrm{softplus}(\alpha)\mathrm{logit}(\mathrm{MSP})\) | 73.09 | 0.0797 | 24.04 | 21.38 | 16.54 | 0.90 | **fail** (core bẩn; \(\alpha\)→1.03) |
| Risk-SC seed42 | soft selective-risk, no contrastive | 73.17 | 0.1111 | 24.61 | 22.42 | 18.79 | 1.40 | **fail** (BCE sụp cuối như tail) |
| Margin-SC seed0 | CE only; score = top1−top2 | 73.03 | 0.0840 | 24.74 | 22.17 | 17.44 | 0.70 | **fail** |
| Margin-SC seed42 | CE only; score = top1−top2 | 73.02 | 0.0839 | 24.84 | 22.39 | 17.11 | 0.60 | **fail** |

**AddMSP:** residual blend không bắt được probe offline \(z(s)+z(\mathrm{MSP})\); @10 xấu.
**Risk:** soft risk không contrastive vẫn collapse BCE — xác nhận soft coverage risk là hướng độc hại.
**Margin:** logit margin thuần kém qw/CCL.

### 4.4 Overnight VAST queue — kết quả

GPU idle; trained queue xong. So với CCL-SC / qw seed42 (official 10k):

| Method | Acc | AURC | @95 | @90 | @80 | @10 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| CCL-SC | 73.45 | — | 23.54 | **20.97** | **16.07** | 0.36 | paper |
| qw acccon seed42 | 73.54 | 0.0763 | 23.54 | 21.19 | 16.29 | 0.20 | baseline |
| **dualaug seed42** | **74.15** | **0.0730** | **23.04** | **20.61** | **15.40** | 0.40 | **CLEAR bar** |
| tcp seed42 | 73.54 | 0.0772 | 23.69 | 21.16 | 16.04 | 0.90 | gần qw @90; core bẩn |
| softauc seed42 | 73.20 | 0.0779 | 23.97 | 21.47 | 16.51 | 0.20 | thua |
| softauc seed0 | 73.56 | 0.0781 | 23.71 | 21.27 | 16.37 | 0.60 | thua |
| scsf seed42 | 73.13 | 0.0786 | 23.96 | 21.62 | 16.80 | 0.30 | thua |
| focal seed42 | 73.25 | 0.0788 | 24.29 | 21.69 | 16.89 | 0.50 | thua |
| valblend seed42 | 72.92 | 0.0787 | 24.27 | 21.64 | 16.66 | 0.30 | λ=0.25; thua (semi post-hoc) |
| temptrain 42/0 | ~72.5 | ~0.082 | ~25 | ~22.2 | ~17.2 | ~0.4 | thua |
| selnet 42/0 | ~72.3 | ~0.084 | ~25.8 | ~23.6 | ~18.3 | 0.20 | thua nặng |

**dualaug** (2-view CE + head học agreement): acc cao hơn, @95/@90/@80 **thắng CCL-SC** trên seed 42. Đang chạy thêm seed 0–3 trên VAST để xác nhận.

CIFAR-100 dualaug (4 seed xong: 0,1,2,42; seed 3 ~epoch 211):

| Seed | Acc | AURC | @95 | @90 | @80 | @10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.44 | 0.0722 | 23.23 | 20.71 | 15.38 | 0.30 |
| 1 | 74.43 | 0.0719 | 22.96 | 20.38 | 15.15 | 0.30 |
| 2 | 74.15 | 0.0733 | 23.28 | 20.78 | 15.62 | 0.20 |
| 42 | 74.15 | 0.0730 | 23.04 | 20.61 | 15.40 | 0.40 |
| **mean±std** | **74.29±0.16** | **0.0726** | **23.13±0.15** | **20.62±0.18** | **15.39±0.19** | **0.30±0.08** |
| CCL-SC | 73.45 | — | 23.54±0.15 | 20.97±0.20 | 16.07±0.15 | 0.36 |

CelebA dualaug (ResNet-18, protocol paper, best val-acc) queued sau seed 3: `run_dualaug_celeba_queue.sh` seeds 42,0,1,2,3.

---

## 5. Các nhánh tìm kiếm sớm (CIFAR-100, chủ yếu split 8k — **không so paper**)

Chỉ để ghi nhận hướng đã loại, không dùng làm số so CCL-SC.

- Search extras (rank, CSC, rank+CSC) không thắng SCSF+acccon.
- CBR: `cbr_micro` (soft 1−p_y trên accepted set) từng thắng các biến thể confusion/duals trên 8k; không thay acccon trên official 10k.
- R3 / DTR: không phải hướng chính sau khi có acccon.
- Smoke test (epoch rất ngắn) bỏ qua.

---

## 6. Việc chưa làm / hạn chế

1. **ImageNet** chưa train (thiếu dung lượng).
2. CIFAR-10 acccon mới **1 seed** official 10k.
3. CelebA và CIFAR-100 acccon **5 run** đã có; CelebA thua chắc, CIFAR-100 thua ở 80–95%.
4. acccon_ds mới 1 seed / dataset; fusion CIFAR-100 collapse.
5. **DSS-ACCon** 2 seed CIFAR-100 xong; **không clear** success bar vs qw/@CCL-SC (@90/@95).
6. **HardRank** và **MSP-ACCon (học)** đều fail bar (§4.2); probe \(s+\mathrm{MSP}\) offline vẫn vượt CCL @90 trên 1 seed.
7. So sánh CelebA/CIFAR-100 với SAT/DG/SR chỉ lấy số **paper**, không re-implement.

---

## 7. Kết luận

1. **Phương pháp ổn định nhất là query-weighted acccon** (SCSF + SupCon acceptance, không huỷ \(w_i\)).
2. Trên **CIFAR-100 (5 run, official 10k)** acccon **không vượt CCL-SC** ở @90/@95/@80; gần SAT+EM+SR. Seed 42 là run lạc quan, không đại diện mean.
3. Trên **CelebA (5 seed, đúng protocol)** acccon **thua CCL-SC ~1 điểm risk** ở 80–95% và core bẩn hơn.
4. Mọi ablation objective (floor, bound, tail, carto, EL2N, temporal/forget/margin) **đều kém** acccon qw. Tail collapse. Không thay hard BCE.
5. Đổi kiến trúc (acccon_ds) **tăng AUROC** trên CIFAR-100 nhưng **mất accuracy**; fusion học gần như bỏ pool3/4. Chưa phải hướng thắng AURC.
6. **DSS-ACCon** (2 seed) giữ acc nhưng **không thắng** qw acccon / CCL-SC ở @90/@95 — xem §4.1. Deep supervision sống không đủ đóng khoảng CCL-SC.
7. **HardRank** và **MSP-ACCon học** đều thua qw (§4.2). Probe offline \(s+\mathrm{MSP}\) trên qw seed 42 vẫn vượt CCL @90 — gợi ý hybrid **lúc eval**, không nhét MSP vào head lúc train.
8. CIFAR-10 bão hòa; 1 seed không đủ để tuyên bố thắng/thua.

**Hướng tiếp:** **dualaug** seed 42 clear bar vs CCL-SC — đang chạy seed 0–3 trên VAST để xác nhận mean±std. Không stack floor/bound/tail. ImageNet vẫn thiếu dữ liệu.
