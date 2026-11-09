# Acceptance-Weighted Contrastive Learning (acccon)

This note describes the method as implemented in `train_next_scsf.py` (`--variant acccon`). Official evaluation uses the last-epoch checkpoint.

acccon trains a standard classifier together with an SCSF confidence head, and adds a supervised contrastive term on penultimate features whose sample weights are the head’s **soft coverage acceptance**. Softmax response is not used as a weight or as the test-time score.

## 1. Selective classification setup

A classifier \(f\) maps an input \(x\) to logits \(\ell(x)\in\mathbb{R}^C\) and a prediction \(\hat y=\arg\max_j \ell_j(x)\). A scalar score \(s(x)\) ranks examples for rejection. At coverage \(c\), the accepted set is the top \(\mathrm{round}(n\cdot c)\) examples by \(s\). The goal is to reduce selective risk on that prefix (and the area under the risk–coverage curve) without changing the ranking rule at test time.

## 2. Architecture

**Backbone.** VGG16-BN with a \(C\)-way linear classifier. There is no reservation class and no auxiliary classification head. The wrapper exposes flattened pool4 and pool5 features. On \(32\times 32\) CIFAR, pool4 is \(2048\)-D and pool5 is \(512\)-D.

**Confidence head.** An MLP produces one unbounded score
\[
s(x)=h\big([\mathrm{pool4}(x),\;\mathrm{pool5}(x),\;\ell(x)_{\mathrm{stop}}]\big).
\]
The logit path into \(h\) is detached (SCSF design): correctness losses cannot edit \(\ell\) through the head. Feature paths stay open. The MLP is
\[
2570{+}C \to 1024 \to 512 \to 256 \to 128 \to 1
\]
with ReLU and dropout \(0.3\) after each hidden layer (for CIFAR-10, \(C=10\)).

**Projection head (train only).** A two-layer MLP maps pool5 to an \(\ell_2\)-normalized embedding
\[
z=\mathrm{normalize}\big(W_2\,\mathrm{ReLU}(W_1\,\mathrm{pool5})\big)\in\mathbb{R}^{128},
\]
with hidden width \(256\). The projector is discarded at test time.

**Memory queue.** A FIFO of size \(3000\) stores detached tuples \((z,y,w)\). It is not a momentum encoder: keys are previous online embeddings.

## 3. Soft coverage acceptance

After the CE warmup, each mini-batch of scores \(s_1,\ldots,s_B\) defines nested soft acceptances at coverages \(\mathcal{C}=\{0.80,0.90,0.95\}\).

For each \(c\in\mathcal{C}\), a scalar threshold \(h_c\) is obtained by \(60\) steps of bisection so that
\[
\frac{1}{B}\sum_{i=1}^{B}\sigma\!\left(\frac{s_i-h_c}{T}\right)=c,
\]
with temperature \(T=0.2\). The soft mask is
\[
a_c(x_i)=\sigma\!\left(\frac{s_i-h_c}{T}\right).
\]
The masks are nested: \(a_{0.80}\le a_{0.90}\le a_{0.95}\). The contrastive weight is the mean acceptance
\[
w_i=\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}} a_c(x_i).
\]

For acccon, \(s\) is **detached** before the bisection and \(w\) is detached before the contrastive loss. Acceptance is a stop-gradient importance weight. It does not provide an extra path from the contrastive term back into \(s\) or \(h_c\).

This is the only use of the coverage machinery. acccon does **not** add soft micro-risk, accepted-set cross-entropy, pairwise ranking, class-pair confusion, or coverage duals.

## 4. Acceptance-weighted supervised contrastive loss

Let \(z_i\) be the projected pool5 embedding of example \(i\) in the current batch. Keys are the batch embeddings concatenated with the filled prefix of the queue. For query \(i\), the positive set \(P(i)\) is every key with the same **true** label except the query itself. That is ordinary supervised contrastive sampling (Khosla et al.), not CCL-SC’s correct-vs-false-positive split.

With temperature \(\tau=0.1\),
\[
p_{ij}=\frac{\exp(z_i^\top z_j/\tau)}{\sum_{k\neq i}\exp(z_i^\top z_k/\tau)}.
\]
Query and key acceptances have separate roles. The per-query term weights positives only by key acceptance,
\[
\mathcal{L}_i^{\mathrm{con}}
=-\frac{\sum_{j\in P(i)} w_j\log p_{ij}}{\sum_{j\in P(i)} w_j+\varepsilon},
\]
and the batch loss is a query-weighted mean over anchors that have at least one positive:
\[
\mathcal{L}_{\mathrm{acccon}}
=\frac{\sum_{i\in\mathcal{V}} w_i\,\mathcal{L}_i^{\mathrm{con}}}{\sum_{i\in\mathcal{V}} w_i+\varepsilon}.
\]
Putting \(w_i\) inside a pair-weighted ratio \(\sum_j w_i w_j\log p_{ij}/\sum_j w_i w_j\) would cancel it whenever \(w_i>0\), so a barely accepted query would count as much as a highly accepted one. The two-level form avoids that: \(w_i\) scales how much the anchor is trained, and \(w_j\) scales how strongly a same-class key acts as a target.

If \(\mathcal{V}\) is empty, the term is zero. After the backward pass, \((z_i,y_i,w_i)\) are enqueued with \(z_i\) and \(w_i\) detached.

**Intended effect.** Reliable anchor + reliable positive \(\rightarrow\) strong attraction. An unreliable (low-acceptance) anchor has little influence on the batch mean. An unreliable positive is a weak prototype. The ranking head still learns \(s\) from correctness BCE; the contrastive term only reshapes pool5.

### Core + boundary queries (`acccon_bound`)

Query-weighting by \(w_i\) alone trains the accepted core and under-trains the 75–85% shoulder, where leftover risk actually jumps. `acccon_bound` keeps **keys** as \(w_j\) and replaces the **query** weight with
\[
q_i = w_i + \beta\, b_i, \qquad b_i = 4w_i(1-w_i).
\]
The boundary term is \(0\) at \(w\in\{0,1\}\) and \(1\) at \(w=0.5\). Default \(\beta=1\). Then
\[
\mathcal{L}_{\mathrm{acccon}}
=\frac{\sum_{i\in\mathcal{V}} q_i\,\mathcal{L}_i^{\mathrm{con}}}{\sum_{i\in\mathcal{V}} q_i+\varepsilon}.
\]
High \(w\) still trains the trusted core. Medium \(w\) (selective decision boundary) gets extra weight. Rejected examples stay near zero. This is not a constant floor on every query.

## 5. Training objective

The batch loss is
\[
\mathcal{L}
=\mathrm{CE}\big(\ell(x),y\big)
+\lambda_t\,\mathrm{BCE}\big(s(x),\;\mathbf{1}[\hat y=y]\big)
+\rho_t\cdot\eta\cdot\mathcal{L}_{\mathrm{acccon}},
\]
with contrastive weight \(\eta=0.5\). The correctness target is the hard argmax indicator and is computed without gradient into \(\ell\).

**Schedule** (300 epochs, batch size 128, seed 42):

| Phase | Epochs | Extra terms |
|---|---|---|
| CE warmup | \(1\)–\(100\) | \(\lambda_t=\rho_t=0\) |
| Joint ramp | \(101\)–\(120\) | \(\rho_t\) cosine-ramps \(0\to 1\); \(\lambda_t\) starts decaying |
| Joint | \(121\)–\(300\) | \(\rho_t=1\); \(\lambda_t\) continues decaying |

The ramp is \(\rho_t=\tfrac12\bigl(1-\cos(\pi\cdot u)\bigr)\) with \(u=\min\bigl(1,(e-100)/20\bigr)\) for \(e>100\). The SCSF coefficient \(\lambda_t\) cosine-decays from \(1.0\) to \(10^{-4}\) over the joint phase (epochs \(101\)–\(300\)).

**Optimizers.** Backbone: SGD, learning rate \(0.1\), momentum \(0.9\), weight decay \(5\times 10^{-4}\), MultiStepLR \(\times 0.5\) every 25 epochs. Confidence head and projector: Adam, learning rate \(10^{-3}\). Gradients are clipped at global norm \(5\).

**Gradient routing.**

| Term | Updates |
|---|---|
| CE | backbone, including the classifier |
| Correctness BCE | confidence head; pool4/pool5 (not \(\ell\) through the head) |
| \(\mathcal{L}_{\mathrm{acccon}}\) | pool5 and the projector (not \(s\), not \(\ell\) via \(w\)) |

## 6. Inference

Only \(s(x)\) is used. Examples are sorted by \(s\) descending (stable sort). For a requested coverage \(c\), the first \(\mathrm{round}(n\cdot c)\) examples are accepted. The projector, queue, and soft masks are training-only.

## 7. Relation to SCSF and CCL-SC

acccon is SCSF plus one feature-level term. It is not CCL-SC with a renamed weight.

- **SCSF** uses the same head and correctness BCE. acccon keeps that and adds acceptance-weighted SupCon on pool5.
- **CCL-SC** uses softmax response as both the test-time score and a scalar CSC weight on the *anchor*; defines positives as *predicted as \(y\) and correct* and negatives as *incorrectly predicted as \(y\)*; maintains two MoCo queues and a momentum encoder; and has no extra ranking head. acccon uses a learned score \(s\), same-true-class positives, a single online FIFO, key-weighted positives, and a query-weighted mean of those terms from soft coverage acceptance.

Implementation: `train_next_scsf.py`, helpers `soft_coverage_masks` in `train_cbr_scsf.py`, `ProjectionHead` / `FeatureQueue` in `train_search_scsf.py`.
