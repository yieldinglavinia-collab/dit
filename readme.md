# Methodology: Clean Cycle-Conditional Score Diffusion with (\log\sigma)-Conditioned Q/K/V Attention

## 1. Method name

**CCSD-QKV: Cycle-Conditional Score Diffusion with Noise-Conditioned Q/K/V Attention**

The core idea is:

> Instead of learning a static density (p(X)) over clean robot cycles, learn a conditional diffusion score (p_\sigma(X_M \mid X_{\bar M})), where (X_M) is an internally withheld subset of measurements and (X_{\bar M}) is the remaining same-cycle context. Benign machine-response drift that is coherent within the same cycle should remain conditionally predictable, while true anomaly should break conditional temporal/channel consistency.

This method is not Stable Diffusion, not reconstruction autoencoding, and not a hand-crafted drift simulator.

The diffusion foundation is:

[
\text{theory: score-based / continuous-noise diffusion}
]

[
\text{implementation: DDPM-style } \epsilon\text{-prediction}
]

The denoising network is a Transformer replacing the usual diffusion U-Net, but the attention operation remains the original clean scaled dot-product attention:

[
\mathrm{Attn}(H,\sigma)
=======================

\mathrm{softmax}
\left(
\frac{Q(\sigma)K(\sigma)^\top}{\sqrt d}
\right)
V(\sigma)
]

No additive attention-logit bias is used.

---

## 2. Data input

Each cycle is represented as:

[
X\in\mathbb{R}^{T\times58}
]

where (T) is the number of time points in one robot cycle and 58 is the number of measurement channels.

Only the 58 directly measured channels are used as detector input.

The following 7 auxiliary columns are never input to the detector:

[
\texttt{time, sample, anomaly, category, setting, action, active}
]

They are used only for split construction, drift simulation, evaluation, or visualization.

### Handling variable length cycles

Because the method uses no padding mask inside attention, the implementation should use one of the following clean choices:

1. **Fixed-length resampling:** resample every cycle to a fixed length (T_0) using only sequence order.
2. **Same-length batching:** group cycles with the same length and run attention without padding.
3. **Single-cycle evaluation:** process one cycle at a time when lengths differ.

For coding and architecture diagrams, the cleanest choice is fixed-length resampling:

[
X\in\mathbb{R}^{T_0\times58}
]

This is preprocessing, not a detector feature. It uses no auxiliary metadata and no anomaly labels.

---

## 3. Problem formulation

Let the clean normal training distribution be:

[
X\sim P_0
]

where (X) comes only from train_normal.

Drifted normal test cycles can be viewed as:

[
X^d\sim P_d
]

where (P_d) is a benign machine-response shifted distribution. These samples are still normal and should not trigger high false alarm.

True anomaly cycles are sampled from:

[
X^a\sim P_A
]

A static anomaly detector learns a clean-normal density or score:

[
S_{\text{static}}(X)=-\log p_0(X)
]

When benign drift moves normal samples away from the clean density support, the static detector gives high scores:

[
S_{\text{static}}(X^d)>\tau
]

causing high FAR.

The goal of CCSD-QKV is not to learn whether the whole cycle matches the clean marginal distribution. Instead, it asks:

[
\text{Are withheld measurements conditionally consistent with the rest of the same cycle?}
]

This is modeled as:

[
p_\sigma(X_M\mid X_{\bar M})
]

where (M) is an internally sampled target mask and (\bar M) is the observed same-cycle context.

---

## 4. Core score formulation

For a clean normal cycle:

[
X\in\mathbb{R}^{T_0\times58}
]

sample a binary target mask:

[
M\in{0,1}^{T_0\times58}
]

where (M_{t,c}=1) means the entry is selected as a target for conditional denoising.

The context part is:

[
X_{\bar M}=(1-M)\odot X
]

The target part is:

[
X_M=M\odot X
]

Sample diffusion noise scale:

[
\sigma>0
]

and Gaussian noise:

[
\epsilon\sim\mathcal{N}(0,I)
]

Construct the noised conditional input:

[
Y_{\sigma,M}
============

(1-M)\odot X
+
M\odot(X+\sigma\epsilon)
]

The model predicts the injected noise only on the target entries:

[
\hat\epsilon_\theta
===================

\epsilon_\theta(Y_{\sigma,M},M,\sigma)
]

The learned conditional score is equivalent to:

[
s_\theta^M(Y_{\sigma,M},X_{\bar M},\sigma)
\approx
\nabla_{Y_M}
\log p_\sigma(Y_M\mid X_{\bar M})
]

with the standard relation:

[
\epsilon_\theta
\approx
-\sigma s_\theta
]

The model is trained by DDPM-style noise prediction:

[
\mathcal{L}(\theta)
===================

\mathbb{E}*{X,M,\sigma,\epsilon}
\left[
\frac{
\left|
M\odot
\left(
\epsilon*\theta(Y_{\sigma,M},M,\sigma)-\epsilon
\right)
\right|_2^2
}{
|M|_0
}
\right]
]

This is the only main training loss.

No reconstruction loss, no classification loss, no contrastive loss, no drift-label loss, no sparsity loss, and no uncertainty loss are used.

---

## 5. Architecture overview

The architecture has four substantive components:

1. Conditional masked diffusion formulation
2. Measurement-token Transformer representation
3. (\log\sigma)-conditioned Q/K/V attention
4. Conditional denoising-risk anomaly scoring

The architecture is:

[
X
\rightarrow
M
\rightarrow
Y_{\sigma,M}
\rightarrow
\text{Token Projection}
\rightarrow
\text{(\log\sigma)-Conditioned Q/K/V Transformer}
\rightarrow
\hat\epsilon
\rightarrow
S(X)
]

---

## 6. Component A: Conditional masked diffusion formulation

### Purpose

Define the anomaly detection problem as conditional score estimation rather than full-cycle density estimation.

### Input

[
X\in\mathbb{R}^{T_0\times58}
]

[
M\in{0,1}^{T_0\times58}
]

[
\sigma,\epsilon
]

### Output

[
Y_{\sigma,M}
]

### Formulation

[
Y_{\sigma,M}
============

(1-M)\odot X
+
M\odot(X+\sigma\epsilon)
]

Only the masked target entries are noised and scored.

### Why needed

If the model learns (p(X)), benign drifted normal may become low-density and trigger false alarms. If the model learns (p(X_M\mid X_{\bar M})), a coherent drifted cycle may still be conditionally predictable.

### Why it helps FAR

Benign drift usually shifts the cycle response coherently. The same-cycle context contains the shifted operating response, so target measurements remain predictable. Therefore drifted normal should not receive as high a score as in static density models.

---

## 7. Component B: Measurement-token Transformer representation

### Purpose

Convert each physical time point into a token representing all 58 measured robot signals at that time.

### Token definition

A token corresponds to one physical time index:

[
t=1,\dots,T_0
]

Each token contains the 58-dimensional noised/context measurement vector plus the internal target mask at the same time:

[
r_t =
[
Y_{\sigma,M}(t,:),\ M(t,:)
]
\in\mathbb{R}^{116}
]

The token embedding is:

[
h_t^{(0)}
=========

W_{\text{in}}r_t
+
p_t
]

where:

* (W_{\text{in}}) is a learned linear projection;
* (p_t) is a standard positional embedding for sequence order;
* (p_t) is not the original metadata `time`;
* no auxiliary metadata is used.

The input token sequence is:

[
H^{(0)}
=======

[h_1^{(0)},\dots,h_{T_0}^{(0)}]
\in\mathbb{R}^{T_0\times d}
]

### Why needed

Robot anomaly detection depends on temporal consistency across the cycle. Each token represents a physical time point, and the Transformer models temporal interactions among these time-indexed measurement vectors.

### What is not included

No channel groups are used.

No torque/current/voltage rules are used.

No active/action/setting labels are used.

No padding mask is used inside attention.

---

## 8. Component C: (\log\sigma)-conditioned Q/K/V attention

### Purpose

Replace the diffusion U-Net with a Transformer denoiser, but make the Transformer noise-level-aware at the Q/K/V level.

The key design is:

> (\log\sigma) modulates Q/K/V representations, while attention itself remains the original scaled dot-product attention.

### Noise embedding

First compute a diffusion noise embedding:

[
e_\sigma
========

\mathrm{MLP}(\log\sigma)
\in\mathbb{R}^{d}
]

For a Transformer layer (l), apply noise-conditioned normalization or modulation:

[
\tilde h_t^{(l)}
================

\gamma^{(l)}(e_\sigma)
\odot
\mathrm{LN}(h_t^{(l)})
+
\beta^{(l)}(e_\sigma)
]

Then compute Q/K/V:

[
Q^{(l)}
=======

\tilde H^{(l)}W_Q^{(l)}
]

[
K^{(l)}
=======

\tilde H^{(l)}W_K^{(l)}
]

[
V^{(l)}
=======

\tilde H^{(l)}W_V^{(l)}
]

The attention is exactly:

[
\mathrm{Attn}^{(l)}(H^{(l)},\sigma)
===================================

\mathrm{softmax}
\left(
\frac{
Q^{(l)}K^{(l)\top}
}{
\sqrt{d_h}
}
\right)
V^{(l)}
]

No extra attention-logit bias is added.

The Transformer block is:

[
H^{(l+1)}
=========

H^{(l)}
+
\mathrm{Attn}^{(l)}(H^{(l)},\sigma)
+
\mathrm{MLP}^{(l)}(\cdot)
]

Implementation can use the standard residual/normalization ordering.

### Important restrictions

Do not add:

[
b_t(t-t')
]

Do not add:

[
b_\sigma(\log\sigma-\log\sigma')
]

Do not add manually designed temporal-distance penalties.

Do not add manually designed noise-scale-distance penalties.

Do not add hand-crafted local windows.

Do not add channel-group attention.

Do not add padding mask inside attention.

The only conditioning is through the token representation and (\log\sigma)-conditioned Q/K/V modulation.

### Why (\log\sigma)-conditioned Q/K/V is reasonable

Diffusion denoising depends on the noise level. At high (\sigma), the model should focus on coarse temporal structure; at low (\sigma), it should focus on fine residual structure. Instead of adding arbitrary attention biases, the model changes the learned query/key/value representations as a function of (\log\sigma).

Thus, the attention mechanism remains clean:

[
\mathrm{softmax}(QK^\top/\sqrt d)V
]

but the representations used to form Q/K/V are diffusion-scale-aware.

### Why this is stronger than plain DiT replacement

Plain DiT-style replacement is:

[
\epsilon_\theta(Y_{\sigma,M},\sigma)
====================================

\mathrm{Transformer}(Y_{\sigma,M},\sigma)
]

CCSD-QKV instead specifies how (\sigma) enters the attention operator through Q/K/V modulation:

[
Q,K,V=f(H,\log\sigma)
]

This is still simple enough to implement, but more principled than merely adding a timestep embedding to the input.

---

## 9. Component D: Conditional denoising-risk anomaly score

### Purpose

Define a reconstruction-free anomaly score based on conditional diffusion residual.

### Output head

After the final Transformer layer:

[
H^{(L)}
\in\mathbb{R}^{T_0\times d}
]

predict noise:

[
\hat\epsilon
============

W_{\text{out}}H^{(L)}
\in\mathbb{R}^{T_0\times58}
]

### Anomaly score

For one mask and one noise scale:

[
S_{M,\sigma,\epsilon}(X)
========================

\frac{
\left|
M\odot
\left(
\epsilon_\theta(Y_{\sigma,M},M,\sigma)-\epsilon
\right)
\right|_2^2
}{
|M|_0
}
]

The final anomaly score averages over several independently sampled masks and noise levels:

[
S(X)
====

\frac{1}{K}
\sum_{k=1}^{K}
S_{M_k,\sigma_k,\epsilon_k}(X)
]

where each (\sigma_k) is sampled from the same noise distribution used during training.

This is not multi-noise-scale attention. Each forward pass uses one (\sigma). The averaging is only Monte Carlo estimation of the expected conditional denoising risk.

### Why this helps FAR

For drifted normal cycles, if the drift is internally coherent, the masked target remains predictable from the same-cycle context. The denoising residual should stay relatively low.

For true anomalies, masked anomalous measurements should be difficult to predict from normal temporal/channel context, increasing the residual.

---

## 10. Training algorithm

### Inputs

Training data:

[
X_i\sim P_0
]

from train_normal only.

### Step-by-step

For each training iteration:

1. Sample a batch of clean normal cycles:

[
X\in\mathbb{R}^{B\times T_0\times58}
]

2. Sample an internal random target mask:

[
M\in{0,1}^{B\times T_0\times58}
]

3. Sample diffusion noise scale:

[
\sigma
]

4. Sample Gaussian noise:

[
\epsilon\sim\mathcal{N}(0,I)
]

5. Construct noised conditional input:

[
Y_{\sigma,M}
============

(1-M)\odot X
+
M\odot(X+\sigma\epsilon)
]

6. Feed ([Y_{\sigma,M},M]) into the Transformer denoiser.

7. Compute predicted noise:

[
\hat\epsilon=
\epsilon_\theta(Y_{\sigma,M},M,\sigma)
]

8. Compute masked noise-prediction loss:

[
\mathcal{L}(\theta)
===================

\frac{
\left|
M\odot(\hat\epsilon-\epsilon)
\right|_2^2
}{
|M|_0
}
]

9. Update parameters.

### What is not used

No anomaly labels.

No test samples.

No P1–P5 drift labels.

No selected alphas.

No active/action/setting metadata.

No hand-crafted channel groups.

No manually designed drift transformation.

No reconstruction target (X).

---

## 11. Inference algorithm

For a test cycle (X):

1. Normalize using train_normal statistics.
2. Resample or represent as fixed length (T_0).
3. For (k=1,\dots,K):

   * sample internal target mask (M_k);
   * sample noise scale (\sigma_k);
   * sample Gaussian noise (\epsilon_k);
   * construct (Y_{\sigma_k,M_k});
   * predict (\hat\epsilon_k);
   * compute masked denoising residual.
4. Average residuals:

[
S(X)
====

\frac{1}{K}
\sum_{k=1}^{K}
\frac{
\left|
M_k\odot(\hat\epsilon_k-\epsilon_k)
\right|_2^2
}{
|M_k|_0
}
]

5. Compare to validation-calibrated threshold:

[
\hat y
======

\mathbb{1}[S(X)>\tau_{0.99}]
]

where:

[
\tau_{0.99}
===========

\operatorname{Quantile}_{0.99}
\left(
{S(X_i): X_i\in \text{clean val_normal}}
\right)
]

No test-set tuning is allowed.

---

## 12. Why this is not reconstruction error

The model does not predict:

[
\hat X
]

and the score is not:

[
|X-\hat X|
]

Instead, the model predicts injected Gaussian noise:

[
\epsilon
]

and the score is:

[
|\hat\epsilon-\epsilon|^2
]

only on internally withheld entries.

Therefore, the method is a conditional denoising score method, not a reconstruction autoencoder.

---

## 13. Why this can reduce false alarms under benign drift

Static density detector:

[
S_{\text{static}}(X)=-\log p(X)
]

penalizes global distribution shift.

CCSD-QKV instead scores:

[
S(X)
\approx
\mathbb{E}*{M,\sigma,\epsilon}
\left[
|\epsilon*\theta(Y_{\sigma,M},M,\sigma)-\epsilon|_M^2
\right]
]

This measures conditional inconsistency.

If benign drift is coherent within the cycle, then:

[
X_M
\text{ remains predictable from }
X_{\bar M}
]

so the score should remain low.

If an anomaly breaks temporal/channel consistency, then:

[
X_M
\text{ becomes hard to predict from }
X_{\bar M}
]

so the score increases.

---

## 14. Evaluation protocol

For every model:

1. Train on train_normal only.
2. Compute scores on clean val_normal.
3. Set:

[
\tau_{0.99}
===========

Q_{0.99}(S_{\text{val-normal}})
]

4. Evaluate on:

   * clean test_normal
   * clean test_anomaly
   * drifted test_normal
   * drifted test_anomaly

Metrics:

[
\mathrm{FAR}
============

\frac{
#{X\in\text{test_normal}:S(X)>\tau_{0.99}}
}{
#{\text{test_normal}}
}
]

[
\mathrm{TPR}
============

\frac{
#{X\in\text{test_anomaly}:S(X)>\tau_{0.99}}
}{
#{\text{test_anomaly}}
}
]

AUROC and AP are computed from raw (S(X)).

Primary metric:

[
\mathrm{FAR}
\text{ on drifted test_normal}
]

Secondary metrics:

[
\mathrm{AUROC},\quad \mathrm{AP},\quad \mathrm{TPR}
]

TPR does not need to exceed drifted MVT-Flow if FAR is substantially reduced and AUROC/AP are preserved.

---

## 15. Ablation plan

### B0. Static MVT-Flow baseline

Existing q=0.99 baseline.

Purpose: main comparison.

---

### B1. Plain unconditional DDPM

Train:

[
\epsilon_\theta(X+\sigma\epsilon,\sigma)\rightarrow\epsilon
]

Score full denoising residual.

Purpose: test whether diffusion alone helps.

Expected: may still suffer from benign drift.

---

### B2. Plain DiT-style Transformer denoiser

Use Transformer as U-Net replacement:

[
\epsilon_\theta(X_\sigma,\sigma)
================================

\mathrm{Transformer}(X_\sigma,\sigma)
]

No conditional masking.

Purpose: show that replacing U-Net with Transformer is not enough.

---

### B3. Conditional score diffusion without Q/K/V (\sigma)-conditioning

Use masked conditional denoising, but (\sigma) only enters as a simple embedding added to tokens.

Purpose: test whether Q/K/V-level (\log\sigma) conditioning matters.

---

### B4. CCSD-QKV without measurement mask input

Use (Y_{\sigma,M}), but do not provide (M) to the model.

Purpose: test whether the model needs to know target/context distinction.

Expected: performance should drop because target and context roles become ambiguous.

---

### B5. Full CCSD-QKV

Use:

[
p_\sigma(X_M\mid X_{\bar M})
]

with:

[
Q,K,V=f(H,\log\sigma)
]

and clean attention:

[
\mathrm{softmax}(QK^\top/\sqrt d)V
]

Purpose: main model.

---

### B6. Reconstruction score diagnostic

Use the same trained model but construct a reconstruction-style score if possible.

Purpose: prove that conditional denoising residual is better than reconstruction-style scoring.

Diagnostic only.

---

### B7. Oracle P1–P5 drift augmentation

Train with exact P1–P5 drift rules or alphas.

Purpose: upper bound only.

This is not allowed as main method evidence.

---

### B8. Mask strategy sensitivity

Compare:

* random element mask
* random time-block mask
* random time-channel patch mask

This is sensitivity analysis. The main method should use one generic mask distribution fixed before final testing.

Do not use P1–P5 performance to select the final mask strategy.

---

## 16. Architecture diagram text

A clean architecture diagram can be drawn as:

[
X\in\mathbb{R}^{T_0\times58}
]

↓

**Random target mask (M)**

↓

[
Y_{\sigma,M}
============

(1-M)\odot X
+
M\odot(X+\sigma\epsilon)
]

↓

**Token construction**

[
r_t=[Y_{\sigma,M}(t,:),M(t,:)]
]

↓

**Linear projection + positional embedding**

[
h_t=W_{\text{in}}r_t+p_t
]

↓

**(\log\sigma)-conditioned Transformer blocks**

[
e_\sigma=\mathrm{MLP}(\log\sigma)
]

[
Q=W_Q\mathrm{AdaLN}(H,e_\sigma)
]

[
K=W_K\mathrm{AdaLN}(H,e_\sigma)
]

[
V=W_V\mathrm{AdaLN}(H,e_\sigma)
]

[
\mathrm{Attn}
=============

\mathrm{softmax}
\left(
QK^\top/\sqrt d
\right)V
]

↓

**Noise prediction head**

[
\hat\epsilon\in\mathbb{R}^{T_0\times58}
]

↓

**Masked denoising residual**

[
S(X)
====

\mathbb{E}
\left[
|M\odot(\hat\epsilon-\epsilon)|^2/|M|_0
\right]
]

↓

**Validation threshold**

[
\tau_{0.99}=Q_{0.99}(S_{\text{val-normal}})
]

↓

**Decision**

[
S(X)>\tau_{0.99}
\Rightarrow
\text{anomaly}
]

---

## 17. What should be emphasized in the paper

The paper should emphasize three points:

1. **Conditional score instead of static density**

[
p_\sigma(X_M\mid X_{\bar M})
\text{ rather than }
p(X)
]

2. **Noise-conditioned Q/K/V instead of arbitrary attention bias**

[
Q,K,V=f(H,\log\sigma)
]

with clean attention:

[
\mathrm{softmax}(QK^\top/\sqrt d)V
]

3. **FAR-oriented anomaly scoring**

The method is designed to reduce false alarms on drifted normal cycles, not to inflate TPR by global score shift.

---

## 18. Main risks

### Risk 1: The method may suppress globally coherent anomalies

If a true anomaly remains conditionally predictable from the rest of the same cycle, CCSD-QKV may assign a low score.

Required experiment:

* category-wise TPR / AP
* compare local vs global anomaly types

---

### Risk 2: Mask strategy strongly affects performance

The method depends on whether the masked entries reveal conditional inconsistency.

Required experiment:

* mask ratio sensitivity
* point mask vs block mask vs patch mask
* fixed before final testing

---

### Risk 3: Q/K/V (\log\sigma)-conditioning may not improve over simple (\sigma) embedding

Required ablation:

[
\text{simple sigma embedding}
\quad \text{vs} \quad
\text{Q/K/V sigma conditioning}
]

If there is no improvement, the Q/K/V design should not be claimed as a major contribution.

---

### Risk 4: Single dataset + synthetic drift limits top-conference claim

For IEEE T-RO / T-ASE, this can be acceptable if experiments are strong.

For NeurIPS / ICLR / ICML, one robot dataset plus synthetic drift is likely not enough unless supported by additional datasets or strong theoretical analysis.

---

## 19. Final practical implementation recommendation

Start with the following minimal implementation:

1. Fixed-length cycle resampling to (T_0).
2. Random element or patch mask (M).
3. Single (\sigma) per forward pass.
4. Transformer over time tokens.
5. (\log\sigma)-conditioned AdaLN before Q/K/V.
6. Standard attention:

[
\mathrm{softmax}(QK^\top/\sqrt d)V
]

7. Masked (\epsilon)-prediction loss.
8. Monte Carlo averaged score over (K) masks/noise samples.
9. q=0.99 threshold from clean val_normal.
10. Evaluate FAR/TPR/AUROC/AP on clean and drifted test sets.

Do not implement multi-noise-scale product attention first.

Do not implement (b_t) or (b_\sigma).

Do not implement P1–P5 training augmentation as main method.

Do not implement reconstruction scoring as the main score.

The first code version should be:

[
\boxed{
\text{conditional DDPM-style score model}
+
\text{Transformer time tokens}
+
\log\sigma\text{-conditioned Q/K/V}
+
\text{masked denoising residual score}
}
]

---

## 20. Implementation layout

The implementation in this folder is intentionally split into six code files:

* `config.py`: paths, dataclass configs, q=0.99 drifted-test path helper.
* `data.py`: parquet loading, train-only StandardScaler, fixed-length resampling, tensor datasets.
* `model.py`: CCSD-QKV denoising Transformer.
* `engine.py`: mask sampling, sigma sampling, train loss, EMA, MC scoring, metrics.
* `train.py`: train on `train_normal.parquet` only.
* `infer.py`: calibrate threshold on `val_normal.parquet`, then evaluate clean and q=0.99 drifted tests.

The code does not use `torch.compile`.

### Recommended training command

```bash
cd /mnt/disk2/CaiShenghao/ITS/DataDrift/src/ours/v1

CUDA_VISIBLE_DEVICES=0 python train.py \
  --train-path /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/train_normal.parquet \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_ccsd_qkv \
  --run-name ccsd_qkv_v1_t512_d128_l4 \
  --target-length 512 \
  --epochs 120 \
  --batch-size 32 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --model-dim 128 \
  --depth 4 \
  --heads 4 \
  --train-mask-ratio 0.25 \
  --eval-mask-ratio 0.25 \
  --mask-strategy mixed \
  --sigma-min 0.02 \
  --sigma-max 1.0 \
  --num-workers 4 \
  --amp
```

### Recommended q=0.99 evaluation command

```bash
cd /mnt/disk2/CaiShenghao/ITS/DataDrift/src/ours/v1

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_ccsd_qkv/ccsd_qkv_v1_t512_d128_l4/checkpoint.pt \
  --calibration /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/val_normal.parquet \
  --include-q099-drifted \
  --clean-test-path /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/test.parquet \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_ccsd_qkv/ccsd_qkv_v1_t512_d128_l4/inference_q099 \
  --threshold-q 0.99 \
  --batch-size 32 \
  --mc-samples 32 \
  --mc-chunk 4 \
  --num-workers 4 \
  --amp
```
