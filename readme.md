## A. One-sentence paper pitch

提出一种 **nuisance-profiled score-based diffusion normality** 方法：先用 score-based SDE 学 clean robot-cycle joint normal density，再用 diffusion denoising geometry 定义非语义、非 P1–P5 的低描述长度 nuisance drift prior，最终用 profile likelihood 评估“观测 cycle 是否能以最小 admissible nuisance correction 回到 clean normal manifold”，目标是在固定 clean-val q=0.99 阈值协议下降低 drifted-normal FAR，同时保留 global anomaly separability。你的输入限制、P1–P5 只作 held-out evaluation、58 measurement channels-only、clean-val q=0.99 threshold 等约束我全部按你给定协议处理。

## B. Method name

**TNP-Diffusion: Tangent-Nuisance Profiled Diffusion**

这个名字只表达方法对象：tangent nuisance、profile likelihood、diffusion normality。它不把 Transformer、QKV、reconstruction 或 drift simulation 包装成贡献。

## C. Problem reframing

令一个 robot cycle 展平为 (x\in\mathbb{R}^{D})，其中 (D=T\times 58)。训练数据来自 clean normal distribution：

[
x \sim P_0.
]

Drifted normal 不是新的 anomaly class，而是 clean normal 经过未观测 benign machine-response nuisance perturbation 后的观测分布：

[
y = z+\delta,\qquad z\sim P_0,\qquad \delta\sim P_{\eta}(\delta\mid z),
]

其中 (\delta) 不由 P1–P5 generator 定义，也不使用 active/action/setting/time/sample/category/anomaly metadata。Anomaly distribution 可写成：

[
y = z+\delta+a,
]

其中 (a) 是不应被 benign nuisance prior 解释的结构性偏离。

static density 或 flow detector 使用：

[
S_{\mathrm{static}}(y)=-\log p_0(y).
]

当 benign drift 把 (y) 推离 clean training support 时，即使 (y) 仍是机器正常响应，(-\log p_0(y)) 也会上升，导致 drifted-normal FAR 高。你给出的 MVT-Flow 结果中，drifted TPR 上升不一定代表更好 detection，而可能只是 global score inflation。这个问题本质上不是 backbone 问题，而是 **normality score 对 nuisance drift 没有 quotient / profiling 机制**。

纯 masked conditional score 的失败也可以数学化：它估计的是

[
p(x_M\mid x_{\bar M}),
]

而不是 joint normality (p(x))。如果 anomaly 在同一个异常 cycle 内部仍然自洽，那么 (p(x_M\mid x_{\bar M})) 可以很高；这解释了“conditional consistency (\neq) normality”。因此新方法不能只做 same-cycle conditional imputation，必须保留 global clean-normal density。

TNP-Diffusion 重新定义 anomaly score：

[
S_{\mathrm{TNP}}(y)
===================

-\log \sup_{z}
p_{\theta}(z),p_{\eta,\theta}(y-z\mid z),
]

等价于

[
S_{\mathrm{TNP}}(y)
===================

\min_{z}
\left[
-\log p_{\theta}(z)
-------------------

\log p_{\eta,\theta}(y-z\mid z)
\right].
]

这不是 arbitrary two-term energy。它是一个 joint latent-variable model：

[
p(y,z)=p_{\theta}(z),p_{\eta,\theta}(y-z\mid z),
]

对 nuisance clean latent (z) 做 profile likelihood。(p_{\theta}) 由 score-based diffusion 学 clean normal joint density；(p_{\eta,\theta}) 由 diffusion denoising geometry 给出 admissible nuisance prior。

## D. Core modeling principle

核心原则是：

[
\textbf{Normality after minimal diffusion-geometric nuisance explanation.}
]

也就是：一个 cycle 应被视为正常，当且仅当存在一个 clean latent trajectory (z) 使得：

[
z \text{ has high clean-normal likelihood}
]

且

[
\delta=y-z \text{ has short nuisance code length under diffusion-induced tangent prior}.
]

这不是“修正后再检测”的 pipeline，而是一个带 nuisance latent variable 的概率模型。benign drift 被看作 nuisance parameter；anomaly score 是 nuisance-profiled negative log likelihood。profile likelihood 是处理 nuisance parameters 的标准统计思想，semiparametric profile likelihood 也有成熟理论基础。([JSTOR][1])

这个原则要求三个条件同时成立：

第一，clean global normality 不能丢。否则会重复 masked conditional failure。

第二，nuisance correction 不能任意自由。否则 true anomaly 会被修掉。

第三，nuisance prior 不能来自 P1–P5、channel semantics、active mask 或手工 drift feature。它必须来自 clean normal score geometry 本身。

## E. Main method: exactly 3 substantive components

### Component 1 — Clean-normal probability-flow diffusion density

**Purpose.** 学习 clean normal robot-cycle 的 joint density (p_{\theta}(x))，作为 global normality anchor。

**Mathematical definition.** 使用 score-based SDE 学

[
s_{\theta}(x_t,t)\approx \nabla_{x_t}\log p_t(x_t),
]

并用 probability-flow ODE 计算 clean negative log likelihood：

[
E_{\theta}(x)=-\log p_{\theta}(x).
]

Score-based SDE 提供连续时间 perturbation、reverse-time score formulation 和 probability-flow ODE likelihood；这比仅用 DDPM sampling 更适合 anomaly scoring，因为我们需要 likelihood-like score，而不是生成样本。([arXiv][2])

**Input/output.** 输入是 train_normal 的 (x\in\mathbb{R}^{T\times 58})。输出是 (s_{\theta}(x_t,t)) 和 (E_{\theta}(x))。

**Why it is necessary.** 没有 (E_{\theta})，方法会退化成 conditional consistency 或 correction cost，无法判断 corrected trajectory 是否是 global clean normal。

**How it contributes to drift robustness.** 它本身不解决 drift FAR；它提供 clean manifold / clean density，让后续 nuisance profiling 有 anchor。drift robustness 来自 Component 2 和 3，但 Component 1 防止模型把任意 corrected sample 当正常。

**How it preserves anomaly sensitivity.** True anomaly 即使局部自洽，只要不能被映射到高 (p_{\theta}) 的 clean latent (z)，仍会得到高 score。

**Assumption.** train_normal 足以覆盖 clean normal operation 的主要 joint dynamics。

**Why the assumption is not dataset-specific.** 单类 clean-normal density modeling 是工业 robot AD 的标准 unsupervised setting；voraus-AD 论文也明确把 anomaly detection 设定为用 normal data 学习 unusual-event detection。([arXiv][3])

**Supporting literature.** DDPM 连接 diffusion latent-variable modeling、variational bound 和 denoising score matching；score-based SDE 给出连续时间 score 和 probability-flow ODE likelihood。([NeurIPS Proceedings][4])

**What would make it invalid.** 如果 probability-flow likelihood 在该 high-dimensional time series 上校准很差，或者 clean train split 本身混入多个未建模 domains，使 (p_{\theta}) 学到错误 typical set，那么 Component 1 不可靠。Likelihood-based deep generative AD 本身存在已知风险：deep generative models 有时会给 OOD 数据更高 likelihood，因此不能把 likelihood 当作无条件可靠的 semantic normality measure。([arXiv][5])

---

### Component 2 — Diffusion-induced tangent nuisance prior

**Purpose.** 定义哪些 perturbation 可以被解释为 benign nuisance drift，且这个定义不使用 P1–P5、active mask、channel groups 或 hand-crafted frequency/physical features。

**Mathematical definition.** 对 VP-SDE forward perturbation：

[
x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I),
]

score model 给出 denoising posterior mean：

[
D_{\theta}(u,t)
===============

\mathbb{E}*{\theta}[x_0\mid x_t=u]
\approx
\frac{u+\sigma_t^2 s*{\theta}(u,t)}{\alpha_t}.
]

这个形式是 Tweedie-type empirical Bayes identity 在 Gaussian perturbation diffusion 中的对应形式；Tweedie’s formula 本身是 empirical Bayes 中连接 posterior correction 与 marginal score 的经典关系。([Brad Efron's Site][6])

定义 diffusion-induced local uncertainty：

[
\Sigma_{\theta}(u,t)
====================

\operatorname{Cov}_{\theta}(x_0\mid x_t=u).
]

在概念上，(\Sigma_{\theta}) 的大方差方向是 denoising posterior 对 clean state 不确定的方向，也就是 clean normal manifold 的 tangent-like directions；小方差方向是 denoising posterior 强烈收缩的 transverse directions。Regularized / denoising autoencoder literature 也指出 denoising mappings 与 score、local manifold structure 有理论联系。([Journal of Machine Learning Research][7])

对候选 clean latent (z)，定义 nuisance prior：

[
p_{\eta,\theta}(\delta\mid z)
=============================

\int
\mathcal{N}!\left(
\delta;,0,,
K_{\theta}(z,t)
\right)
\pi(t),dt,
]

其中

[
K_{\theta}(z,t)
===============

\Sigma_{\theta}(\alpha_t z,t).
]

(\pi(t)) 使用同一个 diffusion time law，而不是为 drift stress tests 手动设计 noise schedule。重要的是，(p_{\eta,\theta}) 包含 covariance determinant / mixture normalization，因此大 correction scale 不免费。对应 nuisance code length 是：

[
C_{\theta}(\delta\mid z)
========================

-\log p_{\eta,\theta}(\delta\mid z).
]

**Input/output.** 输入是 trained score model (s_{\theta})、候选 clean latent (z)、correction (\delta)。输出是 nuisance likelihood (p_{\eta,\theta}(\delta\mid z)) 或 code length (C_{\theta}(\delta\mid z))。

**Why it is necessary.** 直接最小化 (E_{\theta}(y-\delta)) 会允许任意 correction，把 anomaly 修掉。必须给 (\delta) 一个概率 prior，而且这个 prior 不能手工规定 torque/current/voltage、active phase、P1–P5 affected groups 或 spectral ripple。

**How it contributes to drift robustness.** Benign machine-response drift 通常保持 cycle 的 task-level structure，只是改变机器响应的 nuisance degrees of freedom。若 drift 落在 clean score geometry 的 tangent-like uncertain directions，它会有较低 (C_{\theta})，因此不会显著推高 anomaly score。

**How it preserves anomaly sensitivity.** True anomaly 如果破坏了 clean joint dynamics，需要 transverse correction 或大幅 correction；这会导致 (C_{\theta}) 高。即使某个 corrected (z) 有高 (p_{\theta}(z))，若 (y-z) 不符合 tangent nuisance prior，score 仍高。

**Assumption.** Benign drift 是 low-description-length perturbation relative to clean score geometry；true anomalies 包含不可由该 tangent prior 低成本解释的 transverse or high-complexity deviation。

**Why the assumption is not dataset-specific.** 这个 prior 只来自 train_normal diffusion denoising posterior，不知道 active=1、不知道 P1–P5 channel groups、不知道 voltage sag 或 vibration ripple。它表达的是一般机器系统中的 nuisance-vs-structural deviation 区分，而不是 voraus-AD drift generator。

**Supporting literature.** Score-SDE 提供多噪声 score geometry；Tweedie formula 连接 posterior denoising 与 score；denoising autoencoder theory 支持 denoising map 捕获 local density / manifold geometry。([arXiv][2])

**What would make it invalid.** 如果 P1–P5 或真实 drift 主要沿 clean score transverse directions 发生，Component 2 不会降低 FAR。反过来，如果 true anomalies 也只是 tangent-like low-code perturbations，那么没有标签或外部 physics 的 unsupervised method 本来就不可辨识；这种情况下 TNP-Diffusion 不应声称能检测。

---

### Component 3 — Nuisance-profiled diffusion normality score

**Purpose.** 把 clean density 和 nuisance prior 合成一个单一 probability model，而不是把 reconstruction、correction、diffusion score 人为相加。

**Mathematical definition.**

[
p_{\theta}(y,z)
===============

p_{\theta}(z),
p_{\eta,\theta}(y-z\mid z).
]

Final score 是 nuisance-profiled negative log likelihood：

[
S_{\mathrm{TNP}}(y)
===================

# -\log \sup_z p_{\theta}(y,z)

\min_z
\left[
E_{\theta}(z)+C_{\theta}(y-z\mid z)
\right].
]

这里的两项不是 arbitrary weighted sum。它们分别是 joint probability 的两个 log factors：

[
-\log p_{\theta}(z)
\quad\text{and}\quad
-\log p_{\eta,\theta}(y-z\mid z).
]

没有 (\lambda_1,\lambda_2)。penalty 的来源是 negative log nuisance prior / description length，而不是“为了防止乱修所以加 L2”。MDL 的基本思想也是用 model description length 和 data code length 的总长度解释数据；这里的 (C_{\theta}) 就是 nuisance correction 的 code length。([IBM Research][8])

**Input/output.** 输入是观测 cycle (y)、clean diffusion energy (E_{\theta})、nuisance code length (C_{\theta})。输出是 profiled latent (z^\star)、correction (\delta^\star=y-z^\star)、score (S_{\mathrm{TNP}}(y))。

**Why it is necessary.** 如果只用 (E_{\theta}(y))，drifted normal 会被误报。如果只用 (C_{\theta})，就没有 global normality。如果先 reconstruction 再看 residual，就会退化为 autoencoder-style AD。Profile likelihood 是把 nuisance 作为 latent variable 纳入同一个 probability model。

**How it contributes to drift robustness.** Drifted normal 的 (y) 只要能被解释为 high-likelihood (z) 加 low-code nuisance (\delta)，score 不会因 global score inflation 大幅上升。

**How it preserves anomaly sensitivity.** Anomaly 必须同时满足“能回到 high-likelihood clean (z)”和“correction 有低 nuisance code length”。纯 same-cycle self-consistency 不足以降低 score。

**Assumption.** 对 nuisance 做 MAP/profile 是合理近似；marginal likelihood

[
-\log \int p_{\theta}(z)p_{\eta,\theta}(y-z\mid z),dz
]

在高维下可由 dominant mode 近似。Profile likelihood 是标准 nuisance parameter treatment；variational / Bayesian marginalization 是相关但不同的路线。([JSTOR][1])

**Why the assumption is not dataset-specific.** Profiling nuisance parameters 是统计建模原则，不依赖 robot active phase 或 P1–P5 construction。

**Supporting literature.** Profile likelihood、robust statistics、MDL 和 variational inference 都支持“fit quality + nuisance complexity / prior probability”这一建模逻辑。([JSTOR][1])

**What would make it invalid.** 如果 optimization over (z) 经常找到 spurious high-likelihood solutions，或者 (C_{\theta}) 太宽导致 anomaly 被低成本解释，则 score 会塌缩。这个风险不能靠讲故事解决，必须由 score distribution、correction code length 和 anomaly TPR 证据支撑。

## F. Diffusion formulation

Primary foundation: **score-based SDE**.

Implementation parameterization: 可以用 **DDPM-style epsilon prediction** 训练 score network，但理论对象不是 DDPM Markov sampler，而是 continuous-time score model plus probability-flow ODE likelihood。DDPM 本身是 discrete diffusion latent-variable model，Ho et al. 使用 weighted variational bound 并连接 denoising score matching；这里借用的是 epsilon-prediction parameterization，不把 reverse sampling 当 anomaly score。([NeurIPS Proceedings][4])

Forward perturbation 采用 VP-SDE：

[
d x_t
=====

-\frac{1}{2}\beta(t)x_t,dt
+
\sqrt{\beta(t)},dw_t,
\qquad t\in[0,1].
]

其 closed form 是：

[
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
]

Score model：

[
s_{\theta}(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t).
]

Denoising score matching objective：

[
\mathcal{L}_{\mathrm{DSM}}(\theta)
==================================

\mathbb{E}*{x_0,t,\epsilon}
\left[
\lambda(t)
\left|
s*{\theta}(x_t,t)
+
\frac{x_t-\alpha_t x_0}{\sigma_t^2}
\right|_2^2
\right].
]

DDPM-style epsilon prediction 等价写法：

[
\epsilon_{\theta}(x_t,t)\approx \epsilon,
]

[
\mathcal{L}_{\epsilon}(\theta)
==============================

\mathbb{E}*{x_0,t,\epsilon}
\left[
|\epsilon-\epsilon*{\theta}(x_t,t)|_2^2
\right],
]

并用

[
s_{\theta}(x_t,t)
=================

-\frac{\epsilon_{\theta}(x_t,t)}{\sigma_t}.
]

这是唯一主 training loss。没有 reconstruction loss、contrastive loss、classification loss、consistency loss、frequency loss 或 drift simulation loss。

Probability-flow ODE：

[
d x_t
=====

\left[
f(x_t,t)
--------

\frac{1}{2}g(t)^2s_{\theta}(x_t,t)
\right]dt,
]

其中 VP-SDE 下

[
f(x,t)=-\frac12\beta(t)x,\qquad g(t)=\sqrt{\beta(t)}.
]

用 continuous normalizing flow change-of-variables 计算：

[
\log p_{\theta}(x_0)
====================

\log p_1(x_1)
+
\int_0^1
\nabla\cdot
\left[
f(x_t,t)
--------

\frac12 g(t)^2s_{\theta}(x_t,t)
\right]dt.
]

Score-based SDE paper 明确给出 reverse-time SDE 依赖 time-dependent score，并推导等价 probability-flow ODE，可用于 exact likelihood computation。([arXiv][2])

Nuisance variable handling:

[
y=z+\delta.
]

(\delta) 不在 training 中通过 drift labels 学；它在 inference/scoring 时被 profiled。其 prior (p_{\eta,\theta}(\delta\mid z)) 由 trained score model 的 denoising posterior uncertainty 定义。

Reverse sampling: **不需要**。
DDIM: **不使用**；DDIM 是非 Markovian reverse process / accelerated sampling family，训练 objective 可与 DDPM 相同，但它不是这里的 likelihood score。([arXiv][9])
Consistency model: **不作为主方法**；consistency models 主要学习把 noisy states 直接映射到 clean states，用于 one-step/few-step generation。它最多可作为以后加速 profile optimization 的近似器，但不能作为本文核心 score。([Proceedings of Machine Learning Research][10])
DiT: 若使用 Transformer denoiser，只是 backbone。DiT 已证明 diffusion U-Net 可被 Transformer 替换并具有 scaling benefits；因此 plain Transformer denoiser 不能算贡献。([arXiv][11])
Latent diffusion / Stable Diffusion: 不使用。Latent diffusion 是在 learned latent representation 上做 diffusion；Stable Diffusion 是 text-to-image latent diffusion model，不是 diffusion 的泛称。([arXiv][12])

## G. Anomaly score

Final score:

[
\boxed{
S_{\mathrm{TNP}}(y)
===================

\min_{z}
\left[
-\log p_{\theta}(z)
-------------------

\log p_{\eta,\theta}(y-z\mid z)
\right]
}
]

其中

[
p_{\eta,\theta}(\delta\mid z)
=============================

\int
\mathcal{N}
\left(
\delta;0,K_{\theta}(z,t)
\right)
\pi(t),dt.
]

Higher score = more anomalous.

Benign drift handling:

[
y=z+\delta_{\mathrm{benign}},
]

若 (z) 仍是 high-likelihood clean normal，且 (\delta_{\mathrm{benign}}) 在 diffusion-induced tangent nuisance prior 下 code length 低，则：

[
S_{\mathrm{TNP}}(y)
\approx
E_{\theta}(z)+\text{small nuisance code}.
]

因此 drift 不会像 static density 那样直接造成 global score inflation。

True anomaly handling:

[
y=z+\delta+a.
]

若 (a) 破坏 clean joint dynamics，则不存在同时满足以下两点的解释：

[
E_{\theta}(z)\text{ low}
]

和

[
C_{\theta}(y-z\mid z)\text{ low}.
]

所以 score 仍高。

是否会把 anomaly 修掉：会有风险，但不是无约束 correction。修掉 anomaly 只有在 anomaly 本身属于 diffusion-tangent nuisance prior 的 low-code region 时才会发生。这不是方法实现缺陷，而是 identifiability boundary：如果 true anomaly 与 benign nuisance 在 observation-only、normal-only setting 下不可区分，则任何不使用 labels/metadata/physics 的 detector 都不能保证分离。

Penalty 理论依据：

[
C_{\theta}(\delta\mid z)
========================

-\log p_{\eta,\theta}(\delta\mid z).
]

这是 negative log prior / correction code length。它不是手工 L2 penalty。若使用 Gaussian component，log determinant normalization 必须保留，因为它是 Occam / description length penalty 的一部分；不能只保留 quadratic residual。

Threshold:

[
\tau_{99}
=========

\operatorname{Quantile}*{0.99}
\left(
{S*{\mathrm{TNP}}(x_i):x_i\in \mathrm{val_normal}}
\right).
]

测试时固定 (\tau_{99})，分别评估 clean test_normal、clean test_anomaly、drifted test_normal、drifted test_anomaly。不能用 P1–P5 performance 选 nuisance scale、rank、threshold 或 method structure。

## H. Relation to prior work

| Work                                                                 | Relevant contribution                                                                                                                                                     | Relation to TNP-Diffusion                                                                                                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DDPM, Ho et al., NeurIPS 2020                                        | Discrete diffusion probabilistic model; weighted variational bound; practical epsilon prediction; connection to denoising score matching. ([NeurIPS Proceedings][4])      | TNP may use epsilon prediction as implementation, but the anomaly score is not DDPM reverse reconstruction or sampling likelihood.                            |
| Score-Based Generative Modeling through SDEs, Song et al., ICLR 2021 | Continuous-time SDE framework; reverse SDE depends on score; probability-flow ODE enables likelihood computation. ([arXiv][2])                                            | Primary theoretical foundation.                                                                                                                               |
| DDIM, Song et al., ICLR 2021                                         | Non-Markovian diffusion process with same training objective as DDPM, designed for faster sampling. ([arXiv][9])                                                          | Not used. TNP does not need reverse sampling.                                                                                                                 |
| Consistency Models, Song et al., ICML 2023                           | Direct noisy-to-clean consistency mapping; one-step/few-step generation. ([Proceedings of Machine Learning Research][10])                                                 | Not main method. Could only approximate computation later.                                                                                                    |
| Improved Techniques for Training Consistency Models, ICLR 2024       | Direct consistency training improvements, pseudo-Huber loss, lognormal noise schedule, better one-step quality. ([ICLR Proceedings][13])                                  | Not used in main formulation; avoids mixing consistency objective into score-SDE likelihood.                                                                  |
| DiT, Peebles & Xie, ICCV 2023                                        | Replaces diffusion U-Net with Transformer backbone and studies scaling. ([arXiv][11])                                                                                     | Optional denoiser backbone only. Plain DiT is not contribution.                                                                                               |
| Latent Diffusion / Stable Diffusion                                  | Latent diffusion performs diffusion in an autoencoder latent space; Stable Diffusion is a text-to-image latent diffusion model. ([arXiv][12])                             | Not used; “Stable Diffusion” should not be used as generic diffusion terminology.                                                                             |
| CSDI, NeurIPS 2021                                                   | Conditional score-based diffusion for time-series imputation; exploits observed values to impute missing values. ([arXiv][14])                                            | Relevant negative reference: pure conditional imputation can miss self-consistent anomalies.                                                                  |
| TimeGrad, ICML 2021                                                  | Autoregressive denoising diffusion for multivariate probabilistic time-series forecasting. ([arXiv][15])                                                                  | Forecasting model, not nuisance-profiled global AD.                                                                                                           |
| TimeDiff, ICML 2023                                                  | Non-autoregressive conditional diffusion for time-series prediction with future mixup and autoregressive initialization. ([Proceedings of Machine Learning Research][16]) | Conditional prediction, not clean-normal profile likelihood.                                                                                                  |
| Diffusion-TS, ICLR 2024                                              | Time-series generation using transformer and decomposition-oriented representation; direct reconstruction plus Fourier-based loss. ([arXiv][17])                          | Not adopted; TNP avoids hand-designed frequency/reconstruction objectives.                                                                                    |
| D3R, NeurIPS 2023                                                    | Dynamic Decomposition with Diffusion Reconstruction for unstable multivariate time-series AD; targets drift via decomposition/reconstruction. ([NeurIPS Proceedings][18]) | Closest drift-related diffusion AD work. TNP differs by using profile likelihood and score-induced nuisance prior, not decomposition-reconstruction pipeline. |
| On Diffusion Modeling for Anomaly Detection, ICLR 2024               | Studies diffusion for unsupervised/semi-supervised AD; DDPM can work but is expensive; proposes Diffusion Time Estimation. ([ICLR Proceedings][19])                       | Supports diffusion AD relevance; TNP targets drift robustness via nuisance profiling rather than diffusion-time score alone.                                  |
| Profile likelihood                                                   | Profiles nuisance parameters out of likelihood; semiparametric profile likelihood can behave like ordinary likelihood under conditions. ([JSTOR][1])                      | Direct statistical basis for (S_{\mathrm{TNP}}).                                                                                                              |
| Robust statistics                                                    | Provides formal language for robustness rather than ad hoc correction. ([Open Library][20])                                                                               | TNP is robust through a probabilistic nuisance model, not manual outlier clipping.                                                                            |
| MDL / description length                                             | Model selection and inference via shortest description of data; balances fit and complexity. ([IBM Research][8])                                                          | Justifies correction code length (C_{\theta}).                                                                                                                |
| Variational inference / Bayesian marginalization                     | Treats latent variables by approximate posterior inference or marginalization. ([Springer][21])                                                                           | TNP chooses profile likelihood rather than full variational marginalization for a clean anomaly score.                                                        |

## I. Why this is not module stacking

TNP-Diffusion is one latent-variable probability model:

[
p(y,z)=p_{\theta}(z)p_{\eta,\theta}(y-z\mid z).
]

Component 1 defines (p_{\theta}(z)). Component 2 defines (p_{\eta,\theta}(\delta\mid z)). Component 3 profiles (z) out to obtain the anomaly score. Removing any one component changes the mathematical definition of (S_{\mathrm{TNP}}).

It is not “encoder + diffusion + correction + score”。There is no separate encoder objective, no auxiliary reconstruction loss, no contrastive loss, no classification head, no handcrafted drift score. The corrected (z^\star) is not a reconstructed output used as an autoencoder residual; it is the profiled clean latent variable of a joint likelihood.

It is not P1–P5-specific. The nuisance prior never sees P1–P5 formulas, alphas, affected channel groups, active masks, voltage sag, spectral ripple, sample-rank drift schedule, or torque/current/voltage semantic grouping.

It is not hand-crafted feature engineering. No per-cycle mean/slope/variance, no frequency feature, no active-phase anchor, no channel group rule. The admissible directions are induced by score-model denoising posterior geometry.

It is not reconstruction-based AD. Reconstruction AD usually asks whether a network can reconstruct (y). TNP asks whether (y) has high profiled probability under (p_{\theta}(z)p_{\eta,\theta}(y-z\mid z)). A large but easy reconstruction is not automatically normal unless (z) has high clean density and (\delta) has low nuisance code length.

It is not plain DiT. If a Transformer is used, attention remains standard:

[
\operatorname{Attn}(H)
======================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
]

Noise conditioning can be ordinary:

[
e_t=\operatorname{MLP}(\log \sigma_t),
]

then AdaLN/FiLM modulation. No (b_t), no (b_{\sigma}), no active-phase bias, no channel-group bias. Backbone choice is implementation, not contribution.

## J. Reviewer-risk analysis

**1. Synthetic drift evaluation risk.** P1–P5 are useful stress tests, but reviewers can object that they are still synthetic. If the paper only proves FAR reduction on synthetic drifted test_normal, the claim should be framed as “machine-intrinsic drift stress robustness,” not broad real-world drift robustness. Stronger evidence would require real long-horizon drift logs, another robot task, or at least a non-voraus external dataset.

**2. Nuisance correction may repair anomaly.** This is the main theoretical risk. If a true anomaly is a low-code tangent perturbation under (p_{\eta,\theta}), TNP will likely suppress it. The method is valid only when benign drift and anomaly differ in diffusion-geometric description length. If they do not, the problem is non-identifiable under your stated no-label/no-metadata/no-physics setting.

**3. Method may lower FAR by reducing all scores.** Clean-val q=0.99 threshold prevents trivial global score scaling from directly lowering FAR, but it does not guarantee TPR. The decisive question is whether drifted-normal scores move below (\tau_{99}) while drifted-anomaly scores remain separated. If both distributions collapse, method fails.

**4. One dataset is weak for ML venues.** For IEEE T-RO / T-ASE, a strong robotics-specific formulation plus rigorous protocol may be acceptable. For NeurIPS/ICLR/ICML, one dataset and synthetic drift will be viewed as insufficient unless the theory or cross-domain evidence is much stronger.

**5. Penalty could be attacked as hand-crafted.** The defense is that (C_{\theta}=-\log p_{\eta,\theta}), not L2/L1. However reviewers may still ask why diffusion posterior covariance is the right nuisance prior for mechanical drift. This is a real vulnerability.

**6. Diffusion necessity is not automatic.** A reviewer may ask: why not MVT-Flow plus a learned nuisance profile? The answer must be that score-based diffusion provides multi-noise denoising posterior geometry, not just density. If experiments show a flow with a comparable nuisance prior works equally well, the diffusion-specific claim weakens.

**7. Probability-flow likelihood may be unstable.** High-dimensional time-series likelihood can be numerically fragile and may correlate with low-level statistics rather than semantic normality. This is a known concern for likelihood-based deep generative AD.([arXiv][5])

**8. Identifiability problem is fundamental.** If benign drift and true anomaly both appear as smooth machine-response changes in the same measured channels, with no labels, no metadata, no intervention, and no physics prior, no method can reliably distinguish them. TNP should explicitly state this boundary rather than overclaim.

**9. Profile optimization may find spurious (z^\star).** In high dimension, the profiled objective can have undesirable modes. If (p_{\theta}) has likelihood artifacts, profiling may exploit them. This is not a minor engineering issue; it affects the definition of the score.

**10. D3R comparison risk.** D3R already targets drift in unstable multivariate time-series AD with diffusion reconstruction. Reviewers will ask whether TNP is genuinely different. The answer is yes only if the paper emphasizes profile likelihood / nuisance prior / diffusion score geometry, not generic drift reconstruction.

## K. Final judgment

This method is worth doing, but only if you accept a narrow and honest claim:

[
\text{drift-robust normality scoring under diffusion-induced nuisance profiling}.
]

It has paper potential because the modeling object is clear: not a new backbone, not a QKV tweak, not masked imputation, but a nuisance-profiled probability model. It also directly targets your primary objective: reducing drifted-normal FAR without calibrating on P1–P5.

Best venue fit: **T-ASE or T-RO** first. The problem is robotics/industrial AD-specific, and the contribution is methodological but application-grounded. For NeurIPS/ICLR/ICML, the risk is higher: reviewers will expect broader datasets, stronger theory for the nuisance prior, and more general evidence beyond voraus-AD synthetic drift.

Maximum theoretical risk: **benign drift and true anomaly may be non-identifiable under measurement-only normal-only data**. If anomaly lies in the same low-code tangent nuisance family, TNP will suppress it.

Maximum experimental risk: **the nuisance prior is either too narrow or too broad**. Too narrow: FAR remains high under drift. Too broad: TPR/AUROC/AP collapse because anomaly is repaired.

The method stands only if evidence shows three things simultaneously: drifted-normal score inflation is reduced; anomaly scores remain separated after profiling; and learned corrections have low code length for benign drift but high code length for true anomalies. Without those, the method is mathematically clean but empirically unconvincing.

[1]: https://www.jstor.org/stable/pdf/2669386.pdf?utm_source=chatgpt.com "On Profile Likelihood - JSTOR"
[2]: https://arxiv.org/abs/2011.13456?utm_source=chatgpt.com "Score-Based Generative Modeling through Stochastic Differential Equations"
[3]: https://arxiv.org/abs/2311.04765?utm_source=chatgpt.com "The voraus-AD Dataset for Anomaly Detection in Robot Applications"
[4]: https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html?utm_source=chatgpt.com "Denoising Diffusion Probabilistic Models - NeurIPS"
[5]: https://arxiv.org/abs/1810.09136?utm_source=chatgpt.com "Do Deep Generative Models Know What They Don't Know?"
[6]: https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf?utm_source=chatgpt.com "Tweedie’s Formula and Selection Bias"
[7]: https://jmlr.org/papers/volume15/alain14a/alain14a.pdf?utm_source=chatgpt.com "What Regularized Auto-Encoders Learn from the Data-Generating Distribution"
[8]: https://research.ibm.com/publications/modeling-by-shortest-data-description?utm_source=chatgpt.com "Modeling by shortest data description for Automatica"
[9]: https://arxiv.org/abs/2010.02502?utm_source=chatgpt.com "[2010.02502] Denoising Diffusion Implicit Models - arXiv.org"
[10]: https://proceedings.mlr.press/v202/song23a.html?utm_source=chatgpt.com "Consistency Models - PMLR"
[11]: https://arxiv.org/abs/2212.09748?utm_source=chatgpt.com "[2212.09748] Scalable Diffusion Models with Transformers"
[12]: https://arxiv.org/abs/2112.10752?utm_source=chatgpt.com "High-Resolution Image Synthesis with Latent Diffusion Models"
[13]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html?utm_source=chatgpt.com "Improved Techniques for Training Consistency Models"
[14]: https://arxiv.org/abs/2107.03502?utm_source=chatgpt.com "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation"
[15]: https://arxiv.org/abs/2101.12072?utm_source=chatgpt.com "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting"
[16]: https://proceedings.mlr.press/v202/shen23d.html?utm_source=chatgpt.com "Non-autoregressive Conditional Diffusion Models for Time Series ... - PMLR"
[17]: https://arxiv.org/abs/2403.01742?utm_source=chatgpt.com "Diffusion-TS: Interpretable Diffusion for General Time Series Generation"
[18]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html?utm_source=chatgpt.com "Drift doesn't Matter: Dynamic Decomposition with Diffusion ... - NeurIPS"
[19]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/6dfd16ff880a63fee9f6469fee58a496-Abstract-Conference.html?utm_source=chatgpt.com "On Diffusion Modeling for Anomaly Detection - proceedings.iclr.cc"
[20]: https://openlibrary.org/books/OL18207548M/Robust_statistics.?utm_source=chatgpt.com "Robust statistics. by Peter J. Huber | Open Library"
[21]: https://link.springer.com/article/10.1007/s10462-011-9236-8?utm_source=chatgpt.com "A tutorial on variational Bayesian inference | Artificial Intelligence ..."

## L. Implementation files and commands

The implementation under this folder uses only the 58 measurement channels as model input. The seven metadata columns are used only for sample grouping and evaluation labels.

Code files:

- `config.py`: paths, dataclasses, q=0.99 drifted-test path helpers.
- `data.py`: parquet loading, train-only StandardScaler, sample-level resampling.
- `model.py`: time-conditioned Transformer score network.
- `engine.py`: VP diffusion training, EMA, TNP profile scoring, metrics.
- `train.py`: clean-normal training entry point.
- `infer.py`: val-threshold calibration and clean/q=0.99 drifted test evaluation.

Full training command:

```bash
cd /mnt/disk2/CaiShenghao/ITS/DataDrift/src/ours/v1
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train-path /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/train_normal.parquet \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion \
  --run-name tnp_diffusion_v1_t384_d160_l5 \
  --target-length 384 \
  --epochs 120 \
  --batch-size 32 \
  --model-dim 160 \
  --depth 5 \
  --heads 5 \
  --ff-mult 4 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --ema-decay 0.999 \
  --num-workers 4 \
  --amp
```

Full q=0.99 inference command:

```bash
cd /mnt/disk2/CaiShenghao/ITS/DataDrift/src/ours/v1
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion/tnp_diffusion_v1_t384_d160_l5/checkpoint_final.pt \
  --calibration /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/val_normal.parquet \
  --include-q099-drifted \
  --clean-test-path /mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/test.parquet \
  --drifted-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/drifted_test_sets \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion/tnp_diffusion_v1_t384_d160_l5/inference_q099 \
  --threshold-q 0.99 \
  --batch-size 16 \
  --score-probes 8 \
  --posterior-probes 4 \
  --profile-energy-probes 4 \
  --profile-steps 14 \
  --profile-lr 0.08 \
  --num-workers 4 \
  --amp
```

Fast review smoke test:

```bash
cd /mnt/disk2/CaiShenghao/ITS/DataDrift/src/ours/v1
python train.py \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion_smoke \
  --run-name smoke \
  --epochs 1 \
  --batch-size 4 \
  --target-length 64 \
  --model-dim 32 \
  --depth 1 \
  --heads 4 \
  --ff-mult 2 \
  --num-workers 0 \
  --limit-train-samples 8 \
  --no-amp

python infer.py \
  --checkpoint /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion_smoke/smoke/checkpoint_final.pt \
  --eval clean_test=/mnt/disk2/CaiShenghao/ITS/DataDrift/data/splits/test.parquet \
  --output-dir /mnt/disk2/CaiShenghao/ITS/DataDrift/data/outputs/ours_v1_tnp_diffusion_smoke/smoke_inference \
  --batch-size 2 \
  --score-probes 1 \
  --posterior-probes 1 \
  --profile-energy-probes 1 \
  --profile-steps 1 \
  --num-workers 0 \
  --limit-eval-samples 4 \
  --no-amp
```
