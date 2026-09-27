# Cross-sector innovations and their mathematics

Each idea below was proven in another field. The table says what it solves
here, and the sections give the equations exactly as implemented, with the
file that implements them.

| # | Borrowed from | Idea | Problem it solves here | Code |
|---|---|---|---|---|
| 1 | Full-duplex speech (Moshi) + early-exit LLMs (CALM) | Listen and speak at the same time; stop once the answer is known | Belief is updated patch by patch, reading stops early, and the belief stream is itself part of the explanation | `models/duplex.py` |
| 2 | Industrial defect inspection (PatchCore) | Memorise *normal*, score distance from it | Flags anomalies never seen in training and needs no abnormal labels | `normality/memory.py` |
| 3 | Liquid-crystal physics | Nematic order parameter and director misalignment | Detects a *single* nucleus pointing the wrong way, with no training | `physics/order.py` |
| 4 | Symmetry in physics (group-equivariant CNNs) | Build rotation and mirror symmetry into the weights | 8x effective data per weight; exact invariance | `models/equivariant.py` |
| 5 | Self-supervised vision (JEPA, SimSiam, VICReg) | Predict hidden context in representation space | Learns from unlabelled scans, which are plentiful | `models/ssl.py` |
| 6 | Statistics (Stein's paradox) | Shrink noisy class estimates toward a common mean | Stable few-shot classes with 5-10 examples | `models/fewshot.py`, `reasoning/evidence.py` |
| 7 | Wartime cryptanalysis (Turing and Good) | Additive weight of evidence in decibans | Each piece of evidence, *including the rejected ones*, has an exact numeric contribution | `reasoning/evidence.py` |
| 8 | Distribution-free statistics (conformal, Learn-then-Test) | Finite-sample guarantees with no model assumptions | A certified error bound on answered cases, and abstention otherwise | `calibration/conformal.py` |
| 9 | Test-driven software engineering | Every bug becomes a regression test | Failures drive which data to acquire next | `loop/failures.py` |
| 10 | Deep-learning scaling laws | Power-law learning curves | Predicts *how much* data a failing slice needs, or that more of the same will not help | `loop/failures.py` |
| 11 | Control engineering and factory QC (Kalman, CUSUM) | Optimal state tracking, sequential change detection | Growth rate with uncertainty and early drift alarms across scans | `longitudinal/change.py` |
| 12 | Large language models (LoRA) | Low-rank weight updates | Adapts to a new site or cancer type with about 10x fewer trainable parameters | `models/lora.py` |

---

## 1. Full-duplex streaming reasoner

Tokens z_1..z_T (patches, most salient first), causal transformer h_t = f(z_1..z_t):

```
belief_t = softmax(W_b h_t)            (the "speaking" stream, emitted at every step)
halt_t   = sigmoid(w_h . h_t)          (barge-in: "my answer will not change")
L = sum_t w_t CE(belief_t, y) + lambda sum_t BCE(halt_t, 1[argmax belief_t = y]),   w_t increasing in t
```

Stopping policy (min_steps, threshold) is chosen on held-out data as the
cheapest one whose early answer agrees with the full read on at least 98% of
cases. Inference with a KV cache costs O(t) per step and is numerically
identical to the parallel forward pass (`tests/test_models.py`).

## 2. Normality memory

```
s(p) = min_{m in M} ||p - m||_2,        M = greedy k-center coreset of all normal patch features
```

k-center minimises the covering radius `max_p min_m ||p - m||`, so a 10%
coreset keeps near-full coverage. Selection runs in a random Gaussian
projection (Johnson-Lindenstrauss) for speed.

## 3. Nematic order and misalignment

```
J = G_si * (grad I)(grad I)^T,   q = ((Jxx - Jyy), 2 Jxy) / tr J,   |q| = coherence
Q = G_sc * q  (context director),   S = |Q| / (G_sc * |q|)
misalignment m = |q| (1 - cos 2(phi - Phi)) / 2
```

The doubled angle removes the 180-degree ambiguity of an orientation, which
is standard for nematics. Cost is O(HW) with 5 Gaussian filters.

## 4. D4 group convolution

```
lifting:    out(g) = conv(x, T_g psi)
group conv: out(g) = sum_h conv(in(h), T_g psi(g^-1 h))
guarantee:  out[T_u x](g) = T_u out[x](u^-1 g)
```

Group tables are derived numerically from the transform definition, and
exact equivariance is tested to 1e-5.

## 5. Masked-view consistency pretraining

```
L = 1/2 [D(p(z_masked), sg(z_full)) + D(p(z_full), sg(z_masked))] + beta sum_j max(0, gamma - std(z_j))
```

## 6. Shrinkage

```
mu_k_hat = mu_0 + n_k / (n_k + tau) (mean_k - mu_0)           (posterior mean, prior N(mu_0, sigma^2/tau))
v_hat    = (n v_within + nu v_total) / (n + nu)
```

## 7. Weight of evidence

```
WoE_j(H1 : H2) = 10 log10 P(e_j | H1) / P(e_j | H2)   [decibans]
posterior log-odds = prior log-odds + sum_j WoE_j      (exact under conditional independence)
```

Items are labelled supporting (>= +3 db), opposing (<= -3 db) or
uninformative. The sum equals the channel's log-odds exactly (tested).
Because concepts are correlated, the naive sum is over-confident. That is why
the channel gets its own temperature and is fused with two other channels.
A *faithfulness constraint* keeps its fusion weight >= 25%, so the
explanation always describes part of the actual decision.

## 8. Guarantees instead of promises

```
conformal:  q_hat = Quantile_{ceil((n+1)(1-alpha))/n}(1 - p_y),   C(x) = {k : 1 - p_k <= q_hat},   P(y in C) >= 1 - alpha
selective:  answer iff max_k p_k >= lambda;  H0(lambda): risk > alpha;  p = P(Bin(n_ans, alpha) <= errors)
            fixed-sequence testing from strict to loose (untestable lambdas skipped, Tarone 1990)
data need:  n_min = ceil(log delta / log(1 - alpha))
```

| target error on answered cases | confidence | error-free answered cases needed |
|---|---|---|
| 5% | 90% | 45 |
| 1% | 90% | 230 |
| 0.1% | 90% | 2,302 |
| 0.01% | 99% | 46,050 |

This is the honest path toward the "100%" goal: the error rate on answered
cases can be pushed as low as the calibration data supports, and everything
else is referred to a specialist, never silently misclassified.

## 9-10. Failure loop and learning curves

```
slice flagged  iff  Wilson_lower_95(failures, n_slice) > overall failure rate
err(n) = a n^-b + c   ->   n* = ((target - c)/a)^(-1/b),   unreachable if target <= c
```

## 11. Longitudinal

```
level_t = level_{t-1} + dt slope_{t-1} + w1,   slope_t = slope_{t-1} + w2,   y_t = level_t + v
S_t = max(0, S_{t-1} + (y_t - mu_0)/sigma_0 - k),   alarm if S_t > h
doubling time DT = ln 2 / b,   b = d ln V / dt
```

## 12. LoRA

```
y = W x + (alpha / r) B A x,    B = 0 at init   (exact no-op start; r(d_in + d_out) trainable parameters)
```

---

## Why this combination makes learning cheaper

* **Symmetry** removes an 8x augmentation burden from the data.
* **Normality modelling** turns the rarest data (abnormal) into the least needed.
* **Physics priors** (order parameter, nuclear morphometry) give strong
  features with zero parameters.
* **Shrinkage** lets classes with 8 examples stay stable.
* **Early exit** reads about 6-25% of the image on the phantoms.
* **Failure-driven acquisition** spends labelling effort only where the
  model fails, and the learning-curve fit says when to stop.
