# Automating the Anomaly Threshold: Moving Beyond a Hardcoded Constant

## Status
Design discussion for the Anomaly Detection module (Module 1) of the Runtime Security Middleware, addressing a practical deployment concern: the current threshold is a manually-set constant, which doesn't scale to a real client rollout. This document is self-contained and can be used to start a fresh chat.

---

## 1. The Current Problem

Every detection strategy you've built so far (`ClsHeadMahalanobisDetector` and its planned siblings) reduces a per-image anomaly signal down to a single decision via:

```python
is_attack = (score > threshold)
```

Right now, `threshold` is a **hardcoded constant** sitting in `config.json` (currently `15.0`, a placeholder that was never derived from real data — see the earlier debugging discussion where this value caused both a clean image and a patched image to be flagged as attacks). Even once it's replaced with a properly-derived number, the *process* used to derive it still requires:

1. Collecting a batch of clean images.
2. Running the detector across all of them offline.
3. Manually computing a statistic (mean, standard deviation, or percentile) from the resulting scores.
4. Manually choosing a multiplier or percentile cutoff.
5. Hardcoding the resulting number back into `config.json`.

### Why this doesn't work as a real product

In an academic evaluation setting, this is fine — you run it once, on your own dataset, and report results. But your research explicitly frames this middleware as a deployable, model-agnostic component (a "Sidecar"), meant to work across different clients, different cameras, different physical environments, and potentially different YOLO versions. That framing runs directly into the manual-calibration process above:

- **It doesn't scale.** Every new client deployment would need someone (you, or a trained operator) to collect a clean dataset specific to that client's cameras/environment, run an offline script, and manually tune a threshold before the system is usable.
- **It's a maintenance burden, not a one-time cost.** A camera angle change, a new sensor, a different time of day/season with different lighting norms, or a change in the client's typical scene content can all shift what "clean" activations look like — meaning the threshold can silently go stale, and nothing in the current design detects that or re-triggers calibration.
- **It contradicts the "decoupled, low-friction" narrative of your own thesis.** The whole point of the middleware is to avoid tight coupling and manual, model-specific engineering. A hardcoded, manually-tuned threshold reintroduces exactly the kind of brittle, manual-intervention-dependent design the research is trying to move away from — just at the threshold-setting layer instead of the architecture layer.

### The consequence you've already observed directly

In your own testing, a threshold that was never statistically derived (`15.0`) produced a **false positive on a clean image** alongside a correct detection on a patched image — both flagged as "attack," making the output useless for the Recovery Engine to act on. That's the concrete cost of hardcoding: the number either overfits to whatever single test case it was eyeballed against, or (as happened here) isn't grounded in any actual data distribution at all.

The rest of this document lays out eight concrete ways to address this, roughly ordered from "cheapest fix, still requires some calibration" to "restructure the problem so calibration barely matters."

---

## 2. Option 1 — Closed-Form Statistical Threshold (Chi-Squared)

### The idea
You're already computing a **Mahalanobis distance** between a feature vector and a per-class reference distribution:

```
M(x) = sqrt( (x - μ)ᵀ Σ⁻¹ (x - μ) )
```

This isn't just any distance metric — it has a known theoretical property. If the underlying feature vectors `x` are (approximately) drawn from a multivariate Gaussian distribution with mean `μ` and covariance `Σ`, then the **squared** Mahalanobis distance follows a **chi-squared distribution** with degrees of freedom `d` equal to the number of feature dimensions:

```
M(x)² ~ χ²(d)
```

This is a standard result in multivariate statistics (it falls directly out of how the chi-squared distribution is defined: a sum of `d` independent squared standard-normal variables). Practically, it means you don't need a separate empirical study to pick a threshold — you can look up the value directly from the chi-squared distribution's inverse CDF (its "percent-point function") at whatever confidence level you want.

### The math
Choose a confidence level, e.g. `confidence = 0.999` (you want at most 0.1% of clean, in-distribution feature vectors to exceed the threshold). Then:

```
threshold² = χ²⁻¹(confidence, df = d)     # inverse CDF of chi-squared at df=d
threshold  = sqrt(threshold²)              # since your code scores sqrt(...), not the squared distance
```

### Code
```python
from scipy.stats import chi2

d = mean.shape[0]            # feature dimensionality of the tapped activation (number of channels)
confidence = 0.999           # tune this single, interpretable knob
threshold = chi2.ppf(confidence, df=d) ** 0.5
```

### Why this matters for your problem
This threshold comes **directly out of the calibration step you already run** — fitting `mean` and `inv_cov` — with no separate validation dataset, no manual sweep, and no eyeballing. The only parameter left is `confidence`, which has a plain-language meaning ("I accept this false-positive rate on clean data") instead of being an arbitrary score cutoff like `15`.

### The honest caveat
The chi-squared property is exact only if the feature vectors are truly multivariate-Gaussian. Deep network activations are frequently *not* perfectly Gaussian (they can be skewed, multi-modal, or heavy-tailed). So treat this as a **principled default** — a hugely better starting point than a guessed constant — but validate it against a small clean sample if you have one, rather than trusting it blindly in a safety-critical setting.

---

## 3. Option 2 — Percentile / ROC-Based Threshold from a Validation Set

### The idea
Instead of a closed-form formula, derive the threshold empirically from actual score data — but do it systematically (a fixed, repeatable *procedure*) rather than by manual inspection of one or two numbers.

### The math
1. Run the detector over a held-out set of clean images (distinct from whatever images were used to *fit* `mean`/`inv_cov` — this needs to be a separate validation split).
2. Collect the resulting max-per-image scores: `s₁, s₂, ..., sₙ`.
3. Set the threshold at a chosen percentile of this distribution:

```
threshold = percentile(s, 99)     # e.g. 99th percentile of clean scores
```

4. If you also have labeled attacked images, you can instead sweep the threshold across the score range and compute the **ROC curve** (True Positive Rate vs. False Positive Rate at every possible threshold), then pick the threshold that maximizes **Youden's J statistic**:

```
J(t) = TPR(t) - FPR(t)
threshold = argmax_t J(t)
```

### Why this matters
This directly answers "how do I pick a defensible number instead of guessing" — but it's still fundamentally a **procedure that needs to be re-run** whenever the deployment environment changes, which is why it alone doesn't fully solve the "clients can't run calibration repeatedly" problem. It's a good validation tool to pair with Option 1 (use chi-squared as the default, use this as a periodic sanity check), rather than the primary mechanism for a production deployment.

---

## 4. Option 3 — Extreme Value Theory (EVT) / SPOT

### The idea
This is a genuinely different category of solution: instead of assuming a specific parametric shape for the *entire* score distribution (like the Gaussian assumption behind Option 1), EVT only models the **tail** — the extreme, rare values — which is exactly the part of the distribution that matters for anomaly detection. It's designed specifically for the problem of "automatically set an anomaly threshold from a stream of scores, without a human picking a magic constant."

The specific, well-established technique here is **SPOT (Streaming Peaks-Over-Threshold)**, introduced by Siffer et al. ("Anomaly Detection in Streams with Extreme Value Theory," ACM SIGKDD 2017). It's built for exactly your deployment scenario: a live, ongoing stream of scores (your anomaly scores, frame by frame), with no assumption about the overall distribution's shape.

### The math
The theoretical basis is the **Pickands–Balkema–de Haan theorem** (the "second theorem" of Extreme Value Theory): for a wide class of distributions, the values that exceed a sufficiently high threshold `h`, once you subtract off `h`, follow a **Generalized Pareto Distribution (GPD)**:

```
P(X - h > x | X > h) ≈ (1 + γx/σ)^(-1/γ)
```

where `γ` (shape) and `σ` (scale) are parameters fit from the data. SPOT works in two phases:

1. **Initialization:** using an initial batch of `n` observations (the paper suggests `n ≈ 1000`), fit a GPD to the values exceeding some moderate initial threshold, then derive a high quantile `z_q` — this is your working anomaly threshold.
2. **Streaming update:** as new scores arrive one at a time, compare each to `z_q`. If it doesn't exceed `z_q`, quietly incorporate it into the running distribution estimate (refining `γ`, `σ`, and `z_q` over time). If it does exceed `z_q`, flag it as anomalous **and exclude it from updating the model** (so a genuine attack doesn't drag your "normal" threshold upward).

The key user-facing parameter is `q`, a **risk/false-positive-rate target** (e.g., `q = 0.001` means "I'm willing to tolerate roughly a 0.1% false-alarm rate") — a single, interpretable dial, not a raw score cutoff.

### Why this matters for your problem
This is arguably the strongest fit for the specific constraint you raised — it's explicitly designed to run **continuously, in production, on live streaming data**, adapting the threshold automatically as conditions change, without a human re-running an offline script. It also self-corrects for domain drift over time (a different client, a different season, a gradually changing scene) since it keeps updating.

### The trade-off
It's meaningfully more complex to implement correctly than Options 1 or 2 (fitting a GPD via maximum likelihood, handling the streaming update logic). Several open-source implementations exist that follow the original paper, which is worth pointing you toward rather than implementing the GPD-fitting math from scratch.

---

## 5. Option 4 — Conformal Prediction

### The idea
Conformal prediction is a general statistical framework for turning any score into a decision with a **formal, provable guarantee** on the error rate — rather than a threshold that's merely "probably reasonable." Applied to anomaly detection, it's often called **conformal anomaly detection**.

### The math
Given a calibration set of clean scores `s₁, ..., sₙ`, and a desired significance level `α` (e.g. `α = 0.05` for a 95% guarantee), compute the threshold as an order statistic:

```
threshold = the ⌈(n+1)(1-α)⌉-th smallest value in {s₁, ..., sₙ}
```

This gives a threshold with the guarantee: for a *new* clean sample drawn from the same distribution as your calibration set, the probability it's flagged as anomalous is at most `α` — a guarantee that holds under fairly weak assumptions (the calibration and test data need to be *exchangeable*, a milder condition than requiring a specific parametric distribution like Option 1's Gaussian assumption).

### Why this matters
Where Option 1 gives you a threshold that's "theoretically motivated" and Option 2 gives you one that's "empirically observed to work well," conformal prediction gives you one you can defend with an actual mathematical guarantee: *"our false-positive rate is bounded by α with high probability, by construction."* For a thesis evaluation chapter, that's a notably stronger claim than "we picked the 99th percentile and it seemed to work."

### The trade-off
Still needs a calibration set (same practical friction as Options 1–2 for actually *obtaining* the data) — this option strengthens the theoretical justification for the threshold, but doesn't by itself solve the "clients can't provide calibration data" problem. Pair it with Option 5 or 6 below for that.

---

## 6. Option 5 — Ship a Pretrained Default Calibration Profile

### The idea
Stop treating calibration as something each client must do. Instead, your team calibrates once, offline, on a large and diverse dataset (KITTI + BDD100K combined, spanning weather, lighting, and geography), and ships the resulting `mean`/`inv_cov` statistics as a static artifact bundled with the middleware — the same way you'd ship pretrained model weights.

### What changes operationally
A new client installs the middleware and it works immediately, using your default profile — no script to run, no dataset to collect, no manual threshold to set (combine with Option 1's chi-squared formula, computed once from your default profile, to get a ready-to-use default threshold too).

### The honest caveat
This is the most practically convenient option, but it has a real accuracy cost: if a client's deployment domain (a different country's road signage, a very different camera sensor, an unusual environment like a warehouse instead of open roads) diverges significantly from your training data, the "clean" reference distribution won't match what that client's clean images actually look like — the default profile will be miscalibrated for them specifically, in ways your own KITTI/BDD100K validation wouldn't reveal. Name this limitation explicitly in your thesis rather than treating shipped defaults as a complete solution; it motivates the next option.

---

## 7. Option 6 — Automatic Bootstrap Self-Calibration on First Deployment

### The idea
Rather than a human running `calibrate_cls_head.py` as a separate step, make calibration something the middleware does **automatically, to itself, on first startup** — using the client's own live camera feed during an initial settling period (e.g., the first few hours of operation after installation), before the system starts actively flagging attacks.

### How it would work, concretely
1. On first boot, the middleware enters a **"learning" mode** instead of a "monitoring" mode.
2. It runs the object detector normally (as it always does) and, for every detection, extracts the tapped activation and adds it to a running per-class calibration set — exactly the data `calibrate_cls_head.py` currently collects manually, but gathered automatically from real operational footage instead of a prepared dataset.
3. After a set duration or sample count (e.g., "500 samples per class" or "4 hours of operation," whichever comes first), it fits `mean`/`inv_cov` (as before) and computes a threshold (via Option 1 or 3) — then switches into active monitoring mode.

### Why this matters for your problem
This directly removes the "client has to run a manual script" friction — from the client's point of view, they just install the system and it becomes fully operational a few hours later, with **zero data science expertise required on their end**. It also naturally adapts to that specific client's actual cameras and environment, sidestepping the domain-mismatch caveat from Option 5.

### The trade-off to manage
This assumes the bootstrap window itself is attack-free (a reasonable assumption for a fresh installation, but not something to leave unstated) — if an adversarial patch happened to be present during the learning window, it could get baked into the "normal" reference distribution. A simple mitigation: require operator confirmation to start the learning window ("confirm the site is clear, then begin calibration"), or pair this with Option 7's confidence-gating for extra robustness.

---

## 8. Option 7 — Continual / Online Recalibration with Confidence Gating

### The idea
Treat calibration as an ongoing process, not a one-time event (whether that one time was manual, shipped-default, or bootstrap). Keep the reference statistics updated incrementally as the system runs, so it naturally tracks gradual domain drift (seasons changing, a camera being repositioned, etc.) without needing a human to notice staleness and re-trigger calibration.

### The math
Recomputing `mean` and `Σ` from scratch on every new sample would be expensive. Instead, use an **online (streaming) update rule**. For the mean, this is straightforward incremental averaging (Welford's algorithm, which also extends to variance/covariance):

```
n ← n + 1
δ ← x - mean
mean ← mean + δ/n
```

For a version that adapts to drift rather than treating all history equally, use an **exponential moving average** instead, with a decay rate `β` controlling how much weight recent data gets relative to history:

```
mean ← β · mean + (1 - β) · x
```

### The critical safeguard: confidence gating
The obvious risk with any online update is **self-poisoning**: if an actual attack occurs while the system is updating its own baseline, the attack's activations could get folded into "normal," degrading detection over time. Mitigate this by only feeding samples into the update **when there's independent evidence they're genuinely clean** — for example:
- Only update using detections that are **temporally consistent**: the same object, tracked with a stable class and high confidence across several consecutive frames (a real object behaves this way; a single-frame anomaly is more suspicious and should be excluded from the update).
- Skip the update entirely for any frame the detector's own anomaly score already flagged as suspicious (even below the current threshold) — a simple "don't learn from what you're unsure about" rule.

### Why this matters
This is what turns a one-time calibration (however it was originally obtained — manual, shipped, or bootstrapped) into something that stays accurate over the system's operational lifetime, addressing the maintenance-burden part of the original problem, not just the initial setup friction.

---

## 9. Option 8 — Structurally Avoid Calibration: Lean on the Self-Referential Detector (Mechanism 4)

### The idea
All seven options above are ways to make calibration **automatic**. This option asks a different question: can part of your detection pipeline **avoid needing calibration data at all**? You've already designed exactly this: Mechanism 4 (APE-style feature energy) computes its own statistics *from the current image alone*, comparing each spatial location's feature energy against the mean/standard-deviation of energy values within that same image:

```
energy(y, x) = Σ_c  feature_map[c, y, x]²
threshold    = mean(energy_map) + k · std(energy_map)
```

Both the "reference distribution" (`mean(energy_map)`, `std(energy_map)`) and the comparison happen **per image, at inference time** — there's no offline dataset to collect, no calibration artifact to fit or ship, and no domain-mismatch risk from training on one client's data and deploying on another's.

### Why this directly answers the stated constraint
If "we can't ask clients to run calibration" is treated as a hard requirement rather than something to work around, this is the most direct answer: a detection strategy that was never going to need calibration in the first place.

### The trade-off
As discussed when this mechanism was first designed, self-referential, per-image thresholding tends to be more sensitive to natural high-texture false positives (foliage, gravel, cluttered scenes) than a properly-calibrated Mahalanobis approach, precisely because it has no external reference for what a wide variety of "normal" scenes look like — it can only reason about the one image in front of it.

### Combine it, don't just pick it
The strongest design isn't "Mechanism 4 instead of Mechanism 1" — it's an **ensemble**: run Mechanism 4 as the always-on, zero-setup default from the moment the system is installed, and layer in Mechanism 1 (with a threshold from Option 1, 3, or 4 above) as a higher-precision addition once calibration data becomes available — whether from a shipped default (Option 5) or an automatic bootstrap (Option 6). This gives every client working protection on day one, with detection quality improving over time as calibration data accumulates, rather than an all-or-nothing choice between "no protection until calibration is done" and "no calibration, ever."

---

## 10. Tying This Back to Your Architecture

Your existing Strategy pattern already treats detection algorithms as interchangeable (`ClsHeadMahalanobisDetector`, and the planned backbone-tap and feature-energy variants) behind a shared `IAnomalyDetector` interface. The threshold, right now, is *not* similarly decoupled — it's a bare config value with no abstraction around how it's obtained.

A clean extension: introduce an `IThresholdStrategy` interface, with concrete implementations corresponding to the options above —

```python
class IThresholdStrategy(ABC):
    @abstractmethod
    def get_threshold(self, context) -> float:
        ...

class FixedThreshold(IThresholdStrategy): ...        # what you have today — kept as a baseline/fallback
class ChiSquaredThreshold(IThresholdStrategy): ...   # Option 1
class PercentileThreshold(IThresholdStrategy): ...   # Option 2
class EVTThreshold(IThresholdStrategy): ...          # Option 3
class ConformalThreshold(IThresholdStrategy): ...    # Option 4
```

Each concrete `IAnomalyDetector` would take an `IThresholdStrategy` rather than a raw float, and any of the calibration-sourcing options (default profile, bootstrap, online update) become ways of *supplying data* to whichever threshold strategy is active, rather than separate, disconnected concerns.

This isn't just a convenient refactor — it's a direct extension of your thesis's central argument. The research is about decoupling adversarial defense from hardcoded model-specific details; a hardcoded, manually-tuned threshold is the same kind of brittleness the whole architecture was built to eliminate, just showing up one layer over. Framing the fix this way gives you a clean, citable answer to the practical deployment question this document opened with: *"we didn't just decouple the detector from the model architecture — we decoupled the threshold-calibration strategy from the detector itself."*
