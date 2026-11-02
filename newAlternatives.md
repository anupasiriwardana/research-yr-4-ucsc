# New Alternative Mechanisms: Mech-5, Mech-6, Mech-7

## Status
Draft proposals, not yet implemented. Each extends an existing, already-built mechanism rather than starting from scratch. Written to capture the design discussion before any of these are built, so the reasoning (and the risks already identified) isn't lost. Companion to `mech-1-optionB.md`, `mech-4-feature-energy.md`, and `box-dependency-problem-and-solutions.md`.

---

## Why These Exist

Option B (Mechanism 1) works well on YOLOv8 but shows a high false-positive rate on YOLO11, even after recalibrating and retuning the threshold for that model specifically. Raising the threshold suppresses false positives but starts missing real patches (sometimes only partially); lowering it catches patches but false-positives clean images. No single value satisfies both — a strong sign that **one shared global threshold is the wrong tool**, not that the underlying detection signal is broken. All three mechanisms below are responses to that diagnosis, from different angles.

Before any of these: confirm the YOLO11 threshold was genuinely re-derived from a YOLO11-specific clean-image sweep, not inherited from YOLOv8n's calibration. This has already been done for the current investigation, which is what makes the single-global-threshold diagnosis credible rather than just a guess.

---

## Mech-5: Multi-Scale (P3/P4/P5) Classification-Branch Profiling + Learned Threshold

### Problem
Option B taps only the P3 scale of the classification branch. Objects route to different scales by size at training time (small objects → P3, large objects → P5), so a single scale only ever sees part of the picture. Combined with the single-global-threshold problem above, this may be leaving signal on the table that a different scale (or scale combination) would catch more reliably.

### Approach
Extend the existing adapter — which already accepts a list of scales — to tap P3, P4, and P5 simultaneously, and build a full set of per-class (+ background) Mahalanobis profiles **per scale**, not just at P3. At detection time, each scale produces its own per-cell score (minimum distance across classes, as Option B already does); these are then combined across scales.

**Important framing correction:** P3/P4/P5 are not three *depths* of the same signal the way Mechanism 4's backbone layers 1/2/3 are. They're parallel branches, each specialized for a different object-size regime. Whether "P3 and P4 agreeing" is a meaningful signal (the way "layer 1 and layer 2 agreeing" was explored for Mechanism 4) is an open empirical question, not an assumption to carry over — validate it rather than assume it transfers.

### Configuration additions

```json
"mech5_settings": {
    "tap_scales": ["P3", "P4"],
    "stride_per_scale": {"P3": 8, "P4": 16, "P5": 32},
    "min_layer_agreement": 1,
    "adjust_threshold": 0.0,
    "profiles_dir": "...",
    "calibration_profile_filename": "mech5_calibration.pkl"
}
```

`min_layer_agreement`: how many of the tapped scales must independently flag a cell before it counts. Included explicitly as a **testing-only knob** — Mechanism 4's own experiment with this exact pattern only ever worked at the trivial value of 1 (equivalent to "any scale, no agreement required"), so go in with that precedent in mind rather than assuming a different outcome here. Keep it in the config for now since the P3/P4/P5 relationship is structurally different from backbone depth and might behave differently; be ready to remove it if it doesn't.

### Learned threshold (shared design across Mech-5, Mech-6, Mech-7 — see dedicated section below)

### Code sketch

```python
class MultiScaleClsHeadAdapter:
    def __init__(self, model, scales=("P3", "P4")):
        self.scales = list(scales)
        # reuses the existing per-scale hook registration already
        # built into UltralyticsClsHeadAdapter -- no new hook logic needed

class Mech5Detector:
    def __init__(self, scale_profiles, stride_per_scale, thresholds_per_scale,
                 min_layer_agreement=1):
        self.scale_profiles = scale_profiles      # {scale: {class_id: {mean, inv_cov}}}
        self.stride_per_scale = stride_per_scale
        self.thresholds_per_scale = thresholds_per_scale  # {scale: {class_id: threshold}}
        self.min_layer_agreement = min_layer_agreement

    def score_from_activations(self, activations):
        # activations: {"P3": Tensor, "P4": Tensor, ...}
        per_scale_masks = {}
        for scale, feat in activations.items():
            scores = min_mahalanobis_scores(feat, self.scale_profiles[scale])  # reuses Option B logic
            threshold_map = per_class_threshold_lookup(scores, self.thresholds_per_scale[scale])
            per_scale_masks[scale] = (scores > threshold_map)

        # upsample all masks to a common (finest) resolution, then sum votes --
        # same spatial-unpooling pattern already built for Mechanism 4
        agreement_map = aggregate_across_scales(per_scale_masks, self.stride_per_scale)
        final_mask = agreement_map >= self.min_layer_agreement
        return extract_regions(final_mask)  # same connected-component + per-region score pattern as Option B / Mechanism 4
```

### Practicality
Moderate effort — the adapter extension is nearly free (already built for multi-scale), most of the work is in calibration and the cross-scale aggregation step. Main risk: per-(class, scale) sample sparsity. P4/P5 naturally see far fewer samples than P3 in typical driving footage (fewer large/close objects than small/distant ones), so those profiles may be statistically weaker — apply the same sparse-class fallback described under Mech-7.

---

## Mech-6: Per-Class Feature-Energy Profiling for Mechanism 4 + Learned Threshold

### Problem
Mechanism 4's energy threshold is currently a single global value per layer, fit from the pooled energy distribution across all clean images regardless of content. This is exactly why a plain white car (high natural contrast) and an actual adversarial patch can look statistically similar — the threshold has no notion of "what object is normally here," only "what energy is normal anywhere."

### Approach
Build per-class (+ background) energy profiles **per tapped backbone layer** — the same per-class idea already used in Option B, applied to Mechanism 4's scalar energy value instead of a full feature vector. At runtime, score each cell by comparing its energy against the *nearest-fitting* class profile for that layer (minimum normalized distance, never gated by the detector's own current prediction), then compare that against the class-specific threshold.

### Critical design requirement — do not repeat the Option B box-dependency bug

Do **not** decide which class profile to compare a cell against using the detector's current box output. That reintroduces precisely the vulnerability Option B was built to close: if a hiding attack succeeds, there's no box, so a box-gated design would fall back to comparing the corrupted cell only against the (typically loosely-calibrated, high-variance) background profile — which the attack is much more likely to slip under than it would the correct object class's tighter profile. Score every cell against **every** profile and take the minimum, exactly as Option B already does.

### Where class labels for calibration actually come from

This only matters offline, during calibration, and it's a solved problem already — identical to how `calibrate_cls_head.py` projects final detection boxes down onto the P3 grid via the stride ratio. For Mechanism 6, project those same boxes down onto whichever backbone layer is being calibrated, using that layer's own stride. This is a one-time preparation step against a clean dataset; it has no bearing on how detection works at runtime.

### On timing — does detection have to wait for the full forward pass?

No. Once calibration has produced the per-class energy profiles, runtime scoring needs nothing but the tapped layer's activation tensor — no predicted class, no box, no information from later in the network. Because backbone layers compute *before* the detection head, the hook for Mechanism 6 actually fires **earlier** in the forward pass than Option B's classification-branch hook does. In principle this makes Mechanism 6 the most naturally asynchronous of all four mechanisms — scoring could begin the instant the hook fires, in parallel with the rest of the forward pass, rather than waiting for the full `model(image)` call to return. The current single-threaded implementation doesn't yet exploit this, but it's available as a future performance optimization, not a redesign.

### Configuration additions

```json
"mech6_settings": {
    "tap_layers": [1, 2, 3],
    "stride_per_layer": {"1": 4, "2": 4, "3": 8},
    "adjust_threshold": 0.0,
    "profiles_dir": "...",
    "calibration_profile_filename": "mech6_calibration.pkl"
}
```

### Code sketch

```python
def calibrate_mech6(clean_images, model, adapter, tap_layers):
    """Offline only. Projects final detection boxes down onto each
    tapped backbone layer's grid (via that layer's stride) to assign
    class labels for energy-profile fitting -- same projection logic
    already used for Option B's calibration."""
    per_layer_class_energies = {layer: defaultdict(list) for layer in tap_layers}
    for image in clean_images:
        results = model.predict(image)                      # full forward pass, OFFLINE ONLY
        activations = adapter.get_activations(image)
        for layer in tap_layers:
            energy_map = compute_energy(activations[layer])  # squared L1 norm, as in Mechanism 4
            for box, cls_id in results.boxes:
                gx, gy = project_box_to_grid(box, stride_per_layer[layer])
                per_layer_class_energies[layer][cls_id].append(energy_map[gy, gx])
            # background cells, same random-subsample approach as Mechanism 4's calibration
    return fit_profiles(per_layer_class_energies)  # {layer: {class_id: {mean, std}}}


def score_mech6(activations, profiles, thresholds):
    """Runtime. No class lookup, no box dependency -- pure tensor math
    on whatever the hook already captured."""
    per_layer_masks = {}
    for layer, feat in activations.items():
        energy_map = compute_energy(feat)
        best_class_distance = min_normalized_distance(energy_map, profiles[layer])  # per-cell
        threshold_map = per_class_threshold_lookup(best_class_distance, thresholds[layer])
        per_layer_masks[layer] = (best_class_distance > threshold_map)
    return aggregate_across_layers(per_layer_masks)  # same spatial-unpooling + summation already built
```

### Practicality
Cheaper to calibrate than Mech-5 — energy is a single scalar per cell, so each profile is a 1D distribution (mean/std) rather than a full covariance matrix, needing far fewer samples to be stable. Directly targets a known, already-documented weakness of Mechanism 4. Still fundamentally bounded by the fact that this is a low-level, texture-driven signal — within-class variance (chrome trim, strong shadows on an otherwise-ordinary car) can still look anomalous even relative to a class-specific profile. Treat this as a meaningful refinement of Mechanism 4, not a claim that it now matches Option B's precision.

---

## Mech-7: Per-Class Thresholding (Applied to Option B)

### Problem
Same root cause as the others: a single global threshold cannot simultaneously suit low-variance classes (e.g. a stop sign, which barely varies in appearance) and high-variance classes (e.g. "car," which spans an enormous visual range). Tightening it for one makes it wrong for the other.

### Core idea
Option B already computes, for every cell, the distance to *every* calibrated class profile (plus background) and takes the minimum — that minimum already identifies which profile is the best statistical explanation for that cell, entirely from the feature vector itself, with no reference to what the detector currently predicts. Mech-7's entire change is: compare that minimum distance against **that specific profile's own threshold**, instead of one shared global number.

```
for each cell:
    best_class, best_distance = argmin over all profiles c of distance(cell, profile_c)
    is_anomalous = best_distance > threshold[best_class]
```

This preserves the box-independence property Option B already has — nothing here ever consults the detector's current box or predicted class, only which calibrated profile the feature vector actually resembles most. No added computational cost: the per-class distances are already being computed to find the minimum; this only changes what happens with that minimum afterward.

### Variant A — Strict: must fail every profile

Instead of only checking the single nearest profile's threshold, require the cell to exceed **every** profile's threshold (every object class and background) before it counts as anomalous. More conservative — fewer false positives, at the cost of potentially missing an attack that happens to pass some unrelated, loosely-thresholded class's bar it was never actually close to. Treat this as a fallback to reach for only if the base nearest-profile approach still runs too hot on false positives — not a starting point, since it's solving a problem you may not actually have.

### Variant B — Continuous normalized score

Instead of a binary per-class pass/fail, normalize each class's distance into a unitless score using that class's *own* calibration-time distance distribution:

```
normalized_score(cell) = (distance_to_nearest_class - mean_distance_for_that_class) / std_distance_for_that_class
```

This makes different classes comparable on one shared scale (a z-score, essentially), so a single threshold can be applied to the normalized value instead of needing a different raw threshold per class. Functionally the same underlying idea as the base approach, just expressed continuously — useful if you want a graded severity score per region (similar in spirit to Mechanism 4's "how many layers agree" score) rather than a flat yes/no.

### Addition: configurable tap scale, not hardcoded to P3

Currently the stride and tap point are fixed to P3/stride-8 in code. Make both config-driven, so switching to P4 (or P5) is a one-line config change, not a code edit:

```json
"detector_settings": {
    "tap_scale": "P3",
    "stride": 8,
    "threshold_mode": "per_class",
    "adjust_threshold": 0.0,
    "min_class_samples_for_own_threshold": 30
}
```

```python
# cls_head_detector.py / calibrate_cls_head.py
TAP_SCALE = config["detector_settings"].get("tap_scale", "P3")
STRIDE = config["detector_settings"]["stride"]   # must be updated to match tap_scale (P3=8, P4=16, P5=32)

adapter = UltralyticsClsHeadAdapter(model.model, scales=(TAP_SCALE,))
...
feat = activations[TAP_SCALE]   # was hardcoded as activations["P3"]
```

Worth a one-line validation check at startup (`assert STRIDE matches the known stride for TAP_SCALE`) so a config typo (e.g. `tap_scale: "P4"` with `stride: 8` left unchanged) fails loudly at load time instead of silently producing a garbled grid mapping — exactly the kind of coordinate-space bug this project has hit before.

### The `adjust_threshold` knob

Applies to all three mechanisms above, not just Mech-7. After the calibration phase learns a threshold (per class, per scale, or per layer, depending on the mechanism), `adjust_threshold` is added on top of every learned value before it's used:

```python
final_threshold = learned_threshold + config["detector_settings"].get("adjust_threshold", 0.0)
```

A positive value makes detection stricter across the board (fewer false positives, higher risk of missed patches); negative makes it more sensitive. This gives you a single, global "nudge" knob for quick experimentation without having to re-run calibration every time you want to try a slightly different operating point.

### Sample-size fallback (applies to all three mechanisms)

Some classes will always have far fewer calibration samples than others (fewer buses than cars in typical driving footage). A threshold fit on too few samples is noisy and not trustworthy. Use `min_class_samples_for_own_threshold`: below that count, fall back to the pooled/global threshold for that class; only switch to the class-specific threshold once there's enough data to trust it.

### Practicality
Cheapest of the three to implement — no new tap points, no new adapter work, the profiles already exist in Option B today. Main cost is the calibration-side threshold-learning pass (re-running the clean-image sweep, segmented by class instead of pooled) and the sample-size fallback bookkeeping. Given it directly targets the observed symptom (one global threshold can't satisfy both low- and high-variance classes) and is the smallest change to make, this is the natural first build among the three.

---

## Suggested Build Order

1. **Mech-7 first.** Smallest change, directly targets the diagnosed problem (single global threshold), reuses the existing Option B calibration pipeline almost entirely as-is.
2. **Mech-5 next.** Builds cleanly on top of Mech-7's per-class threshold logic; the multi-scale adapter work is nearly free since the adapter already supports a scale list.
3. **Mech-6 as its own track.** Answers a related but distinct question — "can Mechanism 4 be made precise enough to matter on its own" — rather than directly explaining the YOLO11 false-positive problem on Option B. Valuable, but lower priority against the immediate symptom being investigated.

---

## Open Question Carried Over From This Discussion

Before building any of the above: confirm whether the YOLO11 false-positive problem survives a from-scratch, YOLO11-specific threshold re-derivation using the existing clean-image statistical sweep. This has already been done once; if the false-positive/missed-patch tension persists after that, it's good evidence the problem is genuinely the single-global-threshold limitation these three mechanisms are designed to fix, rather than a stale or poorly-swept threshold value.
