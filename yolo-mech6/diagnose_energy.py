"""
Can the energy signal tell patch cells from normal cells, and does per-class profiling help?

Same protocol as yolo-mech7/diagnose_separability.py: every grid cell of a patched image is
labelled patch (centre inside a ground-truth patch box) or normal (centre more than --halo px from
every patch box). Scores compared per tapped layer:

    E            raw energy, one global threshold (= Mech-4)
    nearest_abs  Mech-6: distance (in std units of log-energy) to the nearest class/background profile
    nearest_up   Mech-6, one-sided: same, but only counts energy ABOVE the nearest profile

Reported: AUROC (1.0 perfect, 0.5 coin flip) and the share of patch cells caught at 0.5% / 1% false
alarms, against the normal cells of the patched images and against the cells of clean images.

It also aligns all layers to the 80x80 grid (the finer layers are max-pooled 2x2) and merges them,
after turning each layer's score into its percentile among clean cells (so layers are comparable):
    any   = max over layers (Mech-4's "any layer flags it")
    all   = min over layers (every layer must flag it)
    mean  = mean over layers
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from mech6_energy import LAYER_STRIDE, EnergyProfiles

BOX_RE = re.compile(r"box=\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)")
NAME_PREFIX_RE = re.compile(r"^(\d+-)?(multi-)?patched[_-]")
VARIANTS = ("E", "nearest_abs", "nearest_up")


def gt_boxes(gt_dir, stem):
    for key in (stem, NAME_PREFIX_RE.sub("", stem)):
        p = Path(gt_dir) / f"{key}.txt"
        if p.exists():
            return [tuple(map(int, m.groups())) for m in BOX_RE.finditer(p.read_text())]
    return []


def cell_masks(H, W, stride, boxes, halo):
    cx, cy = (np.arange(W) + 0.5) * stride, (np.arange(H) + 0.5) * stride
    X, Y = np.meshgrid(cx, cy)
    inside, near = np.zeros((H, W), bool), np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        inside |= (X >= x1) & (X < x2) & (Y >= y1) & (Y < y2)
        near |= (X >= x1 - halo) & (X < x2 + halo) & (Y >= y1 - halo) & (Y < y2 + halo)
    return inside, ~near


def auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    sv = allv[order]
    _, start, counts = np.unique(sv, return_index=True, return_counts=True)
    ranks = np.empty(allv.size)
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def summarize(pos, neg, clean):
    out = {"auroc": auroc(pos, neg), "auroc_clean": auroc(pos, clean)}
    for f in (0.005, 0.01):
        out[f"tpr{f * 100:g}"] = float((pos > np.quantile(neg, 1 - f)).mean())
        out[f"tpr{f * 100:g}_clean"] = float((pos > np.quantile(clean, 1 - f)).mean())
    return out


def pool2(a):
    h, w = a.shape
    return a.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def to_rank(a, ref_sorted):
    """Percentile of each value among the clean reference cells (0..1). Values above the largest
    clean cell continue linearly above 1, so strong patch cells do not all tie at 1.0."""
    r = np.searchsorted(ref_sorted, a, side="right") / ref_sorted.size
    top, mid = ref_sorted[-1], ref_sorted[ref_sorted.size // 2]
    return np.where(a > top, 1.0 + (a - top) / (top - mid + 1e-12), r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--energy-dir", required=True, help="folder with patched/ and clean/ energy maps")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--halo", type=int, default=48)
    args = ap.parse_args()

    with open(args.profiles, "rb") as f:
        prof = EnergyProfiles(pickle.load(f))
    layers = sorted(prof.stats)
    root = Path(args.energy_dir)

    def load(folder):
        return {p.stem: {l: np.load(p)[f"e{l}"].astype(np.float64) for l in layers} for p in sorted((root / folder).glob("*.npz"))}

    patched, clean = load("patched"), load("clean")
    gts = {s: gt_boxes(args.gt_dir, s) for s in patched}
    patched = {s: v for s, v in patched.items() if gts[s]}
    n_patches = sum(len(gts[s]) for s in patched)
    print(f"{len(patched)} patched images ({n_patches} patches), {len(clean)} clean images, layers {layers}")

    # ---- per-layer scores, each layer on its own grid
    pos, neg, cl = {}, {}, {}
    for s, maps in patched.items():
        for l in layers:
            sc = prof.scores(l, maps[l])
            inside, far = cell_masks(*maps[l].shape, LAYER_STRIDE[l], gts[s], args.halo)
            for v in VARIANTS:
                pos.setdefault((l, v), []).append(sc[v][inside])
                neg.setdefault((l, v), []).append(sc[v][far])
    for s, maps in clean.items():
        for l in layers:
            sc = prof.scores(l, maps[l])
            for v in VARIANTS:
                cl.setdefault((l, v), []).append(sc[v].ravel())

    print(f"\n{'layer':<7}{'score':<13}{'AUROC':>8}{'TPR@0.5%':>10}{'TPR@1%':>9}   vs clean-image cells: {'AUROC':>7}{'TPR@0.5%':>10}{'TPR@1%':>9}")
    for l in layers:
        for v in VARIANTS:
            r = summarize(np.concatenate(pos[(l, v)]), np.concatenate(neg[(l, v)]), np.concatenate(cl[(l, v)]))
            print(f"L{l} s{LAYER_STRIDE[l]:<3}{v:<13}{r['auroc']:>8.3f}{r['tpr0.5']:>10.3f}{r['tpr1']:>9.3f}"
                  f"   {'':>24}{r['auroc_clean']:>7.3f}{r['tpr0.5_clean']:>10.3f}{r['tpr1_clean']:>9.3f}")

    # ---- all layers aligned to 80x80, merged after percentile-ranking against clean cells
    def aligned(maps, v):
        out = {}
        for l in layers:
            a = prof.scores(l, maps[l])[v]
            out[l] = pool2(a) if a.shape[0] == 160 else a
        return out

    print(f"\nAll layers on the 80x80 grid (finer layers max-pooled 2x2), merged on clean-cell percentiles:")
    print(f"{'score':<13}{'merge':<7}{'AUROC':>8}{'TPR@0.5%':>10}{'TPR@1%':>9}   vs clean-image cells: {'AUROC':>7}{'TPR@0.5%':>10}{'TPR@1%':>9}")
    for v in VARIANTS:
        ref = {l: np.sort(np.concatenate([aligned(m, v)[l].ravel() for m in clean.values()])) for l in layers}
        P, N, C = {k: [] for k in ("any", "all", "mean")}, {k: [] for k in ("any", "all", "mean")}, {k: [] for k in ("any", "all", "mean")}

        def merged(maps):
            a = aligned(maps, v)
            r = np.stack([to_rank(a[l], ref[l]) for l in layers])
            return {"any": r.max(0), "all": r.min(0), "mean": r.mean(0)}
        for s, maps in patched.items():
            m = merged(maps)
            inside, far = cell_masks(80, 80, 8, gts[s], args.halo)
            for k in m:
                P[k].append(m[k][inside])
                N[k].append(m[k][far])
        for maps in clean.values():
            m = merged(maps)
            for k in m:
                C[k].append(m[k].ravel())
        for k in ("any", "all", "mean"):
            r = summarize(np.concatenate(P[k]), np.concatenate(N[k]), np.concatenate(C[k]))
            print(f"{v:<13}{k:<7}{r['auroc']:>8.3f}{r['tpr0.5']:>10.3f}{r['tpr1']:>9.3f}"
                  f"   {'':>24}{r['auroc_clean']:>7.3f}{r['tpr0.5_clean']:>10.3f}{r['tpr1_clean']:>9.3f}")


if __name__ == "__main__":
    main()
