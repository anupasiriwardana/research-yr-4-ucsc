"""
Offline multi-scale combination + box-building sweep for Mech-7 (Mech-5 idea).

Reads the per-image score maps (`.npz`) that detect_mech7_multi_img.py writes
with save_score_maps on -- one folder per tap scale, same images in each --
and never touches the model. For every cell it normalises each scale's
distance by its own threshold:

    r_s = raw_s / (threshold_s[nearest class] + adjust)        (r > 1 = flagged)

then merges the scales (coarser grids are upsampled to the finest grid, e.g. a
P4 cell is repeated 2x2 on the P3 grid) with one of these RULES:

    P3 / P4 / P5 : that scale alone
    and          : min over scales   (every scale must flag the cell)
    or           : max over scales   (any scale flags it)
    mean         : mean over scales
    vote2        : the 2nd-largest ratio, i.e. at least TWO scales must flag the
                   cell (only with 3+ scales)
    confirm      : finest scale, kept only where the coarser scale(s) flag a
                   cell within one cell (3x3 neighbourhood) of it

A cell is flagged when score > 1 + OFFSET. Flagged cells become boxes the same
way the detector does (8-connected regions, min size), with an optional JOIN
step: flagged cells are first dilated JOIN times so scattered patch cells
merge into one region; the box and the size check still use only the original
flagged cells.

Each (rule, offset, join, min_cells) is scored with the same definition as
calc-metrics/compare_patch_detection_metrics.py (positive-area overlap,
one-to-one, highest IoU first) on the patched images, and as "% of images with
any detection" on the clean images. The report picks, per rule, the config
with the best object F1 whose clean-image false-positive rate is <= the cap.

Example:
    python combine_scales.py ^
      --patched P3=score-maps/yolov11/p3-patched P4=score-maps/yolov11/p4-patched ^
      --clean   P3=score-maps/yolov11/p3-clean   P4=score-maps/yolov11/p4-clean
"""

import argparse
import csv
import itertools
import pickle
from pathlib import Path

import cv2
import numpy as np

from mech7_core import load_config
from score_heatmap import find_gt_boxes

KERNEL = np.ones((3, 3), np.uint8)


# ---------- loading ----------

def parse_kv(items, what):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{what}: expected SCALE=PATH, got '{it}'")
        k, v = it.split("=", 1)
        out[k.upper()] = v
    return out


def load_maps(directory):
    maps = {}
    for p in sorted(Path(directory).glob("*.npz")):
        with np.load(p) as d:
            maps[p.stem] = (d["raw"].astype(np.float32), d["best_class"].astype(np.int32), int(d["stride"]))
    if not maps:
        raise SystemExit(f"No .npz score maps in {directory}")
    return maps


def threshold_lut(pkl_path, mode, global_threshold):
    """-> function(best_class ids) -> per-cell threshold (before adjust)."""
    if mode == "global":
        return lambda bc: np.full(bc.shape, global_threshold, np.float32)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pooled = data["pooled"]["threshold"]
    ids = sorted(data["per_class"])
    lut = np.full(max(ids) + 3, pooled, np.float32)          # index = class id + 1
    for cid, e in data["per_class"].items():
        lut[cid + 1] = e["threshold"]
    return lambda bc: lut[np.clip(bc + 1, 0, len(lut) - 1)]


# ---------- scores ----------

def ratio_map(entry, thr_fn, adjust):
    raw, bc, stride = entry
    return raw / np.maximum(thr_fn(bc) + adjust, 1e-6), stride


def combine(rule, ratios, scales):
    """ratios: {scale: (H, W) on the finest grid} -> score (H, W)."""
    stack = np.stack([ratios[s] for s in scales])
    if rule in ratios:
        return ratios[rule]
    if rule == "and":
        return stack.min(0)
    if rule == "or":
        return stack.max(0)
    if rule == "mean":
        return stack.mean(0)
    if rule == "vote2":                       # at least two scales flag the cell (needs >= 3 scales)
        return np.sort(stack, axis=0)[-2]
    if rule == "confirm":
        fine, coarse = ratios[scales[0]], [cv2.dilate(ratios[s], KERNEL) for s in scales[1:]]
        return np.minimum(fine, np.min(coarse, axis=0))
    raise ValueError(rule)


def scores_for(maps_by_scale, thr_fns, adjust, rule, scales):
    """-> {stem: (score map on finest grid, finest stride)} for one rule."""
    out = {}
    for stem in maps_by_scale[scales[0]]:
        per, strides = {}, {}
        for s in scales:
            per[s], strides[s] = ratio_map(maps_by_scale[s][stem], thr_fns[s], adjust)
        fine = min(strides.values())
        for s in scales:
            f = strides[s] // fine
            if f > 1:
                per[s] = np.repeat(np.repeat(per[s], f, axis=0), f, axis=1)
        shapes = {a.shape for a in per.values()}
        if len(shapes) != 1:
            raise SystemExit(f"{stem}: scale grids do not line up after upsampling: {shapes}")
        out[stem] = (combine(rule, per, scales), fine)
    return out


# ---------- boxes ----------

def regions(score, stride, offset, join):
    """-> list of (cells, (x1, y1, x2, y2)) for every connected region."""
    mask = score > 1.0 + offset
    if not mask.any():
        return []
    dil = cv2.dilate(mask.astype(np.uint8), KERNEL, iterations=join) if join else mask.astype(np.uint8)
    _, lab = cv2.connectedComponents(dil, connectivity=8)
    ys, xs = np.nonzero(mask)
    lb = lab[ys, xs]
    order = np.argsort(lb, kind="stable")
    ys, xs, lb = ys[order], xs[order], lb[order]
    _, starts = np.unique(lb, return_index=True)
    cells = np.diff(np.append(starts, lb.size))
    x0, x1 = np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts)
    y0, y1 = np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts)
    return [(int(c), (int(a * stride), int(b * stride), int((d + 1) * stride), int((e + 1) * stride)))
            for c, a, b, d, e in zip(cells, x0, y0, x1, y1)]


# ---------- metrics (same definition as compare_patch_detection_metrics.py) ----------

def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(gt, det):
    cand = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gt) for di, d in enumerate(det) if inter(g, d) > 0),
                  key=lambda t: (-t[0], t[1], t[2]))
    mg, md, ious = set(), set(), []
    for o, gi, di in cand:
        if gi in mg or di in md:
            continue
        mg.add(gi)
        md.add(di)
        ious.append(o)
    return ious


def per_image_stats(patched_regions, patched_stems, gt_boxes, clean_regions, clean_stems, min_cells):
    """-> P [n_patched, 5] = (tp, fp, fn, sum of matched IoUs, boxes); C [n_clean, 2] = (any box, boxes)."""
    P = np.zeros((len(patched_stems), 5))
    for i, stem in enumerate(patched_stems):
        det = [b for c, b in patched_regions[stem] if c >= min_cells]
        m = match(gt_boxes[stem], det)
        P[i] = (len(m), len(det) - len(m), len(gt_boxes[stem]) - len(m), sum(m), len(det))
    C = None
    if clean_regions is not None:
        C = np.zeros((len(clean_stems), 2))
        for i, stem in enumerate(clean_stems):
            n = sum(1 for c, _ in clean_regions[stem] if c >= min_cells)
            C[i] = (n > 0, n)
    return P, C


def metrics(Ps, Cs, n_p, n_c):
    """Ps [..., 5] / Cs [..., 2] are SUMS over a set of images -> dict of arrays (broadcasts over configs)."""
    tp, fp, fn, iou_sum, boxes = (Ps[..., k] for k in range(5))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    out = {"precision": precision, "recall": recall, "f1": f1,
           "mean_iou": np.divide(iou_sum, tp, out=np.zeros_like(tp), where=tp > 0),
           "boxes_per_patched_img": boxes / max(n_p, 1), "tp": tp, "fp": fp, "fn": fn}
    if Cs is not None:
        out["clean_fpr"] = Cs[..., 0] / max(n_c, 1)
        out["clean_boxes_per_img"] = Cs[..., 1] / max(n_c, 1)
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patched", nargs="+", required=True, metavar="SCALE=DIR",
                    help="score-map folders of the PATCHED images, one per scale (same images in each)")
    ap.add_argument("--clean", nargs="+", default=None, metavar="SCALE=DIR",
                    help="score-map folders of CLEAN images, one per scale (strongly recommended)")
    ap.add_argument("--thresholds", nargs="+", default=None, metavar="SCALE=PKL",
                    help="learned-threshold pickles; default: <profiles_dir>/mech7_thresholds_yolo11_<scale>.pkl")
    ap.add_argument("--mode", choices=["per_class", "global"], default="per_class")
    ap.add_argument("--global-threshold", type=float, default=17.0)
    ap.add_argument("--adjust", type=float, default=0.0, help="added to every threshold (distance units)")
    ap.add_argument("--gt-dir", default=r"D:\GitHub\yr-4-research\test-patch-det-multi-img\gen-patched-img\yolov11-patch_coordinates")
    ap.add_argument("--rules", nargs="+", default=None, help="default: every scale alone + and/or/mean/confirm")
    ap.add_argument("--offsets", nargs="+", type=float, default=[-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--joins", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--min-cells", nargs="+", type=int, default=[1, 2, 4, 6, 10])
    ap.add_argument("--max-clean-fpr", type=float, default=0.2,
                    help="per rule, pick the best-F1 config whose clean-image FPR is <= this")
    ap.add_argument("--holdout-splits", type=int, default=0,
                    help="N>0: repeat N random splits; pick the best config on the tuning part, score it on the held-out part")
    ap.add_argument("--train-frac", type=float, default=0.5, help="fraction of images used for tuning in a split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV of every config (default diagnostics/combine_scales.csv)")
    args = ap.parse_args()

    patched_dirs, clean_dirs = parse_kv(args.patched, "--patched"), parse_kv(args.clean, "--clean")
    scales = sorted(patched_dirs, key=lambda s: int(s[1:]))                      # P3, P4, P5 (fine -> coarse)
    if clean_dirs and set(clean_dirs) != set(scales):
        raise SystemExit("--patched and --clean must cover the same scales")
    cfg = load_config()
    profiles_dir = Path(cfg["profiles_dir"])
    thr_paths = parse_kv(args.thresholds, "--thresholds")
    thr_fns = {s: threshold_lut(thr_paths.get(s, profiles_dir / f"mech7_thresholds_yolo11_{s.lower()}.pkl"),
                                args.mode, args.global_threshold) for s in scales}

    patched = {s: load_maps(patched_dirs[s]) for s in scales}
    stems = sorted(set.intersection(*(set(m) for m in patched.values())))
    gt_boxes = {st: find_gt_boxes(args.gt_dir, st) for st in stems}
    stems = [st for st in stems if gt_boxes[st]]
    if not stems:
        raise SystemExit("No patched image with a matching GT file (check --gt-dir / file naming).")
    patched = {s: {st: patched[s][st] for st in stems} for s in scales}
    clean = None
    if clean_dirs:
        clean = {s: load_maps(clean_dirs[s]) for s in scales}
        cstems = sorted(set.intersection(*(set(m) for m in clean.values())))
        clean = {s: {st: clean[s][st] for st in cstems} for s in scales}
    print(f"scales {scales} | {len(stems)} patched images, "
          f"{len(next(iter(clean.values()))) if clean else 0} clean images | mode {args.mode}, adjust {args.adjust}")
    if clean is None:
        print("WARNING: no --clean maps, so false-positive rate is not measured and nothing is capped.")

    rules = args.rules or (scales + (["and", "or", "mean", "confirm"] if len(scales) > 1 else [])
                           + (["vote2"] if len(scales) > 2 else []))
    p_stems = stems
    c_stems = sorted(next(iter(clean.values()))) if clean else []
    cfgs, Plist, Clist = [], [], []
    for rule in rules:
        ps = scores_for(patched, thr_fns, args.adjust, rule, scales)
        cs = scores_for(clean, thr_fns, args.adjust, rule, scales) if clean else None
        for off, join in itertools.product(args.offsets, args.joins):
            preg = {st: regions(sc, strd, off, join) for st, (sc, strd) in ps.items()}
            creg = {st: regions(sc, strd, off, join) for st, (sc, strd) in cs.items()} if cs else None
            for mc in args.min_cells:
                P, C = per_image_stats(preg, p_stems, gt_boxes, creg, c_stems, mc)
                cfgs.append({"rule": rule, "offset": off, "join": join, "min_cells": mc})
                Plist.append(P)
                Clist.append(C)
        print(f"  rule {rule}: done")
    Pall = np.stack(Plist)                                    # [configs, patched images, 5]
    Call = np.stack(Clist) if clean else None                 # [configs, clean images, 2]
    n_p, n_c = len(p_stems), len(c_stems)
    full = metrics(Pall.sum(1), Call.sum(1) if clean else None, n_p, n_c)
    rows = [{**c, **{k: float(v[i]) for k, v in full.items()}} for i, c in enumerate(cfgs)]
    rule_idx = {r: np.array([i for i, c in enumerate(cfgs) if c["rule"] == r]) for r in rules}

    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "diagnostics" / "combine_scales.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pick(m, idx):
        """index (into cfgs) of the best-F1 config in idx with clean FPR <= cap; else the lowest-FPR one."""
        if clean:
            ok = idx[m["clean_fpr"][idx] <= args.max_clean_fpr]
            if not ok.size:
                return int(idx[np.argmin(m["clean_fpr"][idx])]), False
        else:
            ok = idx
        return int(ok[np.argmax(m["f1"][ok])]), True

    # report 1: best config per rule on ALL images (optimistic: tuned and scored on the same images)
    hdr = (f"{'rule':<8}{'offset':>7}{'join':>5}{'minc':>5}{'prec':>7}{'recall':>8}{'f1':>7}{'iou':>7}"
           f"{'box/img':>9}{'cleanFPR':>10}")

    def line(tag, r):
        fpr = f"{r['clean_fpr']:.2f}" if "clean_fpr" in r else "  n/a"
        return (f"{tag:<8}{r['offset']:>7.2f}{r['join']:>5}{r['min_cells']:>5}{r['precision']:>7.3f}"
                f"{r['recall']:>8.3f}{r['f1']:>7.3f}{r['mean_iou']:>7.3f}{r['boxes_per_patched_img']:>9.2f}{fpr:>10}")

    print(f"\nBest config per rule on ALL images (max object F1, clean-image FPR <= {args.max_clean_fpr:g}) "
          f"-- optimistic, tuned and scored on the same images:")
    print(hdr)
    for rule in rules:
        j, ok = pick(full, rule_idx[rule])
        print(line(rule, rows[j]) + ("" if ok else "   (cap not reachable: lowest-FPR config shown)"))
    ref_i = [i for i, c in enumerate(cfgs) if c["rule"] == scales[0] and c["offset"] == 0.0
             and c["join"] == 0 and c["min_cells"] == 10]
    if ref_i:
        print("\nReference (detector as it runs today: finest scale alone, offset 0, join 0, min_cells 10):")
        print(hdr)
        print(line(scales[0], rows[ref_i[0]]))

    # report 2: holdout. Tune on one part of the images, score the chosen config on the rest.
    if args.holdout_splits > 0:
        rng = np.random.default_rng(args.seed)
        keys = ("f1", "precision", "recall", "mean_iou") + (("clean_fpr",) if clean else ())
        res = {r: {k: [] for k in keys + ("tune_f1",)} for r in rules}
        chosen = {r: [] for r in rules}
        ref_res = {k: [] for k in keys}
        for _ in range(args.holdout_splits):
            pp, cc = rng.permutation(n_p), rng.permutation(n_c)
            pt, ph = pp[:int(n_p * args.train_frac)], pp[int(n_p * args.train_frac):]
            ct, ch = cc[:int(n_c * args.train_frac)], cc[int(n_c * args.train_frac):]
            tune = metrics(Pall[:, pt].sum(1), Call[:, ct].sum(1) if clean else None, len(pt), len(ct))
            for rule in rules:
                j, _ = pick(tune, rule_idx[rule])
                held = metrics(Pall[j, ph].sum(0), Call[j, ch].sum(0) if clean else None, len(ph), len(ch))
                chosen[rule].append((cfgs[j]["offset"], cfgs[j]["join"], cfgs[j]["min_cells"]))
                res[rule]["tune_f1"].append(float(tune["f1"][j]))
                for k in keys:
                    res[rule][k].append(float(held[k]))
            if ref_i:
                held = metrics(Pall[ref_i[0], ph].sum(0), Call[ref_i[0], ch].sum(0) if clean else None, len(ph), len(ch))
                for k in keys:
                    ref_res[k].append(float(held[k]))

        def ms(v):
            return f"{np.mean(v):.3f}+-{np.std(v):.3f}"

        print(f"\nHOLDOUT: {args.holdout_splits} random splits, tune on {args.train_frac:.0%} of the images "
              f"({int(n_p * args.train_frac)} patched / {int(n_c * args.train_frac)} clean), "
              f"score the chosen config on the rest. mean+-std over splits.")
        h2 = f"{'rule':<8}{'tuneF1':>8}{'held F1':>13}{'held prec':>13}{'held recall':>13}{'held IoU':>13}"
        print(h2 + (f"{'held cleanFPR':>15}" if clean else ""))
        for rule in rules:
            r = res[rule]
            print(f"{rule:<8}{np.mean(r['tune_f1']):>8.3f}{ms(r['f1']):>13}{ms(r['precision']):>13}"
                  f"{ms(r['recall']):>13}{ms(r['mean_iou']):>13}" + (f"{ms(r['clean_fpr']):>15}" if clean else ""))
        if ref_i:
            print(f"{'today':<8}{'':>8}{ms(ref_res['f1']):>13}{ms(ref_res['precision']):>13}"
                  f"{ms(ref_res['recall']):>13}{ms(ref_res['mean_iou']):>13}"
                  + (f"{ms(ref_res['clean_fpr']):>15}" if clean else "") + "   (fixed config, no tuning)")
        print("\nMost often chosen config per rule (offset, join, min_cells) and how many splits chose it:")
        for rule in rules:
            vals, counts = np.unique(np.array(chosen[rule]), axis=0, return_counts=True)
            k = int(np.argmax(counts))
            print(f"  {rule:<8}{tuple(vals[k].tolist())}  {int(counts[k])}/{args.holdout_splits}")
    print(f"\nAll configs: {out}")


if __name__ == "__main__":
    main()
