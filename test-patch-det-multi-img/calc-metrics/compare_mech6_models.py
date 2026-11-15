"""Patch-detection metrics for several models at once, per localizer panel, plus one bar chart per panel.

Made for the Mech-6 lean-v3 detector (two localizers: panel 2 = energy z, panel 3 = nearest profile), but works for
any detector whose box files look like ``box=(x1, y1, x2, y2) ...`` in 640x640 px.

Inputs (per model, from the JSON config):
    gt_dir                 GT patch boxes, one <key>.txt per patched image
    patched_detections_dir detections on the patched images: <dir>/<panel>/<prefix><key>.txt
    clean_detections_dir   detections on clean images (no patch): <dir>/<panel>/*.txt
    latency_json           output of yolo-mech6-detector-lean-v3/measure_latency.py
Optional config key `panel_titles` ({"panel2": "..."}) renames the panels in the charts and the printout.

Matching: two separate questions (several boxes may sit on one patch).
    1. Is a box correct?  Its intersection over DETECTION (IoD = share of the box's own area that lies inside the GT
       patches, all patches of the image together) must be >= `box_inside_min` (default 0.5). A small box fully
       inside a patch passes (IoD 1.0); a box that only clips a patch corner, or one huge box over half the image,
       fails (most of it is outside). Measured against all patches together, so one box drawn around two touching
       patches (they form one hot blob) is correct too. Correct boxes are never false positives, however many lie on
       the same patch.
    2. Was a patch found?  The union of the correct boxes that overlap it must cover >= `patch_coverage_min`
       (default 0.25) of the patch's area. So one 1-cell box inside a patch is not enough, but several fragments that
       together mark a quarter of it are, and one box around two touching patches finds both.
    Why not IoU: IoU compares a box with the whole patch, so a small box that is right on target gets a low IoU just
    for being small (a 48x48 box inside a 96x96 patch has IoU 0.25), while a box 2x too wide and tall also gets 0.25.
    For comparison the JSON also holds TPR at the `coverage_sweep` levels and the old rule of
    compare_patch_detection_metrics.py (any positive overlap, one box per patch).

Metrics per model and panel:
    TPR   (%)  = found GT patches / all GT patches                           (patched images, object level)
    FPR   (%)  = clean images with at least one box / clean images           (clean images, image level)
    FPPI       = false boxes / images, on patched + clean images together    (boxes not mostly inside a patch on
                 patched images + every box on clean images); patched-only and clean-only values are in the JSON
    Precision  = correct boxes / all boxes on patched images
    Latency    = mean per-image detection latency of that panel (ms), from measure_latency.py
    Also: F1, image-level recall, mean coverage of found patches, mean best-box IoU of found patches (box quality:
    low when patches are found as several fragments).

Usage:
    python compare_mech6_models.py --config compare_mech6_config.json
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
BOX_RE = re.compile(r"(?<![\w])box\s*=\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
PANEL_TITLES = {"panel2": "Panel 2 (energy z)", "panel3": "Panel 3 (nearest profile)"}
COLORS = {"tpr": "#F97316", "fpr": "#9A3412", "latency": "#FDBA74"}


def read_boxes(path):
    if path is None or not path.exists():
        return []
    boxes = []
    for m in BOX_RE.finditer(path.read_text(encoding="utf-8")):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / union if union > 0 else 0.0


def match(gt, det, min_iou):
    """One-to-one, highest IoU first. min_iou None = any positive overlap (the old rule)."""
    pairs = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gt) for di, d in enumerate(det)
                    if (inter(g, d) > 0 if min_iou is None else iou(g, d) >= min_iou)), reverse=True)
    used_g, used_d, out = set(), set(), []
    for v, gi, di in pairs:
        if gi not in used_g and di not in used_d:
            used_g.add(gi)
            used_d.add(di)
            out.append(v)
    return out


def coverage(patch, boxes):
    """Share of the patch's area covered by the union of the boxes (pixel mask over the patch)."""
    x1, y1, x2, y2 = (int(round(v)) for v in patch)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    m = np.zeros((y2 - y1, x2 - x1), bool)
    for b in boxes:
        bx1, by1 = max(int(round(b[0])), x1) - x1, max(int(round(b[1])), y1) - y1
        bx2, by2 = min(int(round(b[2])), x2) - x1, min(int(round(b[3])), y2) - y1
        if bx2 > bx1 and by2 > by1:
            m[by1:by2, bx1:bx2] = True
    return float(m.mean())


def inside_share(box, patches):
    """IoD against ALL patches together: share of the box's area that lies inside the union of the patches."""
    return coverage(box, patches)


def correct_boxes(gt, det, box_inside_min):
    """True for each box whose area lies >= box_inside_min inside the patches (together), else False (a false box).
    Measured against all patches together, so one box drawn around two touching patches is correct too."""
    return [bool(gt) and inside_share(d, gt) >= box_inside_min for d in det]


def ratio(a, b):
    return a / b if b else 0.0


def evaluate_panel(model, panel, prefix, box_inside_min, coverage_min, sweep):
    gt_dir = Path(model["gt_dir"])
    pat_dir = Path(model["patched_detections_dir"]) / panel
    cln_dir = Path(model["clean_detections_dir"]) / panel
    for d in (gt_dir, pat_dir, cln_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"{model['name']} {panel}: folder not found: {d}")
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}
    det_keys = {p.stem[len(prefix):] if p.stem.startswith(prefix) else p.stem for p in pat_dir.glob("*.txt")}
    missing = sorted(set(gt_files) - det_keys)
    extra = sorted(det_keys - set(gt_files))
    if missing or extra:
        raise ValueError(f"{model['name']} {panel}: GT / detection files do not match. "
                         f"No detection file for {missing[:5]}; no GT for {extra[:5]}")

    covs, best_ious = [], []                     # per GT patch: coverage by its correct boxes, best single-box IoU
    correct = false_patched = det_n = img_hit = 0
    old_tp = old_fp = 0
    for key, gpath in sorted(gt_files.items()):
        gt = read_boxes(gpath)
        det = read_boxes(pat_dir / f"{prefix}{key}.txt")
        ok = correct_boxes(gt, det, box_inside_min)
        good = [d for d, o in zip(det, ok) if o]
        det_n += len(det)
        img_hit += bool(det)
        correct += len(good)
        false_patched += len(det) - len(good)
        for g in gt:
            mine = [d for d in good if inter(g, d) > 0]
            covs.append(coverage(g, mine))
            best_ious.append(max((iou(g, d) for d in mine), default=0.0))
        m = match(gt, det, None)                  # the old rule, for comparison
        old_tp += len(m)
        old_fp += len(det) - len(m)
    gt_n = len(covs)
    found = [c >= coverage_min and c > 0 for c in covs]
    tp = sum(found)
    tpr = ratio(tp, gt_n)
    precision = ratio(correct, det_n)
    s = {"gt_patches": gt_n, "patches_found": tp, "patches_missed": gt_n - tp, "predicted_boxes": det_n,
         "correct_boxes": correct, "fp_patched": false_patched, "tpr": tpr, "precision": precision,
         "f1": ratio(2 * precision * tpr, precision + tpr), "image_recall": ratio(img_hit, len(gt_files)),
         "mean_coverage_found": ratio(sum(c for c, f in zip(covs, found) if f), tp),
         "mean_best_iou_found": ratio(sum(v for v, f in zip(best_ious, found) if f), tp)}
    old = {"rule": "any positive overlap, one box per patch (compare_patch_detection_metrics.py)",
           "tpr": ratio(old_tp, gt_n), "precision": ratio(old_tp, old_tp + old_fp), "fp_patched": old_fp}
    tpr_sweep = {str(c): ratio(sum(v >= c and v > 0 for v in covs), gt_n) for c in sweep}

    clean_files = sorted(cln_dir.glob("*.txt"))
    clean_boxes = [len(read_boxes(p)) for p in clean_files]
    n_pat, n_cln = len(gt_files), len(clean_files)
    out = {"model": model["name"], "panel": panel, "box_inside_min": box_inside_min, "patch_coverage_min": coverage_min,
           "patched_images": n_pat, "clean_images": n_cln,
           "tpr": s["tpr"], "precision": s["precision"], "f1": s["f1"], "image_recall": s["image_recall"],
           "mean_coverage_found": s["mean_coverage_found"], "mean_best_iou_found": s["mean_best_iou_found"],
           "fpr_clean_images": ratio(sum(b > 0 for b in clean_boxes), n_cln),
           "fppi": ratio(s["fp_patched"] + sum(clean_boxes), n_pat + n_cln),
           "fppi_patched": ratio(s["fp_patched"], n_pat), "fppi_clean": ratio(sum(clean_boxes), n_cln),
           "tpr_by_coverage": tpr_sweep, "counts": s, "old_rule": old}
    lat_path = model.get("latency_json")
    if lat_path and Path(lat_path).exists():
        lat = json.loads(Path(lat_path).read_text(encoding="utf-8"))
        out["latency_ms"] = lat[f"{panel}_latency"]["mean_ms"]
        out["latency_median_ms"] = lat[f"{panel}_latency"]["median_ms"]
        out["latency_device"] = lat.get("device")
    else:
        out["latency_ms"] = None
        print(f"  warning: no latency file for {model['name']} ({lat_path}); latency left out")
    return out


def label_bars(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)


def plot_panel(rows, panel, path, title=None):
    names = [r["model"] for r in rows]
    x = list(range(len(rows)))
    w = 0.25
    tpr = [100 * r["tpr"] for r in rows]
    fpr = [100 * r["fpr_clean_images"] for r in rows]
    lat = [r["latency_ms"] or 0.0 for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    b1 = ax1.bar([i - w for i in x], tpr, w, color=COLORS["tpr"], label="TPR (%)")
    b2 = ax2.bar(x, fpr, w, color=COLORS["fpr"], label="FPR (%)")
    b3 = ax3.bar([i + w for i in x], lat, w, color=COLORS["latency"], label="Latency (ms)")
    label_bars(ax1, b1, "{:.1f}")
    label_bars(ax2, b2, "{:.1f}")
    label_bars(ax3, b3, "{:.1f}")

    ax1.set_ylim(0, 125)
    ax1.set_yticks(range(0, 101, 20))
    ax2.set_ylim(0, max(10.0, max(fpr) * 1.75))
    ax3.set_ylim(0, max(10.0, max(lat) * 1.5))
    ax1.set_ylabel("TPR (%)")
    ax2.set_ylabel("FPR (%) on clean images")
    ax3.set_ylabel("Latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend(handles=[b1, b2, b3], loc="upper center", ncol=3, frameon=False)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=SCRIPT_DIR / "compare_mech6_config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.get("detection_filename_prefix", "multi-patched_")
    box_inside_min = float(cfg.get("box_inside_min", 0.5))
    coverage_min = float(cfg.get("patch_coverage_min", 0.25))
    sweep = [float(c) for c in cfg.get("coverage_sweep", [0.1, 0.25, 0.5])]
    panels = cfg.get("panels", ["panel2", "panel3"])
    titles = {**PANEL_TITLES, **cfg.get("panel_titles", {})}      # optional per-detector panel names

    all_rows = []
    for panel in panels:
        rows = [evaluate_panel(m, panel, prefix, box_inside_min, coverage_min, sweep) for m in cfg["models"]]
        all_rows += rows
        fig_path = out_dir / f"{cfg.get('figure_prefix', 'mech6_lean_v3')}_{panel}.png"
        plot_panel(rows, panel, fig_path, f"{cfg.get('title', 'Mech-6 lean-v3')}: {titles.get(panel, panel)}"
                   if cfg.get("show_titles", True) else None)
        print(f"\n{titles.get(panel, panel)} (box >= {box_inside_min:.0%} inside the patches; "
              f"patch found if covered >= {coverage_min:.0%})")
        sw = "  ".join(f"cov{c:.0%}" for c in sweep)
        print(f"  {'model':<8} {'TPR %':>6} {'FPR %':>6} {'FPPI':>6} {'prec':>6} {'F1':>6} {'cov':>5} {'IoU':>5}"
              f" {'lat ms':>7}   TPR at {sw}   old-rule TPR")
        for r in rows:
            lat = f"{r['latency_ms']:.1f}" if r["latency_ms"] is not None else "-"
            print(f"  {r['model']:<8} {100 * r['tpr']:6.1f} {100 * r['fpr_clean_images']:6.1f} {r['fppi']:6.2f} "
                  f"{r['precision']:6.3f} {r['f1']:6.3f} {r['mean_coverage_found']:5.2f} {r['mean_best_iou_found']:5.2f}"
                  f" {lat:>7}   " + "  ".join(f"{100 * v:6.1f}" for v in r["tpr_by_coverage"].values())
                  + f"   {100 * r['old_rule']['tpr']:6.1f}")
        print(f"  figure: {fig_path}")

    (out_dir / "metrics.json").write_text(json.dumps({"config": cfg, "results": all_rows}, indent=2), encoding="utf-8")
    keys = ["model", "panel", "tpr", "fpr_clean_images", "fppi", "fppi_patched", "fppi_clean", "precision", "f1",
            "image_recall", "mean_coverage_found", "mean_best_iou_found", "latency_ms", "latency_median_ms",
            "patched_images", "clean_images"] + [f"tpr_cov{c}" for c in sweep] + ["tpr_old_rule"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            row = dict(r, **{f"tpr_cov{c}": r["tpr_by_coverage"][str(c)] for c in sweep},
                       tpr_old_rule=r["old_rule"]["tpr"])
            w.writerow({k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in keys})
    print(f"\nMetrics: {out_dir / 'metrics.json'} and {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
