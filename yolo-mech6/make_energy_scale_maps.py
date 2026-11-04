"""
Turn the layer-1 energy maps into a pseudo "scale" that yolo-mech7/combine_scales.py can combine
with the P3/P4/P5 score maps (same .npz format: raw / best_class / class_ids / stride / tap_scale).

Score per cell: z = (log energy - global mean of log energy) / global std, clipped at 0, taken at
backbone layer 1 (stride 4) and max-pooled 2x2 onto the 80x80 grid (stride 8) so it lines up with P3.
It is a monotone function of energy, so it separates exactly as well as raw energy; the log/z form only
makes the scale comparable (a threshold near 1 = the clean-cell percentile `--percentile`).

The threshold is that percentile of the score over every cell of the clean threshold images
(`--threshold-dir`, disjoint from the calibration and evaluation images), written as a
threshold pickle in the same format as the P-scale ones, so combine_scales divides by it.

Writes <out-root>/e3-patched, <out-root>/e3-clean and <profiles-dir>/<name>.pkl.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from mech6_energy import log_energy


def score(e1, g):
    z = np.clip((log_energy(e1) - g["mean_log"]) / g["std_log"], 0, None)
    h, w = z.shape
    return z.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--energy-dir", required=True, help="has patched/, clean/ and the threshold folder")
    ap.add_argument("--threshold-dir", default="threshold", help="subfolder of --energy-dir with clean threshold images")
    ap.add_argument("--percentile", type=float, default=99.5)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--threshold-out", required=True)
    args = ap.parse_args()

    with open(args.profiles, "rb") as f:
        g = pickle.load(f)[1]["global"]
    root = Path(args.energy_dir)

    vals = np.concatenate([score(np.load(p)["e1"], g).ravel() for p in sorted((root / args.threshold_dir).glob("*.npz"))])
    thr = float(np.percentile(vals, args.percentile))
    entry = {"threshold": thr, "mean": float(vals.mean()), "std": float(vals.std()), "n": int(vals.size), "source": "own"}
    with open(args.threshold_out, "wb") as f:
        pickle.dump({"percentile": args.percentile, "tap_scale": "E3", "pooled": entry, "per_class": {0: entry}}, f)
    print(f"threshold images: {len(list((root / args.threshold_dir).glob('*.npz')))}, p{args.percentile} of z = {thr:.3f}")

    for kind in ("patched", "clean"):
        out = Path(args.out_root) / f"e3-{kind}"
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted((root / kind).glob("*.npz")):
            s = score(np.load(p)["e1"], g)
            np.savez_compressed(out / f"{p.stem}.npz", raw=s, best_class=np.zeros(s.shape, np.int16),
                                class_ids=np.array([0], np.int16), stride=np.int32(8), tap_scale=np.array("E3"))
            n += 1
        print(f"{kind}: {n} maps -> {out}")


if __name__ == "__main__":
    main()
