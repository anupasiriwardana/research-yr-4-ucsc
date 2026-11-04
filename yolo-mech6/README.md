# Mech-6: per-class feature-energy profiling (Mech-4's backbone signal)

Mech-4 looks at **feature energy** in the first backbone layers: for every grid cell, `energy = (L1 norm of the cell's feature vector across channels)^2`.
Adversarial patches are usually busy, high-contrast textures, so their energy is high. Mech-4 flags a cell when its energy is above ONE global threshold per layer.

Mech-6 asks whether a better reference helps: keep a small **profile per object class and for background** (mean and std of the log-energy, learned from clean images) and score a cell by how far
its energy is from the profile it fits best (`nearest_abs`, or `nearest_up` which only counts energy above the profile). The class is chosen from the cell's own energy, never from the detector's boxes.

This folder is the offline experiment for that idea (same style as `yolo-mech7` diagnostics). It is not a runtime detector yet. Results: `findings.md`.

## Files

| file | purpose |
|---|---|
| `mech6_energy.py` | energy map, backbone-layer hooks (layers 1, 2, 3 = strides 4, 4, 8, same for yolov8n and yolo11n), profile class and the three score variants |
| `calibrate_energy_profiles.py` | step 1: per-class + background log-energy profiles per layer, and the global stats Mech-4 uses, from clean `train` images (2000 by default) |
| `extract_energy_maps.py` | saves raw energy maps (`e1`, `e2`, `e3`) per image so scores can be recomputed offline |
| `energy_viz.py` | the per-image picture (see "Visualizations" below) |
| `diagnose_energy.py` | cell-level test: AUROC and share of patch cells caught at 0.5% / 1% false alarms for each layer and score, and for the layers merged on the 80x80 grid |
| `make_energy_scale_maps.py` | turns layer-1 energy into a pseudo scale `E3` (z-score of log-energy, max-pooled to the P3 grid, threshold = percentile of clean `test` cells) in the Mech-7 score-map format, so `yolo-mech7/combine_scales.py` can score it alone or combined with P3/P4/P5 |
| `config_yolov8.json`, `config_yolo11.json` | model, image folders, tapped layers |

## Run (per model)

```
python calibrate_energy_profiles.py --config config_yolo11.json
python extract_energy_maps.py --config config_yolo11.json --input-dir <patched dir> --out-dir energy-maps/yolo11/patched
python extract_energy_maps.py --config config_yolo11.json --input-dir <clean val> --start-index 0 --num-images 100 --out-dir energy-maps/yolo11/clean
python extract_energy_maps.py --config config_yolo11.json --input-dir <clean test> --num-images 300 --out-dir energy-maps/yolo11/threshold
python diagnose_energy.py --profiles profiles/mech6_energy_profiles_yolo11.pkl --energy-dir energy-maps/yolo11 --gt-dir <patch_coordinates>
python make_energy_scale_maps.py --profiles profiles/mech6_energy_profiles_yolo11.pkl --energy-dir energy-maps/yolo11 --out-root energy-maps/yolo11 --threshold-out profiles/mech6_thresholds_yolo11_e3.pkl
```

then `combine_scales.py` in `yolo-mech7` with `--patched E3=energy-maps/yolo11/e3-patched ...` and `--thresholds E3=profiles/mech6_thresholds_yolo11_e3.pkl` (see `findings.md`).
Energy maps are large and are ignored by git.

## Visualizations (`extract_energy_maps.py --save-visualizations`)

Off by default. When on, the extraction step also writes `mech6_<image stem>.png` per image into `--visualizations-dir`, with three panels:

1. the image, with the ground-truth patch boxes in cyan when `--gt-dir` is given (or `gt_coordinates_dir` in the config);
2. the **energy z-score** at `--viz-layer` (default 1): `(log energy - global mean) / global std`, fixed colour scale 0 (blue) to 8 (red), white outline where z > 3.5. This is the Mech-4 signal;
3. the **Mech-6 score**: distance to the nearest class / background energy profile, in std units of log-energy, on the same colour scale.

With GT boxes the header line gives the median and max z inside the patch vs far from it and the share of cells above 3.5 in each. Turn it on or off with `--save-visualizations` / `--no-save-visualizations` (or `save_visualizations`, `visualizations_dir`, `viz_layer` in the config). It needs the profile file from step 1 (it reads the global stats and the class profiles).

Example:

```
python extract_energy_maps.py --config config_yolo11.json --input-dir <patched dir> --num-images 10 --out-dir energy-maps/yolo11/patched --save-visualizations --visualizations-dir energy-viz/yolo11 --gt-dir <patch_coordinates>
```

In these pictures the patches light up in panel 2 (about 85% of patch cells above z = 3.5 against 0.2% of far cells in the example I checked), and are dimmer in panel 3, because a patch has more energy than every class profile, so its nearest profile is the highest-energy class and the distance to it shrinks. That is the cell-level result in `findings.md`: the per-class score is never better than plain energy.

The visualization settings live in the config files (`config_yolo11.json`, `config_yolov8.json`): `save_visualizations` (true/false, default false), `visualizations_dir`, `viz_layer` and `gt_coordinates_dir`. The command-line flags override them for one run.
