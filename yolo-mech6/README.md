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
