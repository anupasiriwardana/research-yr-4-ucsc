# Mech-6: per-class feature-energy profiling (Mech-4's backbone signal)

Mech-4 looks at **feature energy** in the first backbone layers: for every grid cell, `energy = (L1 norm of the cell's feature vector across channels)^2`.
Adversarial patches are usually busy, high-contrast textures, so their energy is high. Mech-4 flags a cell when its energy is above ONE global threshold per layer.

Mech-6 asks whether a better reference helps: keep a small **profile per object class and for background** (mean and std of the log-energy, learned from clean images) and score a cell by how far
its energy is from the profile it fits best (`nearest_abs`, or `nearest_up` which only counts energy above the profile). The class is chosen from the cell's own energy, never from the detector's boxes.

**What this folder is, and is not.** It is the offline experiment for that idea (same style as the `yolo-mech7` diagnostics). There is **no standalone detector** here that reads an image and writes a
box file (like `detect_mech5_multi_img.py` does for Mech-5). You extract energy maps, look at pictures, run a cell-level test, and get box-level numbers through `yolo-mech7/combine_scales.py`.
Results and conclusions: `findings.md`.

Always run the commands below from this folder (`yolo-mech6`) with the `yolo_adv` conda environment.

---

## 1. Which script do I need? (cheat sheet)

| I want to... | script | needs first |
|---|---|---|
| set up a new model (build the energy profiles) | `calibrate_energy_profiles.py` | config |
| compute and save energy maps for a folder of images | `extract_energy_maps.py` | profiles only if pictures are on |
| see pictures of the energy on patched or clean images | `extract_energy_maps.py` with `save_visualizations` on | profiles |
| check how well energy separates patch cells from normal cells (cell level) | `diagnose_energy.py` | energy maps (patched + clean), GT folder, profiles |
| get box-level numbers (precision, recall, IoU, clean false positives) | `make_energy_scale_maps.py`, then `yolo-mech7/combine_scales.py` | energy maps (patched + clean + threshold), GT folder, profiles |
| combine energy with Mech-5's P3/P4/P5 scales | same two steps, with the P maps added | the Mech-7 score maps of the P scales |

---

## 2. What each script does

### `mech6_energy.py` (library, never run directly)
Shared code. The energy formula, the forward hooks on backbone layers 1, 2, 3 (strides 4, 4, 8; the same indices work for `yolov8n` and `yolo11n`), image loading, and `EnergyProfiles`, which turns an energy map into
the three scores: `E` (raw energy, one global threshold, i.e. Mech-4), `nearest_abs` and `nearest_up` (Mech-6).

### `calibrate_energy_profiles.py` (step 1, once per model)
- **Does:** builds the energy profiles. For each clean `train` image it runs the model, projects the detected boxes onto each layer's grid, and collects the log-energy of the cells inside each box (per class) and of a random
  sample of uncovered cells (background). It also stores the global mean/std of energy over all cells (what Mech-4 uses).
- **Reads:** the config (`model_path`, `profile_images_dir`, `tap_layers`, `max_calibration_images` = 2000, `min_calibration_samples`, `background_samples_per_image`).
- **Writes:** `profiles/<profile_filename>` (a pickle). Takes a few minutes.
- **Flags:** `--config <file>` (required), `--max-images N`.

### `extract_energy_maps.py` (the one you will run most)
- **Does:** runs the model on a folder of images and saves the raw energy map of every tapped layer. Optionally also draws a picture per image (see section 5).
- **Reads:** the config (`input_dir`, `out_dir`, `start_index`, `num_images`), and (for pictures only) the profile file and a GT folder.
- **Writes:** `<out-dir>/<image stem>.npz` (keys `e1`, `e2`, `e3`), and, if pictures are on, `mech6_<stem>.png` in the visualizations folder.
- **Flags:** only `--config` is required. Folders and the slice now come from the config; `--input-dir`, `--out-dir`, `--start-index N`, `--num-images N` override them for one run. Also `--save-visualizations` / `--no-save-visualizations`, `--visualizations-dir`,
  `--viz-layer`, `--gt-dir` (override the config for one run).
- You run it once per image set: patched images, clean evaluation images, and clean threshold images.

### `energy_viz.py` (library, called by `extract_energy_maps.py`)
Draws the three-panel picture. Not run directly.

### `diagnose_energy.py` (cell-level test; prints, writes no file)
- **Does:** for every grid cell of every patched image: is it a patch cell (centre inside a ground-truth box) or a normal cell (more than 48 px away from every patch)? Then, for each layer and score, how well does the score
  tell them apart: AUROC and the share of patch cells caught at 0.5% / 1% false alarms. It also merges the layers on the 80x80 grid (`any`, `all`, `mean`).
- **Reads:** `--energy-dir` with the subfolders `patched/` and `clean/`, `--profiles`, `--gt-dir`.
- **Use it to:** judge the signal itself, and to compare Mech-4 energy (`E`) with the Mech-6 per-class scores.

### `make_energy_scale_maps.py` (prepares the box-level test)
- **Does:** turns the layer-1 energy into a pseudo "scale" called `E3`: z-score of the log-energy, clipped at 0, max-pooled to the 80x80 (P3) grid. Its threshold is the 99.5th percentile over all cells of the clean *threshold* images.
  It writes these in the same file format as the Mech-7 score maps, so `combine_scales.py` can treat `E3` like P3, P4 or P5.
- **Reads:** `--energy-dir` with the subfolders `patched/`, `clean/` and `threshold/`, and `--profiles`.
- **Writes:** `<out-root>/e3-patched/`, `<out-root>/e3-clean/`, and the threshold file `--threshold-out`.

### `yolo-mech7/combine_scales.py` (box level; lives in the other folder)
Takes score-map folders (`E3`, and optionally `P3`/`P4`/`P5`), builds boxes the same way Mech-5 does, sweeps the settings (offset / join / min cells), and reports precision, recall, F1, mean IoU and the clean-image
false-positive rate, with the 20-split held-out check. See `../yolo-mech5/findings.md` for how to read its output.

### `config_yolo11.json`, `config_yolov8.json` (one per model)

| key | meaning |
|---|---|
| `model_path` | the YOLO weights |
| `profile_images_dir`, `max_calibration_images` | clean images used to build the profiles (first 2000 of `train`) |
| `profiles_dir`, `profile_filename` | where the profile pickle is written / read |
| `tap_layers` | backbone layers tapped (1, 2, 3) |
| `min_calibration_samples`, `background_samples_per_image` | profile-building details |
| `input_dir` | folder of images that `extract_energy_maps.py` reads (the config ships with the patched set; change it or pass `--input-dir` for clean/threshold sets) |
| `out_dir` | folder where the `.npz` maps are written |
| `start_index`, `num_images` | slice of the folder (`num_images` null = all) |
| `save_visualizations` | true/false: draw a picture per image during extraction (default false) |
| `visualizations_dir` | where the pictures go |
| `viz_layer` | which layer the pictures show (default 1, the one that carries the signal) |
| `gt_coordinates_dir` | patch ground-truth folder, used for the boxes and header in the pictures |

---

## 3. Folders and where things end up

```
yolo-mech6/
  profiles/                         energy profiles (+ the E3 threshold file)         <- calibrate_energy_profiles.py, make_energy_scale_maps.py
  energy-maps/<model>/patched/      raw energy maps of the patched images             <- extract_energy_maps.py
  energy-maps/<model>/clean/        raw energy maps of the clean evaluation images    <- extract_energy_maps.py
  energy-maps/<model>/threshold/    raw energy maps of the clean threshold images     <- extract_energy_maps.py
  energy-maps/<model>/e3-patched/   E3 score maps (Mech-7 format)                     <- make_energy_scale_maps.py
  energy-maps/<model>/e3-clean/     E3 score maps (Mech-7 format)                     <- make_energy_scale_maps.py
  energy-viz/<model>/               the pictures                                      <- extract_energy_maps.py (pictures on)
```
`energy-maps/` and `energy-viz/` are ignored by git.

Image sets used so far (BDD100K images in `D:\GitHub\experiment_data\bdd100k_images_10k`):

| set | YOLO11 | YOLOv8 |
|---|---|---|
| patched images | `D:\GitHub\yr-4-research\test-patch-det-multi-img\gen-patched-img\yolov11-patched-imgs` | `...\gen-patched-img\yolov8-patched-imgs` |
| patch ground truth | `...\gen-patched-img\yolov11-patch_coordinates` | `...\gen-patched-img\yolov8-patch_coordinates` |
| clean evaluation images | `val`, images 0-99 (`--start-index 0 --num-images 100`) | `val`, images 100-199 (`--start-index 100 --num-images 100`) |
| clean threshold images | `test`, first 300 (`--num-images 300`) | same |

(Use the same clean slice for the Mech-5/Mech-7 maps and the energy maps if you want to combine them: the image names must match.)

---

## 4. Step-by-step per use case

Replace `yolo11` with `yolov8` (and the folders from the table above) for the other model. `<patched>`, `<gt>`, `<val>`, `<test>` are the folders from section 3.

### Use case A: starting fresh with a model, run it on patched and clean images, and see pictures

1. **Check the config** (`config_yolo11.json`): `model_path`, `profile_images_dir`. Set `save_visualizations` to `true` and check `visualizations_dir` and `gt_coordinates_dir` if you want pictures during step 3.
2. **Build the profiles** (a few minutes):
   ```
   python calibrate_energy_profiles.py --config config_yolo11.json
   ```
   Check that `profiles/mech6_energy_profiles_yolo11.pkl` exists. (The pictures in step 3 need this file.)
3. **Extract energy maps for the three image sets** (pictures are written during this step when `save_visualizations` is true; put them in separate folders per set with `--visualizations-dir`):
   ```
   python extract_energy_maps.py --config config_yolo11.json --input-dir <patched> --out-dir energy-maps/yolo11/patched --gt-dir <gt> --visualizations-dir energy-viz/yolo11/patched
   python extract_energy_maps.py --config config_yolo11.json --input-dir <val> --start-index 0 --num-images 100 --out-dir energy-maps/yolo11/clean --visualizations-dir energy-viz/yolo11/clean
   python extract_energy_maps.py --config config_yolo11.json --input-dir <test> --num-images 300 --out-dir energy-maps/yolo11/threshold --no-save-visualizations
   ```
   To draw pictures for only a few images, add `--num-images 10` to the first two and `--save-visualizations` if it is off in the config.
4. **Look at the pictures** in `energy-viz/yolo11/patched` and `energy-viz/yolo11/clean` (section 5 explains the panels). For patched images the header shows how many patch cells are above z = 3.5 against how many far cells.
5. **Cell-level numbers:**
   ```
   python diagnose_energy.py --profiles profiles/mech6_energy_profiles_yolo11.pkl --energy-dir energy-maps/yolo11 --gt-dir <gt>
   ```
6. **Box-level numbers** (precision, recall, mean IoU, clean false positives):
   ```
   python make_energy_scale_maps.py --profiles profiles/mech6_energy_profiles_yolo11.pkl --energy-dir energy-maps/yolo11 --out-root energy-maps/yolo11 --threshold-out profiles/mech6_thresholds_yolo11_e3.pkl
   cd ../yolo-mech7
   python combine_scales.py --gt-dir <gt> --holdout-splits 20 --seed 0 --offsets -0.7 -0.5 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.5 0.8 1.2 2.0 --joins 0 1 2 3 --min-cells 4 6 10 15 20 30 40 60 --patched E3=../yolo-mech6/energy-maps/yolo11/e3-patched --clean E3=../yolo-mech6/energy-maps/yolo11/e3-clean --thresholds E3=../yolo-mech6/profiles/mech6_thresholds_yolo11_e3.pkl
   ```
   The "HOLDOUT" table at the end is the honest estimate; the false-positive rate of a run on clean images is the `clean FPR` column.

Note: there is no per-image "detected boxes" output for Mech-6. The pictures show the energy and the cells above z = 3.5, not final boxes; boxes are built inside `combine_scales.py` and only counted, not drawn.

### Use case B: only look at pictures (profiles already exist)

1. In `config_yolo11.json` set `"save_visualizations": true` (or pass `--save-visualizations`).
2. Run, for example on 10 patched images and 10 clean images:
   ```
   python extract_energy_maps.py --config config_yolo11.json --input-dir <patched> --num-images 10 --out-dir energy-maps/yolo11/patched --gt-dir <gt> --visualizations-dir energy-viz/yolo11/patched
   python extract_energy_maps.py --config config_yolo11.json --input-dir <val> --num-images 10 --out-dir energy-maps/yolo11/clean --visualizations-dir energy-viz/yolo11/clean
   ```
3. Open the PNGs. Pass `--viz-layer 3` to see layer 3 instead of layer 1.

(Careful: running this over a folder rewrites its `.npz` maps for those images. Use a different `--out-dir` if you do not want to overwrite maps you already analysed.)

### Use case C: only the cell-level test (maps already exist)
Run step 5 of use case A. Nothing else is needed.

### Use case D: only box-level numbers (maps already exist)
Run step 6 of use case A.

### Use case E: combine energy with Mech-5's P3/P4/P5 scales
1. Have the Mech-7 score maps of the P scales for the same patched and clean images (`yolo-mech7/score-maps/<model>/p3-patched` etc.; made with `detect_mech7_multi_img.py --save-score-maps`).
2. Run step 6 of use case A, adding the P maps and their threshold files to the `combine_scales.py` command, for example for YOLO11 P3+P4+E3:
   ```
   --patched P3=score-maps/yolov11/p3-patched P4=score-maps/yolov11/p4-patched E3=../yolo-mech6/energy-maps/yolo11/e3-patched
   --clean   P3=score-maps/yolov11/p3-clean   P4=score-maps/yolov11/p4-clean   E3=../yolo-mech6/energy-maps/yolo11/e3-clean
   --thresholds E3=../yolo-mech6/profiles/mech6_thresholds_yolo11_e3.pkl P3=profiles/mech7_thresholds_yolo11_p3.pkl P4=profiles/mech7_thresholds_yolo11_p4.pkl
   ```
   (Put P3 first so it is the finest scale in the `confirm` rule. For YOLOv8 the threshold files are `profiles/mech5_thresholds_yolov8_p3.pkl` etc. in `yolo-mech7`.)

### Use case F: a new model
1. Copy a config file, change `model_path`, `profile_filename` and the folders.
2. Check that backbone layers 1, 2, 3 have strides 4, 4, 8 for the new model (a quick forward pass with hooks, as in `mech6_energy.py`); if the layout is different, adjust `LAYER_STRIDE` there.
3. Follow use case A.

---

## 5. The pictures (`save_visualizations`)

Off by default. When on, `extract_energy_maps.py` writes `mech6_<image stem>.png` per image, with three panels:

1. the image, with the ground-truth patch boxes in cyan when a GT folder is given;
2. the **energy z-score** at `viz_layer`: `(log energy - global mean) / global std`, fixed colour scale 0 (blue) to 8 (red), white outline where z > 3.5. This is the Mech-4 signal;
3. the **Mech-6 score**: distance to the nearest class / background energy profile, in std units of log-energy, same colour scale.

The cyan GT boxes are drawn only on panel 1, so the heatmaps in panels 2 and 3 stay unobstructed. With GT boxes the header gives the median and max z inside the patch vs far from it and the share of cells above 3.5 in each; without GT (clean images) it gives overall numbers.
In the example I checked, about 85% of patch cells were above 3.5 against 0.2% of far cells. Panel 3 is dimmer than panel 2 because a patch has more energy than every class profile, so its nearest profile is the
highest-energy class and the distance to it shrinks. That matches the cell-level result in `findings.md`: the per-class score is never better than plain energy.

Settings (all in the config files, overridable on the command line for one run): `save_visualizations`, `visualizations_dir`, `viz_layer`, `gt_coordinates_dir`.
