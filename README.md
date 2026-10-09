# Decoupling Runtime Behavioral Monitoring from Model Architecture

Research code for detecting physical adversarial patches against YOLO object-detection
pipelines using decoupled runtime monitoring. The project explores how a security
middleware layer can observe internal model behavior without modifying the detector
architecture or blocking the primary perception loop.

## Project information

- **Project title:** Decoupling Runtime Behavioral Monitoring from Model Architecture:
  An Adversarial Patch Defense Approach for Object Detection Pipelines
- **Team:** Sakith Thewmika (22002022), Anupa Siriwardana (22001921), Pamali
  Weerasinghe (22002162)
- **Institution:** University of Colombo School of Computing (UCSC)
- **Supervisors:** Prof. Kasun De Zoysa (Internal), Mr. Yasas Mahima (External)
- **Domain:** Software Engineering for Machine Learning (SE4ML) and autonomous-vehicle
  security
- **Current branch:** `multi-patch-detection`
- **Latest recorded commit:** `ec2f782` — `patch generation for yolo11`

## Current repository state

The repository currently contains working research scripts, calibration profiles,
model checkpoints, generated patch examples, and design documents. The main
implementation work is organized as follows:

| Area | Location | Current state |
| :--- | :--- | :--- |
| Generic adversarial patch generation | [`gen-adv-patch/`](./mechanisms-by-claude/gen-adv-patch/) | YOLOv8 white-box patch generation using ART/EoT, saved-patch application, and efficacy verification |
| Object-aware hiding patches | [`gen-adv-patch-hiding/`](./mechanisms-by-claude/gen-adv-patch-hiding/) | Direct PyTorch patch optimization targeted at detected objects, including multi-patch application and verification |
| Mechanism 1: classification-head monitoring | [`mech-1/`](./mechanisms-by-claude/mech-1/) | Baseline decoupled classification-branch tap with Mahalanobis anomaly scoring |
| Mechanism 1 updated | [`mech-1-updated/`](./mechanisms-by-claude/mech-1-updated/) | Box-independent scoring, tighter anomaly boxes, and multi-patch detection variants |
| Mechanism 4: feature-energy monitoring | [`mech-4/`](./mechanisms-by-claude/mech-4/) | Feature-energy detector with multi-layer activation support, calibration experiments, region filtering, and heatmap output |
| Cross-model Mechanism 1 | [`yolo-transferable-mech1/`](./yolo-transferable-mech1/) | Calibration and detection workflow prepared for both YOLOv8 and YOLO11 checkpoints |
| Mechanism 2 | [`mech-2/`](./mechanisms-by-claude/mech-2/) | Pre-PANet backbone-tap design specification; implementation is not yet present |

The generated images, NumPy patches, serialized calibration profiles, and local
checkpoint files under these directories are research artifacts used by the scripts.
They are not a substitute for a clean, reproducible dataset setup.

## Architecture

The intended system separates the detector from the security middleware:

```text
Input frame ──► YOLO detector ──► normal detections
                    │
                    └─ internal activation hook
                                  │
                                  ▼
                    Runtime security middleware
                    ├─ model-specific adapter
                    ├─ anomaly-detection strategy
                    └─ AnomalyResult
                       ├─ is_attack
                       ├─ score
                       └─ bounding_box
                                  │
                                  ▼
                    optional recovery / re-inference
```

Mechanism 1 uses class-conditioned Mahalanobis distances over classification-head
activations. Mechanism 4 computes feature energy and filters anomalous regions.
Both expose the same conceptual result: whether an attack was detected, an anomaly
score, and a pixel-space region for downstream recovery.

## Quick start

Each module has its own configuration and environment notes. The paths in the
checked-in JSON files point to the original local research-data layout, so update
them before running on another machine.

### 1. Set up the environments

The patch-generation modules use a separate environment, while the detection
modules use `yolo_adv`:

```bash
conda create -n patch_gen_yolov8 python=3.10 -y
conda activate patch_gen_yolov8
# Install the dependencies described in the selected patch-generator README.

conda create -n yolo_adv python=3.10 -y
conda activate yolo_adv
# Install PyTorch, Ultralytics, OpenCV, NumPy, and the other dependencies
# described in the selected detector README.
```

### 2. Generate and verify an object-aware hiding patch

```bash
cd mechanisms-by-claude/gen-adv-patch-hiding
python generate_hiding_patch.py
python apply_saved_patch.py
python verify_patch_efficacy.py
```

See [`gen-adv-patch-hiding/README.md`](./mechanisms-by-claude/gen-adv-patch-hiding/README.md)
for the configuration schema, target-class settings, placement modes, and output
locations.

### 3. Run the updated Mechanism 1 detector

Calibrate on clean images before running detection:

```bash
cd mechanisms-by-claude/mech-1-updated
python calibrate_cls_head.py
python cls_head_detector.py
# Optional tighter recovery box:
python cls_head_detector_for_tightPatch.py
```

See [`mech-1-updated/README.md`](./mechanisms-by-claude/mech-1-updated/README.md).
The original [`mech-1/`](./mechanisms-by-claude/mech-1/) module remains available
for comparison with the earlier calibration and scoring workflow.

### 4. Run the feature-energy detector

The current Mechanism 4 directory includes both calibration tooling and the runtime
detector:

```bash
cd mechanisms-by-claude/mech-4
python calibrate_energy_stats.py
python feature_energy_detector.py
```

Review [`mech-4/README.md`](./mechanisms-by-claude/mech-4/README.md) and
[`new-mech-4-detection-process.md`](./mechanisms-by-claude/mech-4/new-mech-4-detection-process.md)
before changing tap layers, stride mappings, or thresholds.

### 5. Compare YOLOv8 and YOLO11 transferability

```bash
cd yolo-transferable-mech1
python calibrate_cls_head.py
python cls_head_detector.py
```

The configuration selects the checkpoint and calibration profile. Use separate
profiles for YOLOv8 and YOLO11; a profile calibrated for one model must not be
reused for the other.

## Repository layout

```text
yr-4-research/
├── LICENSE
├── README.md
├── mechanisms-by-claude/
│   ├── gen-adv-patch/          # Generic YOLOv8 ART/EoT patch workflow
│   ├── gen-adv-patch-hiding/   # Object-aware and multi-patch workflow
│   ├── mech-1/                 # Original classification-head detector
│   ├── mech-1-updated/         # Updated multi-patch/tight-box detector
│   ├── mech-2/                 # Pre-fusion backbone-tap specification
│   └── mech-4/                 # Feature-energy detector and calibration
└── yolo-transferable-mech1/    # YOLOv8/YOLO11 transferability workflow
```

For module-specific implementation details, use the README and design documents
inside each directory rather than assuming that all mechanisms share identical
configuration fields or calibration requirements.

## Reproducibility notes

- Use the same model family and checkpoint for calibration and runtime detection.
- Keep clean calibration images separate from patched evaluation images.
- Re-derive thresholds when changing the scoring rule, tap layer, model version, or
  calibration profile.
- Verify the configured stride and hooked layer against the actual checkpoint before
  interpreting bounding boxes.
- GPU execution is recommended; CPU execution is supported by the scripts but can
  make calibration and patch generation substantially slower.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.
