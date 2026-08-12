# Research Blueprint: Decoupling Runtime Behavioral Monitoring from Model Architecture

## Project Information
* **Project Title:** Decoupling Runtime Behavioral Monitoring from Model Architecture: An Adversarial Patch Defense Approach for Object Detection Pipelines
* **Team Members:** Sakith Thewmika (22002022), Anupa Siriwardana (22001921), Pamali Weerasinghe (22002162)
* **Institution:** University of Colombo School of Computing (UCSC)
* **Supervisors:** Prof. Kasun De Zoysa (Internal), Mr. Yasas Mahima (External)
* **Domain:** Software Engineering for Machine Learning (SE4ML), Autonomous Vehicle Security

---

## 1. Executive Summary & Research Progress

Modern safety-critical systems, such as Autonomous Vehicles (AVs), rely heavily on deep-learning perception pipelines (e.g., YOLO) to identify pedestrians, vehicles, and navigation signs in real time. However, these models are inherently vulnerable to **Physical Adversarial Patch Attacks**, where visual patterns placed on real-world objects cause target detectors to completely ignore critical entities.

Existing ML defenses fail from a **Software Engineering (SE)** perspective because they are either:
1. **Inline / Synchronous:** Blocking the main inference path and introducing unacceptable latency.
2. **Tightly Coupled:** Hardcoded to specific network layer indices, breaking completely when the target model is updated (e.g., upgrading from YOLOv5 to YOLOv8).

**Our Solution:** A formally decoupled **Runtime Security Middleware (Sidecar Pattern)** that monitors internal model behavioral signals asynchronously without altering the underlying model architecture or blocking the primary perception loop.

### Project Roadmap & Implementation Status
| Module / Mechanism | Target Directory | Description | Status |
| :--- | :--- | :--- | :--- |
| **Adversarial Patch Generator** | [`mechanisms-by-claude/gen-adv-patch/`](./mechanisms-by-claude/gen-adv-patch/) | White-box physical patch generator using ART and Expectation Over Transformation (EoT) targeting YOLOv8 | **COMPLETED & VERIFIED** |
| **Mechanism 1: Decoupled-Head Tap** | [`mechanisms-by-claude/mech-1/`](./mechanisms-by-claude/mech-1/) | Anomaly detection targeting the intermediate classification branch (`cv3`) using per-class Mahalanobis distance | **COMPLETED & VERIFIED** |
| **Mechanism 2: Pre-Fusion Backbone Tap** | [`mechanisms-by-claude/mech-2/`](./mechanisms-by-claude/mech-2/) | Tapping backbone features before PANet multi-scale fusion to isolate clean receptive fields | **PLANNED** |
| **Mechanism 3: APE Feature Energy** | [`mechanisms-by-claude/mech-3/`](./mechanisms-by-claude/mech-3/) | Self-referential per-image feature energy thresholding operating on early stem layers | **PLANNED** |

---

## 2. Architectural Overview

```text
                      +-----------------------------------+
                      |      Primary Perception Loop     |
                      |    Input Frame ---> YOLOv8 Model  |
                      +-----------------+-----------------+
                                        | (Asynchronous Forward Hook)
                                        v
+---------------------------------------------------------------------------------+
|                        Runtime Security Middleware (Sidecar)                     |
|                                                                                 |
|  +----------------------------------+     +----------------------------------+  |
|  |     Model-Agnostic Adapter       | --> |   Class-Conditioned Strategy     |  |
|  |  (YOLOv8ClsHeadAdapter: cv3[-2]) |     |  (ClsHeadMahalanobisDetector)    |  |
|  +----------------------------------+     +-----------------+----------------+  |
|                                                              |                  |
|                                                              v                  |
|                                                  +-----------------------+      |
|                                                  | AnomalyResult Schema  |      |
|                                                  | - is_attack: bool     |      |
|                                                  | - score: float        |      |
|                                                  | - bounding_box: tuple |      |
|                                                  +-----------+-----------+      |
+--------------------------------------------------------------|------------------+
                                                               |
                                                               v
                                                +------------------------------+
                                                |       Recovery Engine        |
                                                | (Pixel Blackout / Re-infer)  |
                                                +------------------------------+
```

### The Interface Contract (`AnomalyResult`)
To preserve strict separation of concerns, all anomaly detection strategies must return a unified JSON/dictionary schema:
* `is_attack` *(boolean)*: Flags whether an adversarial anomaly exceeded the calibrated threshold.
* `score` *(float)*: The maximum statistical anomaly score (e.g., Mahalanobis distance) observed in the frame.
* `bounding_box` *(tuple)*: `(x1, y1, x2, y2)` pixel coordinates isolating the patch location for downstream recovery.

---

## 3. Quick Start Guide

### Step 1: Clone Repository
```bash
git clone [https://github.com/AnupaSiriwardhana/yr-4-research.git](https://github.com/AnupaSiriwardhana/yr-4-research.git)
cd yr-4-research
```

### Step 2: Generate an Adversarial Test Patch
Follow the guide in the [`gen-adv-patch`](./mechanisms-by-claude/gen-adv-patch/) folder to build white-box test assets.
```bash
conda activate patch_gen_yolov8
python mechanisms-by-claude/gen-adv-patch/generate_art_patch.py
```

### Step 3: Run the Anomaly Detection Middleware
Follow the guide in the [`mech-1`](./mechanisms-by-claude/mech-1/) folder to calibrate and execute the Decoupled-Head Middleware.
```bash
conda activate yolo_adv
python mechanisms-by-claude/mech-1/calibrate_cls_head.py
python mechanisms-by-claude/mech-1/cls_head_detector.py
```

---

## 4. Repository Structure

```text
yr-4-research/
├── LICENSE
├── README.md                              <-- Main Project Documentation
└── mechanisms-by-claude/
    ├── gen-adv-patch/                     <-- Adversarial Patch Generator Tooling
    │   ├── config.json
    │   ├── generate_art_patch.py
    │   ├── apply_saved_patch.py
    │   ├── verify_patch_efficacy.py
    │   └── README.md
    ├── mech-1/                            <-- Mechanism 1: Decoupled Head Anomaly Module
    │   ├── config.json
    │   ├── cls_head_adapter.py
    │   ├── calibrate_cls_head.py
    │   ├── cls_head_detector.py
    │   ├── mechanism-1-decoupled-head-cls-branch.md
    │   └── README.md
    ├── mech-2/                            <-- Mechanism 2 Specification (Planned)
    └── mech-3/                            <-- Mechanism 3 Specification (Planned)
```

---

## 5. License
Distributed under the MIT License. See `LICENSE` for details.