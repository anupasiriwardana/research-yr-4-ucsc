# Research Blueprint: Decoupling Runtime Behavioral Monitoring from Model Architecture

## Project Information
* **Project Title:** Decoupling Runtime Behavioral Monitoring from Model Architecture: An Adversarial Patch Defense Approach for Object Detection Pipelines
* **Team Members:** Sakith Thewmika (22002022), Anupa Siriwardana (22001921), Pamali Weerasinghe (22002162)
* **Institution:** University of Colombo School of Computing (UCSC)
* **Supervisors:** Prof. Kasun De Zoysa (Internal), Mr. Yasas Mahima (External)
* **Domain:** Software Engineering for Machine Learning (SE4ML), Autonomous Vehicle Security

---

## 1. The Core Idea & Purpose
Modern Artificial Intelligence is deeply embedded in safety-critical systems, notably Autonomous Vehicles (AVs). The foundational software layer, the **Perception Pipeline**, relies heavily on object detection models (like YOLO) to identify pedestrians, vehicles, and traffic signs in real-time. 

**The Purpose:** To architect, engineer, and evaluate a formally decoupled **Runtime Security Middleware (Sidecar)** that monitors internal model behaviors to detect physical adversarial patch attacks and recover reliable detections, without modifying the underlying object detection model or blocking the inference pipeline.

---

## 2. The Threat Model
Object detection models are highly vulnerable to **Adversarial Patch Attacks**.
* **Mechanism:** Specially crafted visual patterns printed as physical stickers or signs and placed in the real-world environment.
* **Impact:** Unlike digital noise, physical patches are robust to lighting, camera angles, and motion blur. A sticker on a stop sign can cause the vehicle's detector to ignore the sign entirely. The AV makes planning decisions based on corrupted data, leading directly to physical collisions.

---

## 3. The Architectural Problem & Research Gap
While the ML community has proposed various defenses, they critically fail from a **Software Engineering (SE) perspective**:

1. **Input-Level Defenses (Preprocessing):** Model-agnostic, but they are synchronous and inline. They block the inference path, introducing mandatory latency on every frame, which destroys real-time performance.
2. **Output-Level Verification:** Decoupled, but heavily domain-specific (e.g., relying on vehicle motion physics) and lack a structured recovery mechanism.
3. **Runtime Monitors (The Core Issue):** Existing runtime monitors are **tightly coupled** to the internal structure of the specific model. They access specific layer names, tensor shapes, and architectures with no abstraction. If the perception model is updated (e.g., YOLOv5 to YOLOv8), the defense breaks entirely.

**The Formal Gap:** There is no independently deployable runtime monitoring component that decouples the adversarial defense from the internal architecture of the object detection model, operates asynchronously without blocking the inference pipeline, and provides a structured detection and recovery response.

---

## 4. The Proposed Solution: System Architecture
To resolve this SE gap, the research proposes a **Run-Time Security Middleware (Sidecar Pattern)**.

### 4.1 Decoupled Abstraction Layer
Instead of hardcoding the defense to YOLO, the middleware uses an **Adapter Pattern**. Access to internal behavioral signals (feature maps) is mediated through a formal abstraction layer. The monitoring logic evaluates the geometry of the latent space (Manifold Theory) without being directly tied to the model’s internal structure.

### 4.2 The Interface Contract (`AnomalyResult`)
To ensure a strict separation of concerns between Detection and Recovery, the Anomaly Detector module must output a standardized `AnomalyResult` schema.
* **Schema Content:** `[is_attack: boolean, score: float, bounding_box: (x1, y1, x2, y2)]`
* **Architectural Benefit:** The Recovery Engine receives exact coordinates for masking but has zero knowledge of *how* the patch was detected, preserving low coupling.

---

## 5. Module 1: Anomaly Detection (Behavioral Profiling)
The detection module profiles internal behavioral biometrics (aligning with the NIST AI 100-2 E2025 standard). The team will implement contrasting detection algorithms that both satisfy the `AnomalyResult` interface. This module will also generate a heatmap which helps to visualize the localized patch during the building and testing purposes.  

---

## 6. Module 2: The Recovery Engine
The Recovery Engine acts as a "dumb" execution engine. It receives the bounding box from the Detection module and triggers a tiered response:

* **Main Recovery attempt:** We would try to attempt the recovery of the patched image asynchronous to the object detection model. If we cannot make the recovery asynchronous, then we'll fall back to tier 2 recovery or tier 3 recovery. Priority should be asynchronous recovery
* **Tier 2 Recovery (Masking & Re-inference):** Operates on the input image using standard masking (a pixel blackout operation over the patch coordinates) and then calls the model's standard forward pass.
* **Tier 3 Recovery (Temporal Substitution):** Operates entirely on buffered historical detection outputs to maintain system stability if re-inference fails.

*(Crucial SE Concept: The recovery engine never attempts to localize the patch itself, preventing redundant detection loops).*

---

## 7. Evaluation & Metrics
The solution will be evaluated not just on security, but on strict **Non-Functional Requirements (NFRs)** necessary for time-critical domains where safety is necessary.

### Security and Reliability Metrics
* **Attack Success Rate (ASR):** How often attacks successfully fool the detector despite the monitor.
* **True Positive Rate (TPR) / False Positive Rate (FPR):** Accuracy of the anomaly detection.
* **Recovery Rate (RR):** Percentage of attacked frames where reliable detections are successfully restored.

### System Performance (Engineering) Metrics
* **Inference Latency Overhead:** Additional delay introduced on clean frames (Target: <15ms).
* **Attacked-Frame Latency:** Total system latency during recovery.
* **Frames Per Second (FPS):** System throughput.
* **Memory Overhead:** RAM/VRAM consumption.

---

## 8. Datasets & Tooling
* **Datasets:** KITTI Vision Benchmark Suite (Baseline urban driving) and BDD100K (Diverse weather/complex environments).
* **Threat Simulation:** Adversarial Robustness Toolbox (ART) and Expectation Over Transformation (EoT) physical patch simulations.
* **Target Model:** Ultralytics YOLO architecture (e.g., YOLOv5/v8/v10).
