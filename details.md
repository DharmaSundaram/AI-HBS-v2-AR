# AI-HBS-v2-AR: Complete Technical Specification & Project Details

> **Project Name:** AI Human Body Measurement & Augmented Reality Engine (Version 2)  
> **Status:** Phase 6.2 Completed (Real-Time AR Measurement Lines & Labels)  
> **Notice:** Prototype computer-vision measurement system. Measurements represent approximate anthropometric estimations and are not certified for medical diagnostics.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [What the System Does](#2-what-the-system-does)
3. [Real-World Use Cases](#3-real-world-use-cases)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture & Data Pipeline](#5-system-architecture--data-pipeline)
6. [Mathematical & Algorithmic Methodology](#6-mathematical--algorithmic-methodology)
7. [Module Breakdown](#7-module-breakdown)
8. [Interactive Controls & HUD](#8-interactive-controls--hud)
9. [Data Outputs & File Schemas](#9-data-outputs--file-schemas)
10. [Assumptions, Limitations & Quality Standards](#10-assumptions-limitations--quality-standards)
11. [Hardware Progression & Future Roadmap](#11-hardware-progression--future-roadmap)

---

## 1. Executive Summary

**AI-HBS-v2-AR** is a markerless, single-webcam computer vision and anthropometry framework designed to extract human body dimensions, construct a continuous 3D volumetric mesh, and refine that mesh using the subject's visible 2D body silhouette.

Unlike legacy computer vision projects that rely solely on flat 2D bounding boxes or sparse joint distances, AI-HBS-v2-AR combines:
1. **Metric 3D World Landmarks** for skeletal length estimation.
2. **Real-Time Semantic Body Segmentation** to extract the subject's actual contour profile.
3. **Regional Silhouette-Constrained Mesh Fitting** across 14 independent anatomical regions.
4. **Planar Cross-Sectional Surface Slicing** for true circumference estimation (chest, waist, hip, arms, thighs).
5. **Statistical Temporal Stability Analysis & Ground-Truth Validation** to measure frame-to-frame consistency and validate against physical measuring tapes.

The system runs locally on standard desktop consumer hardware (laptop/desktop webcam) at real-time speeds ($30\text{–}60\text{ FPS}$) without requiring specialized depth cameras or cloud servers.

---

## 2. What the System Does

### Core Functional Capabilities

- **Automatic Full-Body Posture & Visibility Gating:**  
  Continuously inspects 12 critical keypoints (shoulders, elbows, wrists, hips, knees, ankles). Captures and stabilizes only when the subject is fully visible in a proper frontal stance.
- **Metric 3D Segment Length Calculations:**  
  Calculates Euclidean distances between anatomical joint centers in true metric coordinates (meters converted to centimeters), stabilized over a 60-frame median buffer to reject outliers.
- **Continuous Volumetric 3D Body Mesh:**  
  Generates an anatomically lofted 3D mesh consisting of 18 proportional regions (head, neck, shoulders, chest, abdomen, pelvis, upper arms, forearms, hands, thighs, lower legs, feet) using elliptical ring topology.
- **2D Body Silhouette Extraction:**  
  Extracts a pixel-accurate binary mask ($255$ foreground, $0$ background) using an embedded neural network segmenter with temporal exponential smoothing to eliminate border jitter.
- **Silhouette-Constrained Regional Refinement:**  
  Casts perpendicular search rays from projected 2D segment centers into the segmentation mask to measure actual silhouette width across 14 distinct body zones. Adapts the 3D mesh radii to fit the observed silhouette while strictly preserving 3D joint landmark coordinates.
- **Refined Cross-Sectional Circumferences:**  
  Cuts dynamic 3D slicing planes through the volumetric surface at the Chest, Waist, Hip, Left/Right Upper Arms, and Left/Right Thighs, calculating circumferences using Ramanujan elliptical integration and discrete polygonal contour accumulation.
- **Statistical Temporal Stability Analysis:**  
  Calculates rolling statistics (mean, median, standard deviation, coefficient of variation) for every circumference to quantify measurement consistency across frames.
- **Interactive Developer Validation Mode (`V` Key):**  
  Allows developers and evaluators to input real-world tape measurements, immediately computing absolute error ($\text{cm}$), percentage error ($\%$), and Mean Absolute Error (MAE), and exporting a structured comparison report.
- **Interactive Dual-Pane Visualization:**  
  Displays the webcam feed with an augmented HUD alongside a real-time 3D orbit camera viewport featuring zoom, rotation, slicing planes, and quality badges.

---

## 3. Real-World Use Cases

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AI-HBS-v2-AR USE CASES                          │
├──────────────────┬──────────────────┬──────────────────┬───────────────┤
│ Virtual Fitting  │ Fitness & Health │ Custom Tailoring │ Telehealth    │
│ & E-Commerce     │ Tracking         │ & Apparel        │ & Ergonomics  │
│                  │                  │                  │               │
│ • Size matching  │ • Body dimension │ • Bespoke cloth  │ • Posture     │
│ • Return rate    │   monitoring     │   measurements   │   tracking    │
│   reduction      │ • Circumference  │ • Automated      │ • Ergonomic   │
│ • Virtual sizing │   trends over    │   production     │   workstation │
│   profiles       │   workout blocks │   intake         │   evaluations │
└──────────────────┴──────────────────┴──────────────────┴───────────────┘
```

1. **Virtual Fashion, Sizing & E-Commerce:**
   - Online apparel shoppers experience up to $30\text{–}40\%$ return rates due to incorrect fit.
   - AI-HBS-v2-AR captures chest, waist, hip, and limb dimensions in seconds, enabling accurate size recommendations (S, M, L, XL or custom brand sizing charts) directly from a browser or desktop application.
2. **Fitness, Bodybuilding & Wellness Tracking:**
   - Tracks changes in muscular hypertrophy or fat loss across key areas (chest expansion, waist reduction, arm/thigh circumferences) without manual measuring tapes.
   - Temporal consistency scores confirm whether the user's posture is stable enough for reliable historical tracking.
3. **Custom Tailoring & Made-to-Measure Garments:**
   - Provides tailors and bespoke garment manufacturers with standardized anatomical segment lengths (shoulder span, sleeve length, inseam, outseam) and circumferences without requiring an in-person fitting session.
4. **Ergonomics & Occupational Health:**
   - Calculates segment proportions (torso height, upper arm length, lower leg height) to automatically configure office desk heights, chair ergonomics, and vehicle seat positioning.
5. **Telehealth & Remote Physical Therapy:**
   - Enables physical therapists to monitor patient body proportions, symmetry between left and right limbs, and posture changes during remote rehabilitation.
6. **Metaverse & Personalized 3D Avatar Generation:**
   - Serves as the volumetric body capture front-end for creating dimensionally accurate 3D meshes for VR, gaming engines, and virtual try-on avatars.

---

## 4. Technology Stack

### Languages & Core Runtimes
| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Programming Language** | Python | `3.10` – `3.13` | Core algorithmic and pipeline implementation |
| **Operating System** | Windows / Linux / macOS | 64-bit | Cross-platform desktop runtime |
| **Video & Camera Capture** | OpenCV (`opencv-python`) | `>= 4.8.0, < 5.0.0` | Real-time webcam acquisition, image processing, HUD rendering |
| **Pose & Segmentation AI** | Google MediaPipe | `>= 0.10.9, < 0.11.0` | `PoseLandmarker` (33 3D world joints) & `ImageSegmenter` |
| **Numerical Processing** | NumPy | `>= 1.24.0, < 2.0.0` | Vectorized 3D geometry, transformations, linear algebra |
| **Scientific Computing** | SciPy | `>= 1.11.0, < 1.15.0` | Elliptical integrals, numerical optimizations |
| **Hardware Acceleration** | TensorFlow Lite / XNNPACK | Built-in | CPU SIMD acceleration for real-time neural inference |

### Pretrained Neural Network Models
- **MediaPipe PoseLandmarker (`pose_landmarker_full.task`):**
  Dual-stage neural network (detector + tracker) predicting 33 2D normalized image landmarks and metric 3D world landmarks ($\mathbf{P} \in \mathbb{R}^3$, centered at the pelvis midpoint in meters).
- **MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`):**
  MobileNetV3-based semantic segmenter generating a foreground probability map for human silhouette isolation at $>60\text{ FPS}$.

---

## 5. System Architecture & Data Pipeline

```
                              ┌───────────────────────┐
                              │     Webcam Input      │
                              │ (1280x720 @ 30 FPS)   │
                              └──────────┬────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
      ┌───────────────────────────┐             ┌───────────────────────────┐
      │  2D Person Segmentation   │             │   MediaPipe Pose Engine   │
      │ (SelfieSegmenter .tflite) │             │ (PoseLandmarker .task)    │
      └─────────────┬─────────────┘             └─────────────┬─────────────┘
                    │                                         │
                    ▼                                         ▼
      ┌───────────────────────────┐             ┌───────────────────────────┐
      │  Binary Silhouette Mask   │             │   33 2D Image Landmarks   │
      │  + Temporal EMA Smoothing │             │   33 3D World Landmarks   │
      └─────────────┬─────────────┘             └─────────────┬─────────────┘
                    │                                         │
                    │      ┌──────────────────────────────────┴──────────────────┐
                    │      ▼                                                     ▼
                    │ ┌──────────────────────────┐             ┌───────────────────────────────┐
                    │ │ 12-Point Visibility Gating│             │  3D Segment Length Calculator │
                    │ └──────────────────────────┘             │  + 60-Frame Median Outlier    │
                    │                                          │    Rejection & Stabilization  │
                    │                                          └───────────────┬───────────────┘
                    │                                                          │
                    │                                                          ▼
                    │                                          ┌───────────────────────────────┐
                    │                                          │  data/results/                │
                    │                                          │  final_measurements.json      │
                    │                                          └───────────────────────────────┘
                    │                                                          │
                    ▼                                                          ▼
      ┌────────────────────────────────────────────────────────────────────────┴───────────────┐
      │                      Procedural Continuous Volumetric 3D Mesh                          │
      │                  (18 Anatomical Lofted Regions, Proportional Radii)                    │
      └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │
                                               ▼
      ┌────────────────────────────────────────────────────────────────────────────────────────┐
      │                      Phase 5: Silhouette-Constrained Mesh Refinement                   │
      │       • Bidirectional Normal Ray-Casting across 14 Independent Anatomical Zones        │
      │       • Landmark Joints Fixed in 3D Space (No Artificial Bone Movement)               │
      │       • Anatomical Bounds Clamping [0.75x, 1.35x] & Temporal Radii Smoothing           │
      │       • Automatic Graceful Fallback if Mask is Occluded or Unavailable                 │
      └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │
                                               ▼
      ┌────────────────────────────────────────────────────────────────────────────────────────┐
      │                   Dynamic Cross-Sectional Surface Slicing Planes                       │
      │               (Chest, Waist, Hip, Left/Right Upper Arms, Left/Right Thighs)             │
      │          Calculates Circumferences via Ramanujan Ellipses & Polygonal Contours         │
      └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
      ┌───────────────────────────┐                         ┌───────────────────────────┐
      │ Temporal Stability Engine │                         │ Developer Validation Mode │
      │  (Mean, StdDev, CV, Stab%)│                         │    (Key 'V': Real vs AI)  │
      └─────────────┬─────────────┘                         └─────────────┬─────────────┘
                    │                                                     │
                    ▼                                                     ▼
      ┌───────────────────────────┐                         ┌───────────────────────────┐
      │  data/results/            │                         │  data/results/            │
      │  body_measurements.json   │                         │  validation_results.json  │
      └───────────────────────────┘                         └───────────────────────────┘
```

---

## 6. Mathematical & Algorithmic Methodology

### 6.1 Metric Coordinate Scaling
MediaPipe world landmarks are expressed in meters relative to the hips' center point. The system maps coordinates to metric centimeters:
$$\mathbf{P}_{\text{cm}} = \mathbf{P}_{\text{world}} \times S_{\text{factor}}, \quad \text{where } S_{\text{factor}} = 100.0\text{ cm/m}$$

### 6.2 3D Segment Euclidean Length
For any body segment between proximal joint $A$ and distal joint $B$:
$$L_{AB} = \sqrt{(X_B - X_A)^2 + (Y_B - Y_A)^2 + (Z_B - Z_A)^2} \times C_{\text{factor}}$$
where $C_{\text{factor}}$ is the anthropometric correction factor defined in [`config/correction_factors.json`](file:///e:/AI-HBS-v2-AR/config/correction_factors.json).

### 6.3 Silhouette Ray-Casting & Metric Radius Scaling
For a given anatomical region connecting 2D projected joint coordinates $\mathbf{p}_A = (x_A, y_A)$ and $\mathbf{p}_B = (x_B, y_B)$:

1. **Segment Midpoint & Normal Unit Vector:**
   $$\mathbf{p}_{\text{mid}} = \frac{\mathbf{p}_A + \mathbf{p}_B}{2}$$
   $$\mathbf{v} = \mathbf{p}_B - \mathbf{p}_A, \quad \|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2}$$
   $$\hat{\mathbf{n}} = \left( -\frac{v_y}{\|\mathbf{v}\|},\, \frac{v_x}{\|\mathbf{v}\|} \right)$$

2. **Bidirectional Ray-Casting:**
   Rays are marched outwards from $\mathbf{p}_{\text{mid}}$ along $+\hat{\mathbf{n}}$ and $-\hat{\mathbf{n}}$ inside the binary segmentation mask $M(x, y)$ until hitting background ($0$):
   $$d_+ = \max \{ k \mid M(\mathbf{p}_{\text{mid}} + k \hat{\mathbf{n}}) = 255 \}, \quad d_- = \max \{ k \mid M(\mathbf{p}_{\text{mid}} - k \hat{\mathbf{n}}) = 255 \}$$
   $$\text{Observed Pixel Diameter } W_{px} = d_+ + d_-$$

3. **Metric Radius Conversion:**
   Using the segment's 3D length $L_{3d}$ (meters) and its 2D projected pixel span $L_{px}$:
   $$R_{\text{obs}} = \frac{W_{px}}{2} \times \left( \frac{L_{3d}}{L_{px}} \right)$$

4. **Biological Clamping & Temporal Smoothing:**
   To guarantee anatomical plausibility and filter noise:
   $$R_{\text{clamped}} = \text{clamp}\left( R_{\text{obs}},\, 0.75 \times R_{\text{proc}},\, 1.35 \times R_{\text{proc}} \right)$$
   $$R_t = \alpha \cdot R_{\text{clamped}} + (1 - \alpha) \cdot R_{t-1}, \quad \text{with } \alpha = 0.60$$

### 6.4 Cross-Sectional Circumference Calculation
At each slicing plane, the cross-section is represented as a refined ellipse where:
- Coronal semi-axis: $a = R_{\text{refined}}$ (measured directly from camera silhouette).
- Sagittal semi-axis: $b = a \times \rho_{\text{depth}}$ (anthropometric depth-to-width ratio: Chest $\rho \approx 0.70$, Waist $\rho \approx 0.68$, Hip $\rho \approx 0.72$, Limbs $\rho \approx 0.95$).

The perimeter is calculated using **Ramanujan's First Approximation**:
$$\mathcal{C} \approx \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right]$$
In addition, the 3D viewport computes the discrete polygonal loop circumference by summing Euclidean vertex distances across the sliced 32-vertex ring:
$$\mathcal{C}_{\text{poly}} = \sum_{i=1}^{N} \|\mathbf{v}_i - \mathbf{v}_{i-1}\|$$

### 6.5 Temporal Stability & Error Metrics
For a rolling buffer of $N = 40$ frames:
$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$
$$\text{Coefficient of Variation (CV)} = \frac{\sigma}{\mu}$$
$$\text{Temporal Stability Score} = \max\left(0,\, \min\left(100,\, (1 - 3 \times \text{CV}) \times 100\right)\right)\%$$

In Validation Mode, for manual ground-truth values $A_k$ and AI estimates $E_k$:
$$\text{Absolute Error}_k = |E_k - A_k|\text{ cm}, \quad \text{Percentage Error}_k = \frac{|E_k - A_k|}{A_k} \times 100\%$$
$$\text{Mean Absolute Error (MAE)} = \frac{1}{K}\sum_{k=1}^{K} |E_k - A_k|\text{ cm}$$

---

## 7. Module Breakdown

```
AI-HBS-v2-AR/
├── src/
│   ├── main.py                     # Entry point & application orchestrator
│   ├── body_segmentation.py        # MediaPipe Selfie Segmenter & binary mask generation
│   ├── mesh_refinement.py          # Regional silhouette ray-casting & 14-zone mesh fitting
│   ├── measurement_validation.py   # Statistical temporal stability & ground-truth validation engine
│   ├── body_measurements_3d.py     # Cross-sectional slicing planes & circumference estimation
│   ├── volumetric_body.py          # Continuous procedural 3D body mesh generator (18 regions)
│   ├── visualization_3d.py         # Real-time 3D camera viewport, cutting discs & quality HUD
│   ├── pose_detector.py            # MediaPipe Pose tracking & 12-point visibility validation
│   ├── measurements.py             # 3D Euclidean segment distances & correction factor logic
│   ├── calibration.py              # Camera parameter manager & metric conversion
│   ├── stabilizer.py               # 60-frame median buffer & outlier rejection filter
│   ├── ar_coordinate_mapper.py     # Phase 6.1: Real-time AR coordinate mapper (Normalized -> 2D pixel space)
│   ├── ar_measurement_overlay.py   # Phase 6.2: Real-time AR measurement lines, CAD ticks & anchored labels
│   └── visualization.py            # 2D camera HUD, skeleton rendering, progress bar & overlays
├── config/
│   ├── calibration.json            # Camera resolution, FPS, scale factor
│   └── correction_factors.json     # Segment anthropometric tuning coefficients
├── models/
│   ├── pose_landmarker_full.task   # MediaPipe Pose neural network bundle
│   └── selfie_segmenter.tflite      # MediaPipe Segmentation model
├── data/results/
│   ├── final_measurements.json     # Linear segment lengths
│   ├── body_measurements.json      # Circumferences, confidence, and stability stats
│   └── validation_results.json     # Ground-truth validation error table & MAE
├── requirements.txt                # Python environment specifications
├── run.bat                         # Windows quick-launch script
├── details.md                      # Complete technical specification (this file)
└── README.md                       # Quickstart and overview guide
```

---

## 8. Interactive Controls & HUD

### Keyboard Keybindings

| Key | Action | Description |
|:---:|:---|:---|
| **`B`** | **Toggle 2D Mask** | Toggles cyan/green silhouette overlay on webcam feed |
| **`C`** | **Toggle Circumference Planes** | Shows/hides cutting discs and metric callout cards |
| **`P`** | **Cycle Slicing Plane** | Cycles: `ALL` → `CHEST` → `WAIST` → `HIP` → `ARMS` → `THIGHS` |
| **`M`** | **Toggle 3D Viewport** | Expands/collapses side-by-side 3D volumetric viewer |
| **`S`** | **Toggle Skeleton** | Toggles 2D webcam skeleton & 3D inner anatomical rig |
| **`O`** | **Toggle AR Measurement Overlay** | Toggles Phase 6.1 & 6.2 AR dimension lines, labels, ticks & debug points |
| **`R`** | **Reset 3D Camera** | Restores default orbit yaw, pitch, zoom, and center |
| **`V`** | **Developer Validation** | Opens interactive terminal prompt to test ground-truth values |
| **`+` / `-`** | **Zoom In / Out** | Adjusts virtual 3D camera distance |
| **`A` / `D`** | **Orbit Yaw** | Rotates 3D camera left / right |
| **`W` / `X`** | **Orbit Pitch** | Tilts 3D camera up / down |
| **`Q` / `ESC`** | **Quit** | Exits application and writes results to `data/results/` |

### On-Screen HUD Indicators
- **Mesh Quality Status Badges:**
  - `Excellent`: Silhouette fully detected, $>95\%$ landmarks visible, low jitter.
  - `Good`: Normal stable tracking under standard room lighting.
  - `Poor`: Edge clipping, partial limb occlusion, or high motion blur.
  - `Unavailable`: Person not detected or segmentation disabled (triggers fallback).
- **Fallback Notice:**  
  If segmentation fails, the HUD displays `"Silhouette unavailable - using landmark model"` and uses the base procedural model without generating synthetic or random numbers.
- **Stability Indicator:**  
  Displays temporal consistency (e.g., `Chest: 82.1 cm | Stab: 98%`).

---

## 9. Data Outputs & File Schemas

### 9.1 Segment Lengths: [`data/results/final_measurements.json`](file:///e:/AI-HBS-v2-AR/data/results/final_measurements.json)
```json
{
  "shoulder_width": 38.45,
  "left_upper_arm": 28.12,
  "right_upper_arm": 28.05,
  "left_forearm": 24.30,
  "right_forearm": 24.15,
  "left_thigh": 41.20,
  "right_thigh": 41.10,
  "left_lower_leg": 39.50,
  "right_lower_leg": 39.40
}
```

### 9.2 Circumferences & Stability: [`data/results/body_measurements.json`](file:///e:/AI-HBS-v2-AR/data/results/body_measurements.json)
```json
{
  "measurement_method": "silhouette_refined_mesh",
  "mesh_quality": "Excellent",
  "chest_circumference_cm": 82.5,
  "chest_confidence": 0.88,
  "waist_circumference_cm": 74.2,
  "waist_confidence": 0.94,
  "hip_circumference_cm": 91.8,
  "hip_confidence": 0.94,
  "left_upper_arm_circumference_cm": 29.4,
  "left_upper_arm_confidence": 0.92,
  "right_upper_arm_circumference_cm": 29.3,
  "right_upper_arm_confidence": 0.92,
  "left_thigh_circumference_cm": 52.1,
  "left_thigh_confidence": 0.94,
  "right_thigh_circumference_cm": 52.0,
  "right_thigh_confidence": 0.94,
  "stability_metrics": {
    "chest": {
      "mean": 82.48,
      "median": 82.50,
      "std_dev": 0.32,
      "cv": 0.0039,
      "stability_pct": 98.8
    },
    "waist": {
      "mean": 74.19,
      "median": 74.20,
      "std_dev": 0.28,
      "cv": 0.0038,
      "stability_pct": 98.9
    }
  },
  "metadata": {
    "status": "Approximate 3D body measurement",
    "method": "Silhouette-constrained cross-sectional contour slicing on 3D volumetric surface",
    "disclaimer": "Single-camera procedural estimation. Temporal consistency does not equal real-world medical accuracy.",
    "units": "cm"
  }
}
```

### 9.3 Ground-Truth Validation: [`data/results/validation_results.json`](file:///e:/AI-HBS-v2-AR/data/results/validation_results.json)
```json
{
  "status": "Ground-Truth Validation Completed",
  "mean_absolute_error_cm": 1.25,
  "comparisons": [
    {
      "metric": "chest",
      "actual_cm": 84.0,
      "estimated_cm": 82.5,
      "absolute_error_cm": 1.5,
      "percentage_error_pct": 1.79
    },
    {
      "metric": "waist",
      "actual_cm": 75.0,
      "estimated_cm": 74.0,
      "absolute_error_cm": 1.0,
      "percentage_error_pct": 1.33
    }
  ],
  "disclaimer": "Ground-truth comparison for verification. Calibration factors were NOT automatically modified."
}
```

---

## 10. Assumptions, Limitations & Quality Standards

### Operating Assumptions
1. **Camera Distance & Framing:**  
   The subject stands approximately $1.8\text{ to }2.2\text{ meters}$ from the webcam, facing forward, with head to feet fully within the camera frame.
2. **Clothing:**  
   Fitted clothing yields the highest accuracy. Extremely loose, baggy, or draped apparel expands the 2D silhouette, causing larger cross-sectional estimations.
3. **Lighting & Contrast:**  
   Adequate indoor lighting with contrast between the subject and the background ensures crisp segmentation edges.

### Technical Limitations & Disclaimers
- **Monocular Depth Ambiguity:**  
  A standard single RGB camera cannot measure the depth (anterior-posterior dimension) directly. The system reconstructs cross-sections using coronal width from silhouette rays combined with anthropometric aspect ratios.
- **Temporal Stability $\neq$ Sensor Accuracy:**  
  High stability ($>95\%$) indicates temporal consistency across consecutive video frames, not absolute real-world correctness. Absolute error is verified via Validation Mode (`V`).
- **Non-Medical Disclaimer:**  
  This application is designed for fitness, apparel sizing, and computer vision research. It is **not** certified as a medical device.

---

## 11. Hardware Progression & Future Roadmap

```
Phase 1: 3D Skeleton & Stabilization (Completed)
   │
Phase 2: Segment Capsules (Completed)
   │
Phase 3: Continuous Volumetric Body Mesh (Completed)
   │
Phase 4: Planar Surface Slicing & Circumference (Completed)
   │
Phase 5: Silhouette-Constrained Mesh & Validation (Completed)
   │
Phase 5: Silhouette-Constrained Mesh & Validation (Completed)
   │
Phase 6.1: Real-Time AR Coordinate Mapping (Completed)
   │     • Normalized image landmarks mapped to dynamic camera pixel coordinates
   │     • 3D metric world space preserved intact for measurements
   │     • Standard ARPoint data model & real-time debug visualization
   │
Phase 6.2: Real-Time AR Measurement Lines & Labels (Completed)
   │     • CAD-style dimension lines with perpendicular end-cap ticks
   │     • Anchored glassmorphic metric labels with temporal display smoothing
   │     • Horizontal AR circumference indicators with live confidence & stability
   ▼
Phase 7: True Depth Sensor Integration (Hardware Evolution)
   │     • Time-of-Flight (ToF) / LiDAR / Intel RealSense
   │     • Full 360° point cloud slicing without depth assumptions
   ▼
Phase 8: Unity 3D / ARCore Mobile Deployment
         • Mobile AR measurement app for iOS / Android
         • 3D avatar export (.OBJ / .GLTF) for digital try-on
```

### Phase 6.2: Real-Time AR Measurement Lines & Labels

- **Status:** COMPLETED
- **Description:** "The system now renders body measurements directly on the live camera view using anatomically anchored AR measurement lines and labels. Segment measurements and circumference estimates are displayed using the existing stabilized measurement engine and mapped anatomical landmarks."

#### Core AR Visual Subsystems:
1. **CAD-Style Dimension Lines (`draw_segment_measurement`):**
   - Connects detected joint pairs (Shoulder width, Upper Arms, Forearms, Thighs, Lower Legs).
   - Draws perpendicular end-cap ticks (`├──────────────┤`) and endpoint node dots.
   - Visually distinct vibrant gold/amber color scheme separate from the cyan skeleton.
2. **Anchored Smoothed Metric Labels (`draw_measurement_label`):**
   - Outward perpendicular vector projection away from the torso center prevents body occlusion.
   - Lightweight visual temporal smoothing ($\alpha = 0.35$) ensures jitter-free motion without modifying underlying measurement statistics.
   - Screen edge clamping (`clamp_text_position`) prevents text clipping outside camera frame boundaries.
3. **Horizontal AR Circumference Indicators (`draw_circumference_measurement`):**
   - Renders AR-style horizontal span indicators (`<──────────────>`) anchored at Chest, Waist, and Hip.
   - Color-coded confidence status indicator (`●` Green for $\ge 80\%$, Orange/Yellow for lower).
   - Real-time temporal stability percentage badge (e.g., `Stab: 98%`).

### Phase 6.1: Real-Time AR Coordinate Mapping

- **Status:** COMPLETED
- **Description:** "The system now maps MediaPipe image-space landmarks to real-time camera pixel coordinates while preserving the existing metric 3D world-coordinate measurement pipeline. These mapped coordinates form the foundation for future AR measurement overlays."

#### Coordinate Space Distinction:
- **Image Space:**  
  2D camera coordinates $(x_{pixel}, y_{pixel})$ derived from normalized inputs $(x, y) \in [0.0, 1.0]$ via $x_{pixel} = \text{int}(x \times W)$, $y_{pixel} = \text{int}(y \times H)$ safely clamped to $[0, W-1] \times [0, H-1]$. Used strictly for rendering AR overlays, skeleton debug connections, and labels on the camera image without inverting the Y-axis.
- **World Space:**  
  3D metric coordinates $(x_{meter}, y_{meter}, z_{meter})$ from MediaPipe world landmarks preserved intact for anatomical Euclidean segment lengths, volumetric mesh generation, slicing, and future AR depth positioning.

#### Standard AR Point Structure (`ARPoint`):
```python
@dataclass
class ARPoint:
    name: str              # Anatomical landmark name (e.g. 'LEFT_SHOULDER')
    landmark_index: int    # MediaPipe joint index (0-32)
    pixel_x: int           # Mapped horizontal raster coordinate
    pixel_y: int           # Mapped vertical raster coordinate
    world_x: float         # 3D metric coordinate (meters)
    world_y: float         # 3D metric coordinate (meters)
    world_z: float         # 3D metric coordinate (meters)
    normalized_x: float    # Raw normalized coordinate [0, 1]
    normalized_y: float    # Raw normalized coordinate [0, 1]
    visibility: float      # Confidence score [0, 1]
    valid: bool            # Visibility & boundary validity flag
    short_label: str       # Short label (e.g. 'LS', 'RS', 'LE')
```

### Why Active Depth Sensing Will Enhance Accuracy
1. **Direct Sagittal Dimension Measurement:**  
   LiDAR or ToF depth sensors measure the physical distance to the subject's sternum, spine, abdomen, and lower back, capturing exact 3D cross-sectional contours without relying on statistical depth ratios.
2. **Cloth vs. Body Separation:**  
   Depth surface curvature analysis can identify where clothing drapes away from the body surface.
3. **Arbitrary Angles & Rotations:**  
   Multi-view or rotational depth fusion enables true 360° body scanning and millimetric precision.
