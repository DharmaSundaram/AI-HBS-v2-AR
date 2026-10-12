# AI-HBS-v2-AR (AI Human Body Measurement and AR)

Markerless AI human body measurement, continuous volumetric 3D body mesh, and **silhouette-constrained 3D mesh refinement** using a standard webcam and computer vision.

The system tracks full-body posture, extracts 33 standardized MediaPipe Pose landmarks, calculates 3D body segment dimensions from MediaPipe world coordinates, extracts the visible 2D body silhouette using MediaPipe Selfie Segmentation, **refines the 3D procedural body volume to conform to the person's actual 2D silhouette**, computes cross-sectional body circumferences with temporal stability metrics, and provides an interactive ground-truth validation engine.

---

## Project Structure

```
AI-HBS-v2-AR/
├── src/
│   ├── main.py                  # Main application loop, camera capture, HUD & hotkey controller
│   ├── body_segmentation.py     # Phase 5: BodySegmentation (MediaPipe Selfie Segmenter & binary mask)
│   ├── mesh_refinement.py       # Phase 5: MeshRefiner (silhouette ray-casting & 14-region mesh fitting)
│   ├── measurement_validation.py # Phase 5: MeasurementValidator (stability metrics & MAE validation)
│   ├── body_measurements_3d.py  # Phase 4: BodyMeasurements3D (contour slicing & circumferences)
│   ├── volumetric_body.py       # Phase 3: VolumetricBodyMesh (18 proportional regions, adaptive smoothing)
│   ├── visualization_3d.py      # Phase 4/5 Viewport3D: 3D camera, cutting planes, callouts, quality badges
│   ├── body_mesh.py             # Backwards-compatibility bridge linking previous phases
│   ├── pose_detector.py         # MediaPipe Pose initialization, landmark extraction & visibility checks
│   ├── measurements.py          # 3D body segment Euclidean distance calculations & correction factors
│   ├── calibration.py           # Camera configuration & metric scale factor manager (meters to cm)
│   ├── stabilizer.py            # Multi-frame measurement collector & median outlier rejection
│   ├── ar_coordinate_mapper.py  # Phase 6.1: ARCoordinateMapper (Normalized -> 2D camera pixel mapping)
│   ├── ar_measurement_overlay.py # Phase 6.2: ARMeasurementOverlay (AR dimension lines, CAD ticks & anchored labels)
│   └── visualization.py         # 2D skeleton rendering, dynamic HUD, progress bar & results table
├── config/
│   ├── calibration.json         # Camera configuration and scale factor settings
│   └── correction_factors.json  # Anthropometric segment adjustment ratios
├── models/
│   ├── pose_landmarker_full.task # MediaPipe PoseLandmarker model bundle
│   └── selfie_segmenter.tflite   # MediaPipe Selfie Segmenter model bundle
├── data/
│   └── results/
│       ├── final_measurements.json   # Linear body segment lengths (shoulder, arms, legs, etc.)
│       ├── body_measurements.json    # Silhouette-refined circumferences & stability metrics
│       └── validation_results.json   # Ground-truth validation error report (MAE)
├── requirements.txt             # Python dependencies (opencv-python, mediapipe, numpy, scipy)
├── run.bat                      # Windows interactive launcher
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules for environments and temporary files
```

---

## Pipeline Overview (Phase 5)

```
Webcam Frame
    │
    ├──► 2D Person Body Segmentation ──► Binary Silhouette Mask (Foreground / Background)
    │                                          │
    └──► MediaPipe Pose Estimation    ──► 2D Landmarks + 3D World Landmarks
                                               │
                                  Procedural Volumetric Mesh
                                               │
                                               ▼
                              Silhouette-Constrained Refinement (Ray-Casting)
                                               │
                                               ▼
                                 Refined 3D Body Mesh Surface
                                               │
                                               ▼
                              Cross-Sectional Slicing at Planes
                             (Chest, Waist, Hip, Arms, Thighs)
                                               │
                                               ▼
                             Circumferences + Stability Metrics (CV)
                                               │
                                               ▼
                          Ground-Truth Validation Mode (MAE Analysis)
```

---

## Running the Application

### Option A: Direct Command
Open PowerShell or Command Prompt in the project folder:
```powershell
python src/main.py
```

### Option B: Windows Launcher
Double-click [`run.bat`](file:///e:/AI-HBS-v2-AR/run.bat) in File Explorer.

---

## Interactive Controls

| Key | Action |
|---|---|
| **`B`** | **Toggle 2D Body Segmentation Mask** (renders color silhouette overlay on webcam feed) |
| **`C`** | **Toggle Circumference Slicing Planes** (shows/hides cutting discs & metric callouts) |
| **`P`** | **Cycle Active Slicing Plane** (`ALL` → `CHEST` → `WAIST` → `HIP` → `LEFT_ARM` → `RIGHT_ARM` → `LEFT_THIGH` → `RIGHT_THIGH`) |
| **`M`** | **Toggle 3D Volumetric Mesh** (expands into side-by-side dual-pane) |
| **`S`** | **Toggle Skeleton Overlay** (2D webcam skeleton + 3D inner skeleton) |
| **`O`** | **Toggle AR Measurement & Coordinate Overlay** (toggles CAD measurement lines, anchored metric badges & circumference indicators) |
| **`R`** | **Reset 3D Camera View** (restores default orbit yaw, pitch, distance, and pan) |
| **`V`** | **Enter Ground-Truth Validation Mode** (compares AI estimates against manual ground-truth) |
| **`+` / `-`** | **Zoom In / Out** (adjusts virtual 3D camera distance) |
| **`A` / `D`** | **Rotate Orbit Yaw** (turns 3D view left / right) |
| **`W` / `X`** | **Tilt Orbit Pitch** (tilts 3D view up / down) |
| **`Q`** or **`ESC`** | **Quit** and release camera |

---

## How Silhouette Refinement Works

1. **2D Silhouette Ray-Casting**:
   - For each of the 14 anatomical regions (Head, Neck, Shoulders, Chest, Waist, Pelvis, Arms, Thighs, Calves), the module projects the segment's axis into 2D camera coordinates.
   - At the slice position, it casts rays perpendicular to the segment axis into the binary segmentation mask until encountering background pixels.
   - The observed pixel span is converted to metric meters using the local depth ratio ($\text{meters per pixel} = \frac{L_{3d}}{L_{px}}$).
2. **Anatomical Continuity & Clamping**:
   - Landmark joint endpoints are preserved exactly (landmarks are never displaced).
   - Observed radii are clamped within biological boundaries ($0.75 \times$ to $1.35 \times$ of procedural model) to prevent clothing or background artifacts from distorting the body model.
3. **Temporal Jitter Filtering**:
   - Refined segment radii are smoothed across consecutive video frames ($\alpha = 0.60$).
4. **Fallback Mechanism**:
   - If segmentation is unavailable or confidence is low, the engine automatically falls back to the procedural landmark model and displays:  
     `"Silhouette unavailable - using landmark model"`.

---

## Measurement Output Files

### 1. `data/results/body_measurements.json`
Contains silhouette-refined circumferences, confidence scores, and statistical stability metrics:
```json
{
    "measurement_method": "silhouette_refined_mesh",
    "mesh_quality": "Excellent",
    "chest_circumference_cm": 98.4,
    "chest_confidence": 0.94,
    "waist_circumference_cm": 82.1,
    "waist_confidence": 0.91,
    "hip_circumference_cm": 96.5,
    "hip_confidence": 0.93,
    "left_upper_arm_circumference_cm": 31.2,
    "left_upper_arm_confidence": 0.88,
    "right_upper_arm_circumference_cm": 31.4,
    "right_upper_arm_confidence": 0.88,
    "left_thigh_circumference_cm": 54.0,
    "left_thigh_confidence": 0.90,
    "right_thigh_circumference_cm": 53.8,
    "right_thigh_confidence": 0.90,
    "stability_metrics": {
        "chest": {
            "mean": 98.42,
            "median": 98.40,
            "std_dev": 0.38,
            "cv": 0.0039,
            "stability_pct": 98.8
        }
    },
    "warnings": [],
    "metadata": {
        "status": "Approximate 3D body measurement",
        "method": "Silhouette-constrained cross-sectional contour slicing on 3D volumetric surface",
        "disclaimer": "Single-camera procedural estimation. Temporal consistency does not equal real-world medical accuracy.",
        "units": "cm"
    }
}
```

### 2. `data/results/validation_results.json`
Generated when pressing **`V`** to run developer ground-truth validation:
```json
{
    "status": "Ground-Truth Validation Completed",
    "mean_absolute_error_cm": 1.33,
    "comparisons": [
        {
            "metric": "chest",
            "actual_cm": 80.0,
            "estimated_cm": 78.5,
            "absolute_error_cm": 1.5,
            "percentage_error_pct": 1.88
        },
        {
            "metric": "waist",
            "actual_cm": 75.0,
            "estimated_cm": 74.0,
            "absolute_error_cm": 1.0,
            "percentage_error_pct": 1.33
        },
        {
            "metric": "hip",
            "actual_cm": 92.0,
            "estimated_cm": 90.5,
            "absolute_error_cm": 1.5,
            "percentage_error_pct": 1.63
        }
    ],
    "disclaimer": "Ground-truth comparison for verification. Calibration factors were NOT automatically modified."
}
```

---

## Phase 6.1: Real-Time AR Coordinate Mapping

- **Status:** COMPLETED
- **Description:** "The system now maps MediaPipe image-space landmarks to real-time camera pixel coordinates while preserving the existing metric 3D world-coordinate measurement pipeline. These mapped coordinates form the foundation for future AR measurement overlays."

### Coordinate Spaces:
- **Image Space:** 2D camera raster coordinates $(x_{pixel}, y_{pixel})$ clipped to $[0, W-1] \times [0, H-1]$ used directly for AR rendering, debug skeletons, and visual labels over the webcam frame.
- **World Space:** 3D metric coordinates $(x_{meter}, y_{meter}, z_{meter})$ preserved intact for anatomical Euclidean segment lengths, volumetric mesh generation, slicing, and future AR depth positioning.

### Visual Debug Mode & HUD:
- **Joint Markers:** Displays circles at all valid 2D coordinates with short anatomical tags (`LS`, `RS`, `LE`, `RE`, `LW`, `RW`, `LH`, `RH`, `LK`, `RK`, `LA`, `RA`, `N`).
- **Connected Skeleton:** Draws bone lines between valid joints using newly mapped 2D camera coordinates.
- **Real-Time Debug HUD:** Live panel displaying status (`ACTIVE`), current frame resolution (`W x H`), valid landmarks ratio (`X / 12`), average confidence/visibility score, and pipeline confirmation (`NORMALIZED -> PIXEL`).
- **Toggle Hotkey:** Press `[O]` to toggle AR coordinate mapping overlay on or off at runtime.

---

## Phase 6.2: Real-Time AR Measurement Lines & Labels

- **Status:** COMPLETED
- **Description:** "The system now renders body measurements directly on the live camera view using anatomically anchored AR measurement lines and labels. Segment measurements and circumference estimates are displayed using the existing stabilized measurement engine and mapped anatomical landmarks."

### Features & Styling:
- **CAD-Style Dimension Lines:** Thin amber dimension lines connecting joint pairs with perpendicular end-cap ticks (`├──────────────┤`).
- **Anchored Glassmorphic Badges:** Compact labels positioned via outward perpendicular vectors relative to the torso center to avoid occluding the body.
- **Visual Display Smoothing ($\alpha = 0.35$):** Lightweight temporal smoothing applied strictly to visual label positions, eliminating jitter while moving.
- **Horizontal Circumference Indicators:** AR-style `<──────────────>` span indicators anchored at Chest, Waist, and Hip with real-time confidence indicators (`●`) and temporal stability percentages.
- **Unified Hotkey `[O]`:** Press `[O]` to toggle the entire AR measurement and coordinate overlay layer.

---

## Limitations & Depth Sensing Recommendations

1. **Single RGB Camera Limitations**:
   - The 2D silhouette constrains width in the camera plane. Depth (anterior-posterior thickness) continues to rely on anatomical aspect ratios rather than measured physical depth.
   - Baggy garments expand the perceived 2D silhouette.
2. **Why Depth Sensing (RGB-D / LiDAR) Improves Accuracy**:
   - True 3D depth sensors (Intel RealSense, Azure Kinect, or iPhone LiDAR) directly capture physical point clouds $(X, Y, Z)$ of the front and side surfaces, eliminating the need for elliptical cross-section assumptions.
