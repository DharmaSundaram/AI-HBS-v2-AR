"""
main.py
-------
AI-HBS Automatic Stable Measurement & Silhouette-Constrained Mesh Engine (Phase 5)

Integrates:
- Camera input & 2D Person Body Segmentation (MediaPipe Selfie Segmenter)
- MediaPipe Pose detection & 3D world landmark extraction
- 12-point full-body visibility validation
- 3D anatomical Euclidean segment length calculation
- Multi-frame median outlier filtering & stabilization (60 frames)
- Live HUD with progress bar, layer status, quality indicator, and measurement tables
- Phase 3 Continuous Volumetric 3D Body Mesh
- Phase 4 Cross-Sectional Surface Slicing & Circumference Estimation
- Phase 5 Silhouette-Constrained Mesh Refinement (Independent regional ray-casting)
- Temporal stability percentage metrics (mean, median, std_dev, CV)
- Developer Ground-Truth Validation Mode (Key: 'V') with MAE analysis
- Interactive controls:
  - B = toggle 2D body segmentation mask overlay
  - C = toggle circumference visualization
  - P = cycle measurement planes (All, Chest, Waist, Hip, Arms, Thighs)
  - M = toggle 3D volumetric mesh view
  - S = toggle skeleton layer
  - R = reset 3D camera view
  - V = enter ground-truth validation mode
  - + / - = zoom 3D camera
  - A / D = rotate 3D orbit
  - Q = quit
- Export to:
  - data/results/final_measurements.json (lengths)
  - data/results/body_measurements.json (circumferences + confidence + stability)
  - data/results/validation_results.json (ground-truth validation report)
"""

import os
import sys
import json
import cv2

# Ensure src directory is in module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration import CalibrationManager
from pose_detector import PoseDetector
from measurements import BodyMeasurementCalculator
from stabilizer import MeasurementStabilizer
from visualization import Visualizer
from visualization_3d import Viewport3D
from body_segmentation import BodySegmentation


def print_banner() -> None:
    """Displays startup instructions in the console."""
    print("==================================================")
    print("AI-HBS SILHOUETTE-CONSTRAINED 3D MEASUREMENT ENGINE")
    print("==================================================")
    print("Stand in front of the camera.")
    print("Keep your full body visible.")
    print("Stand still while measurements are captured.")
    print("Controls:")
    print("  [B] Toggle 2D Body Segmentation Mask")
    print("  [C] Toggle Circumference Slicing Planes")
    print("  [P] Cycle Active Slicing Plane")
    print("  [M] Toggle 3D Volumetric Mesh View")
    print("  [S] Toggle Skeleton Layer")
    print("  [R] Reset 3D Camera View")
    print("  [V] Enter Ground-Truth Validation Mode")
    print("  [+ / -] Zoom In / Out")
    print("  [A / D] Orbit Camera Rotate")
    print("  [Q] or ESC: Quit")
    print("Notice: Approximate 3D body measurement (Non-Medical)")
    print("==================================================")


def print_results(final_lengths: dict, final_circs: dict) -> None:
    """Displays formatted results table in the console."""
    print()
    print("==============================================")
    print("FINAL BODY SEGMENT LENGTHS")
    print("==============================================")
    for name, value in final_lengths.items():
        print(f"{name.replace('_', ' ').title():28}: {value:.2f} cm")
    print("==============================================")
    print("FINAL BODY CIRCUMFERENCES (SILHOUETTE-REFINED)")
    print("==============================================")
    circ_keys = [
        "chest_circumference_cm",
        "waist_circumference_cm",
        "hip_circumference_cm",
        "left_upper_arm_circumference_cm",
        "right_upper_arm_circumference_cm",
        "left_thigh_circumference_cm",
        "right_thigh_circumference_cm",
    ]
    for k in circ_keys:
        val = final_circs.get(k, 0.0)
        conf_k = k.replace("_circumference_cm", "_confidence")
        conf = final_circs.get(conf_k, 0.0)
        label = k.replace("_circumference_cm", "").replace("_", " ").title() + " Circ"
        print(f"{label:28}: {val:.1f} cm  (Conf: {int(conf*100)}%)")
    print(f"Refinement Method: {final_circs.get('measurement_method', 'N/A')}")
    print(f"Mesh Quality:     {final_circs.get('mesh_quality', 'N/A')}")
    print("==============================================")
    print("Saved:")
    print("  - data/results/final_measurements.json")
    print("  - data/results/body_measurements.json")
    print("==============================================")


def run_developer_validation(viewport_3d: Viewport3D) -> None:
    """Prompts for manual measurements, calculates errors & MAE, and prints validation report."""
    print("\n" + "=" * 65)
    print("AI-HBS DEVELOPER VALIDATION MODE (Key: 'V')")
    print("==================================================")
    print("Enter manually measured real-world values (cm).")
    print("Press Enter to accept reference default values.")
    print("==================================================")

    estimates = {}
    for name in ["chest", "waist", "hip", "left_upper_arm", "left_thigh"]:
        val, _, _ = viewport_3d.measurement_engine.get_smoothed_circumference(name)
        estimates[name] = val

    actuals = {}
    prompts = [
        ("chest", "Actual Chest (cm)", 82.0),
        ("waist", "Actual Waist (cm)", 76.0),
        ("hip", "Actual Hip (cm)", 94.0),
        ("left_upper_arm", "Actual Upper Arm (cm)", 30.0),
        ("left_thigh", "Actual Thigh (cm)", 52.0),
    ]

    for key, label, default_val in prompts:
        est = estimates.get(key, 0.0)
        try:
            val_str = input(f"{label} [est: {est:.1f} cm, default: {default_val}]: ").strip()
            if val_str:
                actuals[key] = float(val_str)
            else:
                actuals[key] = default_val
        except (ValueError, EOFError):
            actuals[key] = default_val

    report = viewport_3d.validator.run_validation(actuals, estimates)
    viewport_3d.validator.print_validation_summary(report)


def main() -> None:
    print_banner()

    # 1. Load Configurations & Calibrations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")
    results_dir = os.path.join(base_dir, "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    calib_path = os.path.join(config_dir, "calibration.json")
    corrections_path = os.path.join(config_dir, "correction_factors.json")
    results_path = os.path.join(results_dir, "final_measurements.json")
    circ_results_path = os.path.join(results_dir, "body_measurements.json")
    valid_results_path = os.path.join(results_dir, "validation_results.json")

    calibrator = CalibrationManager(calib_path)
    calculator = BodyMeasurementCalculator(corrections_path)
    detector = PoseDetector(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    stabilizer = MeasurementStabilizer(frames_required=60, outlier_threshold=3.0)
    visualizer = Visualizer()
    viewport_3d = Viewport3D()
    viewport_3d.measurement_engine.results_path = circ_results_path
    viewport_3d.validator.results_path = valid_results_path

    segmenter = BodySegmentation()

    # 2. Camera Setup
    camera_cfg = calibrator.camera_settings
    camera_index = camera_cfg.get("device_index", 0)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video device at index {camera_index}.")
        print("Please check that your webcam is connected and accessible.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("height", 720))

    window_name = "AI-HBS Automatic Measurement"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    measurement_complete = False
    display_measurements = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Camera read error or frame dropped.")
                break

            # Mirror image for natural feedback
            frame = cv2.flip(frame, 1)

            # 1. 2D Person Body Segmentation
            binary_mask, mask_quality = segmenter.segment(frame)

            # 2. Pose Landmark Detection
            results = detector.process(frame)
            status = "Searching for body..."

            if detector.is_body_detected(results):
                landmarks = detector.get_landmarks(results)
                world_landmarks = detector.get_world_landmarks(results)

                # Synchronize 3D volumetric engine, silhouette refinement, and slicing
                viewport_3d.update(
                    raw_world_landmarks=world_landmarks,
                    binary_mask=binary_mask,
                    landmarks_2d=landmarks,
                    frame_shape=frame.shape[:2],
                    mask_quality=mask_quality,
                )

                # Check if all 12 key points are visible
                is_fully_visible, _ = detector.check_full_body_visibility(landmarks, min_visibility=0.60)

                if is_fully_visible and not measurement_complete:
                    status = "Stand still... Capturing measurements"

                    # Calculate single-frame 3D segment lengths in cm
                    frame_measurements = calculator.calculate_frame_measurements(
                        world_landmarks,
                        scale_factor=calibrator.scale_factor
                    )
                    stabilizer.add_frame_measurements(frame_measurements)

                    # Check if required frames reached
                    if stabilizer.is_ready():
                        status = "Measurement Complete!"
                        final_lengths = stabilizer.compute_final_measurements()
                        final_circs = viewport_3d.measurement_engine.finalize_and_save_results(
                            mesh_quality=viewport_3d.mesh_refiner.quality_status,
                            validator=viewport_3d.validator,
                        )
                        measurement_complete = True

                        # Save length results to disk
                        with open(results_path, "w", encoding="utf-8") as f:
                            json.dump(final_lengths, f, indent=4)

                        with open(os.path.join(base_dir, "final_measurements.json"), "w", encoding="utf-8") as f:
                            json.dump(final_lengths, f, indent=4)

                        # Combine lengths and main circumferences for on-screen HUD table
                        display_measurements = dict(final_lengths)
                        display_measurements["chest_circumference"] = final_circs.get("chest_circumference_cm", 0.0)
                        display_measurements["waist_circumference"] = final_circs.get("waist_circumference_cm", 0.0)
                        display_measurements["hip_circumference"] = final_circs.get("hip_circumference_cm", 0.0)

                        print_results(final_lengths, final_circs)

                elif measurement_complete:
                    status = "Measurement Complete - Press Q to exit"
                else:
                    status = "Full body not visible"

                # Draw 2D skeleton connections on webcam feed
                visualizer.draw_skeleton(frame, landmarks)
            else:
                viewport_3d.update(None)

            # Draw optional 2D segmentation mask overlay
            if segmenter.is_enabled:
                frame = segmenter.draw_mask_overlay(frame, binary_mask)

            # Get capture progress
            progress, progress_total = stabilizer.get_progress()

            # Render HUD elements on camera feed
            visualizer.draw_hud(
                frame=frame,
                status=status,
                progress=progress,
                progress_total=progress_total,
                is_complete=measurement_complete,
                final_measurements=display_measurements if measurement_complete else None,
                is_3d_active=viewport_3d.is_enabled,
            )

            # Render synchronized 3D dual-pane if enabled
            display_frame = viewport_3d.draw(frame)

            cv2.imshow(window_name, display_frame)

            # Interactive Keyboard Events
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            elif key == ord("b") or key == ord("B"):
                b_state = segmenter.toggle()
                print(f"[INFO] 2D Body Segmentation Mask: {'ENABLED' if b_state else 'DISABLED'}")
            elif key == ord("v") or key == ord("V"):
                run_developer_validation(viewport_3d)
            elif key == ord("c") or key == ord("C"):
                c_state = viewport_3d.toggle_circumference()
                print(f"[INFO] Circumference Visualization: {'ENABLED' if c_state else 'DISABLED'}")
            elif key == ord("p") or key == ord("P"):
                p_mode = viewport_3d.cycle_plane()
                print(f"[INFO] Active Slicing Plane: {p_mode}")
            elif key == ord("m") or key == ord("M"):
                is_on = viewport_3d.toggle()
                print(f"[INFO] 3D Volumetric Mesh: {'ENABLED' if is_on else 'DISABLED'}")
            elif key == ord("s") or key == ord("S"):
                skel_state_2d = visualizer.toggle_skeleton()
                skel_state_3d = viewport_3d.toggle_skeleton()
                print(f"[INFO] Skeleton Overlay: {'ENABLED' if skel_state_2d else 'DISABLED'}")
            elif key == ord("r") or key == ord("R"):
                viewport_3d.reset_view()
                print("[INFO] 3D Camera view reset to defaults.")
            elif key == ord("+") or key == ord("="):
                viewport_3d.zoom_camera(-0.25)
            elif key == ord("-") or key == ord("_"):
                viewport_3d.zoom_camera(+0.25)
            elif key == ord("a") or key == ord("A"):
                viewport_3d.rotate_camera(delta_yaw=-15.0)
            elif key == ord("d") or key == ord("D"):
                viewport_3d.rotate_camera(delta_yaw=+15.0)
            elif key == ord("w") or key == ord("W"):
                viewport_3d.rotate_camera(delta_pitch=+5.0)
            elif key == ord("x") or key == ord("X"):
                viewport_3d.rotate_camera(delta_pitch=-5.0)

    except KeyboardInterrupt:
        print("\n[INFO] Measurement session cancelled by user.")
    finally:
        segmenter.release()
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
        print("\nAI-HBS Automatic Measurement Engine Stopped.")


if __name__ == "__main__":
    main()
