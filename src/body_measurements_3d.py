"""
body_measurements_3d.py
-----------------------
Module: Body Surface & Circumference Measurement Foundation (Phase 4)

Purpose:
Performs cross-sectional slicing of the procedural volumetric 3D body mesh at
standard anatomical measurement levels to calculate perimeters and estimated circumferences:
- Chest Circumference
- Waist Circumference
- Hip Circumference
- Left & Right Upper-Arm Circumferences
- Left & Right Thigh Circumferences

Features:
- Analytical slicing of procedural tapered volumetric geometry at anatomical planes
- Contour perimeter integration (millimeter-level polygonal contour resolution)
- Multi-frame temporal smoothing to prevent fluctuating circumference values
- Confidence score calculation based on landmark visibility, stability, and geometric validity
- Unreliable measurement detection and user warnings
- Export to data/results/body_measurements.json
- Non-medical procedural surface estimation disclaimer
"""

import os
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from pose_detector import PoseLandmark


class MeasurementPlane:
    """Represents an anatomical slicing plane and its extracted cross-sectional contour."""

    def __init__(
        self,
        name: str,
        display_label: str,
        center_3d: np.ndarray,
        normal_3d: np.ndarray,
        contour_points_3d: np.ndarray,
        perimeter_meters: float,
        confidence: float,
        color: Tuple[int, int, int] = (0, 220, 255),
    ) -> None:
        self.name = name
        self.display_label = display_label
        self.center_3d = center_3d
        self.normal_3d = normal_3d
        self.contour_points_3d = contour_points_3d
        self.perimeter_meters = perimeter_meters
        self.confidence = confidence
        self.color = color

    @property
    def circumference_cm(self) -> float:
        """Returns perimeter in centimeters."""
        return self.perimeter_meters * 100.0


class BodyMeasurements3D:
    """
    Slices the volumetric 3D body mesh at anatomical levels, computes contour perimeters,
    evaluates measurement confidence, and smooths readings across video frames.
    """

    PLANE_NAMES = [
        "ALL",
        "CHEST",
        "WAIST",
        "HIP",
        "LEFT_ARM",
        "RIGHT_ARM",
        "LEFT_THIGH",
        "RIGHT_THIGH",
    ]

    def __init__(
        self,
        history_window: int = 15,
        scale_factor: float = 100.0,
        results_path: str = "data/results/body_measurements.json",
    ) -> None:
        self.history_window = history_window
        self.scale_factor = scale_factor
        self.results_path = results_path

        # UI & Display States
        self.show_circumference: bool = True
        self.active_plane_idx: int = 0  # Index into PLANE_NAMES (0 = ALL)

        # Multi-frame measurement history buffers for temporal stabilization
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.confidence_history: Dict[str, List[float]] = defaultdict(list)

        # Cached latest sliced planes
        self.current_planes: Dict[str, MeasurementPlane] = {}
        self.is_finalized: bool = False
        self.final_measurements: Dict[str, Any] = {}

        # Anatomical Plausibility Ranges in cm (for validity checking)
        self.bounds_cm: Dict[str, Tuple[float, float]] = {
            "chest": (55.0, 160.0),
            "waist": (45.0, 150.0),
            "hip": (60.0, 170.0),
            "left_upper_arm": (16.0, 55.0),
            "right_upper_arm": (16.0, 55.0),
            "left_thigh": (28.0, 85.0),
            "right_thigh": (28.0, 85.0),
        }

    def toggle_visualization(self) -> bool:
        """Toggles circumference plane visualization on/off (Key: 'C')."""
        self.show_circumference = not self.show_circumference
        return self.show_circumference

    def cycle_plane(self) -> str:
        """Cycles through active measurement planes (Key: 'P')."""
        self.active_plane_idx = (self.active_plane_idx + 1) % len(self.PLANE_NAMES)
        return self.PLANE_NAMES[self.active_plane_idx]

    @property
    def current_plane_mode(self) -> str:
        """Returns the currently active measurement plane filter name."""
        return self.PLANE_NAMES[self.active_plane_idx]

    @staticmethod
    def _slice_segment_contour(
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        radius_start: float,
        radius_end: float,
        t_pos: float,
        is_elliptical: bool = False,
        aspect_ratio: float = 1.0,
        num_points: int = 36,
    ) -> Tuple[np.ndarray, float]:
        """
        Analytically intersects a tapered 3D segment at parametric position t_pos in [0, 1]
        with a perpendicular slicing plane, extracting the 3D contour and its perimeter in meters.

        Returns:
            contour_3d: NumPy array of shape (N, 3) representing the planar slice contour.
            perimeter: Closed loop perimeter length in meters.
        """
        axis = end_pt - start_pt
        length = float(np.linalg.norm(axis))

        if length < 1e-4:
            return np.array([start_pt], dtype=np.float32), 0.0

        unit_axis = axis / length

        # Construct orthonormal basis perpendicular to segment axis
        arbitrary = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(unit_axis[2]) > 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        n1 = np.cross(unit_axis, arbitrary)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(unit_axis, n1)

        # Slice center and interpolated radius
        center = start_pt + t_pos * axis
        r = radius_start + t_pos * (radius_end - radius_start)

        thetas = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
        depth_scale = aspect_ratio if is_elliptical else 1.0

        contour = []
        for theta in thetas:
            dx = r * math.cos(theta)
            dy = r * math.sin(theta) * depth_scale
            pt = center + dx * n1 + dy * n2
            contour.append(pt)

        contour_arr = np.array(contour, dtype=np.float32)

        # Calculate exact polygon perimeter in 3D
        diffs = np.roll(contour_arr, -1, axis=0) - contour_arr
        segment_lengths = np.linalg.norm(diffs, axis=1)
        perimeter = float(np.sum(segment_lengths))

        return contour_arr, perimeter

    def compute_frame_circumferences(
        self,
        world_landmarks: Optional[List[Any]],
        refined_radii: Optional[Dict[str, float]] = None,
    ) -> Dict[str, MeasurementPlane]:
        """
        Computes 3D cross-sectional contours and perimeters for all 7 anatomical measurement planes
        using silhouette-refined segment dimensions when available.

        Args:
            world_landmarks: MediaPipe 3D world landmark collection.
            refined_radii: Optional map of silhouette-refined segment radii in meters.

        Returns:
            Dictionary mapping plane keys to MeasurementPlane objects.
        """
        if not world_landmarks or len(world_landmarks) < 33:
            return {}

        L = PoseLandmark
        w = world_landmarks

        # Helper to get 3D coords and landmark visibility
        def get_pt(idx: int) -> np.ndarray:
            return np.array([w[idx].x, w[idx].y, w[idx].z], dtype=np.float32)

        def get_vis(idx: int) -> float:
            v = getattr(w[idx], "visibility", 1.0)
            return float(v) if v is not None else 1.0

        # Extract landmarks
        l_sh = get_pt(L.LEFT_SHOULDER)
        r_sh = get_pt(L.RIGHT_SHOULDER)
        l_elb = get_pt(L.LEFT_ELBOW)
        r_elb = get_pt(L.RIGHT_ELBOW)
        l_wri = get_pt(L.LEFT_WRIST)
        r_wri = get_pt(L.RIGHT_WRIST)
        l_hip = get_pt(L.LEFT_HIP)
        r_hip = get_pt(L.RIGHT_HIP)
        l_knee = get_pt(L.LEFT_KNEE)
        r_knee = get_pt(L.RIGHT_KNEE)
        l_ank = get_pt(L.LEFT_ANKLE)
        r_ank = get_pt(L.RIGHT_ANKLE)

        # Proportions derived from detected geometry
        shoulder_w = float(np.linalg.norm(l_sh - r_sh))
        hip_w = float(np.linalg.norm(l_hip - r_hip))
        shoulder_mid = (l_sh + r_sh) * 0.5
        hip_mid = (l_hip + r_hip) * 0.5
        torso_vec = hip_mid - shoulder_mid
        torso_len = float(np.linalg.norm(torso_vec))

        # Clamped anatomical reference dimensions
        sh_w_clamped = max(0.28, min(shoulder_w, 0.58))
        hip_w_clamped = max(0.20, min(hip_w, 0.45))

        # Limb lengths
        l_arm_len = float(np.linalg.norm(l_elb - l_sh))
        r_arm_len = float(np.linalg.norm(r_elb - r_sh))
        l_thigh_len = float(np.linalg.norm(l_knee - l_hip))
        r_thigh_len = float(np.linalg.norm(r_knee - r_hip))

        # Baseline procedural radii in meters
        r_chest = sh_w_clamped * 0.38
        r_waist = (sh_w_clamped + hip_w_clamped) * 0.22
        r_pelvis = hip_w_clamped * 0.42

        l_arm_r_start = max(0.040, min(l_arm_len * 0.20, 0.065))
        l_arm_r_end = l_arm_r_start * 0.75
        r_arm_r_start = max(0.040, min(r_arm_len * 0.20, 0.065))
        r_arm_r_end = r_arm_r_start * 0.75

        l_thigh_r_start = max(0.060, min(l_thigh_len * 0.20, 0.095))
        l_thigh_r_end = l_thigh_r_start * 0.72
        r_thigh_r_start = max(0.060, min(r_thigh_len * 0.20, 0.095))
        r_thigh_r_end = r_thigh_r_start * 0.72

        # Apply silhouette-constrained refinement if available
        if refined_radii:
            if "chest" in refined_radii:
                r_chest = refined_radii["chest"]
            if "waist" in refined_radii:
                r_waist = refined_radii["waist"]
            if "pelvis" in refined_radii:
                r_pelvis = refined_radii["pelvis"]
            if "left_upper_arm" in refined_radii:
                l_arm_r_start = refined_radii["left_upper_arm"]
                l_arm_r_end = l_arm_r_start * 0.75
            if "right_upper_arm" in refined_radii:
                r_arm_r_start = refined_radii["right_upper_arm"]
                r_arm_r_end = r_arm_r_start * 0.75
            if "left_thigh" in refined_radii:
                l_thigh_r_start = refined_radii["left_thigh"]
                l_thigh_r_end = l_thigh_r_start * 0.72
            if "right_thigh" in refined_radii:
                r_thigh_r_start = refined_radii["right_thigh"]
                r_thigh_r_end = r_thigh_r_start * 0.72

        planes: Dict[str, MeasurementPlane] = {}

        # ---------------------------------------------------------
        # 1. CHEST PLANE (Upper torso cross-section ~28% down torso)
        # ---------------------------------------------------------
        chest_t = 0.28
        chest_center = shoulder_mid + torso_vec * chest_t
        chest_r = r_chest * 0.96
        chest_contour, chest_perim = self._slice_segment_contour(
            shoulder_mid, shoulder_mid + torso_vec * 0.50,
            chest_r, chest_r * 0.94, t_pos=0.55,
            is_elliptical=True, aspect_ratio=0.68, num_points=36
        )
        chest_vis = (get_vis(L.LEFT_SHOULDER) + get_vis(L.RIGHT_SHOULDER)) * 0.5
        chest_conf = self._calculate_confidence("chest", chest_perim * 100.0, chest_vis)
        planes["chest"] = MeasurementPlane(
            "chest", "Chest Circumference", chest_center,
            torso_vec / max(torso_len, 1e-4), chest_contour, chest_perim, chest_conf,
            color=(0, 255, 255)
        )

        # ---------------------------------------------------------
        # 2. WAIST PLANE (Narrow mid-torso region ~55% down torso)
        # ---------------------------------------------------------
        waist_t = 0.55
        waist_center = shoulder_mid + torso_vec * waist_t
        waist_contour, waist_perim = self._slice_segment_contour(
            shoulder_mid + torso_vec * 0.35, hip_mid,
            r_chest * 0.88, r_waist, t_pos=0.45,
            is_elliptical=True, aspect_ratio=0.65, num_points=36
        )
        waist_vis = (
            get_vis(L.LEFT_SHOULDER) + get_vis(L.RIGHT_SHOULDER) +
            get_vis(L.LEFT_HIP) + get_vis(L.RIGHT_HIP)
        ) * 0.25
        waist_conf = self._calculate_confidence("waist", waist_perim * 100.0, waist_vis)
        planes["waist"] = MeasurementPlane(
            "waist", "Waist Circumference", waist_center,
            torso_vec / max(torso_len, 1e-4), waist_contour, waist_perim, waist_conf,
            color=(0, 210, 255)
        )

        # ---------------------------------------------------------
        # 3. HIP PLANE (Pelvis region across hip joints)
        # ---------------------------------------------------------
        hip_center = hip_mid + torso_vec * 0.05
        hip_contour, hip_perim = self._slice_segment_contour(
            shoulder_mid + torso_vec * 0.60, hip_mid + torso_vec * 0.15,
            r_waist, r_pelvis, t_pos=0.85,
            is_elliptical=True, aspect_ratio=0.72, num_points=36
        )
        hip_vis = (get_vis(L.LEFT_HIP) + get_vis(L.RIGHT_HIP)) * 0.5
        hip_conf = self._calculate_confidence("hip", hip_perim * 100.0, hip_vis)
        planes["hip"] = MeasurementPlane(
            "hip", "Hip Circumference", hip_center,
            torso_vec / max(torso_len, 1e-4), hip_contour, hip_perim, hip_conf,
            color=(0, 180, 255)
        )

        # ---------------------------------------------------------
        # 4. LEFT UPPER ARM PLANE (Midpoint of shoulder & elbow)
        # ---------------------------------------------------------
        l_arm_center = (l_sh + l_elb) * 0.5
        l_arm_axis = l_elb - l_sh
        l_arm_contour, l_arm_perim = self._slice_segment_contour(
            l_sh, l_elb, l_arm_r_start, l_arm_r_end, t_pos=0.5,
            is_elliptical=False, num_points=28
        )
        l_arm_vis = (get_vis(L.LEFT_SHOULDER) + get_vis(L.LEFT_ELBOW)) * 0.5
        l_arm_conf = self._calculate_confidence("left_upper_arm", l_arm_perim * 100.0, l_arm_vis)
        planes["left_upper_arm"] = MeasurementPlane(
            "left_upper_arm", "Left Upper Arm", l_arm_center,
            l_arm_axis / max(l_arm_len, 1e-4), l_arm_contour, l_arm_perim, l_arm_conf,
            color=(50, 240, 255)
        )

        # ---------------------------------------------------------
        # 5. RIGHT UPPER ARM PLANE (Midpoint of shoulder & elbow)
        # ---------------------------------------------------------
        r_arm_center = (r_sh + r_elb) * 0.5
        r_arm_axis = r_elb - r_sh
        r_arm_contour, r_arm_perim = self._slice_segment_contour(
            r_sh, r_elb, r_arm_r_start, r_arm_r_end, t_pos=0.5,
            is_elliptical=False, num_points=28
        )
        r_arm_vis = (get_vis(L.RIGHT_SHOULDER) + get_vis(L.RIGHT_ELBOW)) * 0.5
        r_arm_conf = self._calculate_confidence("right_upper_arm", r_arm_perim * 100.0, r_arm_vis)
        planes["right_upper_arm"] = MeasurementPlane(
            "right_upper_arm", "Right Upper Arm", r_arm_center,
            r_arm_axis / max(r_arm_len, 1e-4), r_arm_contour, r_arm_perim, r_arm_conf,
            color=(50, 240, 255)
        )

        # ---------------------------------------------------------
        # 6. LEFT THIGH PLANE (Midpoint of hip & knee)
        # ---------------------------------------------------------
        l_thigh_center = (l_hip + l_knee) * 0.5
        l_thigh_axis = l_knee - l_hip
        l_thigh_contour, l_thigh_perim = self._slice_segment_contour(
            l_hip, l_knee, l_thigh_r_start, l_thigh_r_end, t_pos=0.5,
            is_elliptical=False, num_points=32
        )
        l_thigh_vis = (get_vis(L.LEFT_HIP) + get_vis(L.LEFT_KNEE)) * 0.5
        l_thigh_conf = self._calculate_confidence("left_thigh", l_thigh_perim * 100.0, l_thigh_vis)
        planes["left_thigh"] = MeasurementPlane(
            "left_thigh", "Left Thigh", l_thigh_center,
            l_thigh_axis / max(l_thigh_len, 1e-4), l_thigh_contour, l_thigh_perim, l_thigh_conf,
            color=(80, 255, 180)
        )

        # ---------------------------------------------------------
        # 7. RIGHT THIGH PLANE (Midpoint of hip & knee)
        # ---------------------------------------------------------
        r_thigh_center = (r_hip + r_knee) * 0.5
        r_thigh_axis = r_knee - r_hip
        r_thigh_contour, r_thigh_perim = self._slice_segment_contour(
            r_hip, r_knee, r_thigh_r_start, r_thigh_r_end, t_pos=0.5,
            is_elliptical=False, num_points=32
        )
        r_thigh_vis = (get_vis(L.RIGHT_HIP) + get_vis(L.RIGHT_KNEE)) * 0.5
        r_thigh_conf = self._calculate_confidence("right_thigh", r_thigh_perim * 100.0, r_thigh_vis)
        planes["right_thigh"] = MeasurementPlane(
            "right_thigh", "Right Thigh", r_thigh_center,
            r_thigh_axis / max(r_thigh_len, 1e-4), r_thigh_contour, r_thigh_perim, r_thigh_conf,
            color=(80, 255, 180)
        )

        self.current_planes = planes

        # Record history for multi-frame smoothing
        for name, pl in planes.items():
            self.history[name].append(pl.circumference_cm)
            self.confidence_history[name].append(pl.confidence)
            if len(self.history[name]) > self.history_window:
                self.history[name].pop(0)
            if len(self.confidence_history[name]) > self.history_window:
                self.confidence_history[name].pop(0)

        return planes

    def _calculate_confidence(self, name: str, value_cm: float, landmark_vis: float) -> float:
        """
        Calculates a confidence score [0.0, 1.0] for a cross-sectional measurement based on:
        - Landmark visibility
        - Geometric range plausibility
        - Multi-frame variance stability
        """
        # 1. Landmark visibility score (0 to 1)
        score_vis = min(max((landmark_vis - 0.40) / 0.50, 0.0), 1.0)

        # 2. Plausibility bounds score
        min_cm, max_cm = self.bounds_cm.get(name, (20.0, 180.0))
        if min_cm <= value_cm <= max_cm:
            score_range = 1.0
        elif value_cm < min_cm * 0.7 or value_cm > max_cm * 1.3:
            score_range = 0.0
        else:
            score_range = 0.5

        # 3. Stability score (low variance over history window)
        score_stab = 0.8
        if name in self.history and len(self.history[name]) >= 5:
            std_dev = float(np.std(self.history[name]))
            score_stab = max(0.0, min(1.0, 1.0 - (std_dev / 4.0)))

        # Weighted combination
        confidence = score_vis * 0.45 + score_stab * 0.35 + score_range * 0.20
        return round(float(np.clip(confidence, 0.0, 1.0)), 2)

    def get_smoothed_circumference(self, name: str) -> Tuple[float, float, bool]:
        """
        Returns the temporally smoothed circumference in cm and confidence score.

        Returns:
            (value_cm, confidence, is_reliable)
        """
        if name not in self.history or not self.history[name]:
            return 0.0, 0.0, False

        values = self.history[name]
        confs = self.confidence_history[name]

        # Use median to reject high-frequency frame anomalies
        median_val = float(np.median(values))
        filtered = [v for v in values if abs(v - median_val) < 4.0]

        smoothed_cm = float(np.mean(filtered)) if filtered else median_val
        smoothed_conf = float(np.mean(confs)) if confs else 0.0
        is_reliable = smoothed_conf >= 0.50

        return round(smoothed_cm, 1), round(smoothed_conf, 2), is_reliable

    def get_all_smoothed_measurements(self) -> Dict[str, Dict[str, Any]]:
        """Returns smoothed values and warnings for all anatomical planes."""
        results = {}
        for name in self.bounds_cm.keys():
            val_cm, conf, reliable = self.get_smoothed_circumference(name)
            warning = None if reliable else f"{name.replace('_', ' ').title()} measurement unavailable - improve body visibility."
            results[name] = {
                "circumference_cm": val_cm,
                "confidence": conf,
                "is_reliable": reliable,
                "warning": warning,
            }
        return results

    def finalize_and_save_results(
        self,
        mesh_quality: str = "Good",
        validator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generates and saves the final stabilized body circumference report
        to data/results/body_measurements.json.
        """
        smoothed = self.get_all_smoothed_measurements()

        stability_summary = {}
        if validator is not None:
            for name in self.bounds_cm.keys():
                stability_summary[name] = validator.get_stability_metrics(name)

        report: Dict[str, Any] = {
            "measurement_method": "silhouette_refined_mesh",
            "mesh_quality": mesh_quality,
            "chest_circumference_cm": smoothed["chest"]["circumference_cm"],
            "chest_confidence": smoothed["chest"]["confidence"],
            "waist_circumference_cm": smoothed["waist"]["circumference_cm"],
            "waist_confidence": smoothed["waist"]["confidence"],
            "hip_circumference_cm": smoothed["hip"]["circumference_cm"],
            "hip_confidence": smoothed["hip"]["confidence"],
            "left_upper_arm_circumference_cm": smoothed["left_upper_arm"]["circumference_cm"],
            "left_upper_arm_confidence": smoothed["left_upper_arm"]["confidence"],
            "right_upper_arm_circumference_cm": smoothed["right_upper_arm"]["circumference_cm"],
            "right_upper_arm_confidence": smoothed["right_upper_arm"]["confidence"],
            "left_thigh_circumference_cm": smoothed["left_thigh"]["circumference_cm"],
            "left_thigh_confidence": smoothed["left_thigh"]["confidence"],
            "right_thigh_circumference_cm": smoothed["right_thigh"]["circumference_cm"],
            "right_thigh_confidence": smoothed["right_thigh"]["confidence"],
            "stability_metrics": stability_summary,
            "warnings": [
                info["warning"] for info in smoothed.values() if info["warning"] is not None
            ],
            "metadata": {
                "status": "Approximate 3D body measurement",
                "method": "Silhouette-constrained cross-sectional contour slicing on 3D volumetric surface",
                "disclaimer": "Single-camera procedural estimation. Temporal consistency does not equal real-world medical accuracy.",
                "units": "cm",
            },
        }

        # Write to results directory
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        self.final_measurements = report
        self.is_finalized = True
        return report
