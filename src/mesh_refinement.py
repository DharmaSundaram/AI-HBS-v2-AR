"""
mesh_refinement.py
------------------
Module: Body Silhouette-Constrained 3D Mesh Refinement (Phase 5)

Purpose:
Refines procedural 3D volumetric body mesh dimensions by fitting them to the
actual visible 2D body silhouette extracted from camera segmentation.

Pipeline:
Camera → Person Segmentation → 2D Silhouette Mask + Pose Landmarks
       → Procedural Mesh → Silhouette-Constrained Refined Mesh → Accurate Circumference

Features:
- Independent ray-casting measurement of silhouette width across 14 distinct anatomical regions
- Preserves actual 3D MediaPipe landmark joint positions (no artificial displacement)
- Anatomical continuity guarantees (smooth tapering, no holes or sudden radius jumps)
- Temporal smoothing of refined segment radii across frames to eliminate jitter
- Comprehensive mesh quality status: 'Excellent', 'Good', 'Poor', 'Unavailable'
- Automatic graceful fallback to procedural landmark model if segmentation is unavailable
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from pose_detector import PoseLandmark
from volumetric_body import VolumetricSegment, VolumetricBodyMesh


class MeshRefiner:
    """
    Constrains procedural 3D volumetric body segments to match the detected 2D camera silhouette.
    """

    REGIONS = [
        "head",
        "neck",
        "shoulders",
        "chest",
        "waist",
        "pelvis",
        "left_upper_arm",
        "right_upper_arm",
        "left_forearm",
        "right_forearm",
        "left_thigh",
        "right_thigh",
        "left_lower_leg",
        "right_lower_leg",
    ]

    def __init__(self, smoothing_alpha: float = 0.60) -> None:
        """
        Initializes the mesh refiner.

        Args:
            smoothing_alpha: Temporal smoothing weight for refined radii [0.1, 0.9].
        """
        self.smoothing_alpha = smoothing_alpha
        self.refined_radii: Dict[str, float] = {}
        self.quality_status: str = "Unavailable"
        self.quality_score: float = 0.0
        self.fallback_active: bool = False

    def reset(self) -> None:
        """Resets smoothed radii buffers."""
        self.refined_radii.clear()
        self.quality_status = "Unavailable"
        self.quality_score = 0.0

    @staticmethod
    def _measure_silhouette_width_pixels(
        binary_mask: np.ndarray,
        center_px: Tuple[float, float],
        normal_unit: Tuple[float, float],
        max_search_px: int = 150,
    ) -> float:
        """
        Raycasts along the normal axis inside the binary mask from a center point
        to measure the visible 2D silhouette thickness.

        Returns:
            Width in pixels.
        """
        h, w = binary_mask.shape
        cx, cy = center_px
        nx, ny = normal_unit

        if not (0 <= int(cx) < w and 0 <= int(cy) < h):
            return 0.0

        # Positive raycast
        d_pos = 0.0
        for step in range(1, max_search_px):
            px = int(cx + step * nx)
            py = int(cy + step * ny)
            if 0 <= px < w and 0 <= py < h:
                if binary_mask[py, px] > 0:
                    d_pos = float(step)
                else:
                    break
            else:
                break

        # Negative raycast
        d_neg = 0.0
        for step in range(1, max_search_px):
            px = int(cx - step * nx)
            py = int(cy - step * ny)
            if 0 <= px < w and 0 <= py < h:
                if binary_mask[py, px] > 0:
                    d_neg = float(step)
                else:
                    break
            else:
                break

        return d_pos + d_neg

    def refine_mesh(
        self,
        volumetric_engine: VolumetricBodyMesh,
        binary_mask: Optional[np.ndarray],
        landmarks_2d: Optional[List[Any]],
        frame_shape: Tuple[int, int],
        mask_quality: float = 0.0,
    ) -> Tuple[List[VolumetricSegment], Dict[str, float], str]:
        """
        Fits volumetric body segments to the observed 2D silhouette mask.

        Args:
            volumetric_engine: Active VolumetricBodyMesh instance.
            binary_mask: Binary segmentation mask (255 foreground), or None.
            landmarks_2d: 2D normalized MediaPipe landmarks.
            frame_shape: (height, width) of camera image.
            mask_quality: Metric [0.0, 1.0] from segmentation module.

        Returns:
            refined_segments: List of constrained VolumetricSegment objects.
            radii_map: Map of region names to refined metric radii in meters.
            status_text: Quality status ('Excellent', 'Good', 'Poor', 'Unavailable').
        """
        # Base procedural segments, bones, and joints
        procedural_segments, bones, joints = volumetric_engine.generate_body_mesh()

        if not procedural_segments:
            self.quality_status = "Unavailable"
            return [], {}, self.quality_status

        # If mask is unavailable or quality too low, fall back gracefully
        if binary_mask is None or landmarks_2d is None or len(landmarks_2d) < 33 or mask_quality < 0.35:
            self.fallback_active = True
            self.quality_status = "Silhouette unavailable - using landmark model"
            # Return procedural baseline
            radii_map = {seg.name.lower().replace(" ", "_"): seg.radius_start for seg in procedural_segments}
            return procedural_segments, radii_map, self.quality_status

        h, w = frame_shape
        L = PoseLandmark

        # 2D pixel coordinate helper
        def get_2d(idx: int) -> Tuple[float, float]:
            return (landmarks_2d[idx].x * w, landmarks_2d[idx].y * h)

        # Region definitions mapping to corresponding 2D landmark pairs
        region_endpoints_2d = {
            "head": (get_2d(L.NOSE), (get_2d(L.LEFT_EYE)[0] * 0.5 + get_2d(L.RIGHT_EYE)[0] * 0.5, get_2d(L.NOSE)[1] - 30)),
            "neck": (((get_2d(L.LEFT_SHOULDER)[0] + get_2d(L.RIGHT_SHOULDER)[0]) * 0.5, (get_2d(L.LEFT_SHOULDER)[1] + get_2d(L.RIGHT_SHOULDER)[1]) * 0.5), get_2d(L.NOSE)),
            "shoulders": (get_2d(L.LEFT_SHOULDER), get_2d(L.RIGHT_SHOULDER)),
            "chest": (((get_2d(L.LEFT_SHOULDER)[0] + get_2d(L.RIGHT_SHOULDER)[0]) * 0.5, (get_2d(L.LEFT_SHOULDER)[1] + get_2d(L.RIGHT_SHOULDER)[1]) * 0.5), ((get_2d(L.LEFT_HIP)[0] + get_2d(L.RIGHT_HIP)[0]) * 0.5, (get_2d(L.LEFT_HIP)[1] + get_2d(L.RIGHT_HIP)[1]) * 0.5)),
            "waist": (((get_2d(L.LEFT_SHOULDER)[0] + get_2d(L.RIGHT_SHOULDER)[0]) * 0.5, (get_2d(L.LEFT_SHOULDER)[1] + get_2d(L.RIGHT_SHOULDER)[1]) * 0.5), ((get_2d(L.LEFT_HIP)[0] + get_2d(L.RIGHT_HIP)[0]) * 0.5, (get_2d(L.LEFT_HIP)[1] + get_2d(L.RIGHT_HIP)[1]) * 0.5)),
            "pelvis": (get_2d(L.LEFT_HIP), get_2d(L.RIGHT_HIP)),
            "left_upper_arm": (get_2d(L.LEFT_SHOULDER), get_2d(L.LEFT_ELBOW)),
            "right_upper_arm": (get_2d(L.RIGHT_SHOULDER), get_2d(L.RIGHT_ELBOW)),
            "left_forearm": (get_2d(L.LEFT_ELBOW), get_2d(L.LEFT_WRIST)),
            "right_forearm": (get_2d(L.RIGHT_ELBOW), get_2d(L.RIGHT_WRIST)),
            "left_thigh": (get_2d(L.LEFT_HIP), get_2d(L.LEFT_KNEE)),
            "right_thigh": (get_2d(L.RIGHT_HIP), get_2d(L.RIGHT_KNEE)),
            "left_lower_leg": (get_2d(L.LEFT_KNEE), get_2d(L.LEFT_ANKLE)),
            "right_lower_leg": (get_2d(L.RIGHT_KNEE), get_2d(L.RIGHT_ANKLE)),
        }

        # Refine segment radii independently using silhouette ray-casting
        current_refined_radii: Dict[str, float] = {}

        for seg in procedural_segments:
            reg_key = seg.name.lower().replace(" ", "_")
            r_proc = (seg.radius_start + seg.radius_end) * 0.5

            if reg_key in region_endpoints_2d:
                pt_a, pt_b = region_endpoints_2d[reg_key]
                dx = pt_b[0] - pt_a[0]
                dy = pt_b[1] - pt_a[1]
                seg_px_len = math.hypot(dx, dy)

                if seg_px_len > 10:
                    # Segment perpendicular normal in 2D image plane
                    ux = dx / seg_px_len
                    uy = dy / seg_px_len
                    nx = -uy
                    ny = ux

                    # Midpoint in 2D
                    mx = (pt_a[0] + pt_b[0]) * 0.5
                    my = (pt_a[1] + pt_b[1]) * 0.5

                    # Special midpoint placement for waist vs chest
                    if reg_key == "waist":
                        mx = pt_a[0] + dx * 0.55
                        my = pt_a[1] + dy * 0.55
                    elif reg_key == "chest":
                        mx = pt_a[0] + dx * 0.25
                        my = pt_a[1] + dy * 0.25

                    # Measure silhouette width in pixels
                    w_px = self._measure_silhouette_width_pixels(binary_mask, (mx, my), (nx, ny))

                    # 3D metric length of the same segment
                    axis_3d = seg.end_pt - seg.start_pt
                    seg_3d_len = float(np.linalg.norm(axis_3d))
                    meters_per_px = seg_3d_len / max(seg_px_len, 1e-4)

                    r_observed = (w_px * meters_per_px) * 0.5

                    # Anatomical constraint: clamp observed radius within [0.75, 1.35] of procedural
                    if r_observed > 0.01:
                        r_clamped = max(r_proc * 0.75, min(r_observed, r_proc * 1.35))
                    else:
                        r_clamped = r_proc
                else:
                    r_clamped = r_proc
            else:
                r_clamped = r_proc

            # Multi-frame temporal smoothing
            if reg_key in self.refined_radii:
                r_smoothed = self.smoothing_alpha * r_clamped + (1.0 - self.smoothing_alpha) * self.refined_radii[reg_key]
            else:
                r_smoothed = r_clamped

            self.refined_radii[reg_key] = r_smoothed
            current_refined_radii[reg_key] = r_smoothed

        # Construct refined segments maintaining continuity & taper ratios
        refined_segments: List[VolumetricSegment] = []
        for seg in procedural_segments:
            reg_key = seg.name.lower().replace(" ", "_")
            r_target = current_refined_radii.get(reg_key, seg.radius_start)

            # Preserve natural taper ratio
            taper_ratio = seg.radius_end / max(seg.radius_start, 1e-4)
            r_start_refined = r_target
            r_end_refined = r_target * taper_ratio

            refined_seg = VolumetricSegment(
                name=seg.name,
                start_pt=seg.start_pt,
                end_pt=seg.end_pt,
                radius_start=r_start_refined,
                radius_end=r_end_refined,
                color=seg.color,
                num_radial_segments=seg.num_radial_segments,
                num_rings=seg.num_rings,
                is_elliptical=seg.is_elliptical,
                aspect_ratio=seg.aspect_ratio,
            )
            refined_segments.append(refined_seg)

        # Evaluate overall quality score
        vis_scores = [getattr(lm, "visibility", 1.0) for lm in landmarks_2d]
        mean_vis = float(np.mean(vis_scores)) if vis_scores else 0.8

        total_quality = mask_quality * 0.50 + mean_vis * 0.50
        self.quality_score = total_quality

        if total_quality >= 0.85:
            self.quality_status = "Excellent"
        elif total_quality >= 0.68:
            self.quality_status = "Good"
        elif total_quality >= 0.45:
            self.quality_status = "Poor"
        else:
            self.quality_status = "Unavailable"

        self.fallback_active = False
        return refined_segments, current_refined_radii, self.quality_status
