"""
visualization_3d.py
-------------------
Module: 3D Viewport & Silhouette-Constrained Mesh Controller (Phase 5)

Purpose:
Manages the virtual 3D camera (orbit yaw/pitch, zoom, pan, view reset),
renders the silhouette-constrained 3D volumetric body mesh, ground grid, and anatomical slicing planes.
Displays real-time circumference contours, temporal stability percentages, and quality metrics.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from volumetric_body import VolumetricBodyMesh
from body_measurements_3d import BodyMeasurements3D, MeasurementPlane
from mesh_refinement import MeshRefiner
from measurement_validation import MeasurementValidator


class Viewport3D:
    """
    Renders the silhouette-constrained 3D volumetric body mesh, active anatomical slicing planes,
    and real-time circumference measurement callouts with temporal stability scores.
    """

    def __init__(self) -> None:
        # Visibility and Toggle States
        self.is_enabled: bool = False         # Toggled via 'M' key (show/hide 3D view)
        self.show_mesh: bool = True           # Toggled via 'M' key (show/hide volumetric body)
        self.show_skeleton: bool = True       # Toggled via 'S' key (show/hide 3D skeleton)
        self.show_landmarks: bool = True      # Key joint nodes

        # Default Virtual Camera Parameters
        self.DEFAULT_YAW: float = 25.0
        self.DEFAULT_PITCH: float = -6.0
        self.DEFAULT_DISTANCE: float = 2.4
        self.DEFAULT_PAN: Tuple[float, float] = (0.0, 0.0)

        self.camera_yaw: float = self.DEFAULT_YAW
        self.camera_pitch: float = self.DEFAULT_PITCH
        self.camera_distance: float = self.DEFAULT_DISTANCE
        self.pan_x: float = self.DEFAULT_PAN[0]
        self.pan_y: float = self.DEFAULT_PAN[1]
        self.focal_length: float = 380.0

        # Subsystems
        self.mesh_engine = VolumetricBodyMesh(smoothing_factor=0.55)
        self.mesh_refiner = MeshRefiner(smoothing_alpha=0.60)
        self.measurement_engine = BodyMeasurements3D()
        self.validator = MeasurementValidator(buffer_size=40)

        self.has_landmarks: bool = False
        self.refined_segments: List[Any] = []
        self.refined_radii: Dict[str, float] = {}

    def toggle(self) -> bool:
        """Toggles the 3D viewport display on/off."""
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def toggle_mesh(self) -> bool:
        """Toggles the volumetric mesh layer."""
        self.show_mesh = not self.show_mesh
        return self.show_mesh

    def toggle_skeleton(self) -> bool:
        """Toggles the inner skeleton layer."""
        self.show_skeleton = not self.show_skeleton
        return self.show_skeleton

    def toggle_circumference(self) -> bool:
        """Toggles circumference plane visualization (Key: 'C')."""
        return self.measurement_engine.toggle_visualization()

    def cycle_plane(self) -> str:
        """Cycles through active measurement planes (Key: 'P')."""
        return self.measurement_engine.cycle_plane()

    def reset_view(self) -> None:
        """Resets virtual 3D camera orientation, distance, and pan to defaults (Key: 'R')."""
        self.camera_yaw = self.DEFAULT_YAW
        self.camera_pitch = self.DEFAULT_PITCH
        self.camera_distance = self.DEFAULT_DISTANCE
        self.pan_x = self.DEFAULT_PAN[0]
        self.pan_y = self.DEFAULT_PAN[1]

    def rotate_camera(self, delta_yaw: float = 0.0, delta_pitch: float = 0.0) -> None:
        """Rotates camera orbit angles."""
        self.camera_yaw = (self.camera_yaw + delta_yaw) % 360.0
        self.camera_pitch = max(min(self.camera_pitch + delta_pitch, 60.0), -60.0)

    def zoom_camera(self, delta_distance: float) -> None:
        """Adjusts camera zoom distance (closer = zoom in, farther = zoom out)."""
        self.camera_distance = max(min(self.camera_distance + delta_distance, 5.0), 1.0)

    def pan_camera(self, delta_x: float, delta_y: float) -> None:
        """Adjusts 2D viewport pan offset."""
        self.pan_x += delta_x
        self.pan_y += delta_y

    def update(
        self,
        raw_world_landmarks: Optional[List[Any]],
        binary_mask: Optional[np.ndarray] = None,
        landmarks_2d: Optional[List[Any]] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
        mask_quality: float = 0.0,
    ) -> None:
        """
        Updates the volumetric mesh, fits to the 2D silhouette mask, slices cross-sections,
        and computes stability metrics.
        """
        if raw_world_landmarks and len(raw_world_landmarks) >= 33:
            self.has_landmarks = True
            self.mesh_engine.update(raw_world_landmarks)

            # 1. Refine mesh using silhouette constraint
            refined_segs, radii_map, _ = self.mesh_refiner.refine_mesh(
                volumetric_engine=self.mesh_engine,
                binary_mask=binary_mask,
                landmarks_2d=landmarks_2d,
                frame_shape=frame_shape,
                mask_quality=mask_quality,
            )
            self.refined_segments = refined_segs
            self.refined_radii = radii_map

            # 2. Slice cross-sections on silhouette-refined geometry
            planes = self.measurement_engine.compute_frame_circumferences(
                raw_world_landmarks, refined_radii=radii_map
            )

            # 3. Feed samples to stability validator
            for name, pl in planes.items():
                self.validator.add_sample(name, pl.circumference_cm)

        else:
            self.has_landmarks = False
            self.mesh_engine.reset_smoothing()
            self.mesh_refiner.reset()
            self.refined_segments.clear()

    def _project_3d_to_2d(
        self,
        point_3d: np.ndarray,
        center_3d: np.ndarray,
        cx: float,
        cy: float,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Projects a 3D coordinate (meters) to 2D screen coordinates using perspective projection.
        """
        p = point_3d - center_3d
        x = p[0]
        y = -p[1]  # Invert Y so up is positive
        z = p[2]

        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)

        # Yaw rotation (around Y axis)
        x_rot = x * math.cos(rad_yaw) + z * math.sin(rad_yaw)
        z_temp = -x * math.sin(rad_yaw) + z * math.cos(rad_yaw)

        # Pitch rotation (around X axis)
        y_rot = y * math.cos(rad_pitch) - z_temp * math.sin(rad_pitch)
        z_rot = y * math.sin(rad_pitch) + z_temp * math.cos(rad_pitch)

        depth = z_rot + self.camera_distance
        if depth <= 0.15:
            return None

        screen_x = int(cx + (x_rot / depth) * self.focal_length + self.pan_x)
        screen_y = int(cy - (y_rot / depth) * self.focal_length + self.pan_y)

        return screen_x, screen_y, depth

    def render_3d_canvas(self, width: int = 640, height: int = 720) -> cv2.Mat:
        """
        Renders the complete 3D scene (ground grid, silhouette-refined volumetric mesh,
        skeleton, landmarks, circumference slicing planes, callouts, and quality status).
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (16, 16, 20)  # Dark studio backdrop

        cx = width / 2.0 - 25.0
        cy = height / 2.0 + 35.0

        # Pivot around mid-hip center
        center_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.has_landmarks and self.mesh_engine.smoothed_landmarks:
            sm = self.mesh_engine.smoothed_landmarks
            if 23 in sm and 24 in sm:
                center_3d = (sm[23] + sm[24]) * 0.5

        # 1. 3D Isometric Ground Grid
        grid_y = 1.05
        grid_ticks = np.linspace(-1.2, 1.2, 9)
        grid_col = (35, 35, 45)
        axis_col = (60, 75, 100)

        for gx in grid_ticks:
            p1 = self._project_3d_to_2d(np.array([gx, grid_y, -1.2], dtype=np.float32), center_3d, cx, cy)
            p2 = self._project_3d_to_2d(np.array([gx, grid_y, 1.2], dtype=np.float32), center_3d, cx, cy)
            if p1 and p2:
                col = axis_col if abs(gx) < 1e-3 else grid_col
                cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), col, 1, cv2.LINE_AA)

        for gz in grid_ticks:
            p1 = self._project_3d_to_2d(np.array([-1.2, grid_y, gz], dtype=np.float32), center_3d, cx, cy)
            p2 = self._project_3d_to_2d(np.array([1.2, grid_y, gz], dtype=np.float32), center_3d, cx, cy)
            if p1 and p2:
                col = axis_col if abs(gz) < 1e-3 else grid_col
                cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), col, 1, cv2.LINE_AA)

        if self.has_landmarks:
            _, bones, joints = self.mesh_engine.generate_body_mesh()
            segments = self.refined_segments if self.refined_segments else self.mesh_engine.generate_body_mesh()[0]

            # 2. Render Volumetric Mesh Layer (Silhouette-Constrained)
            if self.show_mesh and segments:
                for seg in segments:
                    vertices, edges = seg.generate_mesh()
                    if len(vertices) == 0:
                        continue

                    projected: List[Optional[Tuple[int, int, float]]] = []
                    for v in vertices:
                        projected.append(self._project_3d_to_2d(v, center_3d, cx, cy))

                    for idx1, idx2 in edges:
                        pt1 = projected[idx1]
                        pt2 = projected[idx2]
                        if pt1 is not None and pt2 is not None:
                            avg_depth = (pt1[2] + pt2[2]) * 0.5
                            intensity = max(0.35, min(1.0, 2.6 / avg_depth))
                            edge_col = tuple(int(c * intensity) for c in seg.color)
                            cv2.line(canvas, (pt1[0], pt1[1]), (pt2[0], pt2[1]), edge_col, 1, cv2.LINE_AA)

            # 3. Render Inner Skeleton Bones Layer
            if self.show_skeleton and bones:
                for b_start, b_end in bones:
                    p1 = self._project_3d_to_2d(b_start, center_3d, cx, cy)
                    p2 = self._project_3d_to_2d(b_end, center_3d, cx, cy)
                    if p1 and p2:
                        cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), 2, cv2.LINE_AA)

            # 4. Render Landmark Joint Nodes Layer
            if self.show_landmarks and joints:
                for j_pt in joints:
                    proj = self._project_3d_to_2d(j_pt, center_3d, cx, cy)
                    if proj:
                        cv2.circle(canvas, (proj[0], proj[1]), 4, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (proj[0], proj[1]), 5, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Render Circumference Slicing Planes & Callouts
            if self.measurement_engine.show_circumference:
                mode = self.measurement_engine.current_plane_mode
                planes = self.measurement_engine.current_planes

                planes_to_render = []
                for p_key, p_obj in planes.items():
                    if mode == "ALL":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "CHEST" and p_key == "chest":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "WAIST" and p_key == "waist":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "HIP" and p_key == "hip":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "LEFT_ARM" and p_key == "left_upper_arm":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "RIGHT_ARM" and p_key == "right_upper_arm":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "LEFT_THIGH" and p_key == "left_thigh":
                        planes_to_render.append((p_key, p_obj))
                    elif mode == "RIGHT_THIGH" and p_key == "right_thigh":
                        planes_to_render.append((p_key, p_obj))

                for p_key, p_obj in planes_to_render:
                    contour_pts = p_obj.contour_points_3d
                    if len(contour_pts) < 3:
                        continue

                    proj_contour = []
                    for pt in contour_pts:
                        pr = self._project_3d_to_2d(pt, center_3d, cx, cy)
                        if pr:
                            proj_contour.append((pr[0], pr[1]))

                    if len(proj_contour) >= 3:
                        poly_pts = np.array(proj_contour, dtype=np.int32)

                        # Draw glowing plane contour loop
                        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                        # Cutting plane fill
                        overlay = canvas.copy()
                        cv2.fillPoly(overlay, [poly_pts], color=(0, 180, 255))
                        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

                        # Slice center callout
                        center_pr = self._project_3d_to_2d(p_obj.center_3d, center_3d, cx, cy)
                        if center_pr:
                            c_x, c_y, _ = center_pr

                            val_cm, conf, is_rel = self.measurement_engine.get_smoothed_circumference(p_key)
                            stab_data = self.validator.get_stability_metrics(p_key)
                            stab_pct = stab_data["stability_pct"]

                            is_right_limb = "right" in p_key
                            card_x = c_x - 140 if is_right_limb else c_x + 65
                            card_y = c_y

                            cv2.line(canvas, (c_x, c_y), (card_x, card_y), (0, 220, 255), 1, cv2.LINE_AA)
                            cv2.circle(canvas, (c_x, c_y), 3, (0, 255, 255), -1)

                            short_label = p_obj.display_label.replace("Circumference", "").strip()
                            if is_rel:
                                text_line1 = f"{short_label}"
                                text_line2 = f"{val_cm:.1f} cm | Stab: {int(stab_pct)}%"
                                text_col = (0, 255, 255)
                            else:
                                text_line1 = f"{short_label}"
                                text_line2 = "UNRELIABLE"
                                text_col = (0, 140, 255)

                            cv2.putText(canvas, text_line1, (card_x, card_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
                            cv2.putText(canvas, text_line2, (card_x, card_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.39, text_col, 1, cv2.LINE_AA)

        else:
            cv2.putText(
                canvas,
                "Awaiting Full 3D Pose Landmarks...",
                (int(cx - 165), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (140, 140, 160),
                1,
                cv2.LINE_AA,
            )

        # 6. Header, Status, and Controls HUD
        cv2.putText(
            canvas,
            "AI-HBS SILHOUETTE-CONSTRAINED 3D MESH",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        # Status & Fallback Indicator
        if self.mesh_refiner.fallback_active:
            status_text = "Silhouette unavailable - using landmark model"
            status_col = (0, 140, 255)
        else:
            q_stat = self.mesh_refiner.quality_status
            status_text = f"Status: Approximate 3D body measurement | Quality: {q_stat}"
            if q_stat == "Excellent":
                status_col = (0, 255, 0)
            elif q_stat == "Good":
                status_col = (0, 255, 220)
            elif q_stat == "Poor":
                status_col = (0, 180, 255)
            else:
                status_col = (160, 160, 160)

        cv2.putText(
            canvas,
            status_text,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            status_col,
            1,
            cv2.LINE_AA,
        )

        # Active Plane Mode
        mode_str = self.measurement_engine.current_plane_mode
        circ_tag = "ON" if self.measurement_engine.show_circumference else "OFF"
        plane_status = f"Circumf: [{circ_tag}]  |  Plane: [{mode_str}] [P to cycle]"
        cv2.putText(
            canvas,
            plane_status,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Camera Status
        cam_line = f"Orbit: Yaw {int(self.camera_yaw)} deg | Pitch {int(self.camera_pitch)} deg | Dist {self.camera_distance:.1f}m"
        cv2.putText(
            canvas,
            cam_line,
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (140, 150, 165),
            1,
            cv2.LINE_AA,
        )

        # Interactive Controls Guide
        help_line = "[B] Mask  [C] Circumf  [P] Plane  [M] Mesh  [S] Skel  [R] Reset  [V] Valid"
        cv2.putText(
            canvas,
            help_line,
            (20, height - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        disclaimer = "Silhouette-Constrained 3D Approximation (Temporal Consistency != Real Accuracy)"
        cv2.putText(
            canvas,
            disclaimer,
            (20, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (100, 110, 125),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def draw(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Integrates the 3D viewport side-by-side with the webcam frame when enabled.
        """
        if not self.is_enabled:
            return frame

        h, w, _ = frame.shape
        mesh_canvas = self.render_3d_canvas(width=w, height=h)
        return np.hstack([frame, mesh_canvas])
