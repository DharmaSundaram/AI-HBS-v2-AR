"""
visualization.py
----------------
Module: Visual Overlays & HUD Display

Purpose:
Renders pose skeleton landmarks, connecting bones, capture progress bar,
dynamic status messages, and the final stabilized measurement results table.
Uses direct OpenCV primitives for zero dependency on deprecated drawing utilities.
"""

from typing import Optional, Dict, List, Any
import cv2
from pose_detector import POSE_CONNECTIONS


class Visualizer:
    """
    Renders skeleton connections, landmark points, and HUD status elements on frames.
    """

    def __init__(self) -> None:
        self.show_skeleton: bool = True

    def toggle_skeleton(self) -> bool:
        """Toggles 2D skeleton overlay on/off."""
        self.show_skeleton = not self.show_skeleton
        return self.show_skeleton

    def draw_skeleton(self, frame: cv2.Mat, landmarks: Optional[List[Any]]) -> None:
        """
        Draws 33 pose landmarks and anatomical bone connections onto frame.

        Args:
            frame: OpenCV BGR image.
            landmarks: List of normalized landmarks (x, y, z).
        """
        if not self.show_skeleton or not landmarks:
            return

        h, w, _ = frame.shape

        # 1. Draw connecting bones
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]

                # Convert normalized coords [0, 1] to pixel coords
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)

                # Bone line (Light Blue / Cyan in BGR)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 210, 30), 2, cv2.LINE_AA)

        # 2. Draw landmark joints
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            # Joint circle with border (Green center with white edge)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: cv2.Mat,
        status: str,
        progress: int,
        progress_total: int,
        is_complete: bool,
        final_measurements: Optional[Dict[str, float]] = None,
        is_3d_active: bool = False,
    ) -> None:
        """
        Draws comprehensive HUD: title, status indicator, progress bar, 3D mesh status, and measurement table.
        """
        h, w, _ = frame.shape

        # Semi-transparent top HUD banner
        banner_h = 135
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (480, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Title
        cv2.putText(
            frame,
            "AI-HBS AUTOMATIC BODY MEASUREMENT",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Dynamic Status Color
        if is_complete:
            status_color = (0, 255, 0)      # Green
        elif "Capturing" in status:
            status_color = (0, 255, 255)    # Yellow
        elif "not visible" in status:
            status_color = (0, 140, 255)    # Orange
        else:
            status_color = (200, 200, 200)  # Light Gray

        cv2.putText(
            frame,
            status,
            (25, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Progress text and bar
        if not is_complete:
            progress_text = f"Capture Progress: {progress}/{progress_total}"
            cv2.putText(
                frame,
                progress_text,
                (25, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Progress bar track & fill
            bar_x = 245
            bar_y = 93
            bar_w = 205
            bar_h = 13
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            ratio = min(max(progress / max(progress_total, 1), 0.0), 1.0)
            fill_w = int(bar_w * ratio)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        # 3D Mesh & Skeleton Status Line
        mesh_status_text = "3D: ON" if is_3d_active else "3D: OFF"
        skel_status_text = "Skel: ON" if self.show_skeleton else "Skel: OFF"
        mesh_status_col = (0, 255, 200) if is_3d_active else (160, 160, 160)
        cv2.putText(
            frame,
            f"[{mesh_status_text}] [{skel_status_text}]  [B] Mask  [C] Circ  [P] Plane  [M] 3D  [S] Skel  [O] AR  [V] Valid  [Q] Quit",
            (25, 127),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            mesh_status_col,
            1,
            cv2.LINE_AA,
        )

        # Final Measurements Table
        if is_complete and final_measurements:
            num_rows = len(final_measurements)
            panel_y_start = 140
            panel_h = num_rows * 26 + 45
            overlay_results = frame.copy()
            cv2.rectangle(overlay_results, (15, panel_y_start), (450, panel_y_start + panel_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_results, 0.78, frame, 0.22, 0, frame)

            cv2.putText(
                frame,
                "FINAL BODY MEASUREMENTS & CIRCUMFERENCE",
                (25, panel_y_start + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y = panel_y_start + 50
            for name, value in final_measurements.items():
                label = f"{name.replace('_', ' ').title():22}"
                val_text = f"{value:.2f} cm"
                cv2.putText(
                    frame,
                    label,
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    val_text,
                    (290, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
                y += 26
