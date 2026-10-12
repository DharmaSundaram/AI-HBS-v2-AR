"""
ar_coordinate_mapper.py
------------------------
Module: Real-Time AR Coordinate Mapping Engine (Phase 6.1)

Purpose:
Converts 3D MediaPipe pose landmarks and anatomical measurement coordinates
into accurate 2D camera-image pixel coordinates. Provides the mathematical
and geometric foundation for rendering future Augmented Reality (AR) measurement
lines, guidance cues, and metric labels directly over the user's body.

Coordinate System Architecture:
-------------------------------
1. NORMALIZED IMAGE COORDINATES:
   MediaPipe 2D image coordinates (x, y) normalized to [0.0, 1.0] relative to frame dimensions.
   (0.0, 0.0) is the top-left corner, (1.0, 1.0) is the bottom-right corner.

2. CAMERA PIXEL COORDINATES (IMAGE SPACE):
   Direct OpenCV pixel raster coordinates (pixel_x, pixel_y) where:
     pixel_x = int(normalized_x * frame_width)
     pixel_y = int(normalized_y * frame_height)
   Coordinates are clipped safely to [0, frame_width - 1] and [0, frame_height - 1].
   The Y-axis is NOT inverted.

3. 3D METRIC WORLD COORDINATES (WORLD SPACE):
   MediaPipe 3D world coordinates (world_x, world_y, world_z) in real-world metric meters,
   origin centered near hips. These coordinates are strictly preserved for 3D body
   measurements, volumetric reconstruction, and future AR depth positioning.

4. FUTURE AR OVERLAY:
   By bridging World Space (metrics) and Image Space (raster pixels), this module enables
   direct visual overlays on the real camera feed without disturbing the existing 3D pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from pose_detector import PoseLandmark


@dataclass
class ARPoint:
    """
    Standard AR Point Structure representing a single anatomical landmark
    bridging 2D camera raster coordinates and 3D metric world coordinates.
    """
    name: str
    landmark_index: int
    pixel_x: int
    pixel_y: int
    world_x: float
    world_y: float
    world_z: float
    normalized_x: float
    normalized_y: float
    visibility: float
    valid: bool
    short_label: str = ""


class ARCoordinateMapper:
    """
    High-performance, lightweight coordinate mapping engine for AR rendering.
    Maps normalized and 3D pose landmarks into 2D camera viewport pixel coordinates.
    """

    # Anatomical landmark definitions required for AR overlay and measurements
    REQUIRED_LANDMARKS: Dict[str, Tuple[int, str]] = {
        "NOSE": (PoseLandmark.NOSE, "N"),
        "LEFT_SHOULDER": (PoseLandmark.LEFT_SHOULDER, "LS"),
        "RIGHT_SHOULDER": (PoseLandmark.RIGHT_SHOULDER, "RS"),
        "LEFT_ELBOW": (PoseLandmark.LEFT_ELBOW, "LE"),
        "RIGHT_ELBOW": (PoseLandmark.RIGHT_ELBOW, "RE"),
        "LEFT_WRIST": (PoseLandmark.LEFT_WRIST, "LW"),
        "RIGHT_WRIST": (PoseLandmark.RIGHT_WRIST, "RW"),
        "LEFT_HIP": (PoseLandmark.LEFT_HIP, "LH"),
        "RIGHT_HIP": (PoseLandmark.RIGHT_HIP, "RH"),
        "LEFT_KNEE": (PoseLandmark.LEFT_KNEE, "LK"),
        "RIGHT_KNEE": (PoseLandmark.RIGHT_KNEE, "RK"),
        "LEFT_ANKLE": (PoseLandmark.LEFT_ANKLE, "LA"),
        "RIGHT_ANKLE": (PoseLandmark.RIGHT_ANKLE, "RA"),
    }

    # Anatomical skeleton connections for AR debug visualization
    AR_SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
        # Shoulders & Torso
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        # Arms
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        # Legs
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        min_visibility: float = 0.50,
    ) -> None:
        """
        Initializes the AR coordinate mapper.

        Args:
            frame_width: Default camera frame pixel width.
            frame_height: Default camera frame pixel height.
            min_visibility: Confidence threshold below which landmarks are marked invalid.
        """
        self.frame_width: int = max(1, frame_width)
        self.frame_height: int = max(1, frame_height)
        self.min_visibility: float = min_visibility
        self.is_enabled: bool = True  # Toggle state for AR visual debug overlay

    def initialize(self, frame_width: int, frame_height: int) -> None:
        """
        Dynamically initializes or updates frame dimensions to match the active webcam feed.

        Args:
            frame_width: Width of the camera frame in pixels.
            frame_height: Height of the camera frame in pixels.
        """
        if frame_width > 0 and frame_height > 0:
            self.frame_width = int(frame_width)
            self.frame_height = int(frame_height)

    def toggle(self) -> bool:
        """Toggles AR coordinate mapping visual debug mode on/off."""
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def validate_coordinates(
        self,
        pixel_x: int,
        pixel_y: int,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> bool:
        """
        Verifies if integer pixel coordinates reside inside the valid camera image boundaries.

        Args:
            pixel_x: Horizontal pixel coordinate.
            pixel_y: Vertical pixel coordinate.
            frame_width: Optional width override; defaults to initialized width.
            frame_height: Optional height override; defaults to initialized height.

        Returns:
            True if coordinate is within [0, width - 1] x [0, height - 1], else False.
        """
        w = frame_width if frame_width is not None else self.frame_width
        h = frame_height if frame_height is not None else self.frame_height
        return 0 <= pixel_x < w and 0 <= pixel_y < h

    def landmark_to_pixel(
        self,
        normalized_x: float,
        normalized_y: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Converts normalized coordinates [0.0, 1.0] to clamped camera pixel raster coordinates.

        Formula:
            pixel_x = int(normalized_x * frame_width)
            pixel_y = int(normalized_y * frame_height)

        Guarantees:
            - Clamps pixel_x to [0, frame_width - 1]
            - Clamps pixel_y to [0, frame_height - 1]
            - Does NOT invert the Y-axis.

        Args:
            normalized_x: Horizontal coordinate in range [0.0, 1.0].
            normalized_y: Vertical coordinate in range [0.0, 1.0].
            frame_width: Optional width override.
            frame_height: Optional height override.

        Returns:
            Tuple of (pixel_x, pixel_y) as clamped integers.
        """
        w = frame_width if frame_width is not None else self.frame_width
        h = frame_height if frame_height is not None else self.frame_height

        # Safe fallback if input is None, NaN, or infinite
        if normalized_x is None or np.isnan(normalized_x) or np.isinf(normalized_x):
            normalized_x = 0.0
        if normalized_y is None or np.isnan(normalized_y) or np.isinf(normalized_y):
            normalized_y = 0.0

        # Mathematical scaling: NORMALIZED COORDINATES -> IMAGE PIXEL COORDINATES
        raw_x = int(round(normalized_x * w))
        raw_y = int(round(normalized_y * h))

        # Safe clamping within pixel bounds
        clamped_x = max(0, min(w - 1, raw_x))
        clamped_y = max(0, min(h - 1, raw_y))

        return clamped_x, clamped_y

    def world_to_camera(
        self,
        world_x: float,
        world_y: float,
        world_z: float,
    ) -> Tuple[float, float, float]:
        """
        Preserves 3D metric world landmark space (meters) while providing explicit
        separation from 2D camera pixel coordinates.

        In MediaPipe Pose:
        - World coordinates are metric meters with hip center near origin.
        - X is rightwards (+), Y is downwards (+), Z is away from camera (+/-).
        - Used by the 3D measurement engine and for future AR depth calculations.

        Args:
            world_x: Metric X in meters.
            world_y: Metric Y in meters.
            world_z: Metric Z in meters.

        Returns:
            Tuple of (world_x, world_y, world_z) in meters.
        """
        # Ensure clean float representation without NaN
        wx = float(world_x) if world_x is not None and not np.isnan(world_x) else 0.0
        wy = float(world_y) if world_y is not None and not np.isnan(world_y) else 0.0
        wz = float(world_z) if world_z is not None and not np.isnan(world_z) else 0.0
        return wx, wy, wz

    def measurement_point_to_pixel(
        self,
        point: Any,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Converts an arbitrary measurement point or landmark object to pixel coordinates.

        Supports:
        - ARPoint object
        - MediaPipe landmark object with .x and .y attributes
        - Dict with 'x', 'y' or 'pixel_x', 'pixel_y'
        - Tuple of (norm_x, norm_y)

        Args:
            point: Target point structure.
            frame_width: Optional frame width override.
            frame_height: Optional frame height override.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        if isinstance(point, ARPoint):
            return point.pixel_x, point.pixel_y

        if isinstance(point, (tuple, list)) and len(point) >= 2:
            norm_x, norm_y = float(point[0]), float(point[1])
            return self.landmark_to_pixel(norm_x, norm_y, frame_width, frame_height)

        if isinstance(point, dict):
            if "pixel_x" in point and "pixel_y" in point:
                return int(point["pixel_x"]), int(point["pixel_y"])
            norm_x = float(point.get("x", 0.0))
            norm_y = float(point.get("y", 0.0))
            return self.landmark_to_pixel(norm_x, norm_y, frame_width, frame_height)

        if hasattr(point, "x") and hasattr(point, "y"):
            return self.landmark_to_pixel(float(point.x), float(point.y), frame_width, frame_height)

        return 0, 0

    def landmarks_to_pixels(
        self,
        landmarks: Optional[List[Any]],
        world_landmarks: Optional[List[Any]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Dict[str, ARPoint]:
        """
        Transforms detected pose landmarks and world landmarks into structured ARPoint objects.

        Args:
            landmarks: List of normalized 2D MediaPipe landmarks (primary person).
            world_landmarks: Optional list of 3D metric world landmarks.
            frame_width: Optional current frame width (auto-updates internal dimensions).
            frame_height: Optional current frame height (auto-updates internal dimensions).

        Returns:
            Dictionary mapping landmark name (e.g. 'LEFT_SHOULDER') to an ARPoint instance.
        """
        if frame_width is not None and frame_height is not None:
            self.initialize(frame_width, frame_height)

        mapped_points: Dict[str, ARPoint] = {}

        if not landmarks or len(landmarks) == 0:
            return mapped_points

        num_landmarks = len(landmarks)
        num_world = len(world_landmarks) if world_landmarks else 0

        for name, (idx, short_label) in self.REQUIRED_LANDMARKS.items():
            if idx >= num_landmarks:
                continue

            lm = landmarks[idx]
            norm_x = float(getattr(lm, "x", 0.0))
            norm_y = float(getattr(lm, "y", 0.0))

            # Retrieve visibility / presence confidence
            vis = getattr(lm, "visibility", 1.0)
            if vis is None:
                vis = 1.0
            else:
                vis = float(vis)

            # Retrieve 3D metric world coordinates if available
            world_x, world_y, world_z = 0.0, 0.0, 0.0
            if world_landmarks and idx < num_world:
                wlm = world_landmarks[idx]
                world_x = float(getattr(wlm, "x", 0.0))
                world_y = float(getattr(wlm, "y", 0.0))
                world_z = float(getattr(wlm, "z", 0.0))

            # NORMALIZED COORDINATES -> IMAGE PIXEL COORDINATES
            pixel_x, pixel_y = self.landmark_to_pixel(
                norm_x, norm_y, self.frame_width, self.frame_height
            )

            # Validity evaluation: visibility threshold + within valid bounds
            is_valid = bool(
                vis >= self.min_visibility
                and self.validate_coordinates(pixel_x, pixel_y)
                and not np.isnan(norm_x)
                and not np.isnan(norm_y)
            )

            mapped_points[name] = ARPoint(
                name=name,
                landmark_index=idx,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                normalized_x=norm_x,
                normalized_y=norm_y,
                visibility=vis,
                valid=is_valid,
                short_label=short_label,
            )

        return mapped_points

    def draw_debug_overlay(
        self,
        frame: np.ndarray,
        mapped_points: Dict[str, ARPoint],
    ) -> np.ndarray:
        """
        Renders the Phase 6.1 AR coordinate mapping debug visualization:
        1. Connected skeleton bones using mapped 2D pixel coordinates.
        2. High-contrast landmark joint nodes with short anatomical abbreviations (LS, RS, LE, etc.).
        3. Real-time development HUD showing frame resolution, valid landmark count, and average visibility.

        Args:
            frame: Input OpenCV BGR frame.
            mapped_points: Dictionary of mapped ARPoint instances.

        Returns:
            OpenCV BGR frame with AR coordinate overlay rendered.
        """
        if not self.is_enabled:
            return frame

        h, w = frame.shape[:2]
        self.initialize(w, h)

        if not mapped_points:
            self._draw_debug_hud(frame, w, h, 0, 12, 0.0)
            return frame

        # 1. DRAW BASIC SKELETON CONNECTIONS
        # Connect valid anatomical joint pairs using mapped 2D pixel coordinates
        for start_name, end_name in self.AR_SKELETON_CONNECTIONS:
            p1 = mapped_points.get(start_name)
            p2 = mapped_points.get(end_name)

            if p1 and p2 and p1.valid and p2.valid:
                # Vibrant AR Cyan connection line with anti-aliasing
                cv2.line(
                    frame,
                    (p1.pixel_x, p1.pixel_y),
                    (p2.pixel_x, p2.pixel_y),
                    (255, 200, 0),  # Bright Cyan in BGR
                    2,
                    cv2.LINE_AA,
                )

        # 2. DRAW ANATOMICAL JOINTS & SHORT LABELS (● LS, ● RS, etc.)
        for name, pt in mapped_points.items():
            if not pt.valid:
                continue

            px, py = pt.pixel_x, pt.pixel_y

            # Draw outer ring and filled center for high visibility
            cv2.circle(frame, (px, py), 5, (0, 0, 0), -1, cv2.LINE_AA)          # Dark shadow
            cv2.circle(frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)      # Bright Green core
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)    # Crisp White border

            # Draw short anatomical label (e.g., "LS", "RS", "LE")
            label = pt.short_label if pt.short_label else name[:2]

            # Label offset: slightly above-right with bounding safety
            text_x = min(w - 35, px + 8)
            text_y = max(18, py - 6)

            # Draw text shadow for legibility over clothing / background
            cv2.putText(
                frame,
                label,
                (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. DEBUG STATUS AREA
        # Display coordinate mapper statistics
        body_points = [p for p in mapped_points.values() if p.name != "NOSE"]
        valid_body_count = sum(1 for p in body_points if p.valid)
        total_body_count = len(body_points)

        visibilities = [p.visibility for p in mapped_points.values()]
        avg_vis = float(np.mean(visibilities)) if visibilities else 0.0

        self._draw_debug_hud(frame, w, h, valid_body_count, total_body_count, avg_vis)

        return frame

    def _draw_debug_hud(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        valid_count: int,
        total_count: int,
        avg_vis: float,
    ) -> None:
        """
        Renders a clean development status panel for Phase 6.1 AR Coordinate Mapping.
        """
        # Place HUD panel on the bottom-right corner
        panel_w = 260
        panel_h = 110
        margin = 15

        x1 = width - panel_w - margin
        y1 = height - panel_h - margin
        x2 = width - margin
        y2 = height - margin

        # Background semi-transparent dark card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Card border (neon cyan accent)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)

        # Header
        cv2.putText(
            frame,
            "AR Coordinate Mapping: ACTIVE",
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Frame resolution
        cv2.putText(
            frame,
            f"Frame: {width} x {height}",
            (x1 + 10, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Valid landmarks count
        val_color = (0, 255, 100) if valid_count >= 12 else (0, 180, 255)
        cv2.putText(
            frame,
            f"Valid landmarks: {valid_count} / {total_count}",
            (x1 + 10, y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            val_color,
            1,
            cv2.LINE_AA,
        )

        # Average visibility
        cv2.putText(
            frame,
            f"Average visibility: {avg_vis:.2f}",
            (x1 + 10, y1 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Mapping pipeline description
        cv2.putText(
            frame,
            "Mapping: NORMALIZED -> PIXEL",
            (x1 + 10, y1 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (160, 220, 255),
            1,
            cv2.LINE_AA,
        )
