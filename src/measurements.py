"""
measurements.py
---------------
Module: 3D Body Segment Measurement Calculator

Purpose:
Calculates metric body segment lengths from MediaPipe 3D world landmarks
(pose_world_landmarks) and applies configured empirical correction factors.
"""

import json
import math
from typing import Dict, Any, Optional, List
from pose_detector import PoseLandmark


class BodyMeasurementCalculator:
    """
    Computes anatomical body segment distances and metrics from 3D world landmarks.
    """

    def __init__(self, config_path: Optional[str] = "config/correction_factors.json") -> None:
        """
        Initializes the measurement engine with correction factor adjustments.

        Args:
            config_path: Path to correction_factors.json file.
        """
        self.config_path = config_path
        self.correction_factors: Dict[str, float] = {}
        if config_path:
            self.load_correction_factors(config_path)

        self.L = PoseLandmark

    def load_correction_factors(self, path: str) -> None:
        """Loads correction factors from JSON configuration."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "segment_factors" in data:
                    self.correction_factors = data["segment_factors"]
                else:
                    self.correction_factors = {
                        k: v for k, v in data.items() if isinstance(v, (int, float))
                    }
        except (FileNotFoundError, json.JSONDecodeError):
            self.correction_factors = {}

    @staticmethod
    def distance_3d(a: Any, b: Any) -> float:
        """
        Computes the Euclidean distance between two 3D points (x, y, z).

        Args:
            a: 3D point with x, y, z coordinates.
            b: 3D point with x, y, z coordinates.

        Returns:
            Euclidean distance in meters (using MediaPipe world landmarks).
        """
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    def to_cm(self, distance: float, scale_factor: float = 100.0) -> float:
        """
        Converts distance into centimeters using the configured scale factor.
        """
        return distance * scale_factor

    def apply_correction(self, name: str, value: float) -> float:
        """
        Applies empirical regression/anatomical correction factor if present.
        """
        if name in self.correction_factors:
            return value * self.correction_factors[name]
        return value

    def calculate_frame_measurements(
        self,
        world: List[Any],
        scale_factor: float = 100.0
    ) -> Dict[str, float]:
        """
        Calculates all anatomical segment lengths for a single frame.

        Args:
            world: List of 33 MediaPipe 3D world landmarks for the primary person.
            scale_factor: Conversion scale factor (default 100.0 for meters to cm).

        Returns:
            Dictionary of corrected segment measurements in centimeters.
        """
        L = self.L

        # Extract anatomical keypoints from 3D world coordinate space
        left_shoulder = world[L.LEFT_SHOULDER]
        right_shoulder = world[L.RIGHT_SHOULDER]
        left_elbow = world[L.LEFT_ELBOW]
        right_elbow = world[L.RIGHT_ELBOW]
        left_wrist = world[L.LEFT_WRIST]
        right_wrist = world[L.RIGHT_WRIST]
        left_hip = world[L.LEFT_HIP]
        right_hip = world[L.RIGHT_HIP]
        left_knee = world[L.LEFT_KNEE]
        right_knee = world[L.RIGHT_KNEE]
        left_ankle = world[L.LEFT_ANKLE]
        right_ankle = world[L.RIGHT_ANKLE]

        raw: Dict[str, float] = {
            "shoulder": self.to_cm(
                self.distance_3d(left_shoulder, right_shoulder), scale_factor
            ),
            "left_upper_arm": self.to_cm(
                self.distance_3d(left_shoulder, left_elbow), scale_factor
            ),
            "right_upper_arm": self.to_cm(
                self.distance_3d(right_shoulder, right_elbow), scale_factor
            ),
            "left_forearm": self.to_cm(
                self.distance_3d(left_elbow, left_wrist), scale_factor
            ),
            "right_forearm": self.to_cm(
                self.distance_3d(right_elbow, right_wrist), scale_factor
            ),
            "left_thigh": self.to_cm(
                self.distance_3d(left_hip, left_knee), scale_factor
            ),
            "right_thigh": self.to_cm(
                self.distance_3d(right_hip, right_knee), scale_factor
            ),
            "left_lower_leg": self.to_cm(
                self.distance_3d(left_knee, left_ankle), scale_factor
            ),
            "right_lower_leg": self.to_cm(
                self.distance_3d(right_knee, right_ankle), scale_factor
            ),
        }

        # Apply correction factors
        corrected: Dict[str, float] = {}
        for name, value in raw.items():
            corrected[name] = self.apply_correction(name, value)

        return corrected
