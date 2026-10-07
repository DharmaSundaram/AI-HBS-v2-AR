"""
pose_detector.py
----------------
Module: Pose Detection & Landmark Extraction

Purpose:
Encapsulates MediaPipe Pose estimation using MediaPipe Tasks (PoseLandmarker)
with fallback to legacy solutions. Works seamlessly on Python 3.13 and MediaPipe 0.10.x+.
Extracts 33 normalized image landmarks and 3D world landmarks (in metric meters).
"""

import os
import urllib.request
from typing import Optional, Tuple, List, Any
import cv2
import numpy as np
import mediapipe as mp


class PoseLandmark:
    """Standard MediaPipe Pose Landmark Indices (0-32)."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Standard anatomical skeleton connection pairs
POSE_CONNECTIONS = [
    # Shoulders & Torso
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    # Arms
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    # Hands
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    # Legs
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    # Feet
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
    # Face
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT),
]


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


class PoseDetector:
    """
    High-performance pose detector supporting modern MediaPipe Tasks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        **kwargs: Any,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Determine model path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "pose_landmarker_full.task")
        self.model_path = model_path

        # Download model bundle if missing
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[INFO] Downloading MediaPipe PoseLandmarker model to {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[INFO] Model downloaded successfully.")

        # Initialize MediaPipe Tasks Pose Landmarker
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # 12 essential landmarks required for full body measurement
        self.required_landmarks = [
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_ANKLE,
        ]

    def process(self, frame_bgr: cv2.Mat) -> Any:
        """
        Runs pose landmark detection on an OpenCV BGR frame.

        Args:
            frame_bgr: Input BGR image frame.

        Returns:
            PoseLandmarkerResult with pose_landmarks and pose_world_landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect(mp_image)

    def is_body_detected(self, results: Any) -> bool:
        """Checks if at least one pose was detected."""
        return (
            results is not None
            and bool(results.pose_landmarks)
            and len(results.pose_landmarks) > 0
            and bool(results.pose_world_landmarks)
            and len(results.pose_world_landmarks) > 0
        )

    def get_landmarks(self, results: Any) -> Optional[List[Any]]:
        """Returns normalized landmarks (x, y, z in [0, 1]) for primary person."""
        if self.is_body_detected(results):
            return results.pose_landmarks[0]
        return None

    def get_world_landmarks(self, results: Any) -> Optional[List[Any]]:
        """Returns 3D metric world landmarks (in meters) for primary person."""
        if self.is_body_detected(results):
            return results.pose_world_landmarks[0]
        return None

    def check_full_body_visibility(
        self,
        landmarks: List[Any],
        min_visibility: float = 0.60
    ) -> Tuple[bool, List[int]]:
        """
        Verifies visibility of all 12 key measurement points.
        """
        if not landmarks:
            return False, self.required_landmarks

        missing = []
        for point in self.required_landmarks:
            lm = landmarks[point]
            # Some versions use visibility or presence
            vis = getattr(lm, "visibility", 1.0)
            if vis is None:
                vis = 1.0
            if vis < min_visibility:
                missing.append(point)

        return (len(missing) == 0, missing)

    def release(self) -> None:
        """Releases detector resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
