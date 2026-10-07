"""
body_segmentation.py
--------------------
Module: Person / Body Segmentation (Phase 5)

Purpose:
Extracts a continuous 2D binary body silhouette mask (person foreground vs. background)
using MediaPipe Selfie Segmenter with temporal smoothing and contour filtering.

Features:
- Binary body mask: foreground = 255, background = 0
- Temporal exponential smoothing across frames to eliminate silhouette edge jitter
- Visual mask overlay renderer with customizable transparency
- Segmentation quality and completeness assessment
- Seamless fallback handling if segmentation is unavailable
"""

import os
import urllib.request
from typing import Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp


class BodySegmentation:
    """
    Extracts and stabilizes a 2D person silhouette mask from camera frames.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.50,
        temporal_alpha: float = 0.65,
    ) -> None:
        """
        Initializes the body segmentation engine.

        Args:
            model_path: Path to selfie_segmenter.tflite model.
            threshold: Confidence threshold for person classification.
            temporal_alpha: Weight for temporal mask smoothing [0.0, 1.0].
        """
        self.threshold = threshold
        self.temporal_alpha = temporal_alpha
        self.is_enabled: bool = False  # Toggled via 'B' key
        self.is_initialized: bool = False

        # Temporal smoothing buffer
        self.prev_confidence_mask: Optional[np.ndarray] = None

        # Determine model path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "selfie_segmenter.tflite")
        self.model_path = model_path

        self._init_segmenter()

    def _init_segmenter(self) -> None:
        """Initializes the MediaPipe ImageSegmenter."""
        try:
            if not os.path.exists(self.model_path):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                print(f"[INFO] Downloading segmentation model to {self.model_path}...")
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                print("[INFO] Segmentation model downloaded successfully.")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            self.is_initialized = True
        except Exception as e:
            print(f"[WARNING] Could not initialize MediaPipe segmenter: {e}")
            self.is_initialized = False

    def toggle(self) -> bool:
        """Toggles the 2D segmentation mask overlay display on/off (Key: 'B')."""
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def segment(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Generates a temporally stabilized binary silhouette mask from an input BGR frame.

        Args:
            frame_bgr: Input camera frame.

        Returns:
            mask: Binary uint8 mask (255 foreground, 0 background), or None if unavailable.
            quality_score: Segmentation completeness metric [0.0, 1.0].
        """
        if not self.is_initialized:
            return None, 0.0

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.segmenter.segment(mp_image)

            if result.category_mask is None:
                return None, 0.0

            raw_mask = result.category_mask.numpy_view().astype(np.float32)

            # In selfie segmenter, label 0 is person or confidence mask
            # Normalize to [0.0, 1.0] confidence
            if raw_mask.max() > 0:
                conf_mask = (raw_mask > 0).astype(np.float32)
            else:
                conf_mask = np.zeros_like(raw_mask)

            # Apply temporal smoothing
            if self.prev_confidence_mask is None or self.prev_confidence_mask.shape != conf_mask.shape:
                smoothed_conf = conf_mask
            else:
                smoothed_conf = (
                    self.temporal_alpha * conf_mask
                    + (1.0 - self.temporal_alpha) * self.prev_confidence_mask
                )
            self.prev_confidence_mask = smoothed_conf

            # Threshold into binary mask
            binary_mask = (smoothed_conf >= self.threshold).astype(np.uint8) * 255

            # Morphological opening/closing to eliminate isolated speckles
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            # Evaluate completeness quality score (ratio of person area to frame)
            h, w = cleaned_mask.shape
            area_ratio = float(np.sum(cleaned_mask > 0)) / float(h * w)
            # Normal standing full-body in frame is typically between 8% and 55% of image area
            if 0.06 <= area_ratio <= 0.65:
                quality_score = 0.95
            elif area_ratio > 0.02:
                quality_score = 0.70
            else:
                quality_score = 0.30

            return cleaned_mask, quality_score

        except Exception as e:
            return None, 0.0

    def draw_mask_overlay(
        self,
        frame_bgr: np.ndarray,
        binary_mask: Optional[np.ndarray],
        color: Tuple[int, int, int] = (255, 120, 0),  # Cyan/Blue tint in BGR
        alpha: float = 0.40,
    ) -> np.ndarray:
        """
        Renders an aesthetic silhouette overlay on top of the webcam feed when enabled.

        Args:
            frame_bgr: Input frame to draw onto.
            binary_mask: Binary silhouette mask (255 for person).
            color: Color tint for body mask.
            alpha: Transparency factor.

        Returns:
            Frame with silhouette mask overlay.
        """
        if not self.is_enabled or binary_mask is None:
            return frame_bgr

        overlay = frame_bgr.copy()
        # Find contours of the silhouette for outer edge highlight
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Tint the body area
        mask_indices = binary_mask > 0
        overlay[mask_indices] = (
            (1.0 - alpha) * frame_bgr[mask_indices] + alpha * np.array(color)
        ).astype(np.uint8)

        # Draw crisp contour edge
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw HUD tag in corner
        cv2.putText(
            overlay,
            "BODY SILHOUETTE: ACTIVE",
            (25, frame_bgr.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return overlay

    def release(self) -> None:
        """Frees segmentation resources."""
        if hasattr(self, "segmenter") and self.segmenter is not None:
            self.segmenter.close()
