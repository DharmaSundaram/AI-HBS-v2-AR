"""
calibration.py
--------------
Module: Camera & Scale Calibration

Purpose:
Manages loading camera configuration parameters, video device settings,
and scale conversion factors (converting MediaPipe world coordinate meters to centimeters).
"""

import json
from typing import Dict, Any


class CalibrationManager:
    """
    Manages camera settings and scale factor calibration.
    """

    def __init__(self, config_path: str = "config/calibration.json") -> None:
        """
        Initializes the calibration manager.

        Args:
            config_path: Path to calibration.json configuration file.
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Loads camera and calibration settings from disk."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback default configuration
            self.config_data = {
                "camera": {
                    "device_index": 0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                },
                "scale_factor": 100.0,
            }
        return self.config_data

    @property
    def scale_factor(self) -> float:
        """Returns metric scale factor (converts MediaPipe 3D world meters to cm)."""
        return float(self.config_data.get("scale_factor", 100.0))

    @property
    def camera_settings(self) -> Dict[str, Any]:
        """Returns camera hardware and resolution parameters."""
        return self.config_data.get("camera", {"device_index": 0, "width": 1280, "height": 720})
