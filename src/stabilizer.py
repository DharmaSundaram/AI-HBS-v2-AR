"""
stabilizer.py
-------------
Module: Measurement Stabilization & Outlier Filtering

Purpose:
Aggregates body segment measurements across consecutive frames when a subject
stands still, filters noise/outliers using median absolute deviation, and
produces statistically stabilized final measurements.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


class MeasurementStabilizer:
    """
    Stabilizes measurements across multiple video frames and rejects outliers.
    """

    def __init__(self, frames_required: int = 60, outlier_threshold: float = 3.0) -> None:
        """
        Initializes the stabilizer.

        Args:
            frames_required: Number of valid frames to collect before computing final values.
            outlier_threshold: Maximum allowed delta (cm) from the median before value is flagged as an outlier.
        """
        self.frames_required = frames_required
        self.outlier_threshold = outlier_threshold
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.is_finalized: bool = False
        self.final_results: Dict[str, float] = {}

    def add_frame_measurements(self, frame_measurements: Dict[str, float]) -> None:
        """
        Appends single-frame segment measurements to history buffers.

        Args:
            frame_measurements: Dictionary of segment measurements for current frame.
        """
        if self.is_finalized:
            return

        for name, value in frame_measurements.items():
            self.history[name].append(value)

    def get_progress(self) -> Tuple[int, int]:
        """
        Returns the collection progress.

        Returns:
            Tuple of (collected_frames, frames_required).
        """
        # Base count on shoulder or any tracked segment
        if "shoulder" in self.history:
            count = len(self.history["shoulder"])
        elif self.history:
            count = len(next(iter(self.history.values())))
        else:
            count = 0
        return count, self.frames_required

    def is_ready(self) -> bool:
        """Returns True if sufficient frames have been captured."""
        count, required = self.get_progress()
        return count >= required

    def compute_final_measurements(self) -> Dict[str, float]:
        """
        Filters unstable outlier frames using median comparison and computes mean.

        Returns:
            Dictionary mapping segment names to stabilized measurements in cm (rounded to 2 decimals).
        """
        if self.is_finalized and self.final_results:
            return self.final_results

        final: Dict[str, float] = {}
        for name, values in self.history.items():
            if not values:
                continue

            median = float(np.median(values))
            filtered_values = [
                v for v in values if abs(v - median) < self.outlier_threshold
            ]

            if len(filtered_values) > 0:
                final[name] = round(float(np.mean(filtered_values)), 2)
            else:
                final[name] = round(float(np.mean(values)), 2)

        self.final_results = final
        self.is_finalized = True
        return self.final_results

    def reset(self) -> None:
        """Resets the capture buffer to start a new measurement session."""
        self.history.clear()
        self.final_results.clear()
        self.is_finalized = False
