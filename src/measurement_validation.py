"""
measurement_validation.py
-------------------------
Module: Measurement Validation & Temporal Stability Analysis (Phase 5)

Purpose:
Calculates statistical frame-to-frame temporal stability metrics for all circumference
measurements (mean, median, standard deviation, coefficient of variation, consistency score)
and provides an interactive developer ground-truth validation mode (Key: 'V').

Features:
- Stability Metrics: mean, median, std_dev, CV (std/mean), and temporal consistency percentage
- Developer Validation Mode (Key: 'V'): Compares manual real-world measurements against AI estimates
- Computes Absolute Error (cm), Percentage Error (%), and Mean Absolute Error (MAE)
- Saves validation reports to data/results/validation_results.json
- Preserves calibration: Ground-truth inputs are recorded for verification without automatic distortion
- Disclaimer: Stability score measures temporal frame-to-frame consistency, NOT real-world accuracy
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class MeasurementValidator:
    """
    Computes statistical stability metrics across frames and performs ground-truth validation.
    """

    def __init__(
        self,
        buffer_size: int = 40,
        results_path: str = "data/results/validation_results.json",
    ) -> None:
        self.buffer_size = buffer_size
        self.results_path = results_path
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.is_validating: bool = False

    def add_sample(self, metric_name: str, value_cm: float) -> None:
        """Adds a circumference observation to the statistical stability buffer."""
        if value_cm > 0.0:
            self.history[metric_name].append(value_cm)
            if len(self.history[metric_name]) > self.buffer_size:
                self.history[metric_name].pop(0)

    def get_stability_metrics(self, metric_name: str) -> Dict[str, float]:
        """
        Calculates frame-to-frame stability metrics:
        - Mean
        - Median
        - Standard Deviation
        - Coefficient of Variation (CV = std / mean)
        - Temporal Stability Percentage (100 * max(0, 1 - CV))

        Note: Stability percentage represents temporal consistency across frames only.
        """
        samples = self.history.get(metric_name, [])
        if len(samples) < 3:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "cv": 0.0,
                "stability_pct": 0.0,
            }

        arr = np.array(samples, dtype=np.float32)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr))

        cv = float(std_val / mean_val) if mean_val > 0.0 else 0.0
        # Convert CV to a 0-100% temporal consistency score
        # e.g. std of 1cm on 100cm (CV=0.01) yields ~99% stability
        stability_pct = float(max(0.0, min(100.0, (1.0 - cv * 3.0) * 100.0)))

        return {
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std_dev": round(std_val, 2),
            "cv": round(cv, 4),
            "stability_pct": round(stability_pct, 1),
        }

    def run_validation(
        self,
        actual_measurements: Dict[str, float],
        estimated_measurements: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compares manual real-world measurements against AI estimates and calculates error metrics.

        Args:
            actual_measurements: Ground-truth values in cm provided by developer/user.
            estimated_measurements: Current AI estimated circumferences in cm.

        Returns:
            Validation report dictionary with individual errors and Mean Absolute Error (MAE).
        """
        comparison_table: List[Dict[str, Any]] = []
        abs_errors: List[float] = []

        for metric, actual_val in actual_measurements.items():
            if actual_val <= 0:
                continue

            est_val = estimated_measurements.get(metric, 0.0)
            abs_err = abs(est_val - actual_val)
            pct_err = (abs_err / actual_val) * 100.0 if actual_val > 0 else 0.0
            abs_errors.append(abs_err)

            comparison_table.append({
                "metric": metric,
                "actual_cm": round(actual_val, 2),
                "estimated_cm": round(est_val, 2),
                "absolute_error_cm": round(abs_err, 2),
                "percentage_error_pct": round(pct_err, 2),
            })

        mae = float(np.mean(abs_errors)) if abs_errors else 0.0

        report = {
            "status": "Ground-Truth Validation Completed",
            "mean_absolute_error_cm": round(mae, 2),
            "comparisons": comparison_table,
            "disclaimer": "Ground-truth comparison for verification. Calibration factors were NOT automatically modified.",
        }

        # Save to disk
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return report

    def print_validation_summary(self, report: Dict[str, Any]) -> None:
        """Prints a clean ASCII summary table to the console."""
        print("\n" + "=" * 70)
        print("AI-HBS GROUND-TRUTH MEASUREMENT VALIDATION SUMMARY")
        print("=" * 70)
        print(f"{'Measurement':<18} | {'Actual (cm)':<12} | {'Estimated (cm)':<14} | {'Error (cm)':<10} | {'Error %':<8}")
        print("-" * 70)
        for row in report.get("comparisons", []):
            name = row["metric"].replace("_", " ").title()
            act = f"{row['actual_cm']:.1f}"
            est = f"{row['estimated_cm']:.1f}"
            err = f"{row['absolute_error_cm']:.1f}"
            pct = f"{row['percentage_error_pct']:.1f}%"
            print(f"{name:<18} | {act:<12} | {est:<14} | {err:<10} | {pct:<8}")
        print("-" * 70)
        print(f"Mean Absolute Error (MAE): {report.get('mean_absolute_error_cm', 0.0):.2f} cm")
        print("Results saved to: " + self.results_path)
        print("=" * 70 + "\n")
