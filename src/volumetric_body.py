"""
volumetric_body.py
------------------
Module: Continuous Volumetric 3D Procedural Body Mesh (Phase 3)

Purpose:
Constructs a continuous, anthropometrically proportional, volumetric 3D human body
mesh from MediaPipe 3D pose_world_landmarks.

Regions Generated:
- Head: 3D ellipsoid / sphere volume
- Neck: Volumetric cylinder
- Shoulders & Clavicle line
- Chest / Torso: Anatomically tapered 3D volume
- Waist: Natural narrower mid-torso transition
- Pelvis / Hip region: Tapered lower-torso volume
- Left & Right Upper Arms: Tapered capsules (thicker near shoulder, tapering to elbow)
- Left & Right Forearms: Tapered capsules (tapering towards wrist)
- Left & Right Hands: Volumetric hand shapes (wrist to knuckles)
- Left & Right Thighs: Tapered capsules (thicker near hip, tapering to knee)
- Left & Right Lower Legs: Tapered capsules (thicker near calf, tapering to ankle)
- Left & Right Feet: Elongated 3D foot volumes (ankle to heel & toe)

Features:
- Adaptive exponential smoothing to eliminate landmark jitter without movement lag
- Proportional radius scaling derived from detected user dimensions (shoulder width, hip width, limb lengths)
- Visual distinction between volumetric mesh, inner skeleton, and landmark joints
- Disclaimer: Procedural approximation based on pose landmarks (not a medical scan)
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from pose_detector import PoseLandmark


class VolumetricSegment:
    """
    Represents a 3D volumetric segment with independent start and end radii (tapered capsule/cylinder).
    """

    def __init__(
        self,
        name: str,
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        radius_start: float,
        radius_end: float,
        color: Tuple[int, int, int] = (0, 220, 255),
        num_radial_segments: int = 8,
        num_rings: int = 4,
        is_elliptical: bool = False,
        aspect_ratio: float = 1.0,  # ratio of depth to width for elliptical cross-sections (e.g. torso)
    ) -> None:
        self.name = name
        self.start_pt = np.asarray(start_pt, dtype=np.float32)
        self.end_pt = np.asarray(end_pt, dtype=np.float32)
        self.radius_start = max(radius_start, 0.005)
        self.radius_end = max(radius_end, 0.005)
        self.color = color
        self.num_radial_segments = num_radial_segments
        self.num_rings = num_rings
        self.is_elliptical = is_elliptical
        self.aspect_ratio = aspect_ratio

    def generate_mesh(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Generates 3D surface vertices and wireframe edges for the tapered volumetric segment.

        Returns:
            vertices: NumPy array of shape (V, 3) in meters.
            edges: List of vertex index pairs (idx1, idx2).
        """
        axis = self.end_pt - self.start_pt
        length = float(np.linalg.norm(axis))

        if length < 1e-4:
            return np.array([self.start_pt], dtype=np.float32), []

        unit_axis = axis / length

        # Construct orthonormal coordinate frame perpendicular to cylinder unit axis
        arbitrary = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(unit_axis[2]) > 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        n1 = np.cross(unit_axis, arbitrary)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(unit_axis, n1)

        vertices = []
        edges = []

        t_steps = np.linspace(0.0, 1.0, self.num_rings)
        thetas = np.linspace(0.0, 2 * math.pi, self.num_radial_segments, endpoint=False)

        for ring_idx, t in enumerate(t_steps):
            center = self.start_pt + t * axis
            r_current = self.radius_start + t * (self.radius_end - self.radius_start)
            ring_start_vertex_idx = len(vertices)

            depth_scale = self.aspect_ratio if self.is_elliptical else 1.0

            for theta in thetas:
                # Radial offset along n1 (width) and n2 (depth)
                dx = r_current * math.cos(theta)
                dy = r_current * math.sin(theta) * depth_scale
                v = center + dx * n1 + dy * n2
                vertices.append(v)

            # Circumferential ring edges
            for i in range(self.num_radial_segments):
                next_i = (i + 1) % self.num_radial_segments
                edges.append((ring_start_vertex_idx + i, ring_start_vertex_idx + next_i))

            # Longitudinal rib edges connecting adjacent rings
            if ring_idx > 0:
                prev_ring_start = ring_start_vertex_idx - self.num_radial_segments
                for i in range(self.num_radial_segments):
                    edges.append((prev_ring_start + i, ring_start_vertex_idx + i))

        return np.array(vertices, dtype=np.float32), edges


class VolumetricBodyMesh:
    """
    Constructs an anthropometrically proportional, continuous 3D volumetric body mesh
    from MediaPipe pose_world_landmarks with temporal landmark stabilization.
    """

    def __init__(self, smoothing_factor: float = 0.55) -> None:
        """
        Initializes the volumetric mesh generator.

        Args:
            smoothing_factor: Exponential smoothing weight [0.1, 0.9] for landmark stability.
        """
        self.smoothing_factor = smoothing_factor
        self.smoothed_landmarks: Optional[Dict[int, np.ndarray]] = None

        # Mesh Palette Colors (BGR)
        self.color_head = (230, 120, 255)       # Soft Violet
        self.color_torso = (255, 190, 50)       # Electric Cyan / Azure
        self.color_waist = (240, 170, 70)       # Mid Cyan
        self.color_pelvis = (220, 150, 90)      # Deep Azure
        self.color_arms = (0, 230, 255)         # Warm Golden Yellow
        self.color_hands = (0, 190, 255)        # Light Amber
        self.color_legs = (60, 255, 170)        # Spring Jade
        self.color_feet = (50, 220, 140)        # Deep Mint
        self.color_skeleton = (255, 255, 255)   # Crisp White Inner Skeleton
        self.color_joints = (0, 255, 0)         # Bright Lime Joints

    def reset_smoothing(self) -> None:
        """Clears the temporal filter buffer."""
        self.smoothed_landmarks = None

    def update(self, raw_world_landmarks: Optional[List[Any]]) -> None:
        """
        Smooths incoming 3D world landmarks and updates internal spatial representation.

        Args:
            raw_world_landmarks: Collection of 33 MediaPipe 3D world landmark objects.
        """
        if not raw_world_landmarks or len(raw_world_landmarks) < 33:
            return

        current_points: Dict[int, np.ndarray] = {}
        for idx, lm in enumerate(raw_world_landmarks):
            current_points[idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        if self.smoothed_landmarks is None:
            # Initialize with first valid frame
            self.smoothed_landmarks = current_points
        else:
            # Adaptive exponential smoothing: smooth stationary jitter, track fast movement
            alpha = self.smoothing_factor
            for idx, pt in current_points.items():
                prev = self.smoothed_landmarks[idx]
                delta = float(np.linalg.norm(pt - prev))
                # If sudden fast motion (> 0.12m), increase alpha for instantaneous response
                dynamic_alpha = alpha if delta < 0.12 else min(0.85, alpha + 0.30)
                self.smoothed_landmarks[idx] = dynamic_alpha * pt + (1.0 - dynamic_alpha) * prev

    def _get_point(self, idx: int) -> np.ndarray:
        """Retrieves smoothed 3D landmark coordinate."""
        if self.smoothed_landmarks and idx in self.smoothed_landmarks:
            return self.smoothed_landmarks[idx]
        return np.zeros(3, dtype=np.float32)

    def generate_body_mesh(
        self,
    ) -> Tuple[List[VolumetricSegment], List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
        """
        Builds all 18 volumetric body regions, inner skeleton bones, and key landmark joints.

        Returns:
            segments: List of VolumetricSegment objects.
            skeleton_bones: List of (point_A, point_B) bone pairs.
            joint_positions: List of 3D joint coordinate arrays.
        """
        if not self.smoothed_landmarks or len(self.smoothed_landmarks) < 33:
            return [], [], []

        L = PoseLandmark

        # 1. Extract stabilized anatomical landmarks
        nose = self._get_point(L.NOSE)
        left_eye = self._get_point(L.LEFT_EYE)
        right_eye = self._get_point(L.RIGHT_EYE)
        left_ear = self._get_point(L.LEFT_EAR)
        right_ear = self._get_point(L.RIGHT_EAR)

        left_shoulder = self._get_point(L.LEFT_SHOULDER)
        right_shoulder = self._get_point(L.RIGHT_SHOULDER)
        left_elbow = self._get_point(L.LEFT_ELBOW)
        right_elbow = self._get_point(L.RIGHT_ELBOW)
        left_wrist = self._get_point(L.LEFT_WRIST)
        right_wrist = self._get_point(L.RIGHT_WRIST)
        left_index = self._get_point(L.LEFT_INDEX)
        right_index = self._get_point(L.RIGHT_INDEX)
        left_pinky = self._get_point(L.LEFT_PINKY)
        right_pinky = self._get_point(L.RIGHT_PINKY)

        left_hip = self._get_point(L.LEFT_HIP)
        right_hip = self._get_point(L.RIGHT_HIP)
        left_knee = self._get_point(L.LEFT_KNEE)
        right_knee = self._get_point(L.RIGHT_KNEE)
        left_ankle = self._get_point(L.LEFT_ANKLE)
        right_ankle = self._get_point(L.RIGHT_ANKLE)
        left_heel = self._get_point(L.LEFT_HEEL)
        right_heel = self._get_point(L.RIGHT_HEEL)
        left_foot_idx = self._get_point(L.LEFT_FOOT_INDEX)
        right_foot_idx = self._get_point(L.RIGHT_FOOT_INDEX)

        # 2. Anthropometric Proportional Sizing
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        hip_width = float(np.linalg.norm(left_hip - right_hip))
        shoulder_mid = (left_shoulder + right_shoulder) * 0.5
        hip_mid = (left_hip + right_hip) * 0.5
        torso_vector = hip_mid - shoulder_mid
        torso_len = float(np.linalg.norm(torso_vector))

        # Clamp baseline proportions to human physical constraints
        shoulder_w_clamped = max(0.28, min(shoulder_width, 0.58))
        hip_w_clamped = max(0.20, min(hip_width, 0.45))
        torso_l_clamped = max(0.35, min(torso_len, 0.70))

        # Limb lengths for proportional tapering
        l_uarm_len = float(np.linalg.norm(left_elbow - left_shoulder))
        r_uarm_len = float(np.linalg.norm(right_elbow - right_shoulder))
        l_farm_len = float(np.linalg.norm(left_wrist - left_elbow))
        r_farm_len = float(np.linalg.norm(right_wrist - right_elbow))
        l_thigh_len = float(np.linalg.norm(left_knee - left_hip))
        r_thigh_len = float(np.linalg.norm(right_knee - right_hip))
        l_calf_len = float(np.linalg.norm(left_ankle - left_knee))
        r_calf_len = float(np.linalg.norm(right_ankle - right_knee))

        # Calculated Proportional Radii (meters)
        head_radius = max(0.09, min(float(np.linalg.norm(left_ear - right_ear)) * 0.65, 0.13))
        neck_radius = head_radius * 0.45

        chest_radius = shoulder_w_clamped * 0.38
        waist_radius = (shoulder_w_clamped + hip_w_clamped) * 0.22
        pelvis_radius = hip_w_clamped * 0.42

        # Arm Tapers (thick near joint, tapering down)
        l_uarm_r_start = max(0.040, min(l_uarm_len * 0.20, 0.065))
        l_uarm_r_end = l_uarm_r_start * 0.75
        r_uarm_r_start = max(0.040, min(r_uarm_len * 0.20, 0.065))
        r_uarm_r_end = r_uarm_r_start * 0.75

        l_farm_r_start = l_uarm_r_end * 0.95
        l_farm_r_end = max(0.025, min(l_farm_len * 0.14, 0.045))
        r_farm_r_start = r_uarm_r_end * 0.95
        r_farm_r_end = max(0.025, min(r_farm_len * 0.14, 0.045))

        # Leg Tapers
        l_thigh_r_start = max(0.060, min(l_thigh_len * 0.20, 0.095))
        l_thigh_r_end = l_thigh_r_start * 0.72
        r_thigh_r_start = max(0.060, min(r_thigh_len * 0.20, 0.095))
        r_thigh_r_end = r_thigh_r_start * 0.72

        l_calf_r_start = l_thigh_r_end * 0.92
        l_calf_r_end = max(0.035, min(l_calf_len * 0.12, 0.055))
        r_calf_r_start = r_thigh_r_end * 0.92
        r_calf_r_end = max(0.035, min(r_calf_len * 0.12, 0.055))

        # Hand & Foot sizes
        hand_radius = max(0.025, l_farm_r_end * 0.90)
        foot_radius = max(0.035, l_calf_r_end * 0.85)

        # 3. Intermediate Anatomical Spine & Torso Milestones
        # Waist positioned at 55% down the torso
        waist_pt = shoulder_mid + torso_vector * 0.55
        chest_pt = shoulder_mid + torso_vector * 0.25

        head_top = nose + np.array([0.0, -head_radius * 0.95, 0.0], dtype=np.float32)
        head_bottom = nose + np.array([0.0, head_radius * 0.85, 0.0], dtype=np.float32)

        # Knuckle midpoints for hands
        l_knuckle = (left_index + left_pinky) * 0.5
        r_knuckle = (right_index + right_pinky) * 0.5

        # 4. Construct All 18 Volumetric Segments
        segments: List[VolumetricSegment] = [
            # 1. Head (Volumetric sphere/ellipsoid)
            VolumetricSegment("Head", head_top, head_bottom, head_radius, head_radius, self.color_head, 8, 5),

            # 2. Neck (Cylinder)
            VolumetricSegment("Neck", shoulder_mid, nose, neck_radius, neck_radius * 1.1, self.color_head, 6, 3),

            # 3. Shoulders / Clavicle Line
            VolumetricSegment("Shoulders", left_shoulder, right_shoulder, neck_radius, neck_radius, self.color_torso, 6, 3),

            # 4. Chest / Upper Torso (Tapered 3D volume connecting shoulders to mid-chest)
            VolumetricSegment(
                "Chest", shoulder_mid, chest_pt, chest_radius, chest_radius * 0.92,
                self.color_torso, 8, 4, is_elliptical=True, aspect_ratio=0.68
            ),

            # 5. Waist / Mid Torso (Tapering narrower to waist)
            VolumetricSegment(
                "Waist", chest_pt, waist_pt, chest_radius * 0.92, waist_radius,
                self.color_waist, 8, 4, is_elliptical=True, aspect_ratio=0.65
            ),

            # 6. Pelvis / Lower Torso (Widening to pelvis)
            VolumetricSegment(
                "Pelvis", waist_pt, hip_mid, waist_radius, pelvis_radius,
                self.color_pelvis, 8, 4, is_elliptical=True, aspect_ratio=0.72
            ),

            # 7. Left Upper Arm (Tapered capsule)
            VolumetricSegment("Left Upper Arm", left_shoulder, left_elbow, l_uarm_r_start, l_uarm_r_end, self.color_arms, 6, 4),

            # 8. Right Upper Arm (Tapered capsule)
            VolumetricSegment("Right Upper Arm", right_shoulder, right_elbow, r_uarm_r_start, r_uarm_r_end, self.color_arms, 6, 4),

            # 9. Left Forearm (Tapered capsule)
            VolumetricSegment("Left Forearm", left_elbow, left_wrist, l_farm_r_start, l_farm_r_end, self.color_arms, 6, 4),

            # 10. Right Forearm (Tapered capsule)
            VolumetricSegment("Right Forearm", right_elbow, right_wrist, r_farm_r_start, r_farm_r_end, self.color_arms, 6, 4),

            # 11. Left Hand (Volumetric capsule wrist to knuckles)
            VolumetricSegment("Left Hand", left_wrist, l_knuckle, l_farm_r_end * 0.9, hand_radius * 0.6, self.color_hands, 5, 3),

            # 12. Right Hand (Volumetric capsule wrist to knuckles)
            VolumetricSegment("Right Hand", right_wrist, r_knuckle, r_farm_r_end * 0.9, hand_radius * 0.6, self.color_hands, 5, 3),

            # 13. Left Thigh (Tapered capsule hip to knee)
            VolumetricSegment("Left Thigh", left_hip, left_knee, l_thigh_r_start, l_thigh_r_end, self.color_legs, 7, 5),

            # 14. Right Thigh (Tapered capsule hip to knee)
            VolumetricSegment("Right Thigh", right_hip, right_knee, r_thigh_r_start, r_thigh_r_end, self.color_legs, 7, 5),

            # 15. Left Lower Leg (Tapered capsule knee to ankle)
            VolumetricSegment("Left Lower Leg", left_knee, left_ankle, l_calf_r_start, l_calf_r_end, self.color_legs, 6, 4),

            # 16. Right Lower Leg (Tapered capsule knee to ankle)
            VolumetricSegment("Right Lower Leg", right_knee, right_ankle, r_calf_r_start, r_calf_r_end, self.color_legs, 6, 4),

            # 17. Left Foot (Elongated volumetric shape ankle to toe)
            VolumetricSegment("Left Foot", left_ankle, left_foot_idx, foot_radius, foot_radius * 0.7, self.color_feet, 5, 3),

            # 18. Right Foot (Elongated volumetric shape ankle to toe)
            VolumetricSegment("Right Foot", right_ankle, right_foot_idx, foot_radius, foot_radius * 0.7, self.color_feet, 5, 3),
        ]

        # 5. Inner Skeleton Bones (for distinct anatomical bone visualization)
        skeleton_bones: List[Tuple[np.ndarray, np.ndarray]] = [
            (nose, shoulder_mid),
            (shoulder_mid, hip_mid),
            (left_shoulder, right_shoulder),
            (left_hip, right_hip),
            (left_shoulder, left_elbow),
            (left_elbow, left_wrist),
            (left_wrist, l_knuckle),
            (right_shoulder, right_elbow),
            (right_elbow, right_wrist),
            (right_wrist, r_knuckle),
            (left_hip, left_knee),
            (left_knee, left_ankle),
            (left_ankle, left_foot_idx),
            (right_hip, right_knee),
            (right_knee, right_ankle),
            (right_ankle, right_foot_idx),
        ]

        # 6. Key Joint Centers for Landmark Nodes
        joint_positions: List[np.ndarray] = [
            nose, left_shoulder, right_shoulder,
            left_elbow, right_elbow, left_wrist, right_wrist,
            shoulder_mid, hip_mid,
            left_hip, right_hip, left_knee, right_knee,
            left_ankle, right_ankle,
        ]

        return segments, skeleton_bones, joint_positions
