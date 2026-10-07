"""
body_mesh.py
-------------
Module: 3D Body Representation (Phase 2 & Phase 3 Bridge)

Preserves backwards compatibility for existing imports while delegating
to the modular Phase 3 architecture:
- Volumetric body generation: src/volumetric_body.py
- 3D viewport & camera management: src/visualization_3d.py
"""

from volumetric_body import VolumetricBodyMesh, VolumetricSegment
from visualization_3d import Viewport3D

# Backwards compatibility aliases
BodyMeshVisualizer = Viewport3D
BodySegment = VolumetricSegment

__all__ = [
    "VolumetricBodyMesh",
    "VolumetricSegment",
    "Viewport3D",
    "BodyMeshVisualizer",
    "BodySegment",
]
