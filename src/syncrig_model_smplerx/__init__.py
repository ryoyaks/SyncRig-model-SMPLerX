"""SMPLer-X provider for SyncRig.

Pip-installable plug-in for the SyncRig engine. Adds:

- ``smplx_55`` skeleton topology
- ``smplerx`` provider (SMPL-X body + hands + face from a single image)
- Retarget handler routing ``smplx_55`` frames to ``compute_smpl_pose``

License footprint (non-commercial-research only — read these before
deploying):

- SMPLer-X code + checkpoints: **S-Lab License 1.0** (research only)
- ``smplx`` PyPI package: Max Planck IS research license
- ``SMPLX_NEUTRAL.npz`` parametric model: MPI SMPL-X research license

Commercial deployments need separate licenses from S-Lab AND
Meshcapade. Mocap output remains under the same terms as the model.
"""
from __future__ import annotations

# Order matters:
# 1. ``retarget`` registers the ``smplx_55`` topology + dispatch handler.
# 2. ``provider`` registers the SmplerXProvider class.
# Both run at import time so the engine's autoload_entry_points hook
# only needs to load the package — the class registration and the
# retarget wiring happen automatically.
from . import retarget  # noqa: F401  (side-effect: SkeletonTopology + handler)
from .provider import SmplerXProvider  # noqa: F401  (side-effect: ProviderRegistry)

__version__ = "0.1.0"
__all__ = ["SmplerXProvider"]
