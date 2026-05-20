"""SMPL-X retarget registration.

Three things happen at module-import time:

1. ``SkeletonTopology.register("smplx_55")`` — adds the topology
   identifier to the engine's registry so adapters / probes / wire-
   format validators recognise it.
2. ``register_topology_handler("smplx_55", _handler)`` — wires our
   handler into ``compute_vrm_pose``'s dispatch dict.
3. The handler itself just defers to ``syncrig_engine.retarget.smpl
   .compute_smpl_pose`` — the underlying landmark IK already branches
   on ``smpl.model_type`` to do the right thing for SMPL-X vs SMPL
   bodies, so the in-tree code is the source of truth for the math.

Why import-time side effects rather than an explicit ``init()``: the
engine's plug-in autoload pattern (``ProviderRegistry.autoload_entry_points``)
loads the provider class only, but Python imports the parent package
first, which runs this module via ``syncrig_model_smplerx.__init__``.
By that point the engine is ready to dispatch and we're ready to
serve frames.
"""
from __future__ import annotations

from syncrig_core.skeleton import SkeletonTopology
from syncrig_engine.retarget._dispatch import register_topology_handler
from syncrig_engine.retarget.smpl import compute_smpl_pose

# Register the topology string. Idempotent — no-op if some other
# package or the core already added it.
SMPLX_55 = SkeletonTopology.register("smplx_55")


def _handler(payload, rest, rest_perp):
    """``TopologyRetargetHandler`` for ``smplx_55``.

    Same body-IK as ROMP's ``smpl_24`` path because ``compute_smpl_pose``
    handles both — branching on ``payload.smpl.model_type`` to enable
    the extra SMPL-X joints (eyes, jaw, fingers) when present.

    ``skel.confidence`` is forwarded so off-frame / occluded joints
    (SMPLer-X's PositionNet still emits coordinates for hallucinated
    body parts) don't drive phantom limb rotations. See
    ``compute_smpl_pose``'s ``confidence`` docstring.
    """
    skel = payload.skeleton
    if skel is None or not skel.world_landmarks:
        return None
    return compute_smpl_pose(
        skel.world_landmarks, payload.smpl, rest, rest_perp,
        confidence=list(skel.confidence) if skel.confidence else None,
    )


register_topology_handler("smplx_55", _handler)
