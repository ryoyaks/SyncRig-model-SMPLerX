"""SMPLer-X provider — single-monocular SMPL-X body+hands+face estimator.

Ships as ``syncrig-model-smplerx``: a pip-installable plugin for the
SyncRig engine. License-restricted (S-Lab 1.0 + SMPL-X MPI research)
so it's hosted in its own repo instead of bundled with public SyncRig.

Architecture is a torch-2.x port of the original SMPLer-X (CVPR 2023)
— see ``_smplerx`` subpackage docstring for why we bypass the upstream
mmpose / mmcv / mmdet stack. Person detection uses torchvision
FasterRCNN (BSD-3) via the engine-shared ``_person_detector`` helper.

Required files (under ``<models_root>/smplerx/``; the engine resolves
``models_root`` via ``syncrig_engine.paths.models_root()``):
  - ``smpler_x_<variant>.pth.tar``
  - ``SMPLX_NEUTRAL.npz``   (license-gated; user provides)

Outputs:
  - ``pose_world_landmarks``  55 SMPL-X joints, root-aligned (m)
  - ``pose_landmarks``        same 55 joints, image-space [0,1]
  - ``mesh_vertices`` / ``faces``  10475 vertices, ~21k faces
  - ``smpl_*``  concatenated axis-angle + betas + expression + cam
"""

from __future__ import annotations

import importlib.util as _ilu

for _required in ("torch", "torchvision", "smplx"):
    if _ilu.find_spec(_required) is None:
        raise ImportError(
            f"syncrig-model-smplerx requires its install extras "
            f"(missing module: {_required}). Run "
            "`uv pip install 'syncrig-model-smplerx[runtime]'` or "
            "`pip install syncrig-model-smplerx[runtime]` to pull "
            "torch + torchvision + smplx."
        )

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from syncrig_core.providers import (
    OutputKind,
    Provider,
    ProviderCapabilities,
    ProviderConfigField,
    ProviderOutput,
    ProviderRegistry,
)
from syncrig_core.providers.base import (
    HFDownloadStep,
    InstallStep,
    ManualFileStep,
)
from syncrig_core.skeleton import SkeletonTopology
from syncrig_engine.paths import provider_models_dir

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Models live under the engine's canonical models root — that's
# ``<engine-repo>/models/smplerx/`` by default, ``$SYNCRIG_MODELS_DIR/smplerx/``
# if the user set the override env var. External package doesn't need
# to know where the engine repo is.
_MODELS_DIR = provider_models_dir("smplerx")


# ImageNet normalisation for the encoder input (matches upstream).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


# ── 2D landmark assembly (SMPLX_55 layout) ────────────────────────────────
#
# Maps from PositionNet output sets (POS_JOINT_PART) into SMPLX_55 layout:
#
#   POS body (25 entries, _smplerx.constants.JOINTS_NAME[0:25]):
#     0 Pelvis   1 L_Hip    2 R_Hip    3 L_Knee   4 R_Knee
#     5 L_Ankle  6 R_Ankle  7 Neck     8 L_Shoulder 9 R_Shoulder
#     10 L_Elbow 11 R_Elbow 12 L_Wrist 13 R_Wrist 14 L_Big_toe
#     15 L_Small_toe 16 L_Heel 17 R_Big_toe 18 R_Small_toe 19 R_Heel
#     20 L_Ear   21 R_Ear   22 L_Eye   23 R_Eye   24 Nose
#
#   POS hand per side (20 entries, JOINTS_NAME[25:45] for left):
#     0..3  Thumb_1..4   4..7  Index_1..4   8..11 Middle_1..4
#     12..15 Ring_1..4   16..19 Pinky_1..4
#
# SMPLX_55 layout has 3 phalanx joints per finger (no _4 tip) — we drop the
# tips. SMPLX hand finger order is index/middle/pinky/ring/thumb.

# (smplx55_idx, body_jimg_idx) — direct copies from POS body.
_BODY_DIRECT: tuple[tuple[int, int], ...] = (
    (0, 0),    # pelvis
    (1, 1),    # left_hip
    (2, 2),    # right_hip
    (4, 3),    # left_knee
    (5, 4),    # right_knee
    (7, 5),    # left_ankle
    (8, 6),    # right_ankle
    (10, 14),  # left_foot   ← L_Big_toe (closest)
    (11, 17),  # right_foot  ← R_Big_toe
    (12, 7),   # neck
    (16, 8),   # left_shoulder
    (17, 9),   # right_shoulder
    (18, 10),  # left_elbow
    (19, 11),  # right_elbow
    (20, 12),  # left_wrist
    (21, 13),  # right_wrist
    (23, 22),  # left_eye_smplhf
    (24, 23),  # right_eye_smplhf
)

# (smplx55_idx, hand_jimg_idx) per side — tips dropped, finger order remapped.
_HAND_PER_SIDE: tuple[tuple[int, int], ...] = (
    # index 1/2/3  ← POS Index_1/2/3
    (0, 4), (1, 5), (2, 6),
    # middle      ← POS Middle_1/2/3
    (3, 8), (4, 9), (5, 10),
    # pinky       ← POS Pinky_1/2/3
    (6, 16), (7, 17), (8, 18),
    # ring        ← POS Ring_1/2/3
    (9, 12), (10, 13), (11, 14),
    # thumb       ← POS Thumb_1/2/3
    (12, 0), (13, 1), (14, 2),
)


def _build_pose_landmarks_55(
    body_jimg: np.ndarray,        # (25, 3)
    lhand_jimg: np.ndarray,       # (20, 3)  — feature map was H-flipped
    rhand_jimg: np.ndarray,       # (20, 3)
    lhand_bbox: np.ndarray,       # (4,) xyxy in input_body_shape
    rhand_bbox: np.ndarray,       # (4,) xyxy in input_body_shape
    body_hm_w: float, body_hm_h: float,
    hand_hm_w: float, hand_hm_h: float,
    in_body_w: float, in_body_h: float,
    x1c: float, y1c: float, bw: float, bh: float,
    frame_w: int, frame_h: int,
) -> list[list[float]]:
    """Assemble a 55-entry pose_landmarks list (SMPLX_55 layout) by back-
    projecting model heatmap predictions to the original frame coords,
    normalised to [0, 1] of (frame_w, frame_h).

    Body / hand / face joints come from PositionNet heatmaps; spine
    column, collars, head and jaw are interpolated from neighbours since
    the network doesn't predict them as 2D landmarks.
    """
    out: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(55)]

    # ── body crop coord (output_hm_shape) → original frame ───────────
    def body_to_frame(x_hm: float, y_hm: float) -> tuple[float, float]:
        # output_hm_shape ≡ input_body_shape (proportional) ≡ bbox (proportional)
        nx = (x_hm / body_hm_w) * bw + x1c
        ny = (y_hm / body_hm_h) * bh + y1c
        return nx / max(frame_w, 1), ny / max(frame_h, 1)

    # ── hand crop (output_hand_hm_shape, in hand bbox region) → frame ─
    # Left hand feat map was H-flipped before PositionNet (see HandRoI),
    # so the X axis is inverted relative to the bbox.
    def hand_to_frame(
        x_hm: float, y_hm: float, bbox_in: np.ndarray, *, flip_x: bool,
    ) -> tuple[float, float]:
        bx1, by1, bx2, by2 = bbox_in
        bw_in = bx2 - bx1
        bh_in = by2 - by1
        u = x_hm / hand_hm_w
        v = y_hm / hand_hm_h
        if flip_x:
            u = 1.0 - u
        x_in = bx1 + u * bw_in
        y_in = by1 + v * bh_in
        # input_body_shape → bbox (proportional) → original frame
        nx = (x_in / in_body_w) * bw + x1c
        ny = (y_in / in_body_h) * bh + y1c
        return nx / max(frame_w, 1), ny / max(frame_h, 1)

    # Body (direct mappings).
    for s_idx, b_idx in _BODY_DIRECT:
        nx, ny = body_to_frame(float(body_jimg[b_idx, 0]), float(body_jimg[b_idx, 1]))
        out[s_idx] = [nx, ny, 0.0]

    # Spine column (3 spine1 / 6 spine2 / 9 spine3) — interpolate hips
    # midpoint → neck. Hips midpoint stands in for the lower spine
    # anchor since SMPL-X spine1 sits just above the pelvis.
    pelvis = out[0]
    neck = out[12]
    hip_mid_x = (out[1][0] + out[2][0]) / 2.0
    hip_mid_y = (out[1][1] + out[2][1]) / 2.0
    for s_idx, t in ((3, 0.30), (6, 0.55), (9, 0.85)):
        nx = hip_mid_x * (1 - t) + neck[0] * t
        ny = hip_mid_y * (1 - t) + neck[1] * t
        out[s_idx] = [nx, ny, 0.0]
    # Pelvis sits at hips midpoint when body_jimg's Pelvis drifts (rare
    # but happens when the network is unsure); keep the heatmap value
    # since it's anatomically lower than hips midpoint.
    out[0] = pelvis

    # Collars (13 / 14) — midpoint between neck and corresponding shoulder.
    for s_idx, sh_idx in ((13, 16), (14, 17)):
        sh = out[sh_idx]
        out[s_idx] = [(neck[0] + sh[0]) / 2.0, (neck[1] + sh[1]) / 2.0, 0.0]

    # Head (15) — midpoint of L_Ear / R_Ear. Falls back to nose if ears
    # weren't reliable (rare).
    lear_x, lear_y = body_to_frame(float(body_jimg[20, 0]), float(body_jimg[20, 1]))
    rear_x, rear_y = body_to_frame(float(body_jimg[21, 0]), float(body_jimg[21, 1]))
    out[15] = [(lear_x + rear_x) / 2.0, (lear_y + rear_y) / 2.0, 0.0]

    # Jaw (22) — Nose is the closest predicted landmark; SMPL-X jaw sits
    # just below it. Midway between nose and head approximates the chin.
    nose_x, nose_y = body_to_frame(float(body_jimg[24, 0]), float(body_jimg[24, 1]))
    out[22] = [
        (nose_x + out[15][0]) / 2.0 + (nose_x - out[15][0]) * 0.5,
        (nose_y + out[15][1]) / 2.0 + (nose_y - out[15][1]) * 0.5,
        0.0,
    ]

    # Hands (25..39 left, 40..54 right).
    for s_off, jimg, bbox, flip in (
        (25, lhand_jimg, lhand_bbox, True),    # left side: feat-map was flipped
        (40, rhand_jimg, rhand_bbox, False),
    ):
        for sub_idx, h_idx in _HAND_PER_SIDE:
            nx, ny = hand_to_frame(
                float(jimg[h_idx, 0]), float(jimg[h_idx, 1]), bbox, flip_x=flip,
            )
            out[s_off + sub_idx] = [nx, ny, 0.0]

    return out


# Approximate metric scale used when the SMPL-X parametric layer is
# unavailable — see ``_build_pose_world_landmarks_55_no_smplx``. Picked
# so a typical adult subject (1.7 m, filling ~85 % of the bbox height)
# emits sensible numbers; the renderer's walking-baseline math
# (``rig_state.ts``) takes care of the rest.
_SUBJECT_HEIGHT_M = 1.7
_BBOX_FILL_FRAC = 0.85
# Depth grid coverage in metres. Empirical — SMPLer-X's heatmap-depth
# range covers roughly ±body-depth around the root, and the resulting
# pose-z values are most useful when "limb forward of torso" reads as
# positive z. The exact magnitude isn't critical because consumers
# treat z as relative.
_HM_DEPTH_RANGE_M = 1.5


def _build_pose_world_landmarks_55_no_smplx(
    body_jimg: np.ndarray,        # (25, 3) — x,y in body-hm coords, z in depth bin
    lhand_jimg: np.ndarray,       # (20, 3)
    rhand_jimg: np.ndarray,
    lhand_bbox: np.ndarray,
    rhand_bbox: np.ndarray,
    body_hm_w: float, body_hm_h: float, body_hm_depth: float,
    hand_hm_w: float, hand_hm_h: float, hand_hm_depth: float,
    in_body_w: float, in_body_h: float,
    bw: float, bh: float,
) -> list[list[float]]:
    """Degraded-mode 3D skeleton — used when SMPLX_NEUTRAL.npz is absent.

    The SMPL-X parametric layer's forward gives clean root-aligned 3D
    joints in body-local metres. Without it we reconstruct an
    *approximate* 3D pose from PositionNet's heatmap soft-argmax
    outputs: every joint already comes with (x, y, z) in heatmap-grid
    units, so we just need to convert to metres and root-align to
    pelvis.

    Calibration is intentionally rough — bbox height stands in for
    subject height, depth grid coverage is a rough constant. The
    renderer's walking-baseline math then anchors the first frame to
    the scene origin regardless, so absolute scale doesn't have to be
    pixel-perfect.

    Topology stays SMPLX_55, matching the 2D path's joint indexing
    (``_build_pose_landmarks_55``). Hands inherit the body wrist
    position when their bbox is degenerate; the SMPL-X spine column /
    collars / head / jaw entries are interpolated the same way the
    2D builder does.
    """
    out: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(55)]

    # bbox-pixel → metres. Assume the subject's torso fills ~85% of
    # the bbox vertically and a typical adult is 1.7 m tall.
    px_to_m = (_SUBJECT_HEIGHT_M / max(_BBOX_FILL_FRAC, 1e-3)) / max(bh, 1.0)
    # Heatmap z bin → metres relative to pelvis depth.
    z_body_to_m = _HM_DEPTH_RANGE_M / max(body_hm_depth, 1.0)
    z_hand_to_m = _HM_DEPTH_RANGE_M / max(hand_hm_depth, 1.0)

    pelvis_x_hm = float(body_jimg[0, 0])
    pelvis_y_hm = float(body_jimg[0, 1])
    pelvis_z_hm = float(body_jimg[0, 2])

    def body_to_world(b_idx: int) -> list[float]:
        # body_jimg coords are in body-hm units; convert to "pixels
        # relative to the original-frame bbox" first, then to metres.
        # body_hm grid maps proportionally onto bbox (input_body_shape
        # is bbox-aspect-padded), so the same scale factor works.
        x_hm = float(body_jimg[b_idx, 0])
        y_hm = float(body_jimg[b_idx, 1])
        z_hm = float(body_jimg[b_idx, 2])
        dx_px = (x_hm - pelvis_x_hm) / max(body_hm_w, 1.0) * bw
        # SyncRig uses image-y-down convention (matches MediaPipe's
        # world_landmarks output); renderer Y-flips at draw time.
        dy_px = (y_hm - pelvis_y_hm) / max(body_hm_h, 1.0) * bh
        dz_m = (z_hm - pelvis_z_hm) * z_body_to_m
        return [dx_px * px_to_m, dy_px * px_to_m, dz_m]

    def hand_to_world(
        jimg: np.ndarray, h_idx: int, bbox_in: np.ndarray, flip_x: bool,
    ) -> list[float]:
        bx1, by1, bx2, by2 = (float(v) for v in bbox_in)
        bw_in = bx2 - bx1
        bh_in = by2 - by1
        u = float(jimg[h_idx, 0]) / max(hand_hm_w, 1.0)
        v = float(jimg[h_idx, 1]) / max(hand_hm_h, 1.0)
        if flip_x:
            u = 1.0 - u
        # input_body_shape pixels relative to bbox; convert to original
        # frame pixels via the bw/bh proportional mapping.
        x_in = bx1 + u * bw_in
        y_in = by1 + v * bh_in
        # input_body_shape → bbox pixels (proportional). We use the
        # same body_to_world mapping by translating to a "virtual
        # body_jimg coord" so the metric conversion is consistent
        # across body and hand joints.
        x_px_in_bbox = (x_in / max(in_body_w, 1.0)) * bw
        y_px_in_bbox = (y_in / max(in_body_h, 1.0)) * bh
        # Pelvis in bbox pixels (for the relative offset).
        pelvis_px_x = (pelvis_x_hm / max(body_hm_w, 1.0)) * bw
        pelvis_px_y = (pelvis_y_hm / max(body_hm_h, 1.0)) * bh
        z_hm = float(jimg[h_idx, 2])
        # hand z is in hand_hm depth bins; convert to metres and
        # leave it relative to wrist depth (we don't have a wrist
        # depth from body_jimg in the same depth-grid scale, so just
        # offset from 0 inside hand depth — the wrist limb-link will
        # anchor it in the rendered chain).
        dz_m = (z_hm - hand_hm_depth / 2.0) * z_hand_to_m
        return [
            (x_px_in_bbox - pelvis_px_x) * px_to_m,
            (y_px_in_bbox - pelvis_px_y) * px_to_m,
            dz_m,
        ]

    # Body — direct mappings.
    for s_idx, b_idx in _BODY_DIRECT:
        out[s_idx] = body_to_world(b_idx)

    # Spine column — interpolate hips midpoint → neck on x/y/z.
    pelvis_xyz = out[0]
    neck_xyz = out[12]
    hip_mid = [
        (out[1][0] + out[2][0]) / 2.0,
        (out[1][1] + out[2][1]) / 2.0,
        (out[1][2] + out[2][2]) / 2.0,
    ]
    for s_idx, t in ((3, 0.30), (6, 0.55), (9, 0.85)):
        out[s_idx] = [
            hip_mid[0] * (1 - t) + neck_xyz[0] * t,
            hip_mid[1] * (1 - t) + neck_xyz[1] * t,
            hip_mid[2] * (1 - t) + neck_xyz[2] * t,
        ]
    out[0] = pelvis_xyz

    # Collars (13 / 14) — midpoint of neck and corresponding shoulder.
    for s_idx, sh_idx in ((13, 16), (14, 17)):
        sh = out[sh_idx]
        out[s_idx] = [
            (neck_xyz[0] + sh[0]) / 2.0,
            (neck_xyz[1] + sh[1]) / 2.0,
            (neck_xyz[2] + sh[2]) / 2.0,
        ]

    # Head (15) — midpoint of L_Ear / R_Ear.
    lear = body_to_world(20)
    rear = body_to_world(21)
    out[15] = [
        (lear[0] + rear[0]) / 2.0,
        (lear[1] + rear[1]) / 2.0,
        (lear[2] + rear[2]) / 2.0,
    ]

    # Jaw (22) — extrapolate from head along nose direction.
    nose = body_to_world(24)
    out[22] = [
        (nose[0] + out[15][0]) / 2.0 + (nose[0] - out[15][0]) * 0.5,
        (nose[1] + out[15][1]) / 2.0 + (nose[1] - out[15][1]) * 0.5,
        (nose[2] + out[15][2]) / 2.0 + (nose[2] - out[15][2]) * 0.5,
    ]

    # Hands (25..39 left, 40..54 right).
    for s_off, jimg, bbox, flip in (
        (25, lhand_jimg, lhand_bbox, True),
        (40, rhand_jimg, rhand_bbox, False),
    ):
        for sub_idx, h_idx in _HAND_PER_SIDE:
            out[s_off + sub_idx] = hand_to_world(jimg, h_idx, bbox, flip)

    return out


@ProviderRegistry.register
class SmplerXProvider(Provider):
    """SMPLer-X — SMPL-X full-body single-image regressor."""

    @classmethod
    def capabilities(cls) -> ProviderCapabilities:
        # SMPL-X parametric layer is optional at run time. With the
        # ``SMPLX_NEUTRAL.npz`` on disk we evaluate the full SMPL-X
        # forward and emit a posed mesh + clean root-aligned 3D
        # joints. Without it we degrade to:
        #   * 2D image-space landmarks (renderer's 2D overlay)
        #   * approximate 3D world landmarks reconstructed from the
        #     PositionNet heatmap depth bins — less accurate than the
        #     SMPL-X FK output but enough for the 3D viewport's
        #     wireframe to track the subject
        #   * SMPL-X axis-angle parameters (Blender / Unity SMPL-X rigs)
        # losing only the mesh.
        smplx_npz = _MODELS_DIR / "SMPLX_NEUTRAL.npz"
        has_smplx = smplx_npz.is_file()
        warnings: tuple[tuple[str, str, str], ...] = ()
        if not has_smplx:
            warnings = (
                (
                    "Mesh unavailable",
                    "SMPLX_NEUTRAL.npz is not on disk; the provider runs "
                    "in degraded mode. The 3D viewport still shows a "
                    "55-joint skeleton wireframe (approximated from the "
                    "model's depth heatmaps) and SMPL-X parameters still "
                    "flow to Blender / Unity, but the posed mesh is "
                    f"skipped. Drop the file at {smplx_npz} to enable "
                    "mesh + more accurate 3D joints.",
                    "https://smpl-x.is.tue.mpg.de/download.php",
                ),
            )
        # ``outputs`` also degrades — the engine uses this to grey-out
        # TrackingMask toggles for modalities the provider can't emit.
        outputs = {OutputKind.SKELETON, OutputKind.SMPL}
        if has_smplx:
            outputs.add(OutputKind.MESH)
        return ProviderCapabilities(
            name="smplerx",
            description="SMPLer-X — whole-body SMPL-X (body + 30 finger joints + face) with mesh. Heavier but expressive.",
            skeleton_topology=SkeletonTopology.SMPLX_55,
            outputs=frozenset(outputs),
            requires_gpu=True,  # CPU works but ~10× slower
            requires_extra="smplerx",
            fps_estimate=30,    # ViT-S; H32 ~15 fps
            device_kinds=frozenset({"cuda", "cpu"}),
            min_vram_gb=4.0,
            commercial="non-commercial",
            commercial_note=(
                "S-Lab License 1.0 (code + weights). The optional "
                "SMPL-X parametric model (MPI, research-only) prohibits "
                "commercial use of the mesh / 3D joints derived from "
                "it; the SMPL-X parameters themselves are SMPLer-X "
                "outputs subject to S-Lab terms only."
            ),
            runtime_warnings=warnings,
            user_label="SMPLer-X · whole-body mesh + fingers",
            user_tagline="Research only · CUDA recommended",
            config_schema=(
                ProviderConfigField(
                    name="weights",
                    label="Checkpoint",
                    type="enum",
                    default="smpler_x_h32_correct",
                    options=(
                        # Only the corrected ViT-H is offered. The
                        # smaller variants (s32 / b32 / l32) and the
                        # un-corrected h32 are intentionally removed —
                        # the install pipeline only fetches ONE
                        # checkpoint per install, so listing variants
                        # that aren't on disk leaves the user with a
                        # picker entry that fails on switch.
                        ("smpler_x_h32_correct", "ViT-H corrected (MPE 59.7, ~6 GB)"),
                    ),
                ),
            ),
        )

    @classmethod
    def is_ready(cls) -> tuple[bool, str]:
        """Check that at least one SMPLer-X checkpoint is on disk.

        ``SMPLX_NEUTRAL.npz`` is intentionally NOT required here — the
        provider degrades to a no-mesh, no-3D-joints mode without it (see
        capabilities.runtime_warnings). That mode still emits 2D landmarks
        and SMPL-X axis-angle parameters, which is enough for the Blender
        / Unity driver paths even though the 3D viewport's world-space
        skeleton + mesh are unavailable.
        """
        ckpts = list(_MODELS_DIR.glob("smpler_x_*.pth.tar")) if _MODELS_DIR.exists() else []
        if not ckpts:
            return False, (
                f"SMPLer-X checkpoint missing under {_MODELS_DIR}. "
                "Download at least one variant from HuggingFace "
                "(caizhongang/SMPLer-X)."
            )
        return True, ""

    @classmethod
    def install_steps(cls) -> list[InstallStep]:
        return [
            HFDownloadStep(
                repo="caizhongang/SMPLer-X",
                filename="smpler_x_h32_correct.pth.tar",
                target_dir="models/smplerx",
                gated=False,
                label="Download SMPLer-X ViT-H corrected checkpoint (~6 GB)",
            ),
            ManualFileStep(
                target_path="models/smplerx/SMPLX_NEUTRAL.npz",
                url="https://smpl-x.is.tue.mpg.de/download.php",
                instructions=(
                    "Register at smpl-x.is.tue.mpg.de, download "
                    "models_smplx_v1_1.zip, extract SMPLX_NEUTRAL.npz "
                    "into models/smplerx/. "
                    "COMMERCIAL USE WARNING: SMPLer-X itself is "
                    "released under the S-Lab License 1.0 (research / "
                    "non-commercial). The SMPL-X parametric model is "
                    "additionally licensed for non-commercial research "
                    "only by Max Planck IS. Both layers must be "
                    "separately licensed (via S-Lab and Meshcapade) "
                    "before any commercial use of SMPLer-X outputs."
                ),
                label="SMPL-X parametric model (research-only)",
            ),
        ]

    def __init__(self) -> None:
        self._model = None
        self._detector = None
        self._faces_list: list[list[int]] = []
        self._device: str = "cpu"
        # True only when SMPLX_NEUTRAL.npz was present at setup and
        # the SMPL-X parametric layer loaded successfully. Process()
        # checks this to decide whether to emit mesh + 3D joints.
        self._has_smplx: bool = False

    def setup(self, config: dict | None = None) -> None:
        import torch  # noqa: PLC0415

        cfg = config or {}
        variant = cfg.get("weights") or "smpler_x_h32_correct"

        ckpt_path = _MODELS_DIR / f"{variant}.pth.tar"
        smplx_npz = _MODELS_DIR / "SMPLX_NEUTRAL.npz"

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"SMPLer-X weights missing: {ckpt_path}\n"
                f"Download from HuggingFace: "
                f"`hf download caizhongang/SMPLer-X {variant}.pth.tar "
                f"--local-dir models/smplerx`"
            )

        # SMPLX_NEUTRAL.npz is optional. When present, we get mesh +
        # 3D world joints. When absent, we still get 2D landmarks +
        # SMPL-X axis-angle params (enough for Blender / Unity SMPL-X
        # rig drivers). Capability advertisement + runtime_warnings
        # surface the degraded state to the UI.
        smplx_path: str | None = (
            str(_MODELS_DIR) if smplx_npz.is_file() else None
        )
        if smplx_path is None:
            log.warning(
                "SMPLer-X: SMPLX_NEUTRAL.npz not at %s — running in "
                "degraded mode (no mesh, no 3D world joints). Drop the "
                "file to enable them.",
                smplx_npz,
            )

        # PersonDetector is the engine-shared torchvision FasterRCNN
        # wrapper — provided by syncrig-engine so every body provider
        # (ViTPose, SMPLer-X, …) gets the same person-bbox semantics
        # without each shipping its own copy.
        from syncrig_engine.providers._person_detector import PersonDetector  # noqa: PLC0415
        from ._smplerx.config import VARIANT_CONFIGS  # noqa: PLC0415
        from ._smplerx.model import SmplerXModel  # noqa: PLC0415

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        smplerx_cfg = VARIANT_CONFIGS[variant]
        try:
            model = SmplerXModel(smplerx_cfg, smplx_path=smplx_path)
            model = model.to(self._device).eval()
            ckpt = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
            state = {
                k[len("module."):] if k.startswith("module.") else k: v
                for k, v in ckpt["network"].items()
            }
            # strict=False: SMPL-X parametric layer is loaded separately
            # from the .npz, not stored in the checkpoint.
            missing, unexpected = model.load_state_dict(state, strict=False)
            non_smplx_missing = [k for k in missing if not k.startswith("smplx_layer.")]
            if non_smplx_missing or unexpected:
                log.warning(
                    "SmplerX state_dict load: non-smplx missing=%d unexpected=%d",
                    len(non_smplx_missing), len(unexpected),
                )
            self._model = model
            self._has_smplx = model.smplx_layer is not None

            # Cache mesh faces (constant SMPL-X topology, 20908 triangles).
            # Skipped in degraded mode — we won't be emitting mesh.
            if self._has_smplx:
                faces_np = np.asarray(model.smplx_layer.faces, dtype=np.int64)
                self._faces_list = faces_np.tolist()
            else:
                self._faces_list = []

            # Person detector — torchvision FasterRCNN (BSD-3,
            # AGPL-free). Same detector as ViTPose provider via the
            # shared `_person_detector.PersonDetector` helper.
            det_model = cfg.get("detector", "mobilenet_v3_320")
            self._detector = PersonDetector(model_name=det_model)
            self._detector.setup()

            mode = "full" if self._has_smplx else "degraded (no mesh / 3D joints)"
            log.info(
                "SmplerX loaded variant=%s device=%s faces=%d mode=%s",
                variant, self._device, len(self._faces_list), mode,
            )
        except Exception:  # pylint: disable=broad-except
            log.exception("Failed to initialise SMPLer-X")
            self._has_smplx = False
            self._model = None
            self._detector = None
            self._faces_list = []

    def process(self, frame: "NDArray[np.uint8]") -> ProviderOutput | None:
        if self._model is None or self._detector is None:
            return None
        import torch  # noqa: PLC0415

        # ── Detect the highest-confidence person bbox ─────────────────
        try:
            detections = self._detector.detect(frame)
        except Exception:  # pylint: disable=broad-except
            log.exception("SMPLer-X person detection failed")
            return None
        if not detections:
            return None
        # PersonDetector returns results sorted by descending score —
        # take the first (single-subject mocap assumption, same
        # convention as ROMP `show_largest=True`).
        x1, y1, x2, y2, _ = detections[0]

        # ── Crop + resize to (256, 192) with aspect-ratio padding ──────
        H, W = frame.shape[:2]
        target_h, target_w = 256, 192
        target_aspect = target_w / target_h     # 0.75 (portrait)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        # Match target aspect
        if bw / max(bh, 1) > target_aspect:
            bh = bw / target_aspect
        else:
            bw = bh * target_aspect
        # 1.25x context expansion — matches upstream
        # `process_bbox(ratio=1.25)` in
        # `third_party/SMPLer-X/common/utils/preprocessing.py`. With
        # 1.2 the body fills slightly more of the model input than in
        # training, biasing the predicted cam_t (depth + offset) and
        # producing a ~5 % horizontal drift in the 2D overlay vs the
        # actual person.
        bw *= 1.25
        bh *= 1.25

        x1c = cx - bw / 2.0
        y1c = cy - bh / 2.0
        x2c = cx + bw / 2.0
        y2c = cy + bh / 2.0

        # Pad outside-frame regions with zeros (cv2.copyMakeBorder route),
        # then crop + resize.
        pad_l = max(0, int(np.ceil(-x1c)))
        pad_t = max(0, int(np.ceil(-y1c)))
        pad_r = max(0, int(np.ceil(x2c - W)))
        pad_b = max(0, int(np.ceil(y2c - H)))
        if pad_l + pad_t + pad_r + pad_b > 0:
            padded = cv2.copyMakeBorder(
                frame, pad_t, pad_b, pad_l, pad_r,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        else:
            padded = frame
        x1p = int(round(x1c + pad_l))
        y1p = int(round(y1c + pad_t))
        x2p = int(round(x2c + pad_l))
        y2p = int(round(y2c + pad_t))
        crop = padded[y1p:y2p, x1p:x2p]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        crop_resized = cv2.resize(
            crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR,
        )

        # BGR → RGB → ImageNet-normalised float32 [3, H, W]
        rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - _MEAN) / _STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).unsqueeze(0).to(self._device)

        # ── Forward ────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                out = self._model(tensor)
        except Exception:  # pylint: disable=broad-except
            log.exception("SMPLer-X inference failed")
            return None

        # ── Map outputs to ProviderOutput ──────────────────────────────
        # smplx out.joints is 144-long; the first 55 entries are the
        # SMPL-X kinematic tree we registered as SMPLX_55. ``joints``
        # and ``vertices`` are only present when the SMPL-X parametric
        # layer loaded — see model.py forward.
        joints_tensor = out["joints"]                            # (B, 137, 3) or None
        verts_tensor = out["vertices"]                           # (B, 10475, 3) or None
        cam_t = out["cam_trans"][0]                              # (3,) torch

        # 2D landmark path — back-project the model's heatmap predictions
        # (body_joint_img + hand_joint_img) instead of the SMPL-X mesh
        # reprojection. The mesh path goes through cam_t.z which sigmoid-
        # saturates near 56m when the subject fills the input bbox tightly
        # (see _smplerx/model.py `_get_camera_trans`); the resulting joints
        # project at the wrong scale even though pelvis lands roughly
        # right. Heatmap predictions are direct image-space soft-argmaxes
        # so they sit on the actual body in the source frame, matching the
        # MP / ROMP / ViTPose providers' alignment behaviour.
        body_jimg = out["body_joint_img"][0]                    # (25, 3) in output_hm_shape units
        hand_jimg = out["hand_joint_img"]                       # (2, 20, 3) in output_hand_hm_shape units
        lhand_jimg = hand_jimg[0]                               # (20, 3)
        rhand_jimg = hand_jimg[1]                               # (20, 3)
        lhand_bbox_in = out["lhand_bbox"][0]                    # (4,) xyxy in input_body_shape coords
        rhand_bbox_in = out["rhand_bbox"][0]
        body_hm_w = float(self._model.cfg.output_hm_shape[2])
        body_hm_h = float(self._model.cfg.output_hm_shape[1])
        hand_hm_w = float(self._model.cfg.output_hand_hm_shape[2])
        hand_hm_h = float(self._model.cfg.output_hand_hm_shape[1])
        in_body_h, in_body_w = self._model.cfg.input_body_shape  # (256, 192)
        pose_landmarks = _build_pose_landmarks_55(
            body_jimg.detach().cpu().numpy(),
            lhand_jimg.detach().cpu().numpy(),
            rhand_jimg.detach().cpu().numpy(),
            lhand_bbox_in.detach().cpu().numpy(),
            rhand_bbox_in.detach().cpu().numpy(),
            body_hm_w, body_hm_h, hand_hm_w, hand_hm_h,
            float(in_body_w), float(in_body_h),
            x1c, y1c, bw, bh, W, H,
        )

        provider_out = ProviderOutput(skeleton_topology=SkeletonTopology.SMPLX_55)
        provider_out.pose_landmarks = pose_landmarks
        provider_out.visibility = [1.0] * 55

        # 3D world-space joints + mesh: full mode uses SMPL-X forward
        # output; degraded mode reconstructs a 55-joint skeleton from
        # PositionNet heatmap depth bins (see
        # ``_build_pose_world_landmarks_55_no_smplx``). Mesh is only
        # available in full mode — degraded mode skips it entirely.
        if joints_tensor is not None and verts_tensor is not None:
            joints_local = joints_tensor[0, :55].detach().cpu().numpy()  # (55, 3)
            verts_local = verts_tensor[0].detach().cpu().numpy()         # (10475, 3)
            provider_out.pose_world_landmarks = [
                [float(joints_local[i, 0]),
                 float(joints_local[i, 1]),
                 float(joints_local[i, 2])]
                for i in range(55)
            ]
            provider_out.mesh_vertices = [
                [float(verts_local[i, 0]),
                 float(verts_local[i, 1]),
                 float(verts_local[i, 2])]
                for i in range(verts_local.shape[0])
            ]
            provider_out.mesh_faces = self._faces_list
            provider_out.mesh_topology = "smplx"
        else:
            # Degraded — assemble a 55-joint world skeleton from the
            # heatmap outputs (xy + depth bin). Approximate metric
            # calibration; the renderer's walking-baseline math
            # anchors the first frame regardless.
            body_hm_d = float(self._model.cfg.output_hm_shape[0])
            hand_hm_d = float(self._model.cfg.output_hand_hm_shape[0])
            provider_out.pose_world_landmarks = _build_pose_world_landmarks_55_no_smplx(
                body_jimg.detach().cpu().numpy(),
                lhand_jimg.detach().cpu().numpy(),
                rhand_jimg.detach().cpu().numpy(),
                lhand_bbox_in.detach().cpu().numpy(),
                rhand_bbox_in.detach().cpu().numpy(),
                body_hm_w, body_hm_h, body_hm_d,
                hand_hm_w, hand_hm_h, hand_hm_d,
                float(in_body_w), float(in_body_h),
                bw, bh,
            )

        # SMPL-X parameters — concat in the canonical order for the
        # Blender driver: root_pose(3) + body_pose(63) + lhand(45) +
        # rhand(45) + jaw(3) = 159 axis-angle floats grouped into 53
        # (3,) rows (matches SMPL-X's 22 body + 30 hand + 1 jaw joint
        # rotational set, excluding the 2 eye joints we leave at zero).
        root_pose = out["smplx_root_pose"][0].detach().cpu().numpy()
        body_pose = out["smplx_body_pose"][0].detach().cpu().numpy()
        lhand_pose = out["smplx_lhand_pose"][0].detach().cpu().numpy()
        rhand_pose = out["smplx_rhand_pose"][0].detach().cpu().numpy()
        jaw_pose = out["smplx_jaw_pose"][0].detach().cpu().numpy()
        all_aa = np.concatenate([
            root_pose,                              # (3,)
            body_pose,                              # (63,) — 21 joints
            lhand_pose,                             # (45,) — 15 joints
            rhand_pose,                             # (45,) — 15 joints
            jaw_pose,                               # (3,)
        ], axis=0).reshape(-1, 3)                   # (53, 3)
        provider_out.smpl_rotations = [
            [float(all_aa[j, 0]), float(all_aa[j, 1]), float(all_aa[j, 2])]
            for j in range(all_aa.shape[0])
        ]
        provider_out.smpl_translation = [
            float(cam_t[0]), float(cam_t[1]), float(cam_t[2]),
        ]
        provider_out.smpl_betas = [
            float(v) for v in out["smplx_shape"][0].detach().cpu().numpy()
        ]
        provider_out.smpl_model_type = "smplx"
        return provider_out

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        self._model = None
        self._detector = None
        self._faces_list = []
        self._has_smplx = False
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:  # pylint: disable=broad-except
            pass
