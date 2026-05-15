"""SMPLer-X inference model — assembles encoder + heads + SMPL-X forward.

Port of `third_party/SMPLer-X/main/SMPLer_X.py:Model` (forward only —
training/loss code dropped). Uses the `smplx` PyPI package for the
parametric body model evaluation, no `torchgeometry`, no mmpose / mmcv.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants
from .config import SmplerXConfig
from .heads import (
    BodyRotationNet,
    BoxNet,
    FaceRegressor,
    HandRoI,
    HandRotationNet,
    PositionNet,
)
from .transforms import restore_bbox, rot6d_to_axis_angle
from .vit import ViT


class SmplerXModel(nn.Module):
    """Inference-time SMPLer-X.

    Constructor takes a `SmplerXConfig` so you can swap variants
    (ViT-S/B/L/H) without changing client code. State-dict layer names
    match upstream (after stripping the `module.` prefix), so the
    constructor signature is purely about architecture sizing.

    `smplx_path` should point to the directory containing
    `SMPLX_NEUTRAL.npz`. The constructor lazily imports the `smplx`
    PyPI package so users without it (e.g. running encoder+heads only
    for tests) can still instantiate.
    """

    def __init__(self, cfg: SmplerXConfig, smplx_path: Optional[str | Path] = None):
        super().__init__()
        self.cfg = cfg

        # ── Backbone ───────────────────────────────────────────────
        self.encoder = ViT(
            img_size=cfg.encoder.img_size,
            patch_size=cfg.encoder.patch_size,
            embed_dim=cfg.encoder.embed_dim,
            depth=cfg.encoder.depth,
            num_heads=cfg.encoder.num_heads,
            mlp_ratio=cfg.encoder.mlp_ratio,
            qkv_bias=cfg.encoder.qkv_bias,
            task_tokens_num=cfg.encoder.task_tokens_num,
        )

        # ── Heads ──────────────────────────────────────────────────
        self.body_position_net = PositionNet("body", feat_dim=cfg.feat_dim, cfg=cfg)
        self.body_regressor = BodyRotationNet(feat_dim=cfg.feat_dim)
        self.box_net = BoxNet(feat_dim=cfg.feat_dim, cfg=cfg)

        self.hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale, cfg=cfg)
        self.hand_position_net = PositionNet("hand", feat_dim=cfg.feat_dim, cfg=cfg)
        self.hand_regressor = HandRotationNet("hand", feat_dim=cfg.feat_dim)

        self.face_regressor = FaceRegressor(feat_dim=cfg.feat_dim)

        # ── SMPL-X parametric layer ────────────────────────────────
        # Upstream calls smplx.create(model_path, 'smplx', gender='NEUTRAL',
        # use_pca=False, use_face_contour=True) — replicate exactly.
        self.smplx_layer: Optional[nn.Module] = None
        if smplx_path is not None:
            self._load_smplx(smplx_path)

    def _load_smplx(self, smplx_path: str | Path) -> None:
        """Lazy-load the smplx parametric model. Called by constructor
        when ``smplx_path`` is provided, or manually later."""
        try:
            import smplx as _smplx
        except ImportError as e:
            raise ImportError(
                "smplx PyPI package is required for SMPL-X parametric "
                "evaluation. Install with `uv sync --extra smplerx` "
                "(adds `smplx` to the pin set)."
            ) from e

        # smplx.create probes `<model_path>/smplx/SMPLX_<GENDER>.npz`.
        # We accept either form: the directory holding `smplx/...` or
        # the directory holding `SMPLX_NEUTRAL.npz` directly.
        path = Path(smplx_path)
        if (path / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
            model_root = str(path)
        elif (path / "SMPLX_NEUTRAL.npz").is_file():
            # smplx.create expects a directory tree like
            # <model_root>/smplx/SMPLX_NEUTRAL.npz; build a virtual one.
            model_root = str(path.parent)
            # If `smplx` subdir doesn't exist, smplx.create won't find it;
            # the user is expected to drop the .npz directly under
            # `models/smplerx/` per our docs. Symlink-or-copy on the fly:
            target = path.parent / "smplx"
            target.mkdir(exist_ok=True)
            link = target / "SMPLX_NEUTRAL.npz"
            if not link.is_file():
                # Hardlink first (cheap), fall back to copy on cross-fs.
                try:
                    link.hardlink_to(path / "SMPLX_NEUTRAL.npz")
                except (OSError, AttributeError):
                    import shutil
                    shutil.copyfile(path / "SMPLX_NEUTRAL.npz", link)
        else:
            raise FileNotFoundError(
                f"Cannot find SMPLX_NEUTRAL.npz under {path}. "
                f"Download from https://smpl-x.is.tue.mpg.de/ and place "
                f"as `<smplx_path>/SMPLX_NEUTRAL.npz`."
            )

        self.smplx_layer = _smplx.create(
            model_root, "smplx", gender="NEUTRAL",
            use_pca=False, use_face_contour=True,
            create_global_orient=False, create_body_pose=False,
            create_left_hand_pose=False, create_right_hand_pose=False,
            create_jaw_pose=False, create_leye_pose=False,
            create_reye_pose=False, create_betas=False,
            create_expression=False, create_transl=False,
        )

    # ── Camera back-projection ────────────────────────────────────────

    def _get_camera_trans(self, cam_param: torch.Tensor) -> torch.Tensor:
        """(B, 3) → (B, 3) camera-space translation. Upstream uses a
        sigmoid on the z component scaled by the virtual focal length;
        replicated verbatim."""
        cfg = self.cfg
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = math.sqrt(
            cfg.focal[0] * cfg.focal[1]
            * cfg.camera_3d_size * cfg.camera_3d_size
            / (cfg.input_body_shape[0] * cfg.input_body_shape[1])
        )
        t_z = k_value * gamma
        return torch.cat((t_xy, t_z[:, None]), dim=1)

    # ── SMPL-X evaluation + projection ────────────────────────────────

    def _evaluate_smplx(
        self,
        root_pose: torch.Tensor,    # (B, 3)
        body_pose: torch.Tensor,    # (B, 63)
        lhand_pose: torch.Tensor,   # (B, 45)
        rhand_pose: torch.Tensor,   # (B, 45)
        jaw_pose: torch.Tensor,     # (B, 3)
        shape: torch.Tensor,        # (B, 10)
        expr: torch.Tensor,         # (B, 10)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (verts (B, 10475, 3), joints (B, 144, 3)) in
        SMPL-X canonical body-local frame (root not yet aligned).

        We return the **raw** smplx joint output, not SMPLer-X's
        137-joint permutation. The first 55 entries of
        ``smplx_layer(..).joints`` are the canonical SMPLX_55
        kinematic tree (matches our `syncrig_core.skeleton.topologies.
        smplx_55.LANDMARK_NAMES`); the trailing ~89 entries are face-
        contour and extra landmarks the consumer can slice as needed.
        """
        if self.smplx_layer is None:
            raise RuntimeError(
                "SMPL-X parametric layer not loaded. Pass smplx_path "
                "to the constructor or call _load_smplx() before forward."
            )
        device = root_pose.device
        zero_pose = torch.zeros((root_pose.shape[0], 3), device=device, dtype=root_pose.dtype)
        out = self.smplx_layer(
            betas=shape,
            body_pose=body_pose,
            global_orient=root_pose,
            right_hand_pose=rhand_pose,
            left_hand_pose=lhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=expr,
        )
        return out.vertices, out.joints

    # ── Inference forward ─────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> dict:
        """Single-detection inference.

        ``img`` should be the input_img_shape-sized tensor (B, 3, H, W)
        normalised the way ImageNet expects (mean/std). Returns a dict
        of unbatched-per-detection tensors (the caller is expected to
        index [0] when there's one person).
        """
        cfg = self.cfg

        # 1. Encoder — input_body_shape = (256, 192). Upstream interpolates
        #    img to input_body_shape before encoding; we let the caller
        #    pass an already-cropped tensor of the right shape (cleaner
        #    boundary), but accept any size and bilinear-resize as a
        #    safety net.
        in_h, in_w = cfg.input_body_shape
        if img.shape[-2:] != (in_h, in_w):
            img = F.interpolate(img, (in_h, in_w), mode="bilinear", align_corners=False)
        img_feat, task_tokens = self.encoder(img)

        shape_token = task_tokens[:, 0]
        cam_token = task_tokens[:, 1]
        expr_token = task_tokens[:, 2]
        jaw_pose_token = task_tokens[:, 3]
        # task_tokens[:, 4:6] are the hand tokens — only used during
        # training as a target signal; inference doesn't consume them.
        body_pose_token = task_tokens[:, 6:]

        # 2. Body regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param = self.body_regressor(
            body_pose_token, shape_token, cam_token, body_joint_img.detach(),
        )
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(
            body_pose.shape[0], -1,
        )
        cam_trans = self._get_camera_trans(cam_param)

        # 3. Hand + face bboxes
        lhand_bbox_center, lhand_bbox_size, \
            rhand_bbox_center, rhand_bbox_size, \
            face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())

        hand_aspect = cfg.input_hand_shape[1] / cfg.input_hand_shape[0]
        face_aspect = cfg.input_face_shape[1] / cfg.input_face_shape[0]
        hm_yx = (cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        body_hw = cfg.input_body_shape

        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, hand_aspect, 2.0,
                                  hm_yx, body_hw).detach()
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, hand_aspect, 2.0,
                                  hm_yx, body_hw).detach()
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, face_aspect, 1.5,
                                 hm_yx, body_hw).detach()

        # 4. Hand RoI crop + upsample → joint heatmap → 6D pose
        hand_feat = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)
        _, hand_joint_img = self.hand_position_net(hand_feat)
        hand_pose_6d = self.hand_regressor(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose_6d.reshape(-1, 6)).reshape(
            hand_feat.shape[0], -1,
        )

        # Restore left-hand orientation (network sees flipped left as right)
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size].reshape(
            batch_size, len(constants.ORIG_JOINT_PART["lhand"]), 3,
        )
        lhand_pose = torch.cat(
            (lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), dim=2,
        ).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:]

        # 5. Face regressor (expr + jaw 6D)
        expr, jaw_pose_6d = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose_6d)

        # 6. SMPL-X forward → mesh + joints (root-aligned)
        verts_local, joints_local = self._evaluate_smplx(
            root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr,
        )
        # Root-align (move pelvis to body-local origin). joints_local
        # already is in canonical pose via SMPL-X; subtract the pelvis
        # so downstream code matches MediaPipe's hip-anchored convention.
        root = joints_local[:, constants.ROOT_JOINT_IDX:constants.ROOT_JOINT_IDX + 1, :]
        joints_local = joints_local - root
        verts_local = verts_local - root

        return {
            "vertices": verts_local,                # (B, 10475, 3)
            "joints": joints_local,                 # (B, 137, 3)
            "smplx_root_pose": root_pose,           # (B, 3)
            "smplx_body_pose": body_pose,           # (B, 63)
            "smplx_lhand_pose": lhand_pose,         # (B, 45)
            "smplx_rhand_pose": rhand_pose,         # (B, 45)
            "smplx_jaw_pose": jaw_pose,             # (B, 3)
            "smplx_shape": shape,                   # (B, 10)
            "smplx_expr": expr,                     # (B, 10)
            "cam_trans": cam_trans,                 # (B, 3)
            "lhand_bbox": lhand_bbox,               # (B, 4) xyxy in input_body_shape coords
            "rhand_bbox": rhand_bbox,
            "face_bbox": face_bbox,
            # 2D landmark predictions (heatmap-based, not mesh-projected).
            # Provider-side back-projection avoids cam_t.z saturation
            # artefacts that affect the mesh / joint reprojection path
            # when the subject fills the input frame tightly.
            "body_joint_img": body_joint_img,       # (B, 25, 3) in output_hm_shape
            "hand_joint_img": hand_joint_img,       # (2*B, 20, 3) in output_hand_hm_shape
        }


def load_smplerx_model(
    checkpoint_path: str | Path,
    cfg: SmplerXConfig,
    smplx_path: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
) -> SmplerXModel:
    """Build a SmplerXModel + load weights, return ready-to-eval module.

    Strips the upstream ``module.`` DataParallel prefix on the way in.
    """
    model = SmplerXModel(cfg, smplx_path=smplx_path).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("network", ckpt)
    # Strip ``module.`` prefix.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return model, missing, unexpected
