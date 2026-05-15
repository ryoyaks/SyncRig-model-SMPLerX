"""Head modules — port of `third_party/SMPLer-X/common/nets/smpler_x.py`.

Two changes vs upstream:

1. `mmcv.ops.roi_align` → `torchvision.ops.roi_align` (drop-in
   compatible signature; the upstream call uses spatial_scale=1.0,
   sampling_ratio=0, mode='avg' / aligned=False which torchvision
   reproduces with `aligned=False`).
2. The global `cfg` Singleton is removed; all shape constants get
   passed in at construction. This is the only place where upstream
   reaches into the config from inside a `forward`, so threading it in
   is straightforward.

State-dict layer names match upstream verbatim so checkpoints load
strict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from . import constants
from .config import SmplerXConfig
from .layers import make_conv_layers, make_deconv_layers, make_linear_layers
from .transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d


# ── PositionNet ────────────────────────────────────────────────────────


class PositionNet(nn.Module):
    """Heatmap-based 3D joint regressor.

    `part='body'` → (B, 25, D, H, W) heatmap, soft-argmax → (B, 25, 3).
    `part='hand'` → (B, 20, D', H', W') heatmap → (B, 20, 3).
    """

    def __init__(self, part: str, feat_dim: int, cfg: SmplerXConfig):
        super().__init__()
        if part == "body":
            self.joint_num = len(constants.POS_JOINT_PART["body"])
            self.hm_shape = cfg.output_hm_shape
        elif part == "hand":
            self.joint_num = len(constants.POS_JOINT_PART["rhand"])
            self.hm_shape = cfg.output_hand_hm_shape
        else:
            raise ValueError(f"PositionNet: unsupported part={part!r}")
        self.conv = make_conv_layers(
            [feat_dim, self.joint_num * self.hm_shape[0]],
            kernel=1, stride=1, padding=0, bnrelu_final=False,
        )

    def forward(self, img_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        D, H, W = self.hm_shape
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, D, H, W)
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm_sm = nn.functional.softmax(
            joint_hm.view(-1, self.joint_num, D * H * W), dim=2,
        ).view(-1, self.joint_num, D, H, W)
        return joint_hm_sm, joint_coord


# ── HandRotationNet ────────────────────────────────────────────────────


class HandRotationNet(nn.Module):
    """Per-hand 15-joint 6D rotation regressor."""

    def __init__(self, part: str, feat_dim: int):
        super().__init__()
        del part  # 'hand' only — kept for upstream-call symmetry
        self.joint_num = len(constants.POS_JOINT_PART["rhand"])
        self.hand_conv = make_conv_layers(
            [feat_dim, 512], kernel=1, stride=1, padding=0,
        )
        self.hand_pose_out = make_linear_layers(
            [self.joint_num * 515, len(constants.ORIG_JOINT_PART["rhand"]) * 6],
            relu_final=False,
        )
        self.feat_dim = feat_dim

    def forward(self, img_feat: torch.Tensor,
                joint_coord_img: torch.Tensor) -> torch.Tensor:
        batch_size = img_feat.shape[0]
        img_feat = self.hand_conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:, :, :2])
        feat = torch.cat((img_feat_joints, joint_coord_img), dim=2)
        return self.hand_pose_out(feat.view(batch_size, -1))


# ── BodyRotationNet ────────────────────────────────────────────────────


class BodyRotationNet(nn.Module):
    """Body root + body-pose 6D rotation regressor + shape + cam."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.joint_num = len(constants.POS_JOINT_PART["body"])
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers(
            [self.joint_num * (512 + 3), 6], relu_final=False,
        )
        self.body_pose_out = make_linear_layers(
            [
                self.joint_num * (512 + 3),
                (len(constants.ORIG_JOINT_PART["body"]) - 1) * 6,
            ],
            relu_final=False,
        )
        self.shape_out = make_linear_layers(
            [feat_dim, constants.SHAPE_PARAM_DIM], relu_final=False,
        )
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, body_pose_token: torch.Tensor, shape_token: torch.Tensor,
                cam_token: torch.Tensor, body_joint_img: torch.Tensor):
        batch_size = body_pose_token.shape[0]
        shape_param = self.shape_out(shape_token)
        cam_param = self.cam_out(cam_token)

        body_pose_token = self.body_conv(body_pose_token)
        body_pose_token = torch.cat((body_pose_token, body_joint_img), dim=2)
        root_pose = self.root_pose_out(body_pose_token.view(batch_size, -1))
        body_pose = self.body_pose_out(body_pose_token.view(batch_size, -1))
        return root_pose, body_pose, shape_param, cam_param


# ── FaceRegressor ──────────────────────────────────────────────────────


class FaceRegressor(nn.Module):
    """Expression code + 6D jaw rotation regressor."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.expr_out = make_linear_layers(
            [feat_dim, constants.EXPR_CODE_DIM], relu_final=False,
        )
        self.jaw_pose_out = make_linear_layers([feat_dim, 6], relu_final=False)

    def forward(self, expr_token: torch.Tensor,
                jaw_pose_token: torch.Tensor):
        return self.expr_out(expr_token), self.jaw_pose_out(jaw_pose_token)


# ── BoxNet ─────────────────────────────────────────────────────────────


class BoxNet(nn.Module):
    """Predicts hand+face bbox center + size from img_feat + body heatmap."""

    def __init__(self, feat_dim: int, cfg: SmplerXConfig):
        super().__init__()
        self.joint_num = len(constants.POS_JOINT_PART["body"])
        self.cfg = cfg
        self.deconv = make_deconv_layers([
            feat_dim + self.joint_num * cfg.output_hm_shape[0],
            256, 256, 256,
        ])
        self.bbox_center = make_conv_layers(
            [256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False,
        )
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat: torch.Tensor, joint_hm: torch.Tensor):
        D, H, W = self.cfg.output_hm_shape
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * D, H, W)
        feat = torch.cat((img_feat, joint_hm), dim=1)
        feat = self.deconv(feat)

        bbox_center_hm = self.bbox_center(feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center = bbox_center[:, 0, :]
        rhand_center = bbox_center[:, 1, :]
        face_center = bbox_center[:, 2, :]

        lhand_feat = sample_joint_features(feat, lhand_center[:, None, :].detach())[:, 0, :]
        rhand_feat = sample_joint_features(feat, rhand_center[:, None, :].detach())[:, 0, :]
        face_feat = sample_joint_features(feat, face_center[:, None, :].detach())[:, 0, :]

        lhand_size = self.lhand_size(lhand_feat)
        rhand_size = self.rhand_size(rhand_feat)
        face_size = self.face_size(face_feat)

        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size


# ── HandRoI ────────────────────────────────────────────────────────────


class HandRoI(nn.Module):
    """Differentiable feature-level RoI crop + upsample for the two hands.

    Replaces `mmcv.ops.roi_align` with `torchvision.ops.roi_align`. The
    boxes argument format `(K, 5) = (batch_idx, x0, y0, x1, y1)` is
    identical between the two libraries.
    """

    def __init__(self, feat_dim: int, upscale: int, cfg: SmplerXConfig):
        super().__init__()
        self.upscale = upscale
        self.cfg = cfg
        if upscale == 1:
            self.deconv = make_conv_layers(
                [feat_dim, feat_dim], kernel=1, stride=1, padding=0,
                bnrelu_final=False,
            )
            self.conv = make_conv_layers(
                [feat_dim, feat_dim], kernel=1, stride=1, padding=0,
                bnrelu_final=False,
            )
        elif upscale == 2:
            self.deconv = make_deconv_layers([feat_dim, feat_dim // 2])
            self.conv = make_conv_layers(
                [feat_dim // 2, feat_dim], kernel=1, stride=1, padding=0,
                bnrelu_final=False,
            )
        elif upscale == 4:
            self.deconv = make_deconv_layers(
                [feat_dim, feat_dim // 2, feat_dim // 4]
            )
            self.conv = make_conv_layers(
                [feat_dim // 4, feat_dim], kernel=1, stride=1, padding=0,
                bnrelu_final=False,
            )
        elif upscale == 8:
            self.deconv = make_deconv_layers(
                [feat_dim, feat_dim // 2, feat_dim // 4, feat_dim // 8]
            )
            self.conv = make_conv_layers(
                [feat_dim // 8, feat_dim], kernel=1, stride=1, padding=0,
                bnrelu_final=False,
            )
        else:
            raise ValueError(f"upscale must be 1/2/4/8, got {upscale}")

    def forward(self, img_feat: torch.Tensor,
                lhand_bbox: torch.Tensor,
                rhand_bbox: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        in_h, in_w = cfg.input_body_shape
        _D, hm_h, hm_w = cfg.output_hm_shape
        _Dh, hand_hm_h, hand_hm_w = cfg.output_hand_hm_shape
        device = img_feat.device

        # Prepend per-batch idx column to make (K, 5) RoI tensors.
        idx_l = torch.arange(lhand_bbox.shape[0], device=device, dtype=torch.float32)[:, None]
        idx_r = torch.arange(rhand_bbox.shape[0], device=device, dtype=torch.float32)[:, None]
        lhand_box_full = torch.cat((idx_l, lhand_bbox), dim=1)
        rhand_box_full = torch.cat((idx_r, rhand_bbox), dim=1)

        img_feat = self.deconv(img_feat)

        # Map xyxy from input_body coords → upsampled-feature-map coords.
        # Scale only the 4 coord columns (1..4); leave the batch_idx
        # column 0 untouched.
        sx = hm_w * self.upscale / in_w
        sy = hm_h * self.upscale / in_h
        lhand_box_roi = lhand_box_full.clone()
        lhand_box_roi[:, 1] = lhand_box_full[:, 1] * sx
        lhand_box_roi[:, 2] = lhand_box_full[:, 2] * sy
        lhand_box_roi[:, 3] = lhand_box_full[:, 3] * sx
        lhand_box_roi[:, 4] = lhand_box_full[:, 4] * sy

        assert (hm_h * self.upscale, hm_w * self.upscale) == \
            (img_feat.shape[2], img_feat.shape[3]), \
            f"upsampled feat {img_feat.shape[2:]} != expected " \
            f"{(hm_h * self.upscale, hm_w * self.upscale)}"

        lhand_img_feat = roi_align(
            img_feat, lhand_box_roi,
            output_size=(hand_hm_h, hand_hm_w),
            spatial_scale=1.0, sampling_ratio=0, aligned=False,
        )
        # Flip left → right canonical orientation.
        lhand_img_feat = torch.flip(lhand_img_feat, dims=[3])

        rhand_box_roi = rhand_box_full.clone()
        rhand_box_roi[:, 1] = rhand_box_full[:, 1] * sx
        rhand_box_roi[:, 2] = rhand_box_full[:, 2] * sy
        rhand_box_roi[:, 3] = rhand_box_full[:, 3] * sx
        rhand_box_roi[:, 4] = rhand_box_full[:, 4] * sy

        rhand_img_feat = roi_align(
            img_feat, rhand_box_roi,
            output_size=(hand_hm_h, hand_hm_w),
            spatial_scale=1.0, sampling_ratio=0, aligned=False,
        )

        hand_img_feat = torch.cat((lhand_img_feat, rhand_img_feat), dim=0)
        return self.conv(hand_img_feat)
