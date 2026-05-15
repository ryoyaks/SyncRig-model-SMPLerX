"""Coordinate / rotation utilities used by the heads.

Lifted from `third_party/SMPLer-X/common/utils/transforms.py`. The
upstream `rot6d_to_axis_angle` uses `torchgeometry.rotation_matrix_to_
angle_axis`, a deprecated package whose only stable wheel chain is
torch 1.x. We replace it with a small torch-native quaternion-based
matrix → axis-angle here, no external dep.

All functions operate on CUDA tensors when given CUDA inputs and stay
on CPU otherwise — `.cuda()` calls in the upstream version are
removed because the parent module already places the tensors on the
right device via `.to(device)`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── Soft-argmax heatmap decoders ──────────────────────────────────────


def soft_argmax_2d(heatmap2d: torch.Tensor) -> torch.Tensor:
    """heatmap2d: (B, J, H, W) → (B, J, 2) float (x, y)."""
    batch_size, _, height, width = heatmap2d.shape
    flat = heatmap2d.reshape((batch_size, -1, height * width))
    flat = F.softmax(flat, dim=2)
    hm = flat.reshape((batch_size, -1, height, width))

    accu_x = hm.sum(dim=2)                 # (B, J, W)
    accu_y = hm.sum(dim=3)                 # (B, J, H)

    accu_x = accu_x * torch.arange(width, device=hm.device, dtype=hm.dtype)[None, None, :]
    accu_y = accu_y * torch.arange(height, device=hm.device, dtype=hm.dtype)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    return torch.cat((accu_x, accu_y), dim=2)


def soft_argmax_3d(heatmap3d: torch.Tensor) -> torch.Tensor:
    """heatmap3d: (B, J, D, H, W) → (B, J, 3) float (x, y, z)."""
    batch_size, _, depth, height, width = heatmap3d.shape
    flat = heatmap3d.reshape((batch_size, -1, depth * height * width))
    flat = F.softmax(flat, dim=2)
    hm = flat.reshape((batch_size, -1, depth, height, width))

    accu_x = hm.sum(dim=(2, 3))
    accu_y = hm.sum(dim=(2, 4))
    accu_z = hm.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width, device=hm.device, dtype=hm.dtype)[None, None, :]
    accu_y = accu_y * torch.arange(height, device=hm.device, dtype=hm.dtype)[None, None, :]
    accu_z = accu_z * torch.arange(depth, device=hm.device, dtype=hm.dtype)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    return torch.cat((accu_x, accu_y, accu_z), dim=2)


def sample_joint_features(img_feat: torch.Tensor, joint_xy: torch.Tensor) -> torch.Tensor:
    """img_feat: (B, C, H, W); joint_xy: (B, J, 2). → (B, J, C)."""
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), dim=2)[:, :, None, :]
    sampled = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]
    return sampled.permute(0, 2, 1).contiguous()


# ── Bounding-box restoration ──────────────────────────────────────────


def restore_bbox(
    bbox_center: torch.Tensor,
    bbox_size: torch.Tensor,
    aspect_ratio: float,
    extension_ratio: float,
    output_hm_shape_yx: tuple[int, int],
    input_body_shape_hw: tuple[int, int],
) -> torch.Tensor:
    """Replicates upstream `restore_bbox` without the global `cfg`.

    bbox_center: (B, 2) in heatmap (output_hm_shape) space.
    bbox_size:   (B, 2) heatmap units.
    output_hm_shape_yx: (H, W) of the heatmap (cfg.output_hm_shape[1:]).
    input_body_shape_hw: (H, W) of the encoder input body crop.
    Returns xyxy in input_body_shape coordinates, (B, 4).
    """
    bbox = bbox_center.view(-1, 1, 2) + torch.cat(
        (-bbox_size.view(-1, 1, 2) / 2.0, bbox_size.view(-1, 1, 2) / 2.0), dim=1
    )
    out_h, out_w = output_hm_shape_yx
    in_h, in_w = input_body_shape_hw
    bbox[:, :, 0] = bbox[:, :, 0] / out_w * in_w
    bbox[:, :, 1] = bbox[:, :, 1] / out_h * in_h
    bbox = bbox.view(-1, 4)

    # xyxy → xywh, enforce aspect ratio, then back to xyxy with extension
    xmin, ymin, xmax, ymax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    # match aspect ratio
    w_target = h * aspect_ratio
    h_target = w / aspect_ratio
    use_w = w > w_target
    new_w = torch.where(use_w, w, w_target) * extension_ratio
    new_h = torch.where(use_w, h_target, h) * extension_ratio
    return torch.stack(
        (cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0),
        dim=1,
    )


# ── 6D rotation → axis-angle ──────────────────────────────────────────


def _rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(B, 6) → (B, 3, 3) rotation matrix (Zhou et al. 2019)."""
    x = x.view(-1, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) → (B, 3) axis-angle, via quaternion (numerically stable
    for full SO(3) coverage, including angles near π).

    Uses Shoemake's branched matrix → quaternion conversion: pick the
    branch corresponding to the largest of (1+R00+R11+R22, 1+R00-R11-R22,
    1-R00+R11-R22, 1-R00-R11+R22) so we never divide by a tiny number.
    Then quat → angle-axis via `2*atan2(|v|, w)`.
    """
    # 1) Matrix → quaternion (qw, qx, qy, qz)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22

    # Four candidate denominators (will pick the largest per-row).
    cand = torch.stack([
        1.0 + trace,                # qw branch
        1.0 + m00 - m11 - m22,      # qx branch
        1.0 - m00 + m11 - m22,      # qy branch
        1.0 - m00 - m11 + m22,      # qz branch
    ], dim=-1)
    cand = cand.clamp(min=1e-12)
    sqrt_cand = torch.sqrt(cand)

    # Per-row outputs in each branch:
    qw = torch.stack([
        sqrt_cand[:, 0] * 0.5,
        (m21 - m12) / (sqrt_cand[:, 1] * 2.0),
        (m02 - m20) / (sqrt_cand[:, 2] * 2.0),
        (m10 - m01) / (sqrt_cand[:, 3] * 2.0),
    ], dim=-1)
    qx = torch.stack([
        (m21 - m12) / (sqrt_cand[:, 0] * 2.0),
        sqrt_cand[:, 1] * 0.5,
        (m01 + m10) / (sqrt_cand[:, 2] * 2.0),
        (m02 + m20) / (sqrt_cand[:, 3] * 2.0),
    ], dim=-1)
    qy = torch.stack([
        (m02 - m20) / (sqrt_cand[:, 0] * 2.0),
        (m01 + m10) / (sqrt_cand[:, 1] * 2.0),
        sqrt_cand[:, 2] * 0.5,
        (m12 + m21) / (sqrt_cand[:, 3] * 2.0),
    ], dim=-1)
    qz = torch.stack([
        (m10 - m01) / (sqrt_cand[:, 0] * 2.0),
        (m02 + m20) / (sqrt_cand[:, 1] * 2.0),
        (m12 + m21) / (sqrt_cand[:, 2] * 2.0),
        sqrt_cand[:, 3] * 0.5,
    ], dim=-1)

    branch = cand.argmax(dim=-1, keepdim=True)
    qw = qw.gather(-1, branch).squeeze(-1)
    qx = qx.gather(-1, branch).squeeze(-1)
    qy = qy.gather(-1, branch).squeeze(-1)
    qz = qz.gather(-1, branch).squeeze(-1)

    # Canonicalise to qw >= 0 so axis-angle direction is unique.
    sign = torch.where(qw < 0, -1.0, 1.0)
    qw, qx, qy, qz = qw * sign, qx * sign, qy * sign, qz * sign

    # 2) Quaternion → axis-angle. Magnitude of vector part is sin(θ/2),
    # scalar part is cos(θ/2); θ = 2 * atan2(|v|, w).
    v = torch.stack((qx, qy, qz), dim=-1)
    v_norm = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm.squeeze(-1), qw)
    # When v_norm ≈ 0 the rotation is identity → axis-angle is zero.
    safe = v_norm.clamp(min=1e-8)
    axis = v / safe
    aa = axis * angle.unsqueeze(-1)
    aa = torch.where(v_norm < 1e-8, torch.zeros_like(aa), aa)
    return aa


def rot6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
    """(B*J, 6) → (B*J, 3). Matches upstream signature."""
    R = _rot6d_to_matrix(x)
    aa = _rotation_matrix_to_axis_angle(R)
    aa = torch.where(torch.isnan(aa), torch.zeros_like(aa), aa)
    return aa
