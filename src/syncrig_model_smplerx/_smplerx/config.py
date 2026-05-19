"""Hyperparams that bind the architecture to a particular SMPLer-X size.

Replaces the upstream `cfg` Singleton from `main/config/config.py` +
`main/config/config_smpler_x_<size>.py`. Only inference-relevant fields
are captured; training / dataloading / loss config is dropped.

State-dict layout for a given variant uniquely determines the encoder
dims: see ``CHECKPOINT_VARIANTS`` for the (variant_name, embed_dim,
depth, num_heads) triples we ship support for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    img_size: tuple[int, int]    # (H, W) — body crop input size
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.1
    task_tokens_num: int = 31    # 1 shape + 1 cam + 1 expr + 1 jaw + 2 hand + 25 body


@dataclass(frozen=True)
class SmplerXConfig:
    """Inference-only config for one SMPLer-X variant."""

    variant: str
    encoder: EncoderConfig

    # Input / output shapes
    input_img_shape: tuple[int, int] = (512, 384)        # whole frame after crop
    input_body_shape: tuple[int, int] = (256, 192)        # encoder input (H, W)
    input_hand_shape: tuple[int, int] = (256, 256)
    input_face_shape: tuple[int, int] = (192, 192)
    output_hm_shape: tuple[int, int, int] = (16, 16, 12)   # (D, H, W) — body
    output_hand_hm_shape: tuple[int, int, int] = (16, 16, 16)  # (D, H, W) — hand

    # Camera back-projection (virtual; replaced at inference if known)
    focal: tuple[float, float] = (5000.0, 5000.0)
    princpt: tuple[float, float] = (192.0 / 2, 256.0 / 2)
    camera_3d_size: float = 2.5

    # Head feature dim (matches encoder.embed_dim)
    feat_dim: int = 384

    # Hand RoI upsample factor (output_hm_shape[H,W] × upscale = pre-roi feat map)
    upscale: int = 4


# Per-variant config. Two variants exposed in the picker:
#   ViT-S — 384/12/12, ~384 MB checkpoint, default auto-download
#   ViT-H corrected — 1280/32/16, ~6 GB checkpoint, manual download
# The intermediate b32 / l32 sizes and the un-corrected h32 weren't
# auto-fetched by the installer so they're omitted; adding back is
# a small change (one VARIANT_CONFIGS entry + one HFDownloadStep +
# one dropdown option per variant).
VARIANT_CONFIGS: dict[str, SmplerXConfig] = {
    "smpler_x_s32": SmplerXConfig(
        variant="smpler_x_s32",
        encoder=EncoderConfig(
            img_size=(256, 192), patch_size=16,
            embed_dim=384, depth=12, num_heads=12,
        ),
        feat_dim=384,
    ),
    "smpler_x_h32_correct": SmplerXConfig(
        variant="smpler_x_h32_correct",
        encoder=EncoderConfig(
            img_size=(256, 192), patch_size=16,
            embed_dim=1280, depth=32, num_heads=16,
        ),
        feat_dim=1280,
    ),
}


def variant_from_path(path: str) -> str:
    """Pick the matching VARIANT key from a checkpoint filename.

    Example: ``models/smplerx/smpler_x_s32.pth.tar`` → ``smpler_x_s32``.
    """
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.endswith(".pth.tar"):
        name = name[: -len(".pth.tar")]
    elif name.endswith(".pth"):
        name = name[: -len(".pth")]
    if name not in VARIANT_CONFIGS:
        raise KeyError(
            f"Unknown SMPLer-X variant {name!r}. "
            f"Known: {list(VARIANT_CONFIGS)}"
        )
    return name
