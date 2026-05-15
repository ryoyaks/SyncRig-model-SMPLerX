"""SMPLer-X inference port — torch 2.x, no mmpose / mmcv / mmdet.

Upstream (`third_party/SMPLer-X`) pins to torch 1.12 + mmcv-full 1.7.1
through `mmpose.models.build_posenet` for the encoder and `mmcv.ops.
roi_align` in the hand-crop block. Neither is available on torch 2.x.

This package re-implements only the *inference path* in plain torch:

    vit.py        — ViT-S/B/L/H backbone with task tokens
    heads.py      — PositionNet / BodyRotationNet / BoxNet / HandRoI /
                    HandRotationNet / FaceRegressor (mmcv roi_align
                    swapped for torchvision.ops.roi_align)
    transforms.py — rot6d → axis-angle, soft argmax, joint sampling
    layers.py     — make_conv/linear/deconv_layers verbatim
    constants.py  — SMPL-X joint name / part / index tables
    config.py     — hyperparams from `config_smpler_x_<size>.py`
    model.py      — Model class (inference forward only)

State-dict layouts match upstream exactly so checkpoints load with
``strict=True`` after a uniform ``module.`` prefix strip.
"""
