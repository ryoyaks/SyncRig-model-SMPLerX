"""SMPL-X joint name / part / index tables.

Lifted from `third_party/SMPLer-X/common/utils/human_models.py:SMPLX`.
The upstream class loads parametric body models on import (CUDA, file
paths). We keep only the constant mappings that the inference path
needs: joint counts, joint-name lists, body / lhand / rhand / face
slices, root indices, and the joint-set permutation that maps SMPL-X's
native 144-joint output to the 137-joint set the network supervises.
"""

from __future__ import annotations

# Original SMPL-X joint set (53 joints): 22 body + 30 hand + 1 jaw.
ORIG_JOINT_NUM = 53
ORIG_JOINTS_NAME = (
    "Pelvis", "L_Hip", "R_Hip", "Spine_1", "L_Knee", "R_Knee",
    "Spine_2", "L_Ankle", "R_Ankle", "Spine_3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Index_1", "L_Index_2", "L_Index_3",
    "L_Middle_1", "L_Middle_2", "L_Middle_3",
    "L_Pinky_1", "L_Pinky_2", "L_Pinky_3",
    "L_Ring_1", "L_Ring_2", "L_Ring_3",
    "L_Thumb_1", "L_Thumb_2", "L_Thumb_3",
    "R_Index_1", "R_Index_2", "R_Index_3",
    "R_Middle_1", "R_Middle_2", "R_Middle_3",
    "R_Pinky_1", "R_Pinky_2", "R_Pinky_3",
    "R_Ring_1", "R_Ring_2", "R_Ring_3",
    "R_Thumb_1", "R_Thumb_2", "R_Thumb_3",
    "Jaw",
)
ORIG_JOINT_PART = {
    "body":  range(0, 22),    # Pelvis .. R_Wrist
    "lhand": range(22, 37),   # L_Index_1 .. L_Thumb_3
    "rhand": range(37, 52),   # R_Index_1 .. R_Thumb_3
    "face":  range(52, 53),   # Jaw
}
ORIG_ROOT_JOINT_IDX = 0  # Pelvis

# Supervised SMPL-X joint set (137 joints): 25 body + 40 hand + 72 face.
JOINT_NUM = 137
JOINTS_NAME = (
    # body (0..24): 25 joints
    "Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "Neck", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Big_toe", "L_Small_toe", "L_Heel",
    "R_Big_toe", "R_Small_toe", "R_Heel", "L_Ear", "R_Ear",
    "L_Eye", "R_Eye", "Nose",
    # left hand (25..44): 20 joints (5 fingers × 4)
    "L_Thumb_1", "L_Thumb_2", "L_Thumb_3", "L_Thumb_4",
    "L_Index_1", "L_Index_2", "L_Index_3", "L_Index_4",
    "L_Middle_1", "L_Middle_2", "L_Middle_3", "L_Middle_4",
    "L_Ring_1", "L_Ring_2", "L_Ring_3", "L_Ring_4",
    "L_Pinky_1", "L_Pinky_2", "L_Pinky_3", "L_Pinky_4",
    # right hand (45..64): 20 joints
    "R_Thumb_1", "R_Thumb_2", "R_Thumb_3", "R_Thumb_4",
    "R_Index_1", "R_Index_2", "R_Index_3", "R_Index_4",
    "R_Middle_1", "R_Middle_2", "R_Middle_3", "R_Middle_4",
    "R_Ring_1", "R_Ring_2", "R_Ring_3", "R_Ring_4",
    "R_Pinky_1", "R_Pinky_2", "R_Pinky_3", "R_Pinky_4",
    # face (65..136): 72 keypoints (Face_1 .. Face_72)
    *[f"Face_{i}" for i in range(1, 73)],
)
ROOT_JOINT_IDX = 0
LWRIST_IDX = 12
RWRIST_IDX = 13
NECK_IDX = 7

# Slices into the 137-joint set used by Model.get_coord
JOINT_PART = {
    "body":  range(0, 25),
    "lhand": range(25, 45),
    "rhand": range(45, 65),
    "hand":  range(25, 65),
    "face":  range(65, 137),
}

# joint_idx: permutation that selects 137 joints from SMPL-X's native
# (144-entry) joint output. Verbatim from human_models.py:71-90.
JOINT_IDX = (
    0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21,             # body 14
    60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55,                  # toes/heels/ears/eyes/nose 11 → body total 25
    37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68,              # left hand 12
    34, 35, 36, 69, 31, 32, 33, 70,                               # left hand 8 → lhand total 20
    52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73,              # right hand 12
    49, 50, 51, 74, 46, 47, 48, 75,                               # right hand 8 → rhand total 20
    22, 15,                                                       # jaw, head 2 (face begins)
    57, 56,                                                       # eyeballs 2
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85,                      # eyebrow 10
    86, 87, 88, 89,                                              # nose 4
    90, 91, 92, 93, 94,                                          # below nose 5
    95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,       # eyes 12
    107,                                                          # right mouth 1
    108, 109, 110, 111, 112,                                     # upper mouth 5
    113,                                                          # left mouth 1
    114, 115, 116, 117, 118,                                     # lower mouth 5
    119,                                                          # right lip 1
    120, 121, 122,                                               # upper lip 3
    123,                                                          # left lip 1
    124, 125, 126,                                               # lower lip 3
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143,                                     # face contour 17
)
assert len(JOINT_IDX) == JOINT_NUM, f"{len(JOINT_IDX)} != {JOINT_NUM}"

# PositionNet output joint set: 65 joints (25 body + 40 hand). Used by
# the network heatmap regressors (no face, since face goes through the
# face-regressor head from a token, not heatmap).
POS_JOINT_NUM = 65
POS_JOINT_PART = {
    "body":  range(0, 25),
    "lhand": range(25, 45),
    "rhand": range(45, 65),
    "hand":  range(25, 65),
}

# Constants the heads import directly.
SHAPE_PARAM_DIM = 10
EXPR_CODE_DIM = 10
