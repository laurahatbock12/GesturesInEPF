"""Kinematic feature extraction for gesture clustering analysis, in 3D.

Mirrors analysis_2d/features.py: every spatial quantity is still normalized by each
clip's own shoulder width (now a true 3D distance, in meters, rather than a pixel
distance), and speed/acceleration still additionally divide by fps. Distances that were
2D (x, y) Euclidean distances in analysis_2d are now 3D (x, y, z) Euclidean distances.

Before any feature is computed, every frame's keypoints are re-expressed in a
body-centered coordinate frame (see to_body_frame): origin at the shoulder midpoint, x
axis toward the right shoulder, z axis toward the camera, y axis completing a
right-handed (up) frame. This is done per frame, so it cancels whole-body
translation/rotation relative to the camera -- without it, "x/y/z" would just mean
"camera left-right / camera up-down / camera depth", which mixes gesture shape with
incidental changes in where the person sits or how they're turned. With it, x/y/z
consistently mean "lateral / vertical / forward-back relative to the shoulders",
matching what the 2D pipeline implicitly assumed when it read x/y straight off frames
where participants mostly face the camera.

Two features have no 2D equivalent, added because 3D + this body frame make them
directly measurable for the first time: elbow_angle (how bent the arm is, independent
of camera viewing angle -- a 2D projection of joint angle is heavily confounded by
foreshortening) and hand_shoulder_distance (how far a hand reaches from its own
shoulder, e.g. for reach/extension gestures). hand_speed_z/hand_accel_z (motion toward
or away from the camera) are likewise new: 2D had no depth axis to measure them on.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# See analysis_2d/features.py for why this specific warning is expected/harmless here.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import MERGED_BODYPARTS  # noqa: E402

LIKELIHOOD_THRESHOLD = 0.1
MIN_VALID_FRACTION = 0.5  # a window/clip is dropped if fewer valid frames than this

FEATURE_NAMES = [
    "wrist_distance_mean", "wrist_distance_std",
    "lr_keypoint_distance_mean", "lr_keypoint_distance_std",
    "hand_speed_x_mean", "hand_speed_x_std",
    "hand_speed_y_mean", "hand_speed_y_std",
    "hand_speed_z_mean", "hand_speed_z_std",
    "hand_accel_x_mean", "hand_accel_x_std",
    "hand_accel_y_mean", "hand_accel_y_std",
    "hand_accel_z_mean", "hand_accel_z_std",
    "hand_opening_mean", "hand_opening_std",
    "index_thumb_distance_mean", "index_thumb_distance_std",
    "thumb_pinky_distance_mean", "thumb_pinky_distance_std",
    "hand_y_position_mean", "hand_y_position_std",
    "elbow_angle_mean", "elbow_angle_std",
    "hand_shoulder_distance_mean", "hand_shoulder_distance_std",
]

_IDX = {name: i for i, name in enumerate(MERGED_BODYPARTS)}
_LEFT_SHOULDER, _RIGHT_SHOULDER = _IDX["left_shoulder"], _IDX["right_shoulder"]
_LEFT_ELBOW, _RIGHT_ELBOW = _IDX["left_elbow"], _IDX["right_elbow"]
_LEFT_WRIST_BODY, _RIGHT_WRIST_BODY = _IDX["left_wrist"], _IDX["right_wrist"]
_LEFT_WRIST_HAND, _RIGHT_WRIST_HAND = _IDX["left_wrist_hand"], _IDX["right_wrist_hand"]

_HAND_LANDMARKS = ["wrist_hand"] + [
    f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "pinky") for joint in range(1, 5)
]
_LEFT_HAND_IDX = np.array([_IDX[f"left_{name}"] for name in _HAND_LANDMARKS])
_RIGHT_HAND_IDX = np.array([_IDX[f"right_{name}"] for name in _HAND_LANDMARKS])

_FINGERTIPS = ["thumb_4", "index_4", "middle_4", "ring_4", "pinky_4"]
_LEFT_FINGERTIP_IDX = np.array([_IDX[f"left_{name}"] for name in _FINGERTIPS])
_RIGHT_FINGERTIP_IDX = np.array([_IDX[f"right_{name}"] for name in _FINGERTIPS])

_LEFT_INDEX_TIP, _RIGHT_INDEX_TIP = _IDX["left_index_4"], _IDX["right_index_4"]
_LEFT_THUMB_TIP, _RIGHT_THUMB_TIP = _IDX["left_thumb_4"], _IDX["right_thumb_4"]
_LEFT_PINKY_TIP, _RIGHT_PINKY_TIP = _IDX["left_pinky_4"], _IDX["right_pinky_4"]


def _masked_xyz(keypoints: np.ndarray, index: int) -> np.ndarray:
    """(T, 3) x,y,z with low-confidence points set to NaN."""
    xyz = keypoints[:, index, :3].copy()
    invalid = keypoints[:, index, 3] < LIKELIHOOD_THRESHOLD
    xyz[invalid] = np.nan
    return xyz


def _masked_xyz_multi(keypoints: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(T, len(indices), 3) x,y,z with low-confidence points set to NaN."""
    xyz = keypoints[:, indices, :3].copy()
    invalid = keypoints[:, indices, 3] < LIKELIHOOD_THRESHOLD
    xyz[invalid] = np.nan
    return xyz


def estimate_shoulder_width(keypoints: np.ndarray) -> float:
    """Median 3D shoulder-to-shoulder distance over the whole clip; the scale reference
    used to normalize every other spatial feature for this clip. Computed on the raw
    (pre body-frame) keypoints -- a rigid transform can't change this distance anyway."""
    left = _masked_xyz(keypoints, _LEFT_SHOULDER)
    right = _masked_xyz(keypoints, _RIGHT_SHOULDER)
    dist = np.linalg.norm(left - right, axis=1)
    valid = dist[np.isfinite(dist)]
    if len(valid) == 0:
        return float("nan")
    return float(np.median(valid))


def _shoulder_frame(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame origin (shoulder midpoint) and rotation matrix mapping the original
    camera frame (X=image-right, Y=image-down, Z=depth away from the camera) into a
    body-centered frame. Rotation rows are the new x/y/z basis vectors, expressed in the
    original camera frame. Frames without a confident shoulder detection get NaN."""
    left = keypoints[:, _LEFT_SHOULDER, :3]
    right = keypoints[:, _RIGHT_SHOULDER, :3]
    valid = (
        (keypoints[:, _LEFT_SHOULDER, 3] >= LIKELIHOOD_THRESHOLD)
        & (keypoints[:, _RIGHT_SHOULDER, 3] >= LIKELIHOOD_THRESHOLD)
        & np.isfinite(left).all(axis=1)
        & np.isfinite(right).all(axis=1)
    )

    origin = (left + right) / 2.0

    x_axis = right - origin
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)

    # "Towards the camera" is the direction of decreasing depth, i.e. -Z in the original
    # camera frame; orthogonalized against x_axis (Gram-Schmidt) since the two aren't
    # exactly perpendicular whenever the shoulder line isn't perfectly camera-facing.
    camera_forward = np.zeros_like(x_axis)
    camera_forward[:, 2] = -1.0
    z_axis = camera_forward - np.sum(camera_forward * x_axis, axis=1, keepdims=True) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)

    y_axis = np.cross(z_axis, x_axis)  # right-handed: x cross y == z

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)  # (T, 3, 3)
    origin = np.where(valid[:, None], origin, np.nan)
    rotation = np.where(valid[:, None, None], rotation, np.nan)
    return origin, rotation


def to_body_frame(keypoints: np.ndarray) -> np.ndarray:
    """Re-express every keypoint's (x, y, z) in the per-frame body-centered coordinate
    system described in the module docstring. Likelihood (column 3) is carried over
    unchanged. Frames without a confident shoulder detection get NaN xyz throughout,
    since there is no frame to express them in.
    """
    origin, rotation = _shoulder_frame(keypoints)
    diff = keypoints[..., :3] - origin[:, None, :]
    body_xyz = np.einsum("tij,tkj->tki", rotation, diff)
    transformed = keypoints.copy()
    transformed[..., :3] = body_xyz
    return transformed


def _angle_at_vertex(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T,) angle in degrees at `vertex`, between the rays to `a` and to `b`."""
    v1 = a - vertex
    v2 = b - vertex
    cosine = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _nanmean_std(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) < max(2, MIN_VALID_FRACTION * len(values)):
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def compute_window_features(keypoints: np.ndarray, fps: float, shoulder_width: float) -> np.ndarray:
    """Compute the aggregate kinematic features for one window/clip.

    keypoints: (T, n_bodyparts, 4) x,y,z,likelihood for a single window or full clip, in
    the original camera frame (this function applies the body-frame transform itself).
    Returns an array of len(FEATURE_NAMES); entries are NaN if too much data is missing.
    """
    if not np.isfinite(shoulder_width) or shoulder_width <= 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    keypoints = to_body_frame(keypoints)

    left_shoulder = _masked_xyz(keypoints, _LEFT_SHOULDER)
    right_shoulder = _masked_xyz(keypoints, _RIGHT_SHOULDER)
    left_elbow = _masked_xyz(keypoints, _LEFT_ELBOW)
    right_elbow = _masked_xyz(keypoints, _RIGHT_ELBOW)
    left_wrist_body = _masked_xyz(keypoints, _LEFT_WRIST_BODY)
    right_wrist_body = _masked_xyz(keypoints, _RIGHT_WRIST_BODY)
    left_wrist = _masked_xyz(keypoints, _LEFT_WRIST_HAND)
    right_wrist = _masked_xyz(keypoints, _RIGHT_WRIST_HAND)

    # 1. Distance between the wrists.
    wrist_distance = np.linalg.norm(left_wrist - right_wrist, axis=1) / shoulder_width
    wrist_distance_mean, wrist_distance_std = _nanmean_std(wrist_distance)

    # 2. Distance between matching left/right hand keypoints (21 landmarks per hand).
    left_hand = _masked_xyz_multi(keypoints, _LEFT_HAND_IDX)
    right_hand = _masked_xyz_multi(keypoints, _RIGHT_HAND_IDX)
    lr_keypoint_distance = np.linalg.norm(left_hand - right_hand, axis=2) / shoulder_width  # (T, 21)
    lr_keypoint_distance = np.nanmean(lr_keypoint_distance, axis=1)
    lr_keypoint_distance_mean, lr_keypoint_distance_std = _nanmean_std(lr_keypoint_distance)

    # 3. Hand speed and acceleration in x, y, z (per hand, combined as the mean magnitude
    #    across both hands so that opposite-direction bimanual motion doesn't cancel out).
    left_velocity = np.diff(left_wrist, axis=0) * fps  # (T-1, 3), shoulder-widths handled after
    right_velocity = np.diff(right_wrist, axis=0) * fps
    left_velocity /= shoulder_width
    right_velocity /= shoulder_width
    speed_x = np.nanmean(np.stack([np.abs(left_velocity[:, 0]), np.abs(right_velocity[:, 0])]), axis=0)
    speed_y = np.nanmean(np.stack([np.abs(left_velocity[:, 1]), np.abs(right_velocity[:, 1])]), axis=0)
    speed_z = np.nanmean(np.stack([np.abs(left_velocity[:, 2]), np.abs(right_velocity[:, 2])]), axis=0)
    hand_speed_x_mean, hand_speed_x_std = _nanmean_std(speed_x)
    hand_speed_y_mean, hand_speed_y_std = _nanmean_std(speed_y)
    hand_speed_z_mean, hand_speed_z_std = _nanmean_std(speed_z)

    left_accel = np.diff(left_velocity, axis=0) * fps
    right_accel = np.diff(right_velocity, axis=0) * fps
    accel_x = np.nanmean(np.stack([np.abs(left_accel[:, 0]), np.abs(right_accel[:, 0])]), axis=0)
    accel_y = np.nanmean(np.stack([np.abs(left_accel[:, 1]), np.abs(right_accel[:, 1])]), axis=0)
    accel_z = np.nanmean(np.stack([np.abs(left_accel[:, 2]), np.abs(right_accel[:, 2])]), axis=0)
    hand_accel_x_mean, hand_accel_x_std = _nanmean_std(accel_x)
    hand_accel_y_mean, hand_accel_y_std = _nanmean_std(accel_y)
    hand_accel_z_mean, hand_accel_z_std = _nanmean_std(accel_z)

    # 4. Hand opening: mean fingertip-to-wrist distance, per hand, then averaged.
    left_fingertips = _masked_xyz_multi(keypoints, _LEFT_FINGERTIP_IDX)
    right_fingertips = _masked_xyz_multi(keypoints, _RIGHT_FINGERTIP_IDX)
    left_opening = np.nanmean(np.linalg.norm(left_fingertips - left_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    right_opening = np.nanmean(np.linalg.norm(right_fingertips - right_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    hand_opening = np.nanmean(np.stack([left_opening, right_opening]), axis=0)
    hand_opening_mean, hand_opening_std = _nanmean_std(hand_opening)

    # 5. Index-thumb (pinch) distance, per hand then averaged.
    left_index = _masked_xyz(keypoints, _LEFT_INDEX_TIP)
    right_index = _masked_xyz(keypoints, _RIGHT_INDEX_TIP)
    left_thumb = _masked_xyz(keypoints, _LEFT_THUMB_TIP)
    right_thumb = _masked_xyz(keypoints, _RIGHT_THUMB_TIP)
    left_pinch = np.linalg.norm(left_index - left_thumb, axis=1) / shoulder_width
    right_pinch = np.linalg.norm(right_index - right_thumb, axis=1) / shoulder_width
    pinch = np.nanmean(np.stack([left_pinch, right_pinch]), axis=0)
    index_thumb_distance_mean, index_thumb_distance_std = _nanmean_std(pinch)

    # 6. Thumb-pinky (spread) distance, per hand then averaged.
    left_pinky = _masked_xyz(keypoints, _LEFT_PINKY_TIP)
    right_pinky = _masked_xyz(keypoints, _RIGHT_PINKY_TIP)
    left_spread = np.linalg.norm(left_thumb - left_pinky, axis=1) / shoulder_width
    right_spread = np.linalg.norm(right_thumb - right_pinky, axis=1) / shoulder_width
    spread = np.nanmean(np.stack([left_spread, right_spread]), axis=0)
    thumb_pinky_distance_mean, thumb_pinky_distance_std = _nanmean_std(spread)

    # 7. Vertical hand position. In the body frame the shoulder midpoint is the origin
    #    and both shoulders sit exactly on the x axis (y=0 by construction), so the
    #    hand's own y coordinate already is its height relative to the shoulder line.
    hand_y = np.nanmean(np.stack([left_wrist[:, 1], right_wrist[:, 1]]), axis=0)
    hand_y_position = hand_y / shoulder_width
    hand_y_position_mean, hand_y_position_std = _nanmean_std(hand_y_position)

    # 8. Elbow angle: shoulder-elbow-wrist angle (degrees; 180 = fully extended), per arm
    #    then averaged. Uses the body pose's own wrist/elbow (not the hand model's), the
    #    natural arm skeleton and unaffected by the hand-depth outlier fix.
    left_elbow_angle = _angle_at_vertex(left_shoulder, left_elbow, left_wrist_body)
    right_elbow_angle = _angle_at_vertex(right_shoulder, right_elbow, right_wrist_body)
    elbow_angle = np.nanmean(np.stack([left_elbow_angle, right_elbow_angle]), axis=0)
    elbow_angle_mean, elbow_angle_std = _nanmean_std(elbow_angle)

    # 9. Hand-to-shoulder distance (reach/extension), per side then averaged.
    left_hand_shoulder = np.linalg.norm(left_wrist - left_shoulder, axis=1) / shoulder_width
    right_hand_shoulder = np.linalg.norm(right_wrist - right_shoulder, axis=1) / shoulder_width
    hand_shoulder_distance = np.nanmean(np.stack([left_hand_shoulder, right_hand_shoulder]), axis=0)
    hand_shoulder_distance_mean, hand_shoulder_distance_std = _nanmean_std(hand_shoulder_distance)

    return np.array([
        wrist_distance_mean, wrist_distance_std,
        lr_keypoint_distance_mean, lr_keypoint_distance_std,
        hand_speed_x_mean, hand_speed_x_std,
        hand_speed_y_mean, hand_speed_y_std,
        hand_speed_z_mean, hand_speed_z_std,
        hand_accel_x_mean, hand_accel_x_std,
        hand_accel_y_mean, hand_accel_y_std,
        hand_accel_z_mean, hand_accel_z_std,
        hand_opening_mean, hand_opening_std,
        index_thumb_distance_mean, index_thumb_distance_std,
        thumb_pinky_distance_mean, thumb_pinky_distance_std,
        hand_y_position_mean, hand_y_position_std,
        elbow_angle_mean, elbow_angle_std,
        hand_shoulder_distance_mean, hand_shoulder_distance_std,
    ])


# Per-frame signal names, in the same order/definition as the *_mean/*_std pairs in
# FEATURE_NAMES (each FEATURE_NAMES entry is one of these aggregated with _nanmean_std).
SIGNAL_NAMES = [
    "wrist_distance",
    "lr_keypoint_distance",
    "hand_speed_x",
    "hand_speed_y",
    "hand_speed_z",
    "hand_accel_x",
    "hand_accel_y",
    "hand_accel_z",
    "hand_opening",
    "index_thumb_distance",
    "thumb_pinky_distance",
    "hand_y_position",
    "elbow_angle",
    "hand_shoulder_distance",
]


def compute_raw_signals(keypoints: np.ndarray, fps: float, shoulder_width: float) -> dict[str, np.ndarray]:
    """Per-frame kinematic signals underlying compute_window_features, for plotting a
    clip's kinematics over time. Each returned array has length T (matching keypoints'
    frame axis); velocity/acceleration are NaN for the first 1/2 frames, since there is
    no prior frame to difference against there.
    """
    n_frames = keypoints.shape[0]
    if not np.isfinite(shoulder_width) or shoulder_width <= 0:
        return {name: np.full(n_frames, np.nan) for name in SIGNAL_NAMES}

    keypoints = to_body_frame(keypoints)

    left_shoulder = _masked_xyz(keypoints, _LEFT_SHOULDER)
    right_shoulder = _masked_xyz(keypoints, _RIGHT_SHOULDER)
    left_elbow = _masked_xyz(keypoints, _LEFT_ELBOW)
    right_elbow = _masked_xyz(keypoints, _RIGHT_ELBOW)
    left_wrist_body = _masked_xyz(keypoints, _LEFT_WRIST_BODY)
    right_wrist_body = _masked_xyz(keypoints, _RIGHT_WRIST_BODY)
    left_wrist = _masked_xyz(keypoints, _LEFT_WRIST_HAND)
    right_wrist = _masked_xyz(keypoints, _RIGHT_WRIST_HAND)

    wrist_distance = np.linalg.norm(left_wrist - right_wrist, axis=1) / shoulder_width

    left_hand = _masked_xyz_multi(keypoints, _LEFT_HAND_IDX)
    right_hand = _masked_xyz_multi(keypoints, _RIGHT_HAND_IDX)
    lr_keypoint_distance = np.nanmean(np.linalg.norm(left_hand - right_hand, axis=2) / shoulder_width, axis=1)

    left_velocity = np.diff(left_wrist, axis=0) * fps / shoulder_width
    right_velocity = np.diff(right_wrist, axis=0) * fps / shoulder_width
    speed_x = np.nanmean(np.stack([np.abs(left_velocity[:, 0]), np.abs(right_velocity[:, 0])]), axis=0)
    speed_y = np.nanmean(np.stack([np.abs(left_velocity[:, 1]), np.abs(right_velocity[:, 1])]), axis=0)
    speed_z = np.nanmean(np.stack([np.abs(left_velocity[:, 2]), np.abs(right_velocity[:, 2])]), axis=0)

    left_accel = np.diff(left_velocity, axis=0) * fps
    right_accel = np.diff(right_velocity, axis=0) * fps
    accel_x = np.nanmean(np.stack([np.abs(left_accel[:, 0]), np.abs(right_accel[:, 0])]), axis=0)
    accel_y = np.nanmean(np.stack([np.abs(left_accel[:, 1]), np.abs(right_accel[:, 1])]), axis=0)
    accel_z = np.nanmean(np.stack([np.abs(left_accel[:, 2]), np.abs(right_accel[:, 2])]), axis=0)

    left_fingertips = _masked_xyz_multi(keypoints, _LEFT_FINGERTIP_IDX)
    right_fingertips = _masked_xyz_multi(keypoints, _RIGHT_FINGERTIP_IDX)
    left_opening = np.nanmean(np.linalg.norm(left_fingertips - left_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    right_opening = np.nanmean(np.linalg.norm(right_fingertips - right_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    hand_opening = np.nanmean(np.stack([left_opening, right_opening]), axis=0)

    left_index = _masked_xyz(keypoints, _LEFT_INDEX_TIP)
    right_index = _masked_xyz(keypoints, _RIGHT_INDEX_TIP)
    left_thumb = _masked_xyz(keypoints, _LEFT_THUMB_TIP)
    right_thumb = _masked_xyz(keypoints, _RIGHT_THUMB_TIP)
    left_pinch = np.linalg.norm(left_index - left_thumb, axis=1) / shoulder_width
    right_pinch = np.linalg.norm(right_index - right_thumb, axis=1) / shoulder_width
    index_thumb_distance = np.nanmean(np.stack([left_pinch, right_pinch]), axis=0)

    left_pinky = _masked_xyz(keypoints, _LEFT_PINKY_TIP)
    right_pinky = _masked_xyz(keypoints, _RIGHT_PINKY_TIP)
    left_spread = np.linalg.norm(left_thumb - left_pinky, axis=1) / shoulder_width
    right_spread = np.linalg.norm(right_thumb - right_pinky, axis=1) / shoulder_width
    thumb_pinky_distance = np.nanmean(np.stack([left_spread, right_spread]), axis=0)

    hand_y = np.nanmean(np.stack([left_wrist[:, 1], right_wrist[:, 1]]), axis=0)
    hand_y_position = hand_y / shoulder_width

    left_elbow_angle = _angle_at_vertex(left_shoulder, left_elbow, left_wrist_body)
    right_elbow_angle = _angle_at_vertex(right_shoulder, right_elbow, right_wrist_body)
    elbow_angle = np.nanmean(np.stack([left_elbow_angle, right_elbow_angle]), axis=0)

    left_hand_shoulder = np.linalg.norm(left_wrist - left_shoulder, axis=1) / shoulder_width
    right_hand_shoulder = np.linalg.norm(right_wrist - right_shoulder, axis=1) / shoulder_width
    hand_shoulder_distance = np.nanmean(np.stack([left_hand_shoulder, right_hand_shoulder]), axis=0)

    def _lead_pad(signal: np.ndarray, lead_nans: int) -> np.ndarray:
        return np.concatenate([np.full(lead_nans, np.nan), signal])

    return {
        "wrist_distance": wrist_distance,
        "lr_keypoint_distance": lr_keypoint_distance,
        "hand_speed_x": _lead_pad(speed_x, 1),
        "hand_speed_y": _lead_pad(speed_y, 1),
        "hand_speed_z": _lead_pad(speed_z, 1),
        "hand_accel_x": _lead_pad(accel_x, 2),
        "hand_accel_y": _lead_pad(accel_y, 2),
        "hand_accel_z": _lead_pad(accel_z, 2),
        "hand_opening": hand_opening,
        "index_thumb_distance": index_thumb_distance,
        "thumb_pinky_distance": thumb_pinky_distance,
        "hand_y_position": hand_y_position,
        "elbow_angle": elbow_angle,
        "hand_shoulder_distance": hand_shoulder_distance,
    }
