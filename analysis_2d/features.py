"""Kinematic feature extraction for gesture clustering analysis.

All spatial quantities (distances, positions) are normalized by each clip's own
shoulder width, since reference and participant videos are recorded at different
resolutions and camera distances. Speed/acceleration additionally divide by fps
so they are expressed in shoulder-widths/second (and /second^2).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# A per-frame nanmean over a handful of fingertip/hand landmarks is expected to hit
# all-NaN slices whenever every landmark in that frame was low-confidence; the
# resulting NaN is handled downstream by _nanmean_std, so the warning is just noise.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import MERGED_BODYPARTS  # noqa: E402

LIKELIHOOD_THRESHOLD = 0.1
MIN_VALID_FRACTION = 0.5  # a window/clip is dropped if fewer valid frames than this

FEATURE_NAMES = [
    "wrist_distance_mean", "wrist_distance_std",
    "lr_keypoint_distance_mean", "lr_keypoint_distance_std",
    "hand_speed_x_mean", "hand_speed_x_std",
    "hand_speed_y_mean", "hand_speed_y_std",
    "hand_accel_x_mean", "hand_accel_x_std",
    "hand_accel_y_mean", "hand_accel_y_std",
    "hand_opening_mean", "hand_opening_std",
    "index_thumb_distance_mean", "index_thumb_distance_std",
    "thumb_pinky_distance_mean", "thumb_pinky_distance_std",
    "hand_y_position_mean", "hand_y_position_std",
]

_IDX = {name: i for i, name in enumerate(MERGED_BODYPARTS)}
_LEFT_SHOULDER, _RIGHT_SHOULDER = _IDX["left_shoulder"], _IDX["right_shoulder"]
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


def _masked_xy(keypoints: np.ndarray, index: int) -> np.ndarray:
    """(T, 2) x,y with low-confidence points set to NaN."""
    xy = keypoints[:, index, :2].copy()
    invalid = keypoints[:, index, 2] < LIKELIHOOD_THRESHOLD
    xy[invalid] = np.nan
    return xy


def estimate_shoulder_width(keypoints: np.ndarray) -> float:
    """Median shoulder-to-shoulder distance over the whole clip; the scale reference
    used to normalize every other spatial feature for this clip."""
    left = _masked_xy(keypoints, _LEFT_SHOULDER)
    right = _masked_xy(keypoints, _RIGHT_SHOULDER)
    dist = np.linalg.norm(left - right, axis=1)
    valid = dist[np.isfinite(dist)]
    if len(valid) == 0:
        return float("nan")
    return float(np.median(valid))


def _nanmean_std(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) < max(2, MIN_VALID_FRACTION * len(values)):
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def compute_window_features(keypoints: np.ndarray, fps: float, shoulder_width: float) -> np.ndarray:
    """Compute the 20 aggregate kinematic features for one window/clip.

    keypoints: (T, n_bodyparts, 3) x,y,likelihood for a single window or full clip.
    Returns an array of len(FEATURE_NAMES); entries are NaN if too much data is missing.
    """
    if not np.isfinite(shoulder_width) or shoulder_width <= 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    left_wrist = _masked_xy(keypoints, _LEFT_WRIST_HAND)
    right_wrist = _masked_xy(keypoints, _RIGHT_WRIST_HAND)

    # 1. Distance between the wrists.
    wrist_distance = np.linalg.norm(left_wrist - right_wrist, axis=1) / shoulder_width
    wrist_distance_mean, wrist_distance_std = _nanmean_std(wrist_distance)

    # 2. Distance between matching left/right hand keypoints (21 landmarks per hand).
    left_hand = keypoints[:, _LEFT_HAND_IDX, :2].copy()
    right_hand = keypoints[:, _RIGHT_HAND_IDX, :2].copy()
    left_invalid = keypoints[:, _LEFT_HAND_IDX, 2] < LIKELIHOOD_THRESHOLD
    right_invalid = keypoints[:, _RIGHT_HAND_IDX, 2] < LIKELIHOOD_THRESHOLD
    left_hand[left_invalid] = np.nan
    right_hand[right_invalid] = np.nan
    lr_keypoint_distance = np.linalg.norm(left_hand - right_hand, axis=2) / shoulder_width  # (T, 21)
    lr_keypoint_distance = np.nanmean(lr_keypoint_distance, axis=1)
    lr_keypoint_distance_mean, lr_keypoint_distance_std = _nanmean_std(lr_keypoint_distance)

    # 3. Hand speed and acceleration in x, y (per hand, combined as the mean magnitude
    #    across both hands so that opposite-direction bimanual motion doesn't cancel out).
    left_velocity = np.diff(left_wrist, axis=0) * fps  # (T-1, 2), shoulder-widths handled after
    right_velocity = np.diff(right_wrist, axis=0) * fps
    left_velocity /= shoulder_width
    right_velocity /= shoulder_width
    speed_x = np.nanmean(np.stack([np.abs(left_velocity[:, 0]), np.abs(right_velocity[:, 0])]), axis=0)
    speed_y = np.nanmean(np.stack([np.abs(left_velocity[:, 1]), np.abs(right_velocity[:, 1])]), axis=0)
    hand_speed_x_mean, hand_speed_x_std = _nanmean_std(speed_x)
    hand_speed_y_mean, hand_speed_y_std = _nanmean_std(speed_y)

    left_accel = np.diff(left_velocity, axis=0) * fps
    right_accel = np.diff(right_velocity, axis=0) * fps
    accel_x = np.nanmean(np.stack([np.abs(left_accel[:, 0]), np.abs(right_accel[:, 0])]), axis=0)
    accel_y = np.nanmean(np.stack([np.abs(left_accel[:, 1]), np.abs(right_accel[:, 1])]), axis=0)
    hand_accel_x_mean, hand_accel_x_std = _nanmean_std(accel_x)
    hand_accel_y_mean, hand_accel_y_std = _nanmean_std(accel_y)

    # 4. Hand opening: mean fingertip-to-wrist distance, per hand, then averaged.
    left_fingertips = _masked_xy_multi(keypoints, _LEFT_FINGERTIP_IDX)
    right_fingertips = _masked_xy_multi(keypoints, _RIGHT_FINGERTIP_IDX)
    left_opening = np.nanmean(np.linalg.norm(left_fingertips - left_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    right_opening = np.nanmean(np.linalg.norm(right_fingertips - right_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    hand_opening = np.nanmean(np.stack([left_opening, right_opening]), axis=0)
    hand_opening_mean, hand_opening_std = _nanmean_std(hand_opening)

    # 5. Index-thumb (pinch) distance, per hand then averaged.
    left_index = _masked_xy(keypoints, _LEFT_INDEX_TIP)
    right_index = _masked_xy(keypoints, _RIGHT_INDEX_TIP)
    left_thumb = _masked_xy(keypoints, _LEFT_THUMB_TIP)
    right_thumb = _masked_xy(keypoints, _RIGHT_THUMB_TIP)
    left_pinch = np.linalg.norm(left_index - left_thumb, axis=1) / shoulder_width
    right_pinch = np.linalg.norm(right_index - right_thumb, axis=1) / shoulder_width
    pinch = np.nanmean(np.stack([left_pinch, right_pinch]), axis=0)
    index_thumb_distance_mean, index_thumb_distance_std = _nanmean_std(pinch)

    # 6. Thumb-pinky (spread) distance, per hand then averaged.
    left_pinky = _masked_xy(keypoints, _LEFT_PINKY_TIP)
    right_pinky = _masked_xy(keypoints, _RIGHT_PINKY_TIP)
    left_spread = np.linalg.norm(left_thumb - left_pinky, axis=1) / shoulder_width
    right_spread = np.linalg.norm(right_thumb - right_pinky, axis=1) / shoulder_width
    spread = np.nanmean(np.stack([left_spread, right_spread]), axis=0)
    thumb_pinky_distance_mean, thumb_pinky_distance_std = _nanmean_std(spread)

    # 7. Vertical (y) hand position relative to the shoulder line (positive = below shoulders).
    shoulder_y = np.nanmean(np.stack([
        _masked_xy(keypoints, _LEFT_SHOULDER)[:, 1], _masked_xy(keypoints, _RIGHT_SHOULDER)[:, 1],
    ]), axis=0)
    hand_y = np.nanmean(np.stack([left_wrist[:, 1], right_wrist[:, 1]]), axis=0)
    hand_y_position = (hand_y - shoulder_y) / shoulder_width
    hand_y_position_mean, hand_y_position_std = _nanmean_std(hand_y_position)

    return np.array([
        wrist_distance_mean, wrist_distance_std,
        lr_keypoint_distance_mean, lr_keypoint_distance_std,
        hand_speed_x_mean, hand_speed_x_std,
        hand_speed_y_mean, hand_speed_y_std,
        hand_accel_x_mean, hand_accel_x_std,
        hand_accel_y_mean, hand_accel_y_std,
        hand_opening_mean, hand_opening_std,
        index_thumb_distance_mean, index_thumb_distance_std,
        thumb_pinky_distance_mean, thumb_pinky_distance_std,
        hand_y_position_mean, hand_y_position_std,
    ])


def _masked_xy_multi(keypoints: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(T, len(indices), 2) x,y with low-confidence points set to NaN."""
    xy = keypoints[:, indices, :2].copy()
    invalid = keypoints[:, indices, 2] < LIKELIHOOD_THRESHOLD
    xy[invalid] = np.nan
    return xy


# Per-frame signal names, in the same order/definition as the *_mean/*_std pairs in
# FEATURE_NAMES (each FEATURE_NAMES entry is one of these aggregated with _nanmean_std).
SIGNAL_NAMES = [
    "wrist_distance",
    "lr_keypoint_distance",
    "hand_speed_x",
    "hand_speed_y",
    "hand_accel_x",
    "hand_accel_y",
    "hand_opening",
    "index_thumb_distance",
    "thumb_pinky_distance",
    "hand_y_position",
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

    left_wrist = _masked_xy(keypoints, _LEFT_WRIST_HAND)
    right_wrist = _masked_xy(keypoints, _RIGHT_WRIST_HAND)

    wrist_distance = np.linalg.norm(left_wrist - right_wrist, axis=1) / shoulder_width

    left_hand = keypoints[:, _LEFT_HAND_IDX, :2].copy()
    right_hand = keypoints[:, _RIGHT_HAND_IDX, :2].copy()
    left_invalid = keypoints[:, _LEFT_HAND_IDX, 2] < LIKELIHOOD_THRESHOLD
    right_invalid = keypoints[:, _RIGHT_HAND_IDX, 2] < LIKELIHOOD_THRESHOLD
    left_hand[left_invalid] = np.nan
    right_hand[right_invalid] = np.nan
    lr_keypoint_distance = np.nanmean(np.linalg.norm(left_hand - right_hand, axis=2) / shoulder_width, axis=1)

    left_velocity = np.diff(left_wrist, axis=0) * fps / shoulder_width
    right_velocity = np.diff(right_wrist, axis=0) * fps / shoulder_width
    speed_x = np.nanmean(np.stack([np.abs(left_velocity[:, 0]), np.abs(right_velocity[:, 0])]), axis=0)
    speed_y = np.nanmean(np.stack([np.abs(left_velocity[:, 1]), np.abs(right_velocity[:, 1])]), axis=0)

    left_accel = np.diff(left_velocity, axis=0) * fps
    right_accel = np.diff(right_velocity, axis=0) * fps
    accel_x = np.nanmean(np.stack([np.abs(left_accel[:, 0]), np.abs(right_accel[:, 0])]), axis=0)
    accel_y = np.nanmean(np.stack([np.abs(left_accel[:, 1]), np.abs(right_accel[:, 1])]), axis=0)

    left_fingertips = _masked_xy_multi(keypoints, _LEFT_FINGERTIP_IDX)
    right_fingertips = _masked_xy_multi(keypoints, _RIGHT_FINGERTIP_IDX)
    left_opening = np.nanmean(np.linalg.norm(left_fingertips - left_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    right_opening = np.nanmean(np.linalg.norm(right_fingertips - right_wrist[:, None, :], axis=2), axis=1) / shoulder_width
    hand_opening = np.nanmean(np.stack([left_opening, right_opening]), axis=0)

    left_index = _masked_xy(keypoints, _LEFT_INDEX_TIP)
    right_index = _masked_xy(keypoints, _RIGHT_INDEX_TIP)
    left_thumb = _masked_xy(keypoints, _LEFT_THUMB_TIP)
    right_thumb = _masked_xy(keypoints, _RIGHT_THUMB_TIP)
    left_pinch = np.linalg.norm(left_index - left_thumb, axis=1) / shoulder_width
    right_pinch = np.linalg.norm(right_index - right_thumb, axis=1) / shoulder_width
    index_thumb_distance = np.nanmean(np.stack([left_pinch, right_pinch]), axis=0)

    left_pinky = _masked_xy(keypoints, _LEFT_PINKY_TIP)
    right_pinky = _masked_xy(keypoints, _RIGHT_PINKY_TIP)
    left_spread = np.linalg.norm(left_thumb - left_pinky, axis=1) / shoulder_width
    right_spread = np.linalg.norm(right_thumb - right_pinky, axis=1) / shoulder_width
    thumb_pinky_distance = np.nanmean(np.stack([left_spread, right_spread]), axis=0)

    shoulder_y = np.nanmean(np.stack([
        _masked_xy(keypoints, _LEFT_SHOULDER)[:, 1], _masked_xy(keypoints, _RIGHT_SHOULDER)[:, 1],
    ]), axis=0)
    hand_y = np.nanmean(np.stack([left_wrist[:, 1], right_wrist[:, 1]]), axis=0)
    hand_y_position = (hand_y - shoulder_y) / shoulder_width

    def _lead_pad(signal: np.ndarray, lead_nans: int) -> np.ndarray:
        return np.concatenate([np.full(lead_nans, np.nan), signal])

    return {
        "wrist_distance": wrist_distance,
        "lr_keypoint_distance": lr_keypoint_distance,
        "hand_speed_x": _lead_pad(speed_x, 1),
        "hand_speed_y": _lead_pad(speed_y, 1),
        "hand_accel_x": _lead_pad(accel_x, 2),
        "hand_accel_y": _lead_pad(accel_y, 2),
        "hand_opening": hand_opening,
        "index_thumb_distance": index_thumb_distance,
        "thumb_pinky_distance": thumb_pinky_distance,
        "hand_y_position": hand_y_position,
    }
