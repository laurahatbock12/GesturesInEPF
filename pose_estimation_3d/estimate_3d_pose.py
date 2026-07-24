"""Lift the merged 2D pose (results_*_merged.csv) into 3D using DA3 monocular depth.

DA3METRIC-LARGE (the default model in estimate_monocular_depth.py) does not predict its own
camera intrinsics, so its raw depth output is only relative, not metric (see the NOTE printed
by that script). To get a usable metric scale without any real camera calibration, this script
calibrates a single global depth scale factor per video from shoulder width: the Euclidean
image-plane distance between the left and right shoulder keypoints, at the raw depth sampled at
those points, is assumed to correspond to a real shoulder (biacromial) width of ~40cm.

An earlier version of this calibration used hand length (wrist to middle fingertip, ~20cm)
instead. That was dropped: the projected pixel distance between two points only tracks distance
from the camera if the segment between them keeps a stable real-world length and orientation,
and hands very much don't - fingers curl, hands rotate in and out of plane, constantly changing
the apparent wrist-to-fingertip span independent of how far the hand actually is from the
camera. Measured on real gesture footage, the implied per-frame scale factor from hand length
had a relative spread (std/median) of ~0.8 - useless for a single global constant. Shoulders are
far more rigid and camera-facing for someone seated at a desk; the same measurement on shoulder
width gave a relative spread of ~0.1, with ~20x more usable (confidently-detected) samples per
video.

If the depth was produced by a model that already outputs real metric depth (DA3NESTED-*, or
DA3METRIC-LARGE with intrinsics available), the scale factor is skipped entirely and the depth
is trusted as-is.

Camera intrinsics for backprojecting (x, y, Z) into (X, Y, Z) are not known for these
recordings either, so a pinhole model with fx = fy = frame width in pixels (~53 deg horizontal
FOV, a common default for uncalibrated webcams) and the principal point at the image center is
assumed. This is the same assumed focal length used in the shoulder-width calibration, so the
resulting (X, Y, Z) triplets are self-consistent even though the absolute scale is only as
accurate as that FOV assumption.
"""

from pathlib import Path
import argparse
import json
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the '3d' projection
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_io import load_depth_npz
from pose_estimation_2d.merge_poses import (
    BODY_BODYPARTS,
    HAND_BODYPARTS,
    HIDDEN_VISUALIZATION_INDICES,
    MERGED_BODYPARTS,
    frame_number,
)
from pose_estimation_2d.utils import get_skeleton
from project_config import PROJECT_FOLDER

COORDS_3D = ("x", "y", "z", "likelihood")
LEFT_HAND_INDICES = [MERGED_BODYPARTS.index(bp) for bp in HAND_BODYPARTS if bp.startswith("left_")]
RIGHT_HAND_INDICES = [MERGED_BODYPARTS.index(bp) for bp in HAND_BODYPARTS if bp.startswith("right_")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lift merged 2D pose + DA3 monocular depth into 3D pose."
    )
    parser.add_argument(
        "--project_folder",
        type=str,
        default=PROJECT_FOLDER,
    )
    parser.add_argument(
        "--school",
        type=str,
        default="2026_05_20_KantiMusegg",
        choices=["2026_06_23_KantiHeerbrugg", "2026_06_02_GR_KantiOlten", "2026_05_20_KantiMusegg", "2026_06_08_GymKirschgarten", "2026_06_22_GymImmensee"],
    )
    parser.add_argument("--participant_id", type=str, default="CA78UR")
    parser.add_argument(
        "--process_reference_videos",
        action="store_true",
        help="Process the reference videos in the project_folder instead of participant recordings",
    )
    parser.add_argument(
        "--shoulder_width_m",
        type=float,
        default=0.40,
        help="Assumed real-world left-to-right shoulder (biacromial) width, in meters.",
    )
    parser.add_argument(
        "--focal_length_px",
        type=float,
        default=None,
        help="Assumed pinhole focal length in pixels, at the ORIGINAL video resolution "
        "(shared for fx and fy; principal point is the image center). Defaults to the video's "
        "frame width, i.e. ~53deg horizontal FOV. Used both for the hand-size depth "
        "calibration and for backprojecting every keypoint's (x, y) into (X, Y).",
    )
    parser.add_argument(
        "--min_likelihood",
        type=float,
        default=0.5,
        help="Minimum 2D keypoint likelihood required to use a shoulder pair for the "
        "shoulder-width depth calibration. Does not affect which keypoints get lifted to 3D.",
    )
    parser.add_argument(
        "--hand_outlier_radius_m",
        type=float,
        default=0.15,
        help="DA3's monocular depth often 'bleeds' a fingertip's depth onto the background "
        "when the tip is thin or near an edge, throwing that one keypoint far in front of or "
        "behind the rest of the hand. Per hand and per frame, any keypoint whose raw 3D "
        "position is farther than this radius (meters) from that hand's own (median) center "
        "is treated as a bad depth sample: it's dropped and linearly interpolated in time from "
        "the nearest valid frames for that keypoint, before Z smoothing runs.",
    )
    parser.add_argument(
        "--z_smoothing_window",
        type=int,
        default=10,
        help="Window size (frames) for centered median smoothing applied to the lifted Z "
        "(depth) coordinate only. DA3's per-frame monocular depth is much jitchier "
        "frame-to-frame than the 2D pixel detections, so this window is much larger than "
        "merge_poses.py's default x/y smoothing window. X and Y are not smoothed directly, "
        "but are re-derived from the smoothed Z (via backprojection from the original 2D "
        "pixel positions), so they inherit the stabilization. Use 1 to disable.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save a visualization video with the 3D-lifted skeleton, colored by depth.",
    )
    parser.add_argument(
        "--visualization_frames",
        type=int,
        default=-1,
        help="Number of frames to render for --save_vis, or -1 for all frames.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_runs(project_folder, school, participant_id, process_reference_videos):
    if process_reference_videos:
        video_folders = [i for i in Path(project_folder).iterdir() if i.is_dir() and i.name.startswith("EPF_PL_")]
        runs = sorted(i for folder in video_folders for i in folder.iterdir() if i.name.endswith(".mp4"))
        if not runs:
            raise FileNotFoundError(f"No reference videos found in: {video_folders}")
    else:
        video_folder = Path(project_folder) / school / participant_id
        runs = sorted(
            i for i in video_folder.iterdir() if i.is_dir() and i.name.startswith("recording_") and i.name.endswith(".mp4")
        )
        if not runs:
            raise FileNotFoundError(f"No recording folders found in: {video_folder}")
    return runs


def resolve_run_io(run, process_reference_videos):
    if process_reference_videos:
        pose_dir = run.parent / "pose_estimation" / "merged_predictions"
        depth_dir = run.parent / "pose_estimation" / "depth_predictions"
        output_dir = run.parent / "pose_estimation" / "pose_3d_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = run
    else:
        pose_dir = depth_dir = output_dir = run
        candidates = [
            i for i in run.iterdir() if i.is_file() and i.suffix.lower() in (".mp4", ".avi") and run.name not in i.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No video file found in: {run}")
        video_path = candidates[0]

    merged_csv = pose_dir / f"results_{run.name}_merged.csv"
    depth_npz = depth_dir / f"depth_{run.name}.npz"
    return video_path, merged_csv, depth_npz, output_dir


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_depth_batched(depth, depth_idx, xs, ys, device, chunk_size=256):
    """Bilinear-samples `depth` (n_depth_frames, H, W) at (xs, ys) pixel coordinates - already
    in depth's own resolution - for every (pose-frame, keypoint) pair in one batched pass per
    chunk, instead of a Python loop over individual frames. Uses torch.nn.functional.grid_sample
    (GPU-accelerated when `device` is cuda), which is exactly the batched-bilinear-lookup
    primitive this needs.

    xs/ys: (n_frames, n_points) pixel coords, NaN where a keypoint wasn't detected.
    depth_idx: (n_frames,) index into depth's first axis for that pose frame, or -1 if that
    frame has no matching depth prediction (its row is left as NaN in the result).
    """
    depth_h, depth_w = depth.shape[1:]
    result = np.full(xs.shape, np.nan, dtype=np.float32)

    rows = np.flatnonzero(depth_idx >= 0)
    if rows.size == 0:
        return result

    finite = np.isfinite(xs) & np.isfinite(ys)
    grid_x = np.clip(xs, 0, depth_w - 1) / max(depth_w - 1, 1) * 2 - 1
    grid_y = np.clip(ys, 0, depth_h - 1) / max(depth_h - 1, 1) * 2 - 1
    # grid_sample needs finite input everywhere; positions we'll mask out afterwards.
    grid_x = np.where(finite, grid_x, 0.0).astype(np.float32)
    grid_y = np.where(finite, grid_y, 0.0).astype(np.float32)

    for start in range(0, rows.size, chunk_size):
        chunk = rows[start : start + chunk_size]
        depth_chunk = torch.from_numpy(np.ascontiguousarray(depth[depth_idx[chunk]]))
        depth_chunk = depth_chunk.to(device=device, dtype=torch.float32).unsqueeze(1)  # (B, 1, H, W)
        grid_chunk = np.stack([grid_x[chunk], grid_y[chunk]], axis=-1)  # (B, n_points, 2)
        grid_tensor = torch.from_numpy(grid_chunk).to(device=device).unsqueeze(1)  # (B, 1, n_points, 2)
        sampled = F.grid_sample(
            depth_chunk, grid_tensor, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B, 1, 1, n_points)
        result[chunk] = sampled.squeeze(1).squeeze(1).cpu().numpy()

    result[~finite] = np.nan
    return result


def load_merged_pose(merged_csv):
    if not merged_csv.exists():
        raise FileNotFoundError(f"Merged 2D pose not found (run merge_poses.py first): {merged_csv}")
    df = pd.read_csv(merged_csv, header=[0, 1, 2, 3], index_col=0)
    frame_labels = sorted(df.index, key=frame_number)
    xy = np.stack(
        [df.xs(bp, level="bodyparts", axis=1).to_numpy(dtype=np.float32) for bp in MERGED_BODYPARTS],
        axis=1,
    )  # (n_frames_in_csv, n_bodyparts, 3) in original CSV row order
    # Reindex rows to the sorted frame order.
    row_order = [df.index.get_loc(label) for label in frame_labels]
    xy = xy[row_order]
    return frame_labels, xy  # xy[..., 0]=x, [...,1]=y, [...,2]=likelihood


def load_depth(depth_npz):
    if not depth_npz.exists():
        raise FileNotFoundError(f"Depth predictions not found (run estimate_monocular_depth.py first): {depth_npz}")
    data = load_depth_npz(depth_npz)
    depth = data["depth"].astype(np.float32)
    frame_indices = data["frame_indices"]
    is_metric = bool(data["is_metric"])
    frame_to_depth_idx = {int(f): i for i, f in enumerate(frame_indices)}
    return depth, frame_to_depth_idx, is_metric


def fix_hand_depth_outliers(raw_z, xy, fx, cx, cy, hand_index_groups, radius_m, min_likelihood=0.05):
    """Per hand and per frame, flag any keypoint whose raw 3D position falls outside a
    `radius_m` sphere around that hand's own (median) center as a bad depth sample - a thin or
    edge-of-frame fingertip is where DA3's monocular depth most often 'bleeds' onto the
    background - then replace it by linearly interpolating that keypoint's Z in time from the
    nearest valid frames before/after. Gaps with no valid frame on one side (start/end of the
    run) are left as NaN rather than extrapolated.

    Each group in `hand_index_groups` must list that hand's wrist_hand root first (this is how
    LEFT_HAND_INDICES/RIGHT_HAND_INDICES are built). merge_poses.py already anchors wrist_hand to
    the same pixel position as the (more reliable) body wrist keypoint whenever it's detected, so
    its depth is sampled at that shared position; it's used here as a trusted reference point for
    centering/checking the rest of the hand but is never itself flagged or replaced, so it can't
    be pulled away from the body wrist it's meant to match.
    """
    n_frames = raw_z.shape[0]
    px_all, py_all, likelihood = xy[..., 0], xy[..., 1], xy[..., 2]
    fixed_z = raw_z.copy()
    frame_idx = np.arange(n_frames)

    for side_indices in hand_index_groups:
        side_indices = np.asarray(side_indices)
        z_side = raw_z[:, side_indices]
        x_side = (px_all[:, side_indices] - cx) * z_side / fx
        y_side = (py_all[:, side_indices] - cy) * z_side / fx
        valid_side = np.isfinite(z_side) & (likelihood[:, side_indices] >= min_likelihood)

        outlier_mask = np.zeros_like(valid_side)
        for frame_index in range(n_frames):
            valid_row = valid_side[frame_index]
            if valid_row.sum() < 2:
                continue
            points = np.stack([x_side[frame_index], y_side[frame_index], z_side[frame_index]], axis=-1)
            center = np.median(points[valid_row], axis=0)
            distance = np.linalg.norm(points - center, axis=-1)
            outlier_mask[frame_index] = valid_row & (distance > radius_m)
        outlier_mask[:, 0] = False  # wrist_hand anchor: never flagged/replaced, see docstring

        for local_index, bodypart_index in enumerate(side_indices):
            gap = outlier_mask[:, local_index]
            if not gap.any():
                continue
            col = fixed_z[:, bodypart_index]
            good = np.isfinite(col) & ~gap
            if good.sum() < 2:
                col[gap] = np.nan
                continue
            first, last = np.flatnonzero(good)[[0, -1]]
            interpolatable = gap & (frame_idx >= first) & (frame_idx <= last)
            col[interpolatable] = np.interp(frame_idx[interpolatable], frame_idx[good], col[good])
            col[gap & ~interpolatable] = np.nan

    return fixed_z


def smooth_z(z, window_size):
    """Centered median smoothing on per-keypoint Z (depth) trajectories, frames x bodyparts.
    NaNs are excluded from the window and left as NaN if no valid samples fall inside it."""
    if window_size < 1:
        raise ValueError("z_smoothing_window must be a positive integer.")
    if window_size == 1:
        return z.copy()

    n_frames = z.shape[0]
    half_before = window_size // 2
    half_after = window_size - half_before - 1
    valid = np.isfinite(z)
    smoothed = z.copy()
    for frame_index in range(n_frames):
        start = max(0, frame_index - half_before)
        end = min(n_frames, frame_index + half_after + 1)
        window_valid = valid[start:end]
        window_z = z[start:end]
        for keypoint_index in np.flatnonzero(valid[frame_index]):
            values = window_z[window_valid[:, keypoint_index], keypoint_index]
            if len(values):
                smoothed[frame_index, keypoint_index] = np.median(values)
    return smoothed


def calibrate_scale(frame_labels, xy, depth, frame_to_depth_idx, res_scale, min_likelihood, device):
    """Solve a single global raw-depth -> meters scale factor from shoulder width. See the
    module docstring for why shoulder width was chosen over hand length."""
    scale_x, scale_y = res_scale
    left_idx = MERGED_BODYPARTS.index("left_shoulder")
    right_idx = MERGED_BODYPARTS.index("right_shoulder")

    left = xy[:, left_idx]  # (n_frames, 3): x, y, likelihood
    right = xy[:, right_idx]
    pixel_length = np.hypot(right[:, 0] - left[:, 0], right[:, 1] - left[:, 1])

    depth_idx = np.array(
        [frame_to_depth_idx.get(frame_number(label), -1) for label in frame_labels], dtype=np.int64
    )
    xs = np.stack([left[:, 0], right[:, 0]], axis=1) * scale_x
    ys = np.stack([left[:, 1], right[:, 1]], axis=1) * scale_y
    depths = sample_depth_batched(depth, depth_idx, xs, ys, device)  # (n_frames, 2): left, right
    raw_z = depths.mean(axis=1)

    valid = (
        (left[:, 2] >= min_likelihood)
        & (right[:, 2] >= min_likelihood)
        & np.isfinite(left[:, :2]).all(axis=1)
        & np.isfinite(right[:, :2]).all(axis=1)
        & (pixel_length >= 1e-3)
        & np.isfinite(depths).all(axis=1)
        & (depths > 0).all(axis=1)
    )
    return (pixel_length[valid] * raw_z[valid]).tolist()  # == shoulder_width_m * fx / k


def lift_run(video_path, merged_csv, depth_npz, output_dir, run_name, args, device):
    output_csv = output_dir / f"results_{run_name}_3d.csv"
    output_meta = output_dir / f"results_{run_name}_3d_meta.json"
    output_vis = output_dir / f"visualization_{run_name}_3d.mp4"
    if output_csv.exists() and not args.force:
        print(f"3D pose already exists for {run_name} at {output_csv}. Skipping.")
        return

    frame_labels, xy = load_merged_pose(merged_csv)
    depth, frame_to_depth_idx, already_metric = load_depth(depth_npz)
    depth_h, depth_w = depth.shape[1:]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    res_scale = (depth_w / frame_width, depth_h / frame_height)

    fx = args.focal_length_px if args.focal_length_px is not None else float(frame_width)
    cx, cy = frame_width / 2.0, frame_height / 2.0

    if already_metric:
        scale = 1.0
        n_calibration_samples = 0
    else:
        ratios = calibrate_scale(frame_labels, xy, depth, frame_to_depth_idx, res_scale, args.min_likelihood, device)
        n_calibration_samples = len(ratios)
        if ratios:
            # ratio = pixel_length * raw_z = shoulder_width_m * fx / scale  =>  scale = shoulder_width_m * fx / ratio
            scale = args.shoulder_width_m * fx / float(np.median(ratios))
        else:
            scale = 1.0
            print(
                f"WARNING: no confident shoulder detections in {run_name}; could not calibrate depth "
                "scale. Saving 3D pose with UNCALIBRATED (raw) depth units for Z."
            )

    print(
        f"{run_name}: depth {'already metric' if already_metric else f'calibrated from {n_calibration_samples} shoulder samples, scale={scale:.4g}'}, "
        f"assumed focal length {fx:.1f}px"
    )

    n_frames, n_bodyparts = xy.shape[:2]
    xyz = np.full((n_frames, n_bodyparts, 4), np.nan, dtype=np.float32)
    xyz[..., 3] = xy[..., 2]  # carry over 2D likelihood unchanged

    depth_idx = np.array(
        [frame_to_depth_idx.get(frame_number(label), -1) for label in frame_labels], dtype=np.int64
    )
    raw_z = sample_depth_batched(
        depth, depth_idx, xy[..., 0] * res_scale[0], xy[..., 1] * res_scale[1], device
    ) * scale

    raw_z = fix_hand_depth_outliers(
        raw_z, xy, fx, cx, cy, [LEFT_HAND_INDICES, RIGHT_HAND_INDICES], args.hand_outlier_radius_m
    )

    # DA3's per-frame depth is much jitterier than the 2D pixel detections, so Z gets a heavy
    # dedicated smoothing pass; X/Y are then re-derived from the smoothed Z (backprojection from
    # the original, unsmoothed-here 2D pixel positions) instead of being smoothed directly.
    z = smooth_z(raw_z, args.z_smoothing_window)
    px_all = xy[..., 0]
    py_all = xy[..., 1]
    xyz[..., 0] = (px_all - cx) * z / fx
    xyz[..., 1] = (py_all - cy) * z / fx
    xyz[..., 2] = z

    columns = pd.MultiIndex.from_tuples(
        [("lifted-3d", "idv_0", bodypart, coord) for bodypart in MERGED_BODYPARTS for coord in COORDS_3D],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    pd.DataFrame(xyz.reshape(n_frames, -1), index=frame_labels, columns=columns).to_csv(output_csv)
    print(f"Saved 3D pose to: {output_csv}")

    with open(output_meta, "w") as f:
        json.dump(
            {
                "is_metric": bool(already_metric or n_calibration_samples > 0),
                "depth_scale": scale,
                "assumed_focal_length_px": fx,
                "shoulder_width_m": args.shoulder_width_m,
                "n_calibration_samples": n_calibration_samples,
                "frame_width": frame_width,
                "frame_height": frame_height,
            },
            f,
            indent=2,
        )

    if args.save_vis:
        save_visualization(video_path, output_vis, frame_labels, xy, xyz, frame_rate, args.visualization_frames)
        print(f"Saved 3D visualization to: {output_vis}")

    cap.release()


def compute_axis_limits(xyz, hidden_indices, min_likelihood=0.05, pad_frac=0.08):
    """Fixed (x, y, z) plot ranges (meters) covering every visible keypoint over the whole run,
    so the extra plot panels don't rescale/jitter from frame to frame."""
    visible = [i for i in range(xyz.shape[1]) if i not in hidden_indices]
    pts = xyz[:, visible, :3]
    likelihood = xyz[:, visible, 3]
    mask = np.isfinite(pts).all(axis=-1) & (likelihood >= min_likelihood)
    pts = pts[mask]
    if pts.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    lo = lo - span * pad_frac
    hi = hi + span * pad_frac
    return (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])


def _pose_skeleton_segments(pose, skeleton, hidden_indices, min_likelihood=0.05):
    """(point_1, point_2) xyz triplets for every visible, confident skeleton edge in one frame."""
    segments = []
    for bodypart_1, bodypart_2 in skeleton:
        i1, i2 = bodypart_1 - 1, bodypart_2 - 1
        if i1 in hidden_indices or i2 in hidden_indices:
            continue
        p1, p2 = pose[i1], pose[i2]
        if not np.isfinite(p1[:3]).all() or not np.isfinite(p2[:3]).all():
            continue
        if p1[3] < min_likelihood or p2[3] < min_likelihood:
            continue
        segments.append((p1[:3], p2[:3]))
    return segments


def _figure_to_bgr(fig, panel_w, panel_h):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    if bgr.shape[1] != panel_w or bgr.shape[0] != panel_h:
        # dpi * inches can round to an off-by-one pixel count; force the exact panel size.
        bgr = cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    return bgr


def _render_plane_panel(ax, fig, panel_w, panel_h, segments, pts, colors, x_idx, y_idx, xlim, ylim, xlabel, ylabel, title, invert_y):
    ax.clear()
    for p1, p2 in segments:
        ax.plot([p1[x_idx], p2[x_idx]], [p1[y_idx], p2[y_idx]], color="0.6", linewidth=1, zorder=1)
    if pts.size:
        ax.scatter(pts[:, x_idx], pts[:, y_idx], c=colors, s=18, zorder=2, edgecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    return _figure_to_bgr(fig, panel_w, panel_h)


def _render_3d_panel(ax, fig, panel_w, panel_h, segments, pts, colors, xlim, ylim, zlim, elev, azim):
    """Plots (X, Z, -Y) so the scene reads as X=right, Z=depth, up=up (Y grows downward in
    image/camera coordinates, so it's negated here for display only)."""
    ax.clear()
    for p1, p2 in segments:
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [-p1[1], -p2[1]], color="0.6", linewidth=1, zorder=1)
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=colors, s=18, zorder=2, edgecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(-ylim[1], -ylim[0])
    ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
    ax.set_ylabel("Z (m)", fontsize=7, labelpad=2)
    ax.set_zlabel("Y (m)", fontsize=7, labelpad=2)
    ax.set_title("3D view", fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)
    return _figure_to_bgr(fig, panel_w, panel_h)


def save_visualization(video_path, output_path, frame_labels, xy, xyz, frame_rate, max_frames):
    """4-panel visualization: top-left is the skeleton drawn at its original 2D pixel positions
    (xy), colored/labeled by the lifted depth Z (xyz) - the other 3 panels re-plot the same
    (X, Y, Z) keypoints (meters, not pixels) as an X-Z top view, a Z-Y side view, and a 3D view."""
    if max_frames == 0 or max_frames < -1:
        raise ValueError("visualization_frames must be positive or -1.")
    if max_frames > 0:
        frame_labels = frame_labels[:max_frames]
        xy = xy[:max_frames]
        xyz = xyz[:max_frames]

    labeled_indices = {MERGED_BODYPARTS.index(bodypart) for bodypart in BODY_BODYPARTS}
    _, skeleton = get_skeleton(mode="full_body")
    finite_z = xyz[..., 2][np.isfinite(xyz[..., 2])]
    if finite_z.size:
        z_lo, z_hi = np.percentile(finite_z, [2, 98])
    else:
        z_lo, z_hi = 0.0, 1.0
    z_hi = max(z_hi, z_lo + 1e-6)
    cmap = plt.get_cmap("turbo_r")

    xlim, ylim, zlim = compute_axis_limits(xyz, HIDDEN_VISUALIZATION_INDICES)

    capture = cv2.VideoCapture(str(video_path))
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or frame_rate
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w, panel_h = frame_width // 2, frame_height // 2

    dpi = 100
    fig_xz, ax_xz = plt.subplots(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
    fig_zy, ax_zy = plt.subplots(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
    fig_3d = plt.figure(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    fig_xz.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.90)
    fig_zy.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.90)
    fig_3d.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)

    writer = None
    try:
        for frame_label, pixels, pose in tqdm(zip(frame_labels, xy, xyz), total=len(xyz), desc="Creating 3D visualization"):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number(frame_label))
            success, frame = capture.read()
            if not success:
                continue

            for bodypart_1, bodypart_2 in skeleton:
                if bodypart_1 - 1 in HIDDEN_VISUALIZATION_INDICES or bodypart_2 - 1 in HIDDEN_VISUALIZATION_INDICES:
                    continue
                px1, px2 = pixels[bodypart_1 - 1], pixels[bodypart_2 - 1]
                z1, z2 = pose[bodypart_1 - 1, 2:4], pose[bodypart_2 - 1, 2:4]
                if not np.isfinite(px1).all() or not np.isfinite(px2).all() or z1[1] < 0.05 or z2[1] < 0.05:
                    continue
                cv2.line(
                    frame,
                    tuple(np.rint(px1[:2]).astype(int)),
                    tuple(np.rint(px2[:2]).astype(int)),
                    (200, 200, 200),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            for keypoint_index, (pixel, point) in enumerate(zip(pixels, pose)):
                if keypoint_index in HIDDEN_VISUALIZATION_INDICES:
                    continue
                if not np.isfinite(pixel).all() or not np.isfinite(point).all() or point[3] < 0.05:
                    continue
                z = point[2]
                normalized = np.clip((z - z_lo) / (z_hi - z_lo), 0, 1)
                color = tuple(int(c * 255) for c in cmap(normalized)[:3][::-1])
                center = tuple(np.rint(pixel[:2]).astype(int))
                cv2.circle(frame, center, 4, color, thickness=-1, lineType=cv2.LINE_AA)
                if keypoint_index in labeled_indices and np.isfinite(z):
                    cv2.putText(
                        frame,
                        f"{z:.2f}m",
                        (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            panel_video = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            visible_mask = np.array([i not in HIDDEN_VISUALIZATION_INDICES for i in range(pose.shape[0])])
            valid = visible_mask & np.isfinite(pose[:, :3]).all(axis=1) & (pose[:, 3] >= 0.05)
            pts = pose[valid, :3]
            colors = cmap(np.clip((pts[:, 2] - z_lo) / (z_hi - z_lo), 0, 1)) if pts.size else np.zeros((0, 4))
            segments = _pose_skeleton_segments(pose, skeleton, HIDDEN_VISUALIZATION_INDICES)

            panel_xz = _render_plane_panel(
                ax_xz, fig_xz, panel_w, panel_h, segments, pts, colors, x_idx=0, y_idx=2,
                xlim=xlim, ylim=zlim, xlabel="X (m)", ylabel="Z (m)", title="Top view (X-Z)",
                invert_y=False,
            )
            panel_zy = _render_plane_panel(
                ax_zy, fig_zy, panel_w, panel_h, segments, pts, colors, x_idx=2, y_idx=1,
                xlim=zlim, ylim=ylim, xlabel="Z (m)", ylabel="Y (m)", title="Side view (Z-Y)",
                invert_y=True,
            )
            panel_3d = _render_3d_panel(
                ax_3d, fig_3d, panel_w, panel_h, segments, pts, colors, xlim=xlim, ylim=ylim, zlim=zlim, elev=15, azim=45,
            )

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (panel_w * 2, panel_h * 2)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create visualization video: {output_path}")

            combined = np.empty((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
            combined[:panel_h, :panel_w] = panel_video
            combined[:panel_h, panel_w:] = panel_xz
            combined[panel_h:, :panel_w] = panel_zy
            combined[panel_h:, panel_w:] = panel_3d
            writer.write(combined)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        plt.close(fig_xz)
        plt.close(fig_zy)
        plt.close(fig_3d)


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    runs = find_runs(args.project_folder, args.school, args.participant_id, args.process_reference_videos)
    for run in runs:
        try:
            video_path, merged_csv, depth_npz, output_dir = resolve_run_io(run, args.process_reference_videos)
            print(f"\nProcessing {run.name}")
            lift_run(video_path, merged_csv, depth_npz, output_dir, run.name, args, device)
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
