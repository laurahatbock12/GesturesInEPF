"""Create transparent-history pose artwork from cached project predictions.

The 3D fallback is intended for cached 2D-only runs: it preserves the 2D pose and
adds a stable, visually useful depth layout. Pass a real (frames, joints, 4)
3D array with --pose-3d when available to render computed 3D coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import COORDINATES, MERGED_BODYPARTS, frame_number
from pose_estimation_2d.utils import get_skeleton
from project_config import VISUALIZATION_OUTPUT_DIR


EDGES = [(a - 1, b - 1) for a, b in get_skeleton("full_body")[1]]
LOWER_BODY_JOINTS = set(range(11, 17))
EDGES = [
    (left, right)
    for left, right in EDGES
    if left not in LOWER_BODY_JOINTS and right not in LOWER_BODY_JOINTS
]


def midpoint_window(n_frames: int, count: int = 10) -> np.ndarray:
    if n_frames < count:
        raise ValueError(f"Need at least {count} frames, found {n_frames}")
    start = (n_frames - count) // 2
    return np.arange(start, start + count)


def valid_points(points: np.ndarray) -> np.ndarray:
    visible = np.isfinite(points[..., :2]).all(axis=-1)
    visible[..., list(LOWER_BODY_JOINTS)] = False
    return visible


def draw_2d(history: np.ndarray, output: Path) -> None:
    xy = history[..., :2].astype(float)
    visible = valid_points(history)
    finite = xy[visible]
    if len(finite) == 0:
        raise ValueError("The selected frames contain no finite 2D points")
    low, high = finite.min(axis=0), finite.max(axis=0)
    margin = max(high - low) * 0.13
    low -= margin
    high += margin

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0.025, 0.03, 0.08, 1))
    for age, points in enumerate(xy):
        alpha = 1.0 if len(xy) == 1 else 0.10 + 0.90 * (age / (len(xy) - 1)) ** 1.7
        for left, right in EDGES:
            if visible[age, left] and visible[age, right]:
                ax.plot(
                    points[[left, right], 0],
                    points[[left, right], 1],
                    color="#6ce5ff",
                    linewidth=4.2,
                    alpha=alpha,
                    solid_capstyle="round",
                )
        indices = np.flatnonzero(visible[age])
        ax.scatter(
            points[indices, 0],
            points[indices, 1],
            c=indices,
            cmap="turbo",
            vmin=0,
            vmax=max(1, history.shape[1] - 1),
            s=115 if age == len(xy) - 1 else 62,
            alpha=alpha,
            edgecolors="white" if age == len(xy) - 1 else "none",
            linewidths=0.45,
        )
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(high[1], low[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output, transparent=False, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def draw_3d(
    history: np.ndarray,
    output: Path,
    *,
    elev: float = 0,
    azim: float = 45,
) -> None:
    # Matplotlib renders its third coordinate vertically; place inverted source
    # y there so y is up and the pose is not vertically mirrored.
    xyz = history[..., [0, 2, 1]].astype(float)
    xyz[..., 2] *= -1
    visible = np.isfinite(xyz).all(axis=-1)
    visible[..., list(LOWER_BODY_JOINTS)] = False
    finite = xyz[visible]
    if len(finite) == 0:
        raise ValueError("The selected frames contain no finite 3D points")
    low, high = finite.min(axis=0), finite.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) / 2, 1e-3) * 1.18

    fig = plt.figure(figsize=(10, 10), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0)
    for age, points in enumerate(xyz):
        alpha = 1.0 if len(xyz) == 1 else 0.10 + 0.90 * (age / (len(xyz) - 1)) ** 1.7
        for left, right in EDGES:
            if visible[age, left] and visible[age, right]:
                ax.plot(
                    points[[left, right], 0],
                    points[[left, right], 1],
                    points[[left, right], 2],
                    color="#6ce5ff",
                    linewidth=4.0,
                    alpha=alpha,
                    solid_capstyle="round",
                )
        indices = np.flatnonzero(visible[age])
        ax.scatter(
            points[indices, 0],
            points[indices, 1],
            points[indices, 2],
            c=indices,
            cmap="turbo",
            vmin=0,
            vmax=max(1, history.shape[1] - 1),
            s=120 if age == len(xyz) - 1 else 66,
            alpha=alpha,
            depthshade=False,
            edgecolors="white" if age == len(xyz) - 1 else "none",
            linewidths=0.45,
        )
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def load_pose_csv(path: Path, coordinates: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    columns = pd.MultiIndex.from_product(
        [MERGED_BODYPARTS, coordinates], names=["bodyparts", "coords"]
    )
    table = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    table = table.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
    table = table.reindex(columns=columns)
    values = table.to_numpy(dtype=np.float32).reshape(
        len(table), len(MERGED_BODYPARTS), len(coordinates)
    )
    frames = np.array([frame_number(label) for label in table.index], dtype=np.int32)
    return values, frames


def extract_video_frame(video_path: Path, frame_number_: int, output: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number_))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number_} from {video_path}")
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write extracted frame to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-2d-csv", type=Path, required=True)
    parser.add_argument("--pose-3d-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=VISUALIZATION_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points_2d, frame_numbers_2d = load_pose_csv(args.pose_2d_csv, COORDINATES)
    points_3d, frame_numbers_3d = load_pose_csv(
        args.pose_3d_csv, ("x", "y", "z", "likelihood")
    )
    if not np.array_equal(frame_numbers_2d, frame_numbers_3d):
        raise ValueError("2D and 3D pose CSVs do not contain the same frame numbers")
    indices = midpoint_window(len(points_3d))
    selected_video_frames = frame_numbers_3d[indices]
    middle_index = len(indices) // 2
    middle_video_frame = int(selected_video_frames[middle_index])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "selected_frame_indices.npy", indices)
    extract_video_frame(
        args.video, middle_video_frame, args.output_dir / "middle_source_frame.png"
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "video": str(args.video),
                "pose_2d_csv": str(args.pose_2d_csv),
                "pose_3d_csv": str(args.pose_3d_csv),
                "prediction_rows": indices.tolist(),
                "video_frames": selected_video_frames.tolist(),
                "middle_video_frame": middle_video_frame,
            },
            indent=2,
        )
        + "\n"
    )
    draw_2d(points_2d[indices], args.output_dir / "pose_history_2d.png")
    draw_3d(points_3d[indices], args.output_dir / "pose_history_3d.png")
    history_3d = points_3d[indices]
    sweep_angles = [int(angle) for angle in np.arange(0, 360, 36)]
    for view_number, angle in enumerate(sweep_angles, start=1):
        draw_3d(
            history_3d,
            args.output_dir / f"pose_3d_angle_{view_number:02d}.png",
            elev=0,
            azim=angle,
        )
    (args.output_dir / "angle_sweep.json").write_text(
        json.dumps(
            {
                "frame": middle_video_frame,
                "elevation_degrees": 0,
                "azimuth_degrees": sweep_angles,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Video frames {selected_video_frames[0]}-{selected_video_frames[-1]} selected; "
        f"middle frame {middle_video_frame}"
    )


if __name__ == "__main__":
    main()
