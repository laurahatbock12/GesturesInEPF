"""Fit a fixed-bone-length skeleton to the noisy per-frame 3D pose from estimate_3d_pose.py.

Per-frame independent depth + a single global scale factor still leaves bone lengths (Euclidean
distance between adjacent skeleton joints) jittering frame to frame, even though a real skeleton
is rigid. This script:

1. Builds a reference skeletal model for the participant: for every bone in the full-body
   skeleton (utils.get_skeleton), takes the MEDIAN length of that bone across every frame where
   both endpoints are confidently detected (median chosen over the mean for robustness - a
   single badly mis-detected frame would otherwise skew a bone's reference length permanently).
   Left/right mirrored bones (arms, and every finger bone of both hands) pool their frames into
   one shared reference length, since the two sides of a real skeleton are the same size.
   Also reports the variation (std, coefficient of variation) of each bone's raw per-frame
   length, which is the actual diagnostic for "how physically inconsistent is the raw 3D pose."

2. Fits that fixed-length skeleton to each frame's noisy observed joint positions via
   Position-Based Dynamics (PBD) distance-constraint relaxation: repeatedly nudge each bone's
   two endpoints along the line between them until their distance matches the reference length,
   splitting the correction between the two points in proportion to how little each is trusted
   (inverse of its 2D detection likelihood) - so a confidently-tracked shoulder barely moves
   while an uncertain wrist absorbs most of the correction. This is the same technique used for
   real-time skeleton/cloth constraint solving in physics engines, applied here to a fixed
   topology instead of a simulated one. It's solved independently per frame (no temporal
   smoothing) and vectorized across all frames at once (looping only over bones x iterations,
   not over frames), so it stays fast even for long videos.

A joint with no observed (x, y, z) in a given frame is left untouched (still NaN) rather than
hallucinated from its neighbors.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from estimate_3d_pose import COORDS_3D, save_visualization
from pose_estimation_2d.merge_poses import BODY_BODYPARTS, MERGED_BODYPARTS, frame_number
from pose_estimation_2d.utils import get_skeleton
from project_config import PROJECT_FOLDER

# The recordings only frame the upper body, so knees/ankles are never reliably visible; excluding
# their bones (ankle-knee, knee-hip) keeps the fitted model from being built on untracked joints.
LEG_JOINT_SUFFIXES = {"knee", "ankle"}

# Finger joints only come from the dedicated hand-pose model, whose likelihood is calibrated much
# lower than the body model's (fingers routinely score 0.15-0.3 even when well tracked), so they
# need their own, much lower --min_likelihood_hand floor - consistent with the 0.05 already used
# for hand keypoints elsewhere in this pipeline (e.g. estimate_3d_pose.py's depth-outlier fixup).
N_BODY_BODYPARTS = len(BODY_BODYPARTS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a fixed-bone-length skeletal model to estimate_3d_pose.py output."
    )
    parser.add_argument(
        "--project_folder",
        type=str,
        default=PROJECT_FOLDER,
    )
    parser.add_argument(
        "--school",
        type=str,
        default="2026_06_23_KantiHeerbrugg",
        choices=["2026_06_23_KantiHeerbrugg", "2026_06_02_GR_KantiOlten", "2026_05_20_KantiMusegg", "2026_06_08_GymKirschgarten", "2026_06_22_GymImmensee"],
    )
    parser.add_argument("--participant_id", type=str, default="AL79NI")
    parser.add_argument(
        "--process_reference_videos",
        action="store_true",
        help="Process the reference videos in the project_folder instead of participant recordings",
    )
    parser.add_argument(
        "--min_likelihood",
        type=float,
        default=0.5,
        help="Minimum 2D keypoint likelihood required for a body joint, for a frame's bone "
        "length to count towards that bone's reference length / variation statistics.",
    )
    parser.add_argument(
        "--min_likelihood_hand",
        type=float,
        default=0.05,
        help="Same as --min_likelihood but for hand/finger joints, whose likelihood is "
        "calibrated much lower than the body model's.",
    )
    parser.add_argument("--iterations", type=int, default=30, help="PBD constraint-relaxation sweeps per frame.")
    parser.add_argument("--stiffness", type=float, default=1.0, help="Constraint correction strength, 0-1.")
    parser.add_argument(
        "--post_fit_smoothing_window",
        type=int,
        default=10,
        help="The PBD solve runs independently per frame (see module docstring), so it can "
        "still jitter frame-to-frame even once bone lengths are consistent. This applies "
        "centered median smoothing to the fitted x/y/z of every joint, in a window of this many "
        "frames, after fitting. Use 1 to disable.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save the same 4-panel visualization as estimate_3d_pose.py, for the fitted pose.",
    )
    parser.add_argument("--visualization_frames", type=int, default=-1)
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
        pose3d_dir = run.parent / "pose_estimation" / "pose_3d_predictions"
        output_dir = run.parent / "pose_estimation" / "skeletal_fit_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = run
    else:
        pose_dir = pose3d_dir = output_dir = run
        candidates = [
            i for i in run.iterdir() if i.is_file() and i.suffix.lower() in (".mp4", ".avi") and run.name not in i.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No video file found in: {run}")
        video_path = candidates[0]

    merged_csv = pose_dir / f"results_{run.name}_merged.csv"
    pose3d_csv = pose3d_dir / f"results_{run.name}_3d.csv"
    return video_path, merged_csv, pose3d_csv, output_dir


def load_pose_csv(path, coords):
    if not path.exists():
        raise FileNotFoundError(f"Pose CSV not found: {path}")
    df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    frame_labels = sorted(df.index, key=frame_number)
    columns = pd.MultiIndex.from_product([MERGED_BODYPARTS, coords], names=["bodyparts", "coords"])
    table = df.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
    array = table.reindex(index=frame_labels, columns=columns).to_numpy(dtype=np.float32)
    return frame_labels, array.reshape(len(frame_labels), len(MERGED_BODYPARTS), len(coords))


def _mirror_bodypart(name):
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _mirror_bone_name(bone_name):
    part_a, part_b = bone_name.split("-", 1)
    return f"{_mirror_bodypart(part_a)}-{_mirror_bodypart(part_b)}"


def compute_reference_lengths(xyz, edges, edge_names, min_likelihood_per_joint):
    """Median bone length per edge (robust reference skeleton) plus raw variation stats.

    Left/right mirrored bones (e.g. left_elbow-left_wrist and right_elbow-right_wrist, or the
    matching finger bones of both hands) are pooled into a single shared reference length rather
    than fit independently: a person's left and right limbs have the same bone lengths, so
    treating them separately would just let per-side detection noise make one arm/hand a
    different size than its mirror in the fitted model.

    min_likelihood_per_joint is indexed per joint (not a single scalar) since hand joints need a
    much lower confidence floor than body joints - see N_BODY_BODYPARTS above.
    """
    p = xyz[..., :3]
    likelihood = xyz[..., 3]
    n_edges = len(edges)

    raw_lengths = []
    for i, j in edges:
        confident = (likelihood[:, i] >= min_likelihood_per_joint[i]) & (likelihood[:, j] >= min_likelihood_per_joint[j])
        finite = np.isfinite(p[:, i]).all(axis=-1) & np.isfinite(p[:, j]).all(axis=-1)
        mask = confident & finite
        raw_lengths.append(np.linalg.norm(p[mask, i] - p[mask, j], axis=-1))

    mirror_groups = defaultdict(list)
    for edge_index, name in enumerate(edge_names):
        mirror_groups[min(name, _mirror_bone_name(name))].append(edge_index)

    reference = np.full(n_edges, np.nan, dtype=np.float32)
    stats = []
    for edge_index, name in enumerate(edge_names):
        group = mirror_groups[min(name, _mirror_bone_name(name))]
        pooled = np.concatenate([raw_lengths[member] for member in group])
        if pooled.size == 0:
            stats.append((name, 0, np.nan, np.nan, np.nan))
            continue
        median = float(np.median(pooled))
        std = float(np.std(pooled))
        cv = std / median if median > 1e-9 else np.nan
        reference[edge_index] = median
        stats.append((name, int(pooled.size), median, std, cv))
    return reference, stats


def print_bone_length_report(stats, title):
    print(f"\n=== {title} ===")
    print(f"{'bone':40s} {'n':>6s} {'median (m)':>11s} {'std (m)':>9s} {'CV':>6s}")
    for name, n, median, std, cv in sorted(stats, key=lambda row: (row[4] if np.isfinite(row[4]) else -1), reverse=True):
        cv_str = f"{cv:.3f}" if np.isfinite(cv) else "n/a"
        median_str = f"{median:.3f}" if np.isfinite(median) else "n/a"
        std_str = f"{std:.3f}" if np.isfinite(std) else "n/a"
        print(f"{name:40s} {n:6d} {median_str:>11s} {std_str:>9s} {cv_str:>6s}")


def fit_skeleton(xyz, edges, reference_lengths, iterations, stiffness, min_likelihood_per_joint):
    """PBD distance-constraint relaxation, vectorized across frames. Returns a copy of xyz with
    x, y, z corrected to (approximately) satisfy the reference bone lengths; likelihood and any
    NaN (undetected) joints are left untouched.

    min_likelihood_per_joint is also the floor used to derive each joint's inverse mass (how much
    it gets pushed around by the constraint solve); using the body threshold there for hand
    joints would clip every finger's likelihood to the same floor value and make the solver treat
    all fingers as equally (un)trustworthy instead of weighting by their actual likelihood.
    """
    positions = xyz[..., :3].copy()
    likelihood = xyz[..., 3]
    inv_mass = 1.0 / np.clip(likelihood, min_likelihood_per_joint[None, :], None)
    valid_joint = np.isfinite(positions).all(axis=-1)

    for _ in range(iterations):
        for (i, j), target_length in zip(edges, reference_lengths):
            if not np.isfinite(target_length):
                continue
            valid_edge = valid_joint[:, i] & valid_joint[:, j]
            if not valid_edge.any():
                continue
            p_i = positions[:, i, :]
            p_j = positions[:, j, :]
            delta = p_j - p_i
            dist = np.linalg.norm(delta, axis=-1)
            safe_dist = np.where(dist < 1e-6, 1e-6, dist)
            diff = stiffness * (dist - target_length) / safe_dist
            w_i = inv_mass[:, i]
            w_j = inv_mass[:, j]
            w_sum = np.where((w_i + w_j) < 1e-9, 1e-9, w_i + w_j)
            correction = diff[:, None] * delta
            move_i = (w_i / w_sum)[:, None] * correction
            move_j = -(w_j / w_sum)[:, None] * correction
            mask = valid_edge[:, None]
            positions[:, i, :] = np.where(mask, p_i + move_i, p_i)
            positions[:, j, :] = np.where(mask, p_j + move_j, p_j)

    positions[~valid_joint] = np.nan
    fitted = xyz.copy()
    fitted[..., :3] = positions
    return fitted


def smooth_fitted_pose(xyz, window_size):
    """Centered median smoothing on the fitted x/y/z per joint. The PBD solve in fit_skeleton is
    independent per frame - it makes bone lengths consistent, but the joint positions themselves
    can still jitter frame to frame, which this cleans up post-fit. Likelihood and any NaN
    (undetected) joints are left untouched."""
    if window_size < 1:
        raise ValueError("post_fit_smoothing_window must be a positive integer.")
    if window_size == 1:
        return xyz.copy()

    n_frames = xyz.shape[0]
    half_before = window_size // 2
    half_after = window_size - half_before - 1
    positions = xyz[..., :3]
    valid = np.isfinite(positions).all(axis=-1)
    smoothed = xyz.copy()
    for frame_index in range(n_frames):
        start = max(0, frame_index - half_before)
        end = min(n_frames, frame_index + half_after + 1)
        window_valid = valid[start:end]
        window_positions = positions[start:end]
        for bodypart_index in np.flatnonzero(valid[frame_index]):
            points = window_positions[window_valid[:, bodypart_index], bodypart_index]
            if len(points):
                smoothed[frame_index, bodypart_index, :3] = np.median(points, axis=0)
    return smoothed


def fit_run(video_path, merged_csv, pose3d_csv, output_dir, run_name, args):
    output_csv = output_dir / f"results_{run_name}_3d_fitted.csv"
    output_meta = output_dir / f"results_{run_name}_3d_fitted_meta.json"
    output_vis = output_dir / f"visualization_{run_name}_3d_fitted.mp4"
    if output_csv.exists() and not args.force:
        print(f"Fitted 3D pose already exists for {run_name} at {output_csv}. Skipping.")
        return

    frame_labels, xyz = load_pose_csv(pose3d_csv, COORDS_3D)
    _, skeleton = get_skeleton(mode="full_body")
    edges = [(bp1 - 1, bp2 - 1) for bp1, bp2 in skeleton]
    edges = [
        (i, j)
        for i, j in edges
        if not (LEG_JOINT_SUFFIXES & {MERGED_BODYPARTS[i].split("_", 1)[-1], MERGED_BODYPARTS[j].split("_", 1)[-1]})
    ]
    edge_names = [f"{MERGED_BODYPARTS[i]}-{MERGED_BODYPARTS[j]}" for i, j in edges]

    min_likelihood_per_joint = np.full(len(MERGED_BODYPARTS), args.min_likelihood, dtype=np.float32)
    min_likelihood_per_joint[N_BODY_BODYPARTS:] = args.min_likelihood_hand

    reference_lengths, stats = compute_reference_lengths(xyz, edges, edge_names, min_likelihood_per_joint)
    print_bone_length_report(stats, f"{run_name}: raw per-frame bone length (before fitting)")

    fitted = fit_skeleton(xyz, edges, reference_lengths, args.iterations, args.stiffness, min_likelihood_per_joint)

    _, post_stats = compute_reference_lengths(fitted, edges, edge_names, min_likelihood_per_joint)
    print_bone_length_report(post_stats, f"{run_name}: bone length after fitting (should be ~constant)")

    fitted = smooth_fitted_pose(fitted, args.post_fit_smoothing_window)

    columns = pd.MultiIndex.from_tuples(
        [("skeletal-fit", "idv_0", bodypart, coord) for bodypart in MERGED_BODYPARTS for coord in COORDS_3D],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    n_frames = fitted.shape[0]
    pd.DataFrame(fitted.reshape(n_frames, -1), index=frame_labels, columns=columns).to_csv(output_csv)
    print(f"Saved fitted 3D pose to: {output_csv}")

    with open(output_meta, "w") as f:
        json.dump(
            {
                "iterations": args.iterations,
                "stiffness": args.stiffness,
                "min_likelihood": args.min_likelihood,
                "min_likelihood_hand": args.min_likelihood_hand,
                "post_fit_smoothing_window": args.post_fit_smoothing_window,
                "reference_lengths_m": {name: (float(length) if np.isfinite(length) else None) for name, length in zip(edge_names, reference_lengths)},
            },
            f,
            indent=2,
        )

    if args.save_vis:
        _, xy = load_pose_csv(merged_csv, ("x", "y", "likelihood"))
        frame_rate = 30.0
        cap = cv2.VideoCapture(str(video_path))
        frame_rate = cap.get(cv2.CAP_PROP_FPS) or frame_rate
        cap.release()
        save_visualization(video_path, output_vis, frame_labels, xy, fitted, frame_rate, args.visualization_frames)
        print(f"Saved fitted 3D visualization to: {output_vis}")


def main(args):
    runs = find_runs(args.project_folder, args.school, args.participant_id, args.process_reference_videos)
    for run in runs:
        try:
            video_path, merged_csv, pose3d_csv, output_dir = resolve_run_io(run, args.process_reference_videos)
            print(f"\nProcessing {run.name}")
            fit_run(video_path, merged_csv, pose3d_csv, output_dir, run.name, args)
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
