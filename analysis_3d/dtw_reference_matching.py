"""Prototype: subsequence DTW template matching, using a reference clip as the query
and searching for its best-matching subsequence inside every participant recording.

3D counterpart of analysis_2d/dtw_reference_matching.py: identical algorithm, over the
3D kinematic signals from analysis_3d/features.py (computed in a per-frame
body-centered coordinate frame) instead of 2D pixel-based ones. Video overlays still
need pixel coordinates, so they're rendered from each clip's companion 2D keypoints
(dataloader.Clip.keypoints_2d) rather than the lifted 3D ones.

This is an alternative to knn_reference_clusters.py's approach (aggregate mean/std
features per fixed-size window, clustered via t-SNE). Aggregate features throw away
the temporal order/shape of the motion and bin the recording into rigid, fixed-length,
50%-overlap windows. DTW instead compares the actual per-frame kinematic signal shape
and allows local time-warping, so it can find a precise onset/offset for where the
reference's motion pattern occurs even if the participant performs it faster or slower.

An earlier version of this script used unconstrained "open begin" subsequence DTW
(the query could start anywhere, no bound on how much it compressed/expanded). That
version failed in practice: every top match collapsed to a ~30-frame candidate window
regardless of the reference's own length (75 or 179 frames) — unconstrained warping
let the query fold onto short, low-variance "quiet" segments rather than a genuinely
time-warped counterpart, and equal-weighting the noisy accel channels let them drown
out the smoother, more gesture-defining channels (wrist distance, keypoint distance).

Algorithm (per reference, per participant clip):
  1. Z-score 8 of the 10 kinematic channels (features.SIGNAL_NAMES, excluding the
     jittery hand_accel_x/hand_accel_y) using the reference clip's own per-channel
     mean/std; apply that same affine transform to the participant clip's channels.
  2. Scan candidate windows of length in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} x the
     reference's frame count, at a stride of ~m/5, across the whole clip. For each
     candidate window, run standard (fixed-length) DTW between the reference and that
     window, cost normalized by (reference_length + window_length). This bounds
     compression/expansion to the 0.5x-2x band instead of leaving it unconstrained.
  3. Take the clip's single lowest-cost window across all scanned lengths/positions.
  4. Rank all participant clips by that cost; keep the top K as "DTW matches."
For each top match: an overlay plot (reference vs. matched excerpt, all 10 channels —
including the 2 excluded from the cost, for context — aligned by fractional progress
since matched duration can differ from the reference's) and a pose-overlaid video clip
of the matched window, mirroring knn_reference_clusters.py for side-by-side comparison.

Usage:
    python dtw_reference_matching.py --references rDA1_DeviationDistance,iDA3 [--k 5]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from clustering_analysis import (
    COLOR_AXIS,
    COLOR_BACKGROUND,
    COLOR_IRRELEVANT_REF,
    COLOR_RELEVANT_REF,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    OUTPUT_DIR_DEFAULT,
    _style_axes,
)
from dataloader import CACHE_DIR_DEFAULT, PROJECT_FOLDER_DEFAULT, REFERENCE_FOLDER_DEFAULT, Clip, load_dataset, resolve_video_path
from features import SIGNAL_NAMES, compute_raw_signals, estimate_shoulder_width

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import create_visualization  # noqa: E402

DTW_OUTPUT_SUBDIR = "dtw"
DEFAULT_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.1

# Channels used in the DTW cost itself. hand_accel_x/y are excluded: they're the
# jitteriest (second-derivative) signals and, weighted equally with the rest, drowned
# out the smoother, more gesture-defining channels (wrist distance, keypoint distance)
# in the naive version of this script. They're still plotted in the overlay for context.
DTW_CHANNEL_NAMES = [name for name in SIGNAL_NAMES if name not in ("hand_accel_x", "hand_accel_y")]
_DTW_CHANNEL_INDICES = np.array([SIGNAL_NAMES.index(name) for name in DTW_CHANNEL_NAMES])

# Candidate window lengths, as multiples of the reference's own frame count. Bounds
# compression/expansion to 0.5x-2x instead of the unconstrained version's free collapse.
LENGTH_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@numba.njit(cache=True)
def _closed_dtw_cost(query: np.ndarray, candidate: np.ndarray, weights: np.ndarray) -> float:
    """Standard fixed-length weighted DTW total cost between query (m, C) and candidate
    (L, C), both already z-scored/imputed (no NaN). `weights` (length C, sums to 1)
    controls how much each channel's squared error counts toward the local cost -- so a
    channel this specific reference barely moves in can't out-vote one it moves a lot in.
    Both ends are fixed (no open begin/end)."""
    m, c = query.shape
    length = candidate.shape[0]
    cost = np.empty((m, length))

    total = 0.0
    for k in range(c):
        diff = query[0, k] - candidate[0, k]
        total += weights[k] * diff * diff
    cost[0, 0] = np.sqrt(total)
    for j in range(1, length):
        total = 0.0
        for k in range(c):
            diff = query[0, k] - candidate[j, k]
            total += weights[k] * diff * diff
        cost[0, j] = np.sqrt(total) + cost[0, j - 1]

    for i in range(1, m):
        total = 0.0
        for k in range(c):
            diff = query[i, k] - candidate[0, k]
            total += weights[k] * diff * diff
        cost[i, 0] = np.sqrt(total) + cost[i - 1, 0]
        for j in range(1, length):
            total = 0.0
            for k in range(c):
                diff = query[i, k] - candidate[j, k]
                total += weights[k] * diff * diff
            local = np.sqrt(total)
            best_prev = cost[i - 1, j]
            if cost[i, j - 1] < best_prev:
                best_prev = cost[i, j - 1]
            if cost[i - 1, j - 1] < best_prev:
                best_prev = cost[i - 1, j - 1]
            cost[i, j] = local + best_prev
    return cost[m - 1, length - 1]


@numba.njit(cache=True)
def _scan_best_window(query: np.ndarray, series: np.ndarray, weights: np.ndarray, length_options: np.ndarray, stride: int):
    """Best (start, length, normalized_cost) over all candidate window lengths/positions."""
    m = query.shape[0]
    n = series.shape[0]
    best_cost = np.inf
    best_start = -1
    best_length = -1
    for option_index in range(length_options.shape[0]):
        length = length_options[option_index]
        if length < 2 or length > n:
            continue
        start = 0
        while start + length <= n:
            raw_cost = _closed_dtw_cost(query, series[start:start + length, :], weights)
            normalized = raw_cost / (m + length)
            if normalized < best_cost:
                best_cost = normalized
                best_start = start
                best_length = length
            start += stride
    return best_start, best_length, best_cost


def best_match_in_clip(
    query_full: np.ndarray, series_full: np.ndarray, weights: np.ndarray,
) -> tuple[int, int, float] | None:
    """Lowest-cost fixed-length window (0.5x-2x the query's length) of `query_full`
    against `series_full` (both (T, len(SIGNAL_NAMES))). Returns (start_index,
    end_index, normalized_cost), or None if the series is too short to scan."""
    query = np.ascontiguousarray(query_full[:, _DTW_CHANNEL_INDICES])
    series = np.ascontiguousarray(series_full[:, _DTW_CHANNEL_INDICES])
    if series.shape[0] < 2:
        return None
    m = query.shape[0]
    length_options = np.array(sorted({max(2, round(m * factor)) for factor in LENGTH_FACTORS}), dtype=np.int64)
    stride = max(1, m // 5)
    start, length, cost = _scan_best_window(query, series, weights, length_options, stride)
    if start < 0:
        return None
    return int(start), int(start + length - 1), float(cost)


def reference_channel_stats(reference_clip: Clip) -> tuple[dict[str, float], dict[str, float]]:
    shoulder_width = estimate_shoulder_width(reference_clip.keypoints)
    signals = compute_raw_signals(reference_clip.keypoints, reference_clip.fps, shoulder_width)
    means, stds = {}, {}
    for name in SIGNAL_NAMES:
        values = signals[name]
        valid = values[np.isfinite(values)]
        means[name] = float(np.mean(valid)) if len(valid) else 0.0
        std = float(np.std(valid)) if len(valid) else 1.0
        stds[name] = std if std > 1e-6 else 1.0
    return means, stds


def compute_global_channel_scale(reference_clips: list[Clip]) -> dict[str, float]:
    """Per-channel std pooled across every reference clip's raw (shoulder-width
    normalized) signal -- the 'typical' amount of movement in each channel across the
    whole reference set, used to make one reference's channel weights comparable across
    channel types (a channel's raw std alone can't be compared to another channel's,
    since they're in different units/have different natural magnitudes)."""
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in SIGNAL_NAMES}
    for clip in reference_clips:
        shoulder_width = estimate_shoulder_width(clip.keypoints)
        signals = compute_raw_signals(clip.keypoints, clip.fps, shoulder_width)
        for name in SIGNAL_NAMES:
            values = signals[name]
            pooled[name].append(values[np.isfinite(values)])
    scales = {}
    for name in SIGNAL_NAMES:
        concatenated = np.concatenate(pooled[name]) if pooled[name] else np.array([])
        std = float(np.std(concatenated)) if len(concatenated) else 1.0
        scales[name] = std if std > 1e-6 else 1.0
    return scales


def reference_channel_weights(reference_clip: Clip, global_scale: dict[str, float]) -> np.ndarray:
    """How much this specific reference moves in each DTW channel, relative to that
    channel's typical variation across the whole reference set, normalized to sum to 1.
    Returned in DTW_CHANNEL_NAMES order, ready to pass to best_match_in_clip."""
    shoulder_width = estimate_shoulder_width(reference_clip.keypoints)
    signals = compute_raw_signals(reference_clip.keypoints, reference_clip.fps, shoulder_width)
    importance = {}
    for name in DTW_CHANNEL_NAMES:
        values = signals[name]
        valid = values[np.isfinite(values)]
        std = float(np.std(valid)) if len(valid) else 0.0
        importance[name] = std / global_scale[name]
    total = sum(importance.values()) or 1.0
    weights = np.array([importance[name] / total for name in DTW_CHANNEL_NAMES], dtype=np.float64)
    return weights


def build_channel_matrix(clip: Clip, means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    """(T, len(SIGNAL_NAMES)) z-scored against the reference's own per-channel stats;
    remaining NaN (occlusion, leading pad) imputed to 0 (= the reference's mean)."""
    shoulder_width = estimate_shoulder_width(clip.keypoints)
    signals = compute_raw_signals(clip.keypoints, clip.fps, shoulder_width)
    channels = []
    for name in SIGNAL_NAMES:
        z = (signals[name] - means[name]) / stds[name]
        channels.append(np.where(np.isfinite(z), z, 0.0))
    return np.ascontiguousarray(np.stack(channels, axis=1), dtype=np.float64)


def plot_match_overlay(
    reference_clip: Clip, reference_channels: np.ndarray, matched_channels: np.ndarray,
    start_index: int, end_index: int, normalized_cost: float, output_path: Path,
) -> None:
    line_color = COLOR_RELEVANT_REF if reference_clip.category == "relevant" else COLOR_IRRELEVANT_REF
    reference_progress = np.linspace(0, 1, reference_channels.shape[0])
    matched_progress = np.linspace(0, 1, end_index - start_index + 1)

    fig, axes = plt.subplots(
        len(SIGNAL_NAMES), 1, figsize=(8.5, 1.5 * len(SIGNAL_NAMES)),
        sharex=True, facecolor=COLOR_BACKGROUND,
    )
    for axis_index, (ax, signal_name) in enumerate(zip(axes, SIGNAL_NAMES)):
        _style_axes(ax)
        ax.plot(
            reference_progress, reference_channels[:, axis_index],
            color=COLOR_TEXT_PRIMARY, linewidth=1.2, linestyle="--", alpha=0.8,
            label="Reference (z-scored)" if axis_index == 0 else None,
        )
        ax.plot(
            matched_progress, matched_channels[start_index:end_index + 1, axis_index],
            color=line_color, linewidth=1.6,
            label="Matched excerpt (z-scored)" if axis_index == 0 else None,
        )
        ax.set_ylabel(signal_name.replace("_", " "), fontsize=8.5, color=COLOR_TEXT_SECONDARY,
                       rotation=0, ha="right", va="center", labelpad=8)

    axes[-1].set_xlabel("Fractional progress through the gesture", fontsize=10, color=COLOR_TEXT_SECONDARY)
    legend = axes[0].legend(loc="upper right", frameon=True, facecolor=COLOR_BACKGROUND, edgecolor=COLOR_AXIS, fontsize=8)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)
    fig.suptitle(
        f"DTW match vs. reference '{reference_clip.clip_id}' (normalized cost {normalized_cost:.3f})",
        fontsize=12.5, color=COLOR_TEXT_PRIMARY, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0.1, 0, 1, 0.97))
    fig.savefig(output_path, dpi=170, facecolor=COLOR_BACKGROUND)
    plt.close(fig)


def render_match_video(clip: Clip, start_index: int, end_index: int, output_path: Path, confidence_threshold: float) -> None:
    video_path = resolve_video_path(clip)
    if video_path is None or not video_path.exists():
        print(f"    ! source video not found for {clip.clip_id}; skipping video")
        return
    if clip.keypoints_2d is None:
        print(f"    ! no 2D pixel keypoints (companion merged CSV) for {clip.clip_id}; skipping video")
        return
    frame_labels = [f"frame_{n:04d}" for n in clip.frame_numbers[start_index:end_index + 1]]
    poses = clip.keypoints_2d[start_index:end_index + 1]
    create_visualization(video_path, output_path, frame_labels, poses, confidence_threshold)


def run(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    force_reload: bool = False,
    reference_clip_ids: list[str] = (),
    k: int = DEFAULT_K,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    print("Loading clips...")
    participant_clips, reference_clips = load_dataset(
        project_folder=project_folder, reference_folder=reference_folder,
        cache_dir=cache_dir, force_reload=force_reload,
    )
    print(f"  {len(participant_clips)} participant clips, {len(reference_clips)} reference clips")

    reference_by_id = {clip.clip_id: clip for clip in reference_clips}
    unknown = sorted(set(reference_clip_ids) - set(reference_by_id))
    if unknown:
        raise ValueError(f"Unknown reference clip_id(s): {unknown}. Available: {sorted(reference_by_id)}")

    print("Computing per-channel global scale across the reference set...")
    global_scale = compute_global_channel_scale(reference_clips)

    dtw_dir = output_dir / DTW_OUTPUT_SUBDIR
    for reference_clip_id in reference_clip_ids:
        reference_clip = reference_by_id[reference_clip_id]
        print(f"\n=== {reference_clip_id} ({reference_clip.n_frames} frames, {reference_clip.duration_s:.2f}s) ===")
        means, stds = reference_channel_stats(reference_clip)
        reference_channels = build_channel_matrix(reference_clip, means, stds)
        weights = reference_channel_weights(reference_clip, global_scale)
        weights_report = ", ".join(
            f"{name}={weight:.2f}" for name, weight in sorted(zip(DTW_CHANNEL_NAMES, weights), key=lambda kv: -kv[1])
        )
        print(f"  channel weights: {weights_report}")

        results = []
        start_time = time.time()
        for index, clip in enumerate(participant_clips):
            candidate_channels = build_channel_matrix(clip, means, stds)
            match = best_match_in_clip(reference_channels, candidate_channels, weights)
            if match is None:
                continue
            start_index, end_index, normalized_cost = match
            results.append({
                "clip_id": clip.clip_id,
                "participant_id": clip.participant_id,
                "school": clip.school,
                "start_index": start_index,
                "end_index": end_index,
                "start_frame": int(clip.frame_numbers[start_index]),
                "end_frame": int(clip.frame_numbers[end_index]),
                "window_length_frames": end_index - start_index + 1,
                "normalized_cost": normalized_cost,
                "_clip": clip,
                "_channels": candidate_channels,
            })
            if (index + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"    matched {index + 1}/{len(participant_clips)} clips ({elapsed:.1f}s elapsed)")

        results.sort(key=lambda row: row["normalized_cost"])
        top_results = results[:k]
        elapsed = time.time() - start_time
        print(f"  matched against all {len(participant_clips)} participant clips in {elapsed:.1f}s;"
              f" best normalized cost {top_results[0]['normalized_cost']:.3f}" if top_results else "  no matches found")

        reference_dir = dtw_dir / reference_clip_id
        windows_dir = reference_dir / "windows"
        reference_dir.mkdir(parents=True, exist_ok=True)
        windows_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for rank, row in enumerate(top_results, start=1):
            clip = row["_clip"]
            print(f"    rank {rank}: {row['participant_id']}/{row['clip_id']}"
                  f" frames {row['start_frame']}-{row['end_frame']} (cost {row['normalized_cost']:.3f})")

            overlay_path = reference_dir / f"rank{rank:02d}_overlay_{row['participant_id']}_{row['clip_id']}.png"
            plot_match_overlay(
                reference_clip, reference_channels, row["_channels"],
                row["start_index"], row["end_index"], row["normalized_cost"], overlay_path,
            )

            video_path = windows_dir / (
                f"rank{rank:02d}_{row['participant_id']}_{row['clip_id']}"
                f"_f{row['start_frame']:05d}-{row['end_frame']:05d}.mp4"
            )
            render_match_video(clip, row["start_index"], row["end_index"], video_path, confidence_threshold)

            manifest_rows.append({
                "rank": rank, "clip_id": row["clip_id"], "participant_id": row["participant_id"],
                "school": row["school"], "start_frame": row["start_frame"], "end_frame": row["end_frame"],
                "window_length_frames": row["window_length_frames"], "normalized_cost": row["normalized_cost"],
            })

        pd.DataFrame(manifest_rows).to_csv(reference_dir / "matches.csv", index=False)

    print(f"\nSaved DTW outputs to {dtw_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-folder", type=Path, default=PROJECT_FOLDER_DEFAULT)
    parser.add_argument("--reference-folder", type=Path, default=REFERENCE_FOLDER_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--references", type=str, required=True, help="Comma-separated reference clip_ids.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top DTW matches per reference.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        project_folder=args.project_folder,
        reference_folder=args.reference_folder,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_reload=args.force_reload,
        reference_clip_ids=args.references.split(","),
        k=args.k,
        confidence_threshold=args.confidence_threshold,
    )
