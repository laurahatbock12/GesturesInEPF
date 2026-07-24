"""KNN inspection: which participant windows sit closest to each reference gesture,
in the t-SNE embedding produced by clustering_analysis.py?

3D counterpart of analysis_2d/knn_reference_clusters.py: identical logic, over the 3D
t-SNE embedding built from analysis_3d/clustering_analysis.py's 3D kinematic features.
Video overlays still need pixel coordinates, so they're rendered from each clip's
companion 2D keypoints (dataloader.Clip.keypoints_2d) rather than the lifted 3D ones.

For every reference clip:
  1. Run K-nearest-neighbors (Euclidean, in the 2D t-SNE coordinates) against all
     participant windows to find the K windows t-SNE places closest to that reference.
  2. Plot the full t-SNE scatter with those K windows highlighted and their convex-hull
     extent shaded, so you can see how tight or spread the "would sample" region is.
  3. Cut each of the K windows out of its source recording as a short pose-overlaid
     video clip, so the sampled windows can be eyeballed directly.
A combined overview plot (all processed references at once) is also produced.

Usage:
    python knn_reference_clusters.py [--k 10] [--references rDA1_DeviationDistance,iDA3] [--skip-videos]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors

from clustering_analysis import (
    COLOR_AXIS,
    COLOR_BACKGROUND,
    COLOR_GRID,
    COLOR_IRRELEVANT_REF,
    COLOR_RELEVANT_REF,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    OUTPUT_DIR_DEFAULT,
    AnalysisResult,
    _style_axes,
    prepare_analysis,
    reference_id_map,
)
from dataloader import CACHE_DIR_DEFAULT, PROJECT_FOLDER_DEFAULT, REFERENCE_FOLDER_DEFAULT, Clip, resolve_video_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import create_visualization  # noqa: E402

KNN_OUTPUT_SUBDIR = "knn"
COLOR_BACKGROUND_MUTED = "#c3c2b7"
COLOR_OVERVIEW_HIGHLIGHT = "#4a3aa7"  # categorical slot 7, violet
DEFAULT_K = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.1


def find_nearest_windows(table: pd.DataFrame, tsne_embedding: np.ndarray, reference_clip_id: str, k: int) -> pd.DataFrame:
    """The K participant-window rows whose t-SNE coordinates are closest to the given
    reference's t-SNE coordinates. Returns those rows plus `tsne_distance` and `rank`."""
    is_reference = table["is_reference"].to_numpy()
    reference_positions = np.flatnonzero(is_reference & (table["clip_id"] == reference_clip_id).to_numpy())
    if len(reference_positions) != 1:
        raise ValueError(f"Expected exactly one reference row for '{reference_clip_id}', found {len(reference_positions)}.")
    reference_point = tsne_embedding[reference_positions[0]].reshape(1, -1)

    participant_positions = np.flatnonzero(~is_reference)
    n_neighbors = min(k, len(participant_positions))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(tsne_embedding[participant_positions])
    distances, local_indices = neighbors.kneighbors(reference_point)

    selected_positions = participant_positions[local_indices[0]]
    result = table.iloc[selected_positions].copy()
    result["tsne_distance"] = distances[0]
    result["rank"] = np.arange(1, len(result) + 1)
    result["_position"] = selected_positions
    return result


def _draw_hull(ax: plt.Axes, points: np.ndarray, color: str, alpha: float = 0.15) -> None:
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except QhullError:
        return
    polygon = plt.Polygon(points[hull.vertices], closed=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5, zorder=2)
    ax.add_patch(polygon)


def plot_reference_highlight(
    tsne_embedding: np.ndarray, table: pd.DataFrame, reference_clip_id: str, neighbors: pd.DataFrame, output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=COLOR_BACKGROUND)
    _style_axes(ax)

    is_reference = table["is_reference"].to_numpy()
    selected_mask = np.zeros(len(table), dtype=bool)
    selected_mask[neighbors["_position"].to_numpy()] = True
    background_mask = ~is_reference & ~selected_mask

    category = table.loc[is_reference & (table["clip_id"] == reference_clip_id), "category"].iloc[0]
    highlight_color = COLOR_RELEVANT_REF if category == "relevant" else COLOR_IRRELEVANT_REF

    ax.scatter(
        tsne_embedding[background_mask, 0], tsne_embedding[background_mask, 1],
        s=10, c=COLOR_BACKGROUND_MUTED, alpha=0.4, linewidths=0, zorder=1,
    )

    points = tsne_embedding[selected_mask]
    _draw_hull(ax, points, highlight_color)
    ax.scatter(
        points[:, 0], points[:, 1], s=55, c=highlight_color, edgecolors=COLOR_TEXT_PRIMARY, linewidths=0.7,
        zorder=3, label=f"K={len(points)} nearest windows",
    )
    for rank, (x, y) in zip(neighbors["rank"], points):
        ax.annotate(str(rank), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=COLOR_TEXT_PRIMARY, zorder=4)

    reference_position = np.flatnonzero(is_reference & (table["clip_id"] == reference_clip_id).to_numpy())[0]
    reference_point = tsne_embedding[reference_position]
    ax.scatter(
        [reference_point[0]], [reference_point[1]], s=340, marker="*", c=highlight_color,
        edgecolors=COLOR_TEXT_PRIMARY, linewidths=1.1, zorder=5, label="Reference",
    )

    ax.set_title(f"t-SNE neighbors of reference '{reference_clip_id}'", fontsize=13, color=COLOR_TEXT_PRIMARY, loc="left", pad=12)
    ax.set_xlabel("Component 1", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("Component 2", fontsize=10, color=COLOR_TEXT_SECONDARY)
    legend = ax.legend(loc="best", frameon=True, facecolor=COLOR_BACKGROUND, edgecolor=COLOR_AXIS, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=COLOR_BACKGROUND)
    plt.close(fig)


def plot_overview(
    tsne_embedding: np.ndarray, table: pd.DataFrame, neighbors_by_reference: dict[str, pd.DataFrame],
    ref_ids: dict[str, int], output_path: Path,
) -> None:
    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(13.5, 7.5), facecolor=COLOR_BACKGROUND, gridspec_kw={"width_ratios": [3.2, 1]},
    )
    _style_axes(ax)

    is_reference = table["is_reference"].to_numpy()
    any_selected = np.zeros(len(table), dtype=bool)
    for neighbors in neighbors_by_reference.values():
        any_selected[neighbors["_position"].to_numpy()] = True
    background_mask = ~is_reference & ~any_selected

    ax.scatter(
        tsne_embedding[background_mask, 0], tsne_embedding[background_mask, 1],
        s=10, c=COLOR_BACKGROUND_MUTED, alpha=0.35, linewidths=0, zorder=1,
    )

    for reference_clip_id, neighbors in neighbors_by_reference.items():
        category = table.loc[is_reference & (table["clip_id"] == reference_clip_id), "category"].iloc[0]
        color = COLOR_RELEVANT_REF if category == "relevant" else COLOR_IRRELEVANT_REF
        points = tsne_embedding[neighbors["_position"].to_numpy()]
        _draw_hull(ax, points, color, alpha=0.10)
        ax.scatter(points[:, 0], points[:, 1], s=26, c=color, alpha=0.75, linewidths=0, zorder=3)

        reference_position = np.flatnonzero(is_reference & (table["clip_id"] == reference_clip_id).to_numpy())[0]
        reference_point = tsne_embedding[reference_position]
        ax.scatter(
            [reference_point[0]], [reference_point[1]], s=260, marker="*", c=color,
            edgecolors=COLOR_TEXT_PRIMARY, linewidths=0.8, zorder=5,
        )
        ax.annotate(
            str(ref_ids[reference_clip_id]), reference_point, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=6,
        )

    ax.set_title("K-NN neighborhoods of all references in t-SNE space", fontsize=13, color=COLOR_TEXT_PRIMARY, loc="left", pad=12)
    ax.set_xlabel("Component 1", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("Component 2", fontsize=10, color=COLOR_TEXT_SECONDARY)

    legend_ax.axis("off")
    legend_ax.set_facecolor(COLOR_BACKGROUND)
    legend_ax.text(0, 1.0, "References", fontsize=11, fontweight="bold", color=COLOR_TEXT_PRIMARY,
                    transform=legend_ax.transAxes, va="top")
    ordered_ids = sorted(neighbors_by_reference.keys(), key=lambda clip_id: ref_ids[clip_id])
    for line_index, clip_id in enumerate(ordered_ids):
        category = table.loc[is_reference & (table["clip_id"] == clip_id), "category"].iloc[0]
        color = COLOR_RELEVANT_REF if category == "relevant" else COLOR_IRRELEVANT_REF
        y = 0.94 - line_index * 0.045
        legend_ax.text(0, y, f"{ref_ids[clip_id]:>2}", fontsize=8.5, fontweight="bold", color=color,
                        transform=legend_ax.transAxes, va="top", family="monospace")
        legend_ax.text(0.09, y, clip_id, fontsize=8.5, color=COLOR_TEXT_PRIMARY,
                        transform=legend_ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=COLOR_BACKGROUND)
    plt.close(fig)


def render_window_video(
    clip: Clip, video_path: Path, start_frame: int, n_frames: int, output_path: Path, confidence_threshold: float,
) -> None:
    frame_labels = [f"frame_{n:04d}" for n in clip.frame_numbers[start_frame:start_frame + n_frames]]
    poses = clip.keypoints_2d[start_frame:start_frame + n_frames]
    create_visualization(video_path, output_path, frame_labels, poses, confidence_threshold)


def render_neighbor_videos(
    neighbors: pd.DataFrame, clip_by_key: dict[tuple[str, str], Clip], output_dir: Path, confidence_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in neighbors.itertuples(index=False):
        clip = clip_by_key.get((row.participant_id, row.clip_id))
        if clip is None:
            print(f"    ! no loaded clip for {row.participant_id}/{row.clip_id}; skipping video")
            continue
        video_path = resolve_video_path(clip)
        if video_path is None or not video_path.exists():
            print(f"    ! source video not found for {row.clip_id}; skipping video")
            continue
        if clip.keypoints_2d is None:
            print(f"    ! no 2D pixel keypoints (companion merged CSV) for {row.clip_id}; skipping video")
            continue
        start_frame, n_frames = int(row.window_start_frame), int(row.n_frames)
        output_path = output_dir / (
            f"rank{row.rank:02d}_{row.participant_id}_{row.clip_id}"
            f"_f{start_frame:05d}-{start_frame + n_frames:05d}.mp4"
        )
        print(f"    rendering rank {row.rank}: {row.participant_id}/{row.clip_id}"
              f" frames {start_frame}-{start_frame + n_frames} -> {output_path.name}")
        render_window_video(clip, video_path, start_frame, n_frames, output_path, confidence_threshold)


def run(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    force_reload: bool = False,
    k: int = DEFAULT_K,
    reference_clip_ids: list[str] | None = None,
    skip_videos: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    result: AnalysisResult = prepare_analysis(project_folder, reference_folder, cache_dir, output_dir, force_reload)
    table = result.table
    tsne_embedding = result.projections["tsne"]

    all_reference_ids = sorted(table.loc[table["is_reference"], "clip_id"].unique())
    if reference_clip_ids is None:
        reference_clip_ids = all_reference_ids
    else:
        unknown = sorted(set(reference_clip_ids) - set(all_reference_ids))
        if unknown:
            raise ValueError(f"Unknown reference clip_id(s): {unknown}. Available: {all_reference_ids}")

    knn_dir = output_dir / KNN_OUTPUT_SUBDIR
    knn_dir.mkdir(parents=True, exist_ok=True)
    clip_by_key = {(clip.participant_id, clip.clip_id): clip for clip in result.participant_clips}

    print(f"\nFinding K={k} nearest participant windows (t-SNE space) for {len(reference_clip_ids)} reference(s)...")
    neighbors_by_reference: dict[str, pd.DataFrame] = {}
    for reference_clip_id in reference_clip_ids:
        print(f"  {reference_clip_id}")
        neighbors = find_nearest_windows(table, tsne_embedding, reference_clip_id, k)
        neighbors_by_reference[reference_clip_id] = neighbors

        reference_dir = knn_dir / reference_clip_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        neighbors.drop(columns="_position").to_csv(reference_dir / "neighbors.csv", index=False)
        plot_reference_highlight(tsne_embedding, table, reference_clip_id, neighbors, reference_dir / "tsne_highlight.png")

        if not skip_videos:
            render_neighbor_videos(neighbors, clip_by_key, reference_dir / "windows", confidence_threshold)

    plot_overview(tsne_embedding, table, neighbors_by_reference, result.ref_ids, knn_dir / "overview_tsne.png")
    print(f"\nSaved KNN outputs to {knn_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-folder", type=Path, default=PROJECT_FOLDER_DEFAULT)
    parser.add_argument("--reference-folder", type=Path, default=REFERENCE_FOLDER_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of nearest participant windows per reference.")
    parser.add_argument("--references", type=str, default=None, help="Comma-separated reference clip_ids (default: all).")
    parser.add_argument("--skip-videos", action="store_true", help="Only produce plots/manifests, skip video rendering.")
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
        k=args.k,
        reference_clip_ids=args.references.split(",") if args.references else None,
        skip_videos=args.skip_videos,
        confidence_threshold=args.confidence_threshold,
    )
