"""Per-frame kinematic feature time series for each reference clip.

For every reference clip, plots each of the 10 underlying kinematic signals (the same
ones aggregated into the *_mean/*_std columns used everywhere else in this analysis)
across the clip's frames, one subplot per signal, and saves it into that reference's
output/knn/<clip_id>/ folder alongside the existing t-SNE/KNN outputs.

Usage:
    python reference_timeseries.py [--references rDA1_DeviationDistance,iDA3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering_analysis import (
    COLOR_BACKGROUND,
    COLOR_IRRELEVANT_REF,
    COLOR_RELEVANT_REF,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    OUTPUT_DIR_DEFAULT,
    _style_axes,
)
from dataloader import CACHE_DIR_DEFAULT, PROJECT_FOLDER_DEFAULT, REFERENCE_FOLDER_DEFAULT, Clip, load_dataset
from features import SIGNAL_NAMES, compute_raw_signals, estimate_shoulder_width

KNN_OUTPUT_SUBDIR = "knn"


def plot_kinematic_timeseries(clip: Clip, output_path: Path) -> dict[str, np.ndarray]:
    shoulder_width = estimate_shoulder_width(clip.keypoints)
    signals = compute_raw_signals(clip.keypoints, clip.fps, shoulder_width)
    time_s = (clip.frame_numbers - clip.frame_numbers[0]) / clip.fps

    line_color = COLOR_RELEVANT_REF if clip.category == "relevant" else COLOR_IRRELEVANT_REF

    fig, axes = plt.subplots(
        len(SIGNAL_NAMES), 1, figsize=(8.5, 1.5 * len(SIGNAL_NAMES)),
        sharex=True, facecolor=COLOR_BACKGROUND,
    )
    for ax, signal_name in zip(axes, SIGNAL_NAMES):
        _style_axes(ax)
        ax.plot(time_s, signals[signal_name], color=line_color, linewidth=1.4)
        ax.axhline(0, color=COLOR_TEXT_SECONDARY, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_ylabel(
            signal_name.replace("_", " "), fontsize=8.5, color=COLOR_TEXT_SECONDARY,
            rotation=0, ha="right", va="center", labelpad=8,
        )

    axes[-1].set_xlabel("Time (s)", fontsize=10, color=COLOR_TEXT_SECONDARY)
    fig.suptitle(
        f"Kinematic feature time series — {clip.clip_id} ({clip.category})",
        fontsize=13, color=COLOR_TEXT_PRIMARY, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0.1, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, facecolor=COLOR_BACKGROUND)
    plt.close(fig)

    return signals


def run(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    force_reload: bool = False,
    reference_clip_ids: list[str] | None = None,
) -> None:
    print("Loading reference clips...")
    _, reference_clips = load_dataset(
        project_folder=project_folder, reference_folder=reference_folder,
        cache_dir=cache_dir, force_reload=force_reload,
    )
    if not reference_clips:
        raise RuntimeError("No reference clips found.")

    if reference_clip_ids is not None:
        available = {clip.clip_id for clip in reference_clips}
        unknown = sorted(set(reference_clip_ids) - available)
        if unknown:
            raise ValueError(f"Unknown reference clip_id(s): {unknown}. Available: {sorted(available)}")
        reference_clips = [clip for clip in reference_clips if clip.clip_id in reference_clip_ids]

    knn_dir = output_dir / KNN_OUTPUT_SUBDIR
    for clip in sorted(reference_clips, key=lambda c: c.clip_id):
        reference_dir = knn_dir / clip.clip_id
        reference_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {clip.clip_id} ({clip.n_frames} frames, {clip.duration_s:.2f}s)")
        signals = plot_kinematic_timeseries(clip, reference_dir / "kinematic_timeseries.png")

        time_s = (clip.frame_numbers - clip.frame_numbers[0]) / clip.fps
        timeseries_table = pd.DataFrame({"frame": clip.frame_numbers, "time_s": time_s, **signals})
        timeseries_table.to_csv(reference_dir / "kinematic_timeseries.csv", index=False)

    print(f"\nSaved {len(reference_clips)} time series plot(s) under {knn_dir}/<clip_id>/kinematic_timeseries.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-folder", type=Path, default=PROJECT_FOLDER_DEFAULT)
    parser.add_argument("--reference-folder", type=Path, default=REFERENCE_FOLDER_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--references", type=str, default=None, help="Comma-separated reference clip_ids (default: all).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        project_folder=args.project_folder,
        reference_folder=args.reference_folder,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_reload=args.force_reload,
        reference_clip_ids=args.references.split(",") if args.references else None,
    )
