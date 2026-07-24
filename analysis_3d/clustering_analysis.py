"""Clustering analysis: can kinematic features locate reference gestures inside
participant recordings?

3D counterpart of analysis_2d/clustering_analysis.py: identical pipeline, but the
underlying clips are lifted-3D poses (dataloader.py) and the kinematic features
(features.py) are computed in 3D, in a per-frame body-centered coordinate frame.

Pipeline:
  1. Load all lifted-3D clips (participants + references) via dataloader.py.
  2. Report reference clip length vs. the mean reference length (they set the window size).
  3. Slide a window (mean reference duration, 50% overlap) over every participant clip and
     extract the aggregate kinematic features (features.py) from each window; extract the
     same features once from each full reference clip.
  4. Standardize the combined feature matrix and project it with PCA, t-SNE and UMAP.
  5. Plot each projection, visually emphasizing where the reference clips land.

Usage:
    python clustering_analysis.py [--project-folder ...] [--reference-folder ...] [--force-reload]
"""
from __future__ import annotations

import argparse
import colorsys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from dataloader import CACHE_DIR_DEFAULT, PROJECT_FOLDER_DEFAULT, REFERENCE_FOLDER_DEFAULT, Clip, load_dataset
from features import FEATURE_NAMES, compute_window_features, estimate_shoulder_width

OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "output"
OVERLAP = 0.5

# Palette (validated categorical set, see dataviz skill references/palette.md).
COLOR_BACKGROUND = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_RELEVANT_REF = "#e34948"  # categorical slot 8, red
COLOR_IRRELEVANT_REF = "#eb6834"  # categorical slot 6, orange
# Categorical slots 1,2,3,4,5,7 (orange/red are reserved for reference markers above).
SCHOOL_HUES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#4a3aa7"]
PARTICIPANT_MIN_LIGHTNESS = 0.30
PARTICIPANT_MAX_LIGHTNESS = 0.72
PARTICIPANT_ALPHA = 0.55


def build_school_palette(schools: list[str]) -> dict[str, str]:
    """Assign each school a fixed categorical hue, cycling with a warning past 6 schools."""
    if len(schools) > len(SCHOOL_HUES):
        print(f"Warning: {len(schools)} schools but only {len(SCHOOL_HUES)} distinguishable hues; colors will repeat.")
    return {school: SCHOOL_HUES[index % len(SCHOOL_HUES)] for index, school in enumerate(sorted(schools))}


def _with_lightness(hex_color: str, lightness: float) -> tuple[float, float, float]:
    hue, _, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def participant_row_colors(table: pd.DataFrame, school_palette: dict[str, str]) -> np.ndarray:
    """RGBA per row: hue identifies the school, lightness ranks the participant within it
    (alphabetical, dark to light). Reference rows get an unused placeholder color/alpha."""
    colors = np.zeros((len(table), 4))
    colors[:, 3] = PARTICIPANT_ALPHA

    participant_table = table.loc[~table["is_reference"], ["school", "participant_id"]]
    for school in participant_table["school"].dropna().unique():
        base_hue = school_palette[school]
        participants = sorted(participant_table.loc[participant_table["school"] == school, "participant_id"].unique())
        for index, participant in enumerate(participants):
            fraction = 0.5 if len(participants) == 1 else index / (len(participants) - 1)
            lightness = PARTICIPANT_MIN_LIGHTNESS + fraction * (PARTICIPANT_MAX_LIGHTNESS - PARTICIPANT_MIN_LIGHTNESS)
            row_mask = ((table["school"] == school) & (table["participant_id"] == participant)).to_numpy()
            colors[row_mask, :3] = _with_lightness(base_hue, lightness)
    return colors


def reference_length_table(reference_clips: list[Clip]) -> pd.DataFrame:
    rows = [
        {
            "clip_id": clip.clip_id,
            "category": clip.category,
            "n_frames": clip.n_frames,
            "fps": round(clip.fps, 2),
            "duration_s": clip.duration_s,
        }
        for clip in reference_clips
    ]
    table = pd.DataFrame(rows).sort_values(["category", "clip_id"]).reset_index(drop=True)
    mean_frames = table["n_frames"].mean()
    mean_duration = table["duration_s"].mean()
    table["diff_from_mean_frames"] = table["n_frames"] - mean_frames
    table["diff_from_mean_s"] = table["duration_s"] - mean_duration
    return table


def extract_window_rows(clip: Clip, window_frames: int, step_frames: int) -> list[dict]:
    shoulder_width = estimate_shoulder_width(clip.keypoints)
    n_frames = clip.n_frames
    if n_frames < window_frames:
        starts = [0]
        window_frames = n_frames
    else:
        starts = list(range(0, n_frames - window_frames + 1, step_frames))

    rows = []
    for window_index, start in enumerate(starts):
        window_keypoints = clip.keypoints[start:start + window_frames]
        features = compute_window_features(window_keypoints, clip.fps, shoulder_width)
        row = {
            "clip_id": clip.clip_id,
            "participant_id": clip.participant_id,
            "school": clip.school,
            "category": clip.category,
            "is_reference": False,
            "window_index": window_index,
            "window_start_frame": start,
            "n_frames": window_keypoints.shape[0],
        }
        row.update(zip(FEATURE_NAMES, features))
        rows.append(row)
    return rows


def extract_reference_rows(reference_clips: list[Clip]) -> list[dict]:
    rows = []
    for clip in reference_clips:
        shoulder_width = estimate_shoulder_width(clip.keypoints)
        features = compute_window_features(clip.keypoints, clip.fps, shoulder_width)
        row = {
            "clip_id": clip.clip_id,
            "participant_id": None,
            "school": None,
            "category": clip.category,
            "is_reference": True,
            "window_index": 0,
            "window_start_frame": 0,
            "n_frames": clip.n_frames,
        }
        row.update(zip(FEATURE_NAMES, features))
        rows.append(row)
    return rows


def build_feature_table(
    participant_clips: list[Clip], reference_clips: list[Clip], window_frames_by_fps: dict[float, int]
) -> pd.DataFrame:
    rows = extract_reference_rows(reference_clips)
    for clip in participant_clips:
        window_frames = window_frames_by_fps[clip.fps]
        step_frames = max(1, round(window_frames * (1 - OVERLAP)))
        rows.extend(extract_window_rows(clip, window_frames, step_frames))
    return pd.DataFrame(rows)


def normalize_per_video(table: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature per its own group instead of on the pooled dataset.

    Participant windows are normalized against their own clip's mean/std across that
    clip's own windows (removes each recording's baseline scale/speed so windows are
    compared on relative dynamics). A reference clip is only ever one window (the full
    clip), so it can't supply its own std; references are instead normalized against
    the pooled mean/std across all reference clips. Clips with fewer than 2 windows,
    or a feature with zero within-clip variance, fall back to the pooled participant
    statistics for that feature so nothing divides by zero.
    """
    normalized = table.copy()
    feature_cols = FEATURE_NAMES

    reference_mask = table["is_reference"].to_numpy()
    participant_mask = ~reference_mask

    reference_values = table.loc[reference_mask, feature_cols]
    reference_mean = reference_values.mean()
    reference_std = reference_values.std(ddof=0).replace(0, np.nan)
    normalized.loc[reference_mask, feature_cols] = (reference_values - reference_mean) / reference_std

    participant_values = table.loc[participant_mask, feature_cols]
    pooled_mean = participant_values.mean()
    pooled_std = participant_values.std(ddof=0).replace(0, np.nan)

    for _, group_index in table.loc[participant_mask].groupby("clip_id").groups.items():
        group = table.loc[group_index, feature_cols]
        if len(group) >= 2:
            clip_mean = group.mean()
            clip_std = group.std(ddof=0).replace(0, np.nan).fillna(pooled_std)
        else:
            clip_mean, clip_std = pooled_mean, pooled_std
        normalized.loc[group_index, feature_cols] = (group - clip_mean) / clip_std

    normalized[feature_cols] = normalized[feature_cols].fillna(0.0)
    return normalized


def _project(features: np.ndarray, n_neighbors_hint: int) -> dict[str, np.ndarray]:
    scaled = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2, random_state=0)
    pca_embedding = pca.fit_transform(scaled)

    perplexity = min(30, max(5, n_neighbors_hint // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0)
    tsne_embedding = tsne.fit_transform(scaled)

    import umap  # imported lazily: heavier import, only needed here

    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(2, n_neighbors_hint - 1)), random_state=0)
    umap_embedding = reducer.fit_transform(scaled)

    return {
        "pca": pca_embedding,
        "tsne": tsne_embedding,
        "umap": umap_embedding,
        "pca_explained_variance": pca.explained_variance_ratio_,
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(COLOR_BACKGROUND)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
    ax.set_axisbelow(True)


def reference_id_map(table: pd.DataFrame) -> dict[str, int]:
    """Stable clip_id -> integer marker label, shared across all three plots."""
    reference_ids = table.loc[table["is_reference"], ["category", "clip_id"]].drop_duplicates()
    reference_ids = reference_ids.sort_values(["category", "clip_id"]).reset_index(drop=True)
    return {row.clip_id: index + 1 for index, row in reference_ids.iterrows()}


def plot_embedding(
    embedding: np.ndarray,
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    ref_ids: dict[str, int],
    row_colors: np.ndarray,
    school_palette: dict[str, str],
) -> None:
    import matplotlib.patheffects as path_effects

    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(13.5, 7.5), facecolor=COLOR_BACKGROUND, gridspec_kw={"width_ratios": [3.2, 1]},
    )
    _style_axes(ax)

    is_reference = table["is_reference"].to_numpy()
    participant_mask = ~is_reference
    relevant_mask = is_reference & (table["category"] == "relevant")
    irrelevant_mask = is_reference & (table["category"] == "irrelevant")

    ax.scatter(
        embedding[participant_mask, 0], embedding[participant_mask, 1],
        s=14, c=row_colors[participant_mask], linewidths=0, zorder=2,
    )
    ax.scatter(
        embedding[irrelevant_mask, 0], embedding[irrelevant_mask, 1],
        s=260, c=COLOR_IRRELEVANT_REF, marker="*", edgecolors=COLOR_TEXT_PRIMARY, linewidths=0.8,
        zorder=4,
    )
    ax.scatter(
        embedding[relevant_mask, 0], embedding[relevant_mask, 1],
        s=260, c=COLOR_RELEVANT_REF, marker="*", edgecolors=COLOR_TEXT_PRIMARY, linewidths=0.8,
        zorder=5,
    )

    for mask in (relevant_mask, irrelevant_mask):
        for (x, y), clip_id in zip(embedding[mask], table.loc[mask, "clip_id"]):
            text = ax.text(
                x, y, str(ref_ids[clip_id]), fontsize=7.5, fontweight="bold", color="white",
                ha="center", va="center", zorder=6,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground=COLOR_TEXT_PRIMARY)])

    ax.set_title(title, fontsize=13, color=COLOR_TEXT_PRIMARY, loc="left", pad=12)
    ax.set_xlabel("Component 1", fontsize=10, color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("Component 2", fontsize=10, color=COLOR_TEXT_SECONDARY)

    school_handles = [
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=7,
            markerfacecolor=_with_lightness(base_hue, 0.5), markeredgewidth=0, label=school,
        )
        for school, base_hue in school_palette.items()
    ]
    reference_handles = [
        mlines.Line2D([], [], marker="*", linestyle="None", markersize=12, markerfacecolor=COLOR_RELEVANT_REF,
                       markeredgecolor=COLOR_TEXT_PRIMARY, label="Reference (relevant)"),
        mlines.Line2D([], [], marker="*", linestyle="None", markersize=12, markerfacecolor=COLOR_IRRELEVANT_REF,
                      markeredgecolor=COLOR_TEXT_PRIMARY, label="Reference (irrelevant)"),
    ]
    legend = ax.legend(
        handles=school_handles + reference_handles, loc="best", frameon=True,
        facecolor=COLOR_BACKGROUND, edgecolor=COLOR_AXIS, fontsize=9, title="School (shade = participant)",
    )
    legend.get_title().set_color(COLOR_TEXT_PRIMARY)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_PRIMARY)

    # Side panel: numbered reference legend, since markers cluster too tightly for direct labels.
    legend_ax.axis("off")
    legend_ax.set_facecolor(COLOR_BACKGROUND)
    reference_rows = table.loc[is_reference, ["clip_id", "category"]].drop_duplicates()
    reference_rows = sorted(reference_rows.itertuples(index=False), key=lambda row: ref_ids[row.clip_id])
    legend_ax.text(0, 1.0, "References", fontsize=11, fontweight="bold", color=COLOR_TEXT_PRIMARY,
                    transform=legend_ax.transAxes, va="top")
    for line_index, row in enumerate(reference_rows):
        color = COLOR_RELEVANT_REF if row.category == "relevant" else COLOR_IRRELEVANT_REF
        y = 0.94 - line_index * 0.045
        legend_ax.text(0, y, f"{ref_ids[row.clip_id]:>2}", fontsize=8.5, fontweight="bold", color=color,
                        transform=legend_ax.transAxes, va="top", family="monospace")
        legend_ax.text(0.09, y, row.clip_id, fontsize=8.5, color=COLOR_TEXT_PRIMARY,
                        transform=legend_ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=COLOR_BACKGROUND)
    plt.close(fig)


@dataclass
class AnalysisResult:
    participant_clips: list[Clip]
    reference_clips: list[Clip]
    table: pd.DataFrame  # normalized features + metadata, one row per window/reference, RangeIndex
    projections: dict[str, np.ndarray]  # "pca" / "tsne" / "umap" (each aligned to table's row order)
    ref_ids: dict[str, int]
    school_palette: dict[str, str]
    row_colors: np.ndarray


def prepare_analysis(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    force_reload: bool = False,
) -> AnalysisResult:
    """Load clips, extract + normalize kinematic features, and project to PCA/t-SNE/UMAP.

    Shared by clustering_analysis.py (plots the projections) and knn_reference_clusters.py
    (finds participant windows near each reference in the t-SNE embedding this returns).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clips...")
    participant_clips, reference_clips = load_dataset(
        project_folder=project_folder, reference_folder=reference_folder,
        cache_dir=cache_dir, force_reload=force_reload,
    )
    print(f"  {len(participant_clips)} participant clips, {len(reference_clips)} reference clips")
    if not reference_clips:
        raise RuntimeError("No reference clips found; cannot size the analysis window.")

    length_table = reference_length_table(reference_clips)
    length_table.to_csv(output_dir / "reference_length_table.csv", index=False)
    print("\nReference clip lengths vs. mean:")
    print(length_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    mean_duration_s = length_table["duration_s"].mean()
    print(f"\nMean reference duration: {mean_duration_s:.3f}s -> used as the sliding-window size"
          f" ({int(OVERLAP * 100)}% overlap) over participant clips.")

    clip_fps_values = {clip.fps for clip in participant_clips} | {clip.fps for clip in reference_clips}
    window_frames_by_fps = {fps: max(2, round(mean_duration_s * fps)) for fps in clip_fps_values}

    print("Extracting kinematic features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        table = build_feature_table(participant_clips, reference_clips, window_frames_by_fps)

    n_total = len(table)
    table = table.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    n_dropped = n_total - len(table)
    print(f"  {len(table)} feature rows kept ({n_dropped} dropped for insufficient valid keypoints)"
          f" [{(table['is_reference']).sum()} references,"
          f" {(~table['is_reference']).sum()} participant windows]")

    table.to_csv(output_dir / "kinematic_features_raw.csv", index=False)

    print("Normalizing features per video (participant windows against their own clip;"
          " references against the reference pool)...")
    table = normalize_per_video(table)
    table.to_csv(output_dir / "kinematic_features_normalized.csv", index=False)

    features = table[FEATURE_NAMES].to_numpy()
    projections = _project(features, n_neighbors_hint=len(table))

    print("\nPCA explained variance (components 1-2):",
          [f"{v:.1%}" for v in projections["pca_explained_variance"]])

    ref_ids = reference_id_map(table)
    school_palette = build_school_palette(sorted({clip.school for clip in participant_clips if clip.school}))
    row_colors = participant_row_colors(table, school_palette)

    return AnalysisResult(
        participant_clips=participant_clips,
        reference_clips=reference_clips,
        table=table,
        projections=projections,
        ref_ids=ref_ids,
        school_palette=school_palette,
        row_colors=row_colors,
    )


def run(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    output_dir: Path = OUTPUT_DIR_DEFAULT,
    force_reload: bool = False,
) -> None:
    result = prepare_analysis(project_folder, reference_folder, cache_dir, output_dir, force_reload)

    plot_embedding(
        result.projections["pca"], result.table, "PCA of gesture kinematic features",
        output_dir / "pca.png", result.ref_ids, result.row_colors, result.school_palette,
    )
    plot_embedding(
        result.projections["tsne"], result.table, "t-SNE of gesture kinematic features",
        output_dir / "tsne.png", result.ref_ids, result.row_colors, result.school_palette,
    )
    plot_embedding(
        result.projections["umap"], result.table, "UMAP of gesture kinematic features",
        output_dir / "umap.png", result.ref_ids, result.row_colors, result.school_palette,
    )

    print(f"\nSaved outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-folder", type=Path, default=PROJECT_FOLDER_DEFAULT)
    parser.add_argument("--reference-folder", type=Path, default=REFERENCE_FOLDER_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        project_folder=args.project_folder,
        reference_folder=args.reference_folder,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_reload=args.force_reload,
    )
