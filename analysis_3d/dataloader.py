"""Efficient loading of estimate_3d_pose.py output (participant recordings + reference clips).

Mirrors analysis_2d/dataloader.py, but reads the lifted-3D prediction CSVs
(results_*_3d.csv, scorer="lifted-3d") instead of the merged 2D CSVs: keypoints are
(x, y, z, likelihood) in meters, in the camera's own frame (see estimate_3d_pose.py's
module docstring for how that scale/frame is calibrated). x/y/z are re-expressed in a
body-centered frame later, in features.py, right before kinematic features are computed.

Each clip also carries along its original 2D pixel keypoints (from the sibling merged
CSV that estimate_3d_pose.py was lifted from), purely so the DTW/KNN scripts can still
overlay a skeleton on the source video -- the lifted (X, Y, Z) are metric coordinates,
not pixel positions, and can't be drawn on a frame directly.

Like the 2D loader, each CSV is parsed once into a compact array and cached as .npz next
to a JSON metadata sidecar, keyed by a hash of the source path.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_estimation_2d.merge_poses import COORDINATES, MERGED_BODYPARTS, frame_number  # noqa: E402
from project_config import PROJECT_FOLDER, REFERENCE_FOLDER  # noqa: E402

PROJECT_FOLDER_DEFAULT = Path(PROJECT_FOLDER)
REFERENCE_FOLDER_DEFAULT = Path(REFERENCE_FOLDER)
CACHE_DIR_DEFAULT = Path(__file__).resolve().parent / "cache"

DEFAULT_FPS = 30.0
COORDS_3D = ("x", "y", "z", "likelihood")
_BODYPART_COLUMNS_3D = pd.MultiIndex.from_product([MERGED_BODYPARTS, COORDS_3D], names=["bodyparts", "coords"])
_BODYPART_COLUMNS_2D = pd.MultiIndex.from_product([MERGED_BODYPARTS, COORDINATES], names=["bodyparts", "coords"])


@dataclass
class Clip:
    """One lifted-3D pose sequence: a participant recording or a single reference gesture."""

    clip_id: str
    keypoints: np.ndarray  # (n_frames, n_bodyparts, 4) float32, columns = x, y, z, likelihood (meters)
    keypoints_2d: np.ndarray | None  # (n_frames, n_bodyparts, 3) float32 x, y, likelihood (pixels); None if the
    # companion 2D merged CSV wasn't found. Only used to overlay a skeleton on the source video.
    frame_numbers: np.ndarray  # (n_frames,) int32, original frame indices (may have gaps)
    fps: float
    is_reference: bool
    category: str  # "relevant" / "irrelevant" for references, task name for participants
    participant_id: str | None
    school: str | None
    source_csv: Path

    @property
    def n_frames(self) -> int:
        return self.keypoints.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


def _cache_paths(csv_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    key = hashlib.sha1(str(csv_path).encode()).hexdigest()[:20]
    return cache_dir / f"{key}.npz", cache_dir / f"{key}.json"


def _read_fps(video_path: Path | None) -> float:
    if video_path is None or not video_path.exists():
        return DEFAULT_FPS
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return float(fps) if fps and fps > 1e-3 else DEFAULT_FPS


def _find_participant_video(recording_dir: Path) -> Path | None:
    candidates = [
        path for path in recording_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".mp4", ".avi") and not path.name.startswith("visualization_")
        and not path.name.startswith("results_") and not path.name.startswith("bboxes_")
    ]
    return candidates[0] if candidates else None


def _parse_participant_meta(csv_path: Path) -> dict:
    # Directory layout is identical to the merged 2D CSV (same recording folder), so this
    # is unchanged from analysis_2d/dataloader.py's version.
    recording_dir = csv_path.parent
    participant_dir = recording_dir.parent
    school_dir = participant_dir.parent

    participant_id = participant_dir.name
    recording_name = recording_dir.name.removesuffix(".mp4")
    task = re.sub(rf"^recording_{re.escape(participant_id)}_", "", recording_name, flags=re.IGNORECASE)
    task = re.sub(r"__?\d+$", "", task)

    return {
        "clip_id": recording_name,
        "category": task,
        "participant_id": participant_id,
        "school": school_dir.name,
        "video_path": _find_participant_video(recording_dir),
    }


def _parse_reference_meta(csv_path: Path) -> dict:
    category_dir = csv_path.parents[2]  # EPF_PL_<category>DA/pose_estimation/pose_3d_predictions
    category = "relevant" if "irrelevant" not in category_dir.name.lower() else "irrelevant"
    clip_name = csv_path.name.removeprefix("results_").removesuffix("_3d.csv").removesuffix(".mp4")
    video_path = category_dir / f"{clip_name}.mp4"

    return {
        "clip_id": clip_name,
        "category": category,
        "participant_id": None,
        "school": None,
        "video_path": video_path,
    }


def _companion_2d_csv(csv_path_3d: Path, is_reference: bool) -> Path:
    """Path to the merged 2D CSV that estimate_3d_pose.py lifted `csv_path_3d` from."""
    merged_name = csv_path_3d.name.removesuffix("_3d.csv") + "_merged.csv"
    if is_reference:
        # .../pose_estimation/pose_3d_predictions/results_X_3d.csv -> .../pose_estimation/merged_predictions/...
        return csv_path_3d.parents[1] / "merged_predictions" / merged_name
    return csv_path_3d.parent / merged_name


def _parse_csv_3d(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    table = table.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
    table = table.reindex(columns=_BODYPART_COLUMNS_3D)
    keypoints = table.to_numpy(dtype=np.float32).reshape(len(table), len(MERGED_BODYPARTS), len(COORDS_3D))
    frame_numbers = np.array([frame_number(label) for label in table.index], dtype=np.int32)
    return keypoints, frame_numbers


def _parse_csv_2d(csv_path: Path) -> np.ndarray:
    """Same parsing as analysis_2d/dataloader.py's _parse_csv, but only the keypoints
    (not frame_numbers: the 3D array's own frame_numbers are used, see load note below)."""
    table = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    table = table.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
    table = table.reindex(columns=_BODYPART_COLUMNS_2D)
    return table.to_numpy(dtype=np.float32).reshape(len(table), len(MERGED_BODYPARTS), len(COORDINATES))


def _load_clip(csv_path: Path, is_reference: bool, cache_dir: Path, force_reload: bool) -> Clip:
    cache_npz, cache_json = _cache_paths(csv_path, cache_dir)
    source_mtime = csv_path.stat().st_mtime

    if not force_reload and cache_npz.exists() and cache_json.exists():
        meta = json.loads(cache_json.read_text())
        if meta.get("source_mtime") == source_mtime:
            arrays = np.load(cache_npz)
            return Clip(
                clip_id=meta["clip_id"],
                keypoints=arrays["keypoints"],
                keypoints_2d=arrays["keypoints_2d"] if "keypoints_2d" in arrays.files else None,
                frame_numbers=arrays["frame_numbers"],
                fps=meta["fps"],
                is_reference=is_reference,
                category=meta["category"],
                participant_id=meta["participant_id"],
                school=meta["school"],
                source_csv=csv_path,
            )

    meta = _parse_reference_meta(csv_path) if is_reference else _parse_participant_meta(csv_path)
    keypoints, frame_numbers = _parse_csv_3d(csv_path)
    fps = _read_fps(meta["video_path"])

    companion_csv = _companion_2d_csv(csv_path, is_reference)
    keypoints_2d = None
    if companion_csv.exists():
        keypoints_2d = _parse_csv_2d(companion_csv)
        if keypoints_2d.shape[0] != keypoints.shape[0]:
            print(
                f"Warning: {companion_csv.name} has {keypoints_2d.shape[0]} frames, "
                f"3D pose has {keypoints.shape[0]}; skipping 2D pixel overlay for {csv_path.name}."
            )
            keypoints_2d = None

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"keypoints": keypoints, "frame_numbers": frame_numbers}
    if keypoints_2d is not None:
        save_kwargs["keypoints_2d"] = keypoints_2d
    np.savez_compressed(cache_npz, **save_kwargs)
    cache_json.write_text(json.dumps({
        "clip_id": meta["clip_id"],
        "category": meta["category"],
        "participant_id": meta["participant_id"],
        "school": meta["school"],
        "fps": fps,
        "source_mtime": source_mtime,
    }))

    return Clip(
        clip_id=meta["clip_id"],
        keypoints=keypoints,
        keypoints_2d=keypoints_2d,
        frame_numbers=frame_numbers,
        fps=fps,
        is_reference=is_reference,
        category=meta["category"],
        participant_id=meta["participant_id"],
        school=meta["school"],
        source_csv=csv_path,
    )


def find_participant_csvs(project_folder: Path) -> list[Path]:
    return sorted(project_folder.glob("*/*/recording_*.mp4/results_*_3d.csv"))


def find_reference_csvs(reference_folder: Path) -> list[Path]:
    return sorted(reference_folder.glob("*/pose_estimation/pose_3d_predictions/results_*_3d.csv"))


def _load_many(csv_paths: list[Path], is_reference: bool, cache_dir: Path, force_reload: bool, n_jobs: int) -> list[Clip]:
    if not csv_paths:
        return []
    clips: list[Clip] = [None] * len(csv_paths)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(_load_clip, path, is_reference, cache_dir, force_reload): index
            for index, path in enumerate(csv_paths)
        }
        for future in as_completed(futures):
            clips[futures[future]] = future.result()
    return clips


def resolve_video_path(clip: Clip) -> Path | None:
    """Locate the source video a clip's lifted-3D predictions were computed from."""
    meta = _parse_reference_meta(clip.source_csv) if clip.is_reference else _parse_participant_meta(clip.source_csv)
    return meta["video_path"]


def load_dataset(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    force_reload: bool = False,
    n_jobs: int = 8,
) -> tuple[list[Clip], list[Clip]]:
    """Load all participant recordings and reference clips that have lifted 3D pose.

    Returns (participant_clips, reference_clips).
    """
    participant_csvs = find_participant_csvs(project_folder)
    reference_csvs = find_reference_csvs(reference_folder)

    participant_clips = _load_many(participant_csvs, False, cache_dir, force_reload, n_jobs)
    reference_clips = _load_many(reference_csvs, True, cache_dir, force_reload, n_jobs)

    return participant_clips, reference_clips
