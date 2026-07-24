"""Efficient loading of merge_poses.py output (participant recordings + reference clips).

Merged prediction CSVs use a 4-row multi-index header, which is slow for pandas to
parse repeatedly. This module parses each CSV once into a compact (frames, bodyparts, 3)
float32 array and caches it as .npz next to a JSON metadata sidecar, keyed by a hash of
the source path. Subsequent loads just read the cache, which is orders of magnitude
faster once the cache is warm.
"""
from __future__ import annotations

import hashlib
import json
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
_BODYPART_COLUMNS = pd.MultiIndex.from_product([MERGED_BODYPARTS, COORDINATES], names=["bodyparts", "coords"])


@dataclass
class Clip:
    """One pose sequence: a participant recording or a single reference gesture."""

    clip_id: str
    keypoints: np.ndarray  # (n_frames, n_bodyparts, 3) float32, columns = x, y, likelihood
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
    import re

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
    category_dir = csv_path.parents[2]  # merged_predictions/pose_estimation/EPF_PL_<category>DA
    category = "relevant" if "irrelevant" not in category_dir.name.lower() else "irrelevant"
    clip_name = csv_path.name.removeprefix("results_").removesuffix("_merged.csv").removesuffix(".mp4")
    video_path = category_dir / f"{clip_name}.mp4"

    return {
        "clip_id": clip_name,
        "category": category,
        "participant_id": None,
        "school": None,
        "video_path": video_path,
    }


def _parse_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    table = table.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
    table = table.reindex(columns=_BODYPART_COLUMNS)
    keypoints = table.to_numpy(dtype=np.float32).reshape(len(table), len(MERGED_BODYPARTS), len(COORDINATES))
    frame_numbers = np.array([frame_number(label) for label in table.index], dtype=np.int32)
    return keypoints, frame_numbers


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
                frame_numbers=arrays["frame_numbers"],
                fps=meta["fps"],
                is_reference=is_reference,
                category=meta["category"],
                participant_id=meta["participant_id"],
                school=meta["school"],
                source_csv=csv_path,
            )

    meta = _parse_reference_meta(csv_path) if is_reference else _parse_participant_meta(csv_path)
    keypoints, frame_numbers = _parse_csv(csv_path)
    fps = _read_fps(meta["video_path"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, keypoints=keypoints, frame_numbers=frame_numbers)
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
        frame_numbers=frame_numbers,
        fps=fps,
        is_reference=is_reference,
        category=meta["category"],
        participant_id=meta["participant_id"],
        school=meta["school"],
        source_csv=csv_path,
    )


def find_participant_csvs(project_folder: Path) -> list[Path]:
    return sorted(project_folder.glob("*/*/recording_*.mp4/results_*_merged.csv"))


def find_reference_csvs(reference_folder: Path) -> list[Path]:
    return sorted(reference_folder.glob("*/pose_estimation/merged_predictions/results_*_merged.csv"))


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
    """Locate the source video a clip's merged predictions were computed from."""
    meta = _parse_reference_meta(clip.source_csv) if clip.is_reference else _parse_participant_meta(clip.source_csv)
    return meta["video_path"]


def load_dataset(
    project_folder: Path = PROJECT_FOLDER_DEFAULT,
    reference_folder: Path = REFERENCE_FOLDER_DEFAULT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    force_reload: bool = False,
    n_jobs: int = 8,
) -> tuple[list[Clip], list[Clip]]:
    """Load all participant recordings and reference clips that have merged predictions.

    Returns (participant_clips, reference_clips).
    """
    participant_csvs = find_participant_csvs(project_folder)
    reference_csvs = find_reference_csvs(reference_folder)

    participant_clips = _load_many(participant_csvs, False, cache_dir, force_reload, n_jobs)
    reference_clips = _load_many(reference_csvs, True, cache_dir, force_reload, n_jobs)

    return participant_clips, reference_clips
