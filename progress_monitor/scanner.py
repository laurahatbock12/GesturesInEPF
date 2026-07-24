"""Fast filesystem scan of the EmbodiedProductiveFailure data tree.

Folder layout on disk: project_folder/<school>/<participant>/<recording_*.mp4>/<output files>
Each recording is itself a directory (despite the .mp4 suffix) holding the source
video plus whatever pipeline outputs have been produced so far. Completion of each
pipeline stage is decided the same way the processing scripts themselves decide
whether to skip a run: by the presence of its expected output file.
"""

import os
import time
from pathlib import Path

# (key, label) in pipeline order. Grouped in the UI as "2D pose estimation"
# (body/full_body/hand/merged) followed by "3D pipeline" (depth/pose_3d/fitted).
STAGES = [
    ("body", "2D Body"),
    ("full_body", "2D Full-Body"),
    ("hand", "2D Hands"),
    ("merged", "Merged"),
    ("depth", "Depth"),
    ("pose_3d", "3D Pose"),
    ("fitted", "3D Fitted"),
]
STAGE_KEYS = [key for key, _label in STAGES]


def _stage_filenames(run_name):
    return {
        "body": f"results_{run_name}_body.csv",
        "full_body": f"results_{run_name}_full_body.csv",
        "hand": f"results_{run_name}_hand.csv",
        "merged": f"results_{run_name}_merged.csv",
        "depth": f"depth_{run_name}.npz",
        "pose_3d": f"results_{run_name}_3d.csv",
        "fitted": f"results_{run_name}_3d_fitted.csv",
    }


def _is_recording_dir(entry):
    return (
        entry.is_dir(follow_symlinks=False)
        and entry.name.startswith("recording_")
        and entry.name.endswith(".mp4")
    )


def _scan_recording(entry):
    run_name = entry.name
    try:
        names = {child.name for child in os.scandir(entry.path)}
    except OSError:
        names = set()
    stages = {key: filename in names for key, filename in _stage_filenames(run_name).items()}
    return {
        "name": run_name,
        "stages": stages,
        "completed": sum(stages.values()),
    }


def _scan_participant(entry):
    try:
        recording_entries = [child for child in os.scandir(entry.path) if _is_recording_dir(child)]
    except OSError:
        recording_entries = []
    if not recording_entries:
        return None
    recordings = sorted((_scan_recording(child) for child in recording_entries), key=lambda r: r["name"])
    return {"name": entry.name, "recordings": recordings}


def _scan_school(entry):
    try:
        candidate_entries = [child for child in os.scandir(entry.path) if child.is_dir(follow_symlinks=False)]
    except OSError:
        candidate_entries = []
    participants = []
    for candidate in candidate_entries:
        participant = _scan_participant(candidate)
        if participant is not None:
            participants.append(participant)
    participants.sort(key=lambda p: p["name"])
    return {"name": entry.name, "participants": participants}


def scan(project_folder):
    """Walk the whole project tree and return a fully-materialized status snapshot.

    Cheap enough to run synchronously: one os.scandir() per school, participant,
    and recording folder (roughly 2 syscalls per recording), well under a second
    for datasets in the thousands of recordings.
    """
    start = time.monotonic()
    project_folder = Path(project_folder)
    try:
        school_entries = [child for child in os.scandir(project_folder) if child.is_dir(follow_symlinks=False)]
    except OSError:
        school_entries = []

    schools = sorted((_scan_school(entry) for entry in school_entries), key=lambda s: s["name"])

    total_recordings = sum(len(p["recordings"]) for s in schools for p in s["participants"])
    duration_ms = (time.monotonic() - start) * 1000

    return {
        "project_folder": str(project_folder),
        "generated_at": time.time(),
        "scan_duration_ms": round(duration_ms, 1),
        "stages": STAGES,
        "total_schools": len(schools),
        "total_participants": sum(len(s["participants"]) for s in schools),
        "total_recordings": total_recordings,
        "schools": schools,
    }
