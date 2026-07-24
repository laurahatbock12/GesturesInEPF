from pathlib import Path
import argparse
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import PROJECT_FOLDER, REFERENCE_FOLDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a fixed-bone-length skeletal model (fit_skeletal_model.py) for every "
        "participant recording and every reference video. Runs without a matching "
        "results_*_3d.csv (see estimate_3d_pose.py) are skipped."
    )
    parser.add_argument(
        "--project_folder",
        type=Path,
        default=Path(PROJECT_FOLDER),
    )
    parser.add_argument(
        "--reference_project_folder",
        type=Path,
        default=Path(REFERENCE_FOLDER),
    )
    parser.add_argument(
        "--schools",
        nargs="*",
        default=["2026_06_23_KantiHeerbrugg", "2026_06_02_GR_KantiOlten", "2026_05_20_KantiMusegg", "2026_06_08_GymKirschgarten", "2026_06_22_GymImmensee"],
        help="Optional school folder names. Defaults to every school folder in project_folder.",
    )
    parser.add_argument(
        "--participant_ids",
        nargs="*",
        default=None,
        help="Optional participant IDs. Defaults to every participant with recordings.",
    )
    parser.add_argument("--min_likelihood", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_participants",
        action="store_true",
        help="Skip per-participant recordings, only process reference videos.",
    )
    parser.add_argument(
        "--skip_reference",
        action="store_true",
        help="Skip reference videos, only process participant recordings.",
    )
    return parser.parse_args()


def has_recordings(folder):
    return any(
        path.is_dir() and path.name.startswith("recording_") and path.name.endswith(".mp4")
        for path in folder.iterdir()
    )


def find_participants(project_folder, schools, participant_ids):
    for school_folder in sorted(path for path in project_folder.iterdir() if path.is_dir()):
        if schools is not None and school_folder.name not in schools:
            continue
        for participant_folder in sorted(path for path in school_folder.iterdir() if path.is_dir()):
            if participant_ids is not None and participant_folder.name not in participant_ids:
                continue
            if has_recordings(participant_folder):
                yield school_folder.name, participant_folder.name


def run_command(command, dry_run):
    print(" ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def main(args):
    script_folder = Path(__file__).resolve().parent
    fit_script = script_folder / "fit_skeletal_model.py"

    common_args = [
        "--min_likelihood", str(args.min_likelihood),
        "--iterations", str(args.iterations),
        "--stiffness", str(args.stiffness),
    ]
    if args.save_vis:
        common_args.append("--save_vis")
    if args.force:
        common_args.append("--force")

    if not args.skip_participants:
        project_folder = args.project_folder.resolve()
        if not project_folder.is_dir():
            raise FileNotFoundError(f"Project folder does not exist: {project_folder}")
        participants = list(find_participants(project_folder, args.schools, args.participant_ids))
        if not participants:
            print("No participant folders with recording directories were found.")
        for school, participant_id in participants:
            print(f"\nProcessing {school}/{participant_id}")
            command = [
                sys.executable, str(fit_script),
                "--project_folder", str(project_folder),
                "--school", school,
                "--participant_id", participant_id,
                *common_args,
            ]
            run_command(command, args.dry_run)

    if not args.skip_reference:
        reference_project_folder = args.reference_project_folder.resolve()
        if not reference_project_folder.is_dir():
            raise FileNotFoundError(f"Reference project folder does not exist: {reference_project_folder}")
        print(f"\nProcessing reference videos in {reference_project_folder}")
        command = [
            sys.executable, str(fit_script),
            "--project_folder", str(reference_project_folder),
            "--process_reference_videos",
            *common_args,
        ]
        run_command(command, args.dry_run)


if __name__ == "__main__":
    main(parse_args())
