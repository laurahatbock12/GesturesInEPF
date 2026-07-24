from pathlib import Path
import argparse
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import PROJECT_FOLDER


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run body, full-body, hand pose estimation and merged visualization for all recordings."
	)
	parser.add_argument(
		"--project_folder",
		type=Path,
		default=Path(PROJECT_FOLDER),
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
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--visualization_frames", type=int, default=200)
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--dry_run", action="store_true")
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
	print(" ".join(command))
	if not dry_run:
		subprocess.run(command, check=True)


def main(args):
	project_folder = args.project_folder.resolve()
	if not project_folder.is_dir():
		raise FileNotFoundError(f"Project folder does not exist: {project_folder}")

	script_folder = Path(__file__).resolve().parent
	inference_script = script_folder / "run_2d_pose_video.py"
	merge_script = script_folder / "merge_poses.py"
	participants = list(find_participants(project_folder, args.schools, args.participant_ids))
	if not participants:
		raise FileNotFoundError("No participant folders with recording directories were found.")

	for school, participant_id in participants:
		print(f"\nProcessing {school}/{participant_id}")
		common_args = [
			"--project_folder", str(project_folder),
			"--school", school,
			"--participant_id", participant_id,
		]
		for model in ("body", "full_body", "hand"):
			command = [
				sys.executable, str(inference_script), *common_args,
				"--model", model,
				"--batch_size", str(args.batch_size),
			]
			if args.force:
				command.append("--force")
			run_command(command, args.dry_run)
		try:
			merge_command = [
				sys.executable, str(merge_script), *common_args,
				"--visualization_frames", str(args.visualization_frames),
			]
			if args.force:
				merge_command.append("--force")
			run_command(merge_command, args.dry_run)
		except Exception as e:
			print(f"Error merging poses for {school}/{participant_id}: {e}")
			continue


if __name__ == "__main__":
	main(parse_args())
