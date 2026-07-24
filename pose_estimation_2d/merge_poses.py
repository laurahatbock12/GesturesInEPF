from pathlib import Path
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_estimation_2d.utils import get_skeleton
from project_config import PROJECT_FOLDER


BODY_BODYPARTS = [
	"nose",
	"left_eye",
	"right_eye",
	"left_ear",
	"right_ear",
	"left_shoulder",
	"right_shoulder",
	"left_elbow",
	"right_elbow",
	"left_wrist",
	"right_wrist",
	"left_hip",
	"right_hip",
	"left_knee",
	"right_knee",
	"left_ankle",
	"right_ankle",
]


def get_hand_bodyparts(hand_label):
	finger_joints = [
		f"{finger}_{joint}"
		for finger in ("thumb", "index", "middle", "ring", "pinky")
		for joint in range(1, 5)
	]
	return [f"{hand_label}_wrist_hand"] + [f"{hand_label}_{joint}" for joint in finger_joints]


HAND_BODYPARTS = get_hand_bodyparts("left") + get_hand_bodyparts("right")
MERGED_BODYPARTS = BODY_BODYPARTS + HAND_BODYPARTS
COORDINATES = ("x", "y", "likelihood")
HIDDEN_VISUALIZATION_BODYPARTS = {
	"left_hip",
	"right_hip",
	"left_knee",
	"right_knee",
	"left_ankle",
	"right_ankle",
}
HIDDEN_VISUALIZATION_INDICES = {
	MERGED_BODYPARTS.index(bodypart)
	for bodypart in HIDDEN_VISUALIZATION_BODYPARTS
}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Merge body, full-body, and hand pose-estimation results."
	)
	parser.add_argument(
		"--project_folder",
		type=Path,
		default=Path(PROJECT_FOLDER),
	)
	parser.add_argument(
		"--school",
		default="2026_06_23_KantiHeerbrugg",
	)
	parser.add_argument("--participant_id", default="AL79NI")
	parser.add_argument(
		"--confidence_threshold",
		type=float,
		default=0.1,
		help="Minimum likelihood to draw a merged keypoint in the visualization.",
	)
	parser.add_argument(
		"--smoothing_window",
		type=int,
		default=5,
		help="Odd number of centered frames used for temporal median smoothing.",
	)
	parser.add_argument(
		"--visualization_frames",
		type=int,
		default=-1,
		help="Number of merged frames to render, or -1 to render all frames.",
	)
	parser.add_argument("--force", action="store_true")
	parser.add_argument(
		"--process_reference_videos",
		action="store_true",
		help="Process the reference videos in the project_folder"
	)
	return parser.parse_args()


def load_predictions(path):
	if not path.exists():
		return None
	return pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)


def frame_number(frame_label):
	try:
		return int(str(frame_label).rsplit("_", maxsplit=1)[1])
	except (IndexError, ValueError):
		raise ValueError(f"Unexpected frame label: {frame_label}") from None


def prediction_array(predictions, frame_labels, bodyparts):
	"""Align one DLC table to frames/bodyparts and return (frames, points, x/y/confidence)."""
	if predictions is None:
		return np.full((len(frame_labels), len(bodyparts), len(COORDINATES)), np.nan, dtype=np.float32)

	# DLC scorer and individual levels are not relevant after inference; collapse them
	# to one bodypart/coordinate table before performing a single aligned reindex.
	table = predictions.T.groupby(level=["bodyparts", "coords"], sort=False).first().T
	columns = pd.MultiIndex.from_product(
		[bodyparts, COORDINATES], names=["bodyparts", "coords"]
	)
	return table.reindex(index=frame_labels, columns=columns).to_numpy(dtype=np.float32).reshape(
		len(frame_labels), len(bodyparts), len(COORDINATES)
	)


def select_first_available(*sources):
	"""Select the first source with finite x, y, and likelihood at each location."""
	selected = np.full_like(sources[0], np.nan)
	for source in reversed(sources):
		selected = np.where(np.isfinite(source).all(axis=-1, keepdims=True), source, selected)
	return selected


def merge_predictions(body_predictions, full_body_predictions, hand_predictions):
	available_indices = [
		predictions.index
		for predictions in (body_predictions, full_body_predictions, hand_predictions)
		if predictions is not None
	]
	if not available_indices:
		raise FileNotFoundError("No body, full-body, or hand result CSV was found.")
	frame_labels = sorted(set().union(*[set(index) for index in available_indices]), key=frame_number)
	print(f"Aligning {len(frame_labels)} frames from body, full-body, and hand result tables.")
	body_points = prediction_array(body_predictions, frame_labels, BODY_BODYPARTS)
	full_body_points = prediction_array(full_body_predictions, frame_labels, BODY_BODYPARTS)
	hand_points = prediction_array(hand_predictions, frame_labels, HAND_BODYPARTS)
	full_hand_points = prediction_array(full_body_predictions, frame_labels, HAND_BODYPARTS)

	merged = np.full((len(frame_labels), len(MERGED_BODYPARTS), len(COORDINATES)), np.nan, dtype=np.float32)
	merged[:, :len(BODY_BODYPARTS)] = select_first_available(body_points, full_body_points)
	merged[:, len(BODY_BODYPARTS):] = select_first_available(hand_points, full_hand_points)

	for hand_bodypart in ("left_wrist_hand", "right_wrist_hand"):
		side = hand_bodypart.removesuffix("_wrist_hand")
		target_index = len(BODY_BODYPARTS) + HAND_BODYPARTS.index(hand_bodypart)
		body_wrist_index = BODY_BODYPARTS.index(f"{side}_wrist")
		merged[:, target_index] = select_first_available(
			body_points[:, body_wrist_index:body_wrist_index + 1],
			full_hand_points[:, HAND_BODYPARTS.index(hand_bodypart):HAND_BODYPARTS.index(hand_bodypart) + 1],
			full_body_points[:, body_wrist_index:body_wrist_index + 1],
		)[:, 0]

	return frame_labels, merged


def smooth_poses(poses, window_size=5):
	"""Median-smooth valid x/y coordinates in a centered temporal window."""
	if window_size < 1 or window_size % 2 == 0:
		raise ValueError("smoothing_window must be a positive odd number.")
	if window_size == 1:
		return poses.copy()

	smoothed = poses.copy()
	half_window = window_size // 2
	for frame_index, center_pose in enumerate(poses):
		start_index = max(0, frame_index - half_window)
		end_index = min(len(poses), frame_index + half_window + 1)
		window = poses[start_index:end_index]
		valid_center = np.isfinite(center_pose).all(axis=1)
		valid_window = np.isfinite(window).all(axis=2)
		for keypoint_index in np.flatnonzero(valid_center):
			valid_points = window[valid_window[:, keypoint_index], keypoint_index, :2]
			if len(valid_points):
				smoothed[frame_index, keypoint_index, :2] = np.median(valid_points, axis=0)
	return smoothed


def save_merged_predictions(output_path, frame_labels, poses):
	columns = pd.MultiIndex.from_tuples(
		[
			("merged-body-hand", "idv_0", bodypart, coordinate)
			for bodypart in MERGED_BODYPARTS
			for coordinate in COORDINATES
		],
		names=["scorer", "individuals", "bodyparts", "coords"],
	)
	pd.DataFrame(poses.reshape(len(poses), -1), index=frame_labels, columns=columns).to_csv(output_path)


def find_video(run):
	videos = [
		path for path in run.iterdir()
		if path.is_file() and path.suffix.lower() in (".mp4", ".avi") and not path.name.startswith("visualization_")
	]
	if not videos:
		raise FileNotFoundError(f"No source video found in {run}")
	return videos[0]


def create_visualization(video_path, output_path, frame_labels, poses, confidence_threshold, max_frames=-1):
	if max_frames == 0 or max_frames < -1:
		raise ValueError("visualization_frames must be positive or -1.")
	if max_frames > 0:
		frame_labels = frame_labels[:max_frames]
		poses = poses[:max_frames]
	_, skeleton = get_skeleton(mode="full_body")
	skeleton_colors = [
		tuple(int(channel * 255) for channel in color[:3][::-1])
		for color in plt.get_cmap("rainbow_r")(np.linspace(0, 1, len(skeleton)))
	]
	capture = cv2.VideoCapture(str(video_path))
	frame_rate = capture.get(cv2.CAP_PROP_FPS) or 30.0
	writer = None
	try:
		for frame_label, pose in tqdm(zip(frame_labels, poses), total=len(poses), desc="Creating merged visualization"):
			capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number(frame_label))
			success, frame = capture.read()
			if not success:
				raise RuntimeError(f"Could not read {frame_label} from {video_path}")
			if writer is None:
				height, width = frame.shape[:2]
				writer = cv2.VideoWriter(
					str(output_path),
					cv2.VideoWriter_fourcc(*"mp4v"),
					frame_rate,
					(width, height),
				)
				if not writer.isOpened():
					raise RuntimeError(f"Could not create visualization video: {output_path}")

			keypoints = pose
			for (bodypart_1, bodypart_2), color in zip(skeleton, skeleton_colors):
				if bodypart_1 - 1 in HIDDEN_VISUALIZATION_INDICES or bodypart_2 - 1 in HIDDEN_VISUALIZATION_INDICES:
					continue
				point_1 = keypoints[bodypart_1 - 1]
				point_2 = keypoints[bodypart_2 - 1]
				if (
					not np.isfinite(point_1).all()
					or not np.isfinite(point_2).all()
					or point_1[2] < confidence_threshold
					or point_2[2] < confidence_threshold
				):
					continue
				cv2.line(
					frame,
					tuple(np.rint(point_1[:2]).astype(int)),
					tuple(np.rint(point_2[:2]).astype(int)),
					color,
					thickness=2,
					lineType=cv2.LINE_AA,
				)

			for keypoint_index, point in enumerate(keypoints):
				if keypoint_index in HIDDEN_VISUALIZATION_INDICES:
					continue
				if not np.isfinite(point).all() or point[2] < confidence_threshold:
					continue
				cv2.circle(
					frame,
					tuple(np.rint(point[:2]).astype(int)),
					4,
					(0, 255, 255),
					thickness=-1,
					lineType=cv2.LINE_AA,
				)
			writer.write(frame)
	finally:
		capture.release()
		if writer is not None:
			writer.release()


def main(args):
	if args.process_reference_videos:
		video_folders = [i for i in args.project_folder.iterdir() if i.is_dir() and i.name.startswith("EPF_PL_")]
		runs = sorted(i for folder in video_folders for i in folder.iterdir() if i.name.endswith(".mp4"))
		if not runs:
			raise FileNotFoundError(f"No recording folders found in: {video_folders}")
	else:
		participant_folder = args.project_folder / args.school / args.participant_id
		runs = sorted(
			path for path in participant_folder.iterdir()
			if path.is_dir() and path.name.startswith("recording_") and path.name.endswith(".mp4")
		)
		if not runs:
			raise FileNotFoundError(f"No recording directories found in {participant_folder}")

	for run in runs:
		if args.process_reference_videos:
			output_dir = run.parent / "pose_estimation" / "merged_predictions"
			output_dir.mkdir(parents=True, exist_ok=True)
			body_path = run.parent / "pose_estimation" / "body_predictions" / f"results_{run.name}_body.csv"
			full_body_path = run.parent / "pose_estimation" / "full_body_predictions" / f"results_{run.name}_full_body.csv"
			hand_path = run.parent / "pose_estimation" / "hand_predictions" / f"results_{run.name}_hand.csv"
			video_file = run
		else:
			output_dir = run
			body_path = run / f"results_{run.name}_body.csv"
			full_body_path = run / f"results_{run.name}_full_body.csv"
			hand_path = run / f"results_{run.name}_hand.csv"
			video_file = find_video(run)

		output_csv = output_dir / f"results_{run.name}_merged.csv"
		output_video = output_dir / f"visualization_{run.name}_merged.mp4"
		if output_csv.exists() and output_video.exists() and not args.force:
			print(f"Merged results already exist for {run.name}; skipping.")
			continue

		body_predictions = load_predictions(body_path)
		full_body_predictions = load_predictions(full_body_path)
		hand_predictions = load_predictions(hand_path)
		frame_labels, poses = merge_predictions(body_predictions, full_body_predictions, hand_predictions)
		poses = smooth_poses(poses, args.smoothing_window)
		save_merged_predictions(output_csv, frame_labels, poses)
		create_visualization(
			video_file,
			output_video,
			frame_labels,
			poses,
			args.confidence_threshold,
			args.visualization_frames,
		)
		print(f"Saved merged predictions to: {output_csv}")
		print(f"Saved merged visualization to: {output_video}")


if __name__ == "__main__":
	main(parse_args())
