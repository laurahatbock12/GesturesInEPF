from pathlib import Path
import os
import sys
import cv2
import deeplabcut.pose_estimation_pytorch as dlc_torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.detection as detection
from tqdm import tqdm
import argparse
import pandas as pd

# Add project root and current package directory to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).parent))

# Now we can import utils from the same directory
from pose_estimation_2d.utils import get_skeleton
from project_config import MODEL_PATH, PROJECT_FOLDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 2D pose estimation on videos from multiple cameras"
    )
    parser.add_argument(
        "--project_folder",
        type=str,
        default=PROJECT_FOLDER,
        help="Path to the folder containing video file (video name will be camera_name.mp4)",
    )
    parser.add_argument(
        "--school",
        type=str,
        default="2026_06_23_KantiHeerbrugg",
        choices=["2026_06_23_KantiHeerbrugg", "2026_06_02_GR_KantiOlten", "2026_05_20_KantiMusegg", "2026_06_08_GymKirschgarten", "2026_06_22_GymImmensee"],
        help="School name (used to locate video file)",
    )
    parser.add_argument(
        "--participant_id",
        type=str,
        default="AL79NI",
        help="Participant ID (used to locate video file)",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save visualizations (optional)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing videos",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Frame index to stop processing (default: -1 for all frames)"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Frame index to start processing (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="body",
        choices=["body", "full_body", "hand"],
        help="Model to use for pose estimation",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing and overwrite existing predictions (optional)",
    )

    args = parser.parse_args()
    return args


def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        try:
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
        except Exception:
            pass

def get_hand_bodyparts(hand_label):
    finger_joints = [
        f"{finger}_{joint}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for joint in range(1, 5)
    ]
    return [f"{hand_label}_wrist_hand"] + [f"{hand_label}_{joint}" for joint in finger_joints]


HAND_BODYPARTS = get_hand_bodyparts("left") + get_hand_bodyparts("right")


def load_full_body_predictions(run):
    full_body_path = run / f"results_{run.name}_full_body.csv"
    if not full_body_path.exists():
        raise FileNotFoundError(
            f"Hand inference requires full-body predictions. Run --model full_body first: {full_body_path}"
        )
    return pd.read_csv(full_body_path, header=[0, 1, 2, 3], index_col=0)


def get_hand_bboxes(
    full_body_predictions,
    frame_idx,
    frame_width,
    frame_height,
    likelihood_threshold=0.1,
    average_likelihood_threshold=0.3,
    scale_factor=2,
):
    """Create separate left and right hand boxes from confident full-body keypoints."""
    frame_label = f"frame_{frame_idx:04d}"
    if frame_label not in full_body_predictions.index:
        return np.empty((0, 4), dtype=np.float32), []

    frame_predictions = full_body_predictions.loc[frame_label]
    bboxes = []
    hand_labels = []
    for hand_label in ("left", "right"):
        points = []
        likelihoods = []
        for bodypart in get_hand_bodyparts(hand_label):
            try:
                coordinates = frame_predictions.xs(bodypart, level="bodyparts")
                x = float(coordinates.xs("x", level="coords").iloc[0])
                y = float(coordinates.xs("y", level="coords").iloc[0])
                likelihood = float(coordinates.xs("likelihood", level="coords").iloc[0])
            except KeyError:
                continue
            if likelihood > likelihood_threshold and np.isfinite((x, y, likelihood)).all():
                points.append((x, y))
                likelihoods.append(likelihood)

        if len(points) < 2 or np.mean(likelihoods) < average_likelihood_threshold:
            continue

        points = np.asarray(points)
        min_x, min_y = np.maximum(points.min(axis=0), (0, 0))
        max_x, max_y = np.minimum(points.max(axis=0), (frame_width - 1, frame_height - 1))
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            continue

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        half_width = width * scale_factor / 2
        half_height = height * scale_factor / 2
        min_x = max(0, center_x - half_width)
        min_y = max(0, center_y - half_height)
        max_x = min(frame_width - 1, center_x + half_width)
        max_y = min(frame_height - 1, center_y + half_height)

        bboxes.append([min_x, min_y, max_x - min_x, max_y - min_y])
        hand_labels.append(hand_label)

    return np.asarray(bboxes, dtype=np.float32), hand_labels



def load_pose_runner(path_model_config, path_snapshot, max_detections=10, batch_size=32):
    pose_cfg = dlc_torch.config.read_config_as_dict(path_model_config)
    runner = dlc_torch.get_pose_inference_runner(
        pose_cfg, snapshot_path=path_snapshot, batch_size=batch_size, max_individuals=max_detections
    )
    return pose_cfg, runner


def run_pose_estimation_batch(frames_batch, context_batch, runner, mode="body"):
    """Run pose estimation on a batch of frames."""
    # Convert BGR to RGB
    images = []
    for frame in frames_batch:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(rgb_frame)
    predictions = runner.inference(zip(images, context_batch))
    if mode == "full_body":
        body_indices = np.array(list(range(17)) + list(range(91,133)))
        predictions = [{"bboxes": pred["bboxes"], "bodyparts": pred["bodyparts"][:, body_indices, :]} for pred in predictions]
    elif mode == "hand":
        for i, (pred, context) in enumerate(zip(predictions, context_batch)):
            merged_pose = np.full((1, len(HAND_BODYPARTS), 3), np.nan, dtype=np.float32)
            for hand_index, hand_label in enumerate(context["hand_labels"]):
                if hand_index >= len(pred["bodyparts"]):
                    break
                start_index = 0 if hand_label == "left" else 21
                merged_pose[0, start_index:start_index + 21] = pred["bodyparts"][hand_index]
            pred["bodyparts"] = merged_pose
            pred.pop("unique_bodyparts", None)
            pred["bboxes"] = context["bboxes"]
            if "hand_labels" in context:
                predictions[i]["hand_labels"] = context["hand_labels"]
    return predictions


def save_predictions(predictions, frame_indices, pose_cfg, output_dir, mode, run, max_detections=10):
    print("Saving the predictions to a CSV file")
    # Create frame labels like frame_0000, frame_0001, etc.
    frame_labels = [f"frame_{idx:04d}" for idx in frame_indices]
    bodyparts = pose_cfg["metadata"]["bodyparts"]
    unique_bodyparts = pose_cfg["metadata"]["unique_bodyparts"]
    if mode == "hand":
        bodyparts = HAND_BODYPARTS
        unique_bodyparts = []
        max_detections = 1

    df = dlc_torch.build_predictions_dataframe(
        scorer="rtmpose-body7",
        predictions={frame_label: img_predictions for frame_label, img_predictions in zip(frame_labels, predictions)},
        parameters=dlc_torch.PoseDatasetParameters(
            bodyparts=bodyparts,
            unique_bpts=unique_bodyparts,
            individuals=[f"idv_{i}" for i in range(max_detections)],
        ),
    )
    df.to_csv(output_dir / f"results_{run}_{mode}.csv")

    print(len(predictions), len(frame_indices))
    #Save bounding boxes as well
    pred = {
        "frame_label": frame_labels,
        "bboxes": [np.asarray(img_predictions.get("bboxes", [])).tolist() for img_predictions in predictions],
        "hand_labels": [img_predictions.get("hand_labels", []) for img_predictions in predictions],
    }
    df_bboxes = pd.DataFrame(pred)
    df_bboxes.to_csv(output_dir / f"bboxes_{run}_{mode}.csv", index=False)

    return df, df_bboxes


def create_visualization_video(
    frames,
    predictions,
    video_writer,
    skeleton,
    cmap_skeleton,
    marker_size=4,
    conf_threshold=0.05,
    mode=None,
):
    """Overlay pose predictions onto frames and append them to an MP4 video."""
    bbox_colors = [(0, 0, 255), (255, 0, 255), (255, 0, 0)]
    skeleton_colors = [
        tuple(int(channel * 255) for channel in color[:3][::-1])
        for color in cmap_skeleton(np.linspace(0, 1, len(skeleton)))
    ]

    for frame, image_predictions in zip(frames, predictions):
        visualization = frame.copy()
        pose = image_predictions["bodyparts"]

        for idv_pose in pose:
            skeleton_offsets = (0, 21) if mode == "hand" else (0,)
            for offset in skeleton_offsets:
                for (bpt_1, bpt_2), color in zip(skeleton, skeleton_colors):
                    point_1 = idv_pose[offset + bpt_1 - 1]
                    point_2 = idv_pose[offset + bpt_2 - 1]
                    if (
                        not np.isfinite(point_1).all()
                        or not np.isfinite(point_2).all()
                        or point_1[2] < conf_threshold
                        or point_2[2] < conf_threshold
                    ):
                        continue
                    cv2.line(
                        visualization,
                        tuple(np.rint(point_1[:2]).astype(int)),
                        tuple(np.rint(point_2[:2]).astype(int)),
                        color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            for bodypart_index, point in enumerate(idv_pose):
                if not np.isfinite(point).all() or point[2] < conf_threshold:
                    continue
                marker_color = plt.get_cmap("rainbow")(bodypart_index / max(len(idv_pose) - 1, 1))
                marker_color_bgr = tuple(int(channel * 255) for channel in marker_color[:3][::-1])
                cv2.circle(
                    visualization,
                    tuple(np.rint(point[:2]).astype(int)),
                    marker_size,
                    marker_color_bgr,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        bboxes = image_predictions.get("bboxes", [])
        hand_labels = image_predictions.get("hand_labels", [])
        for index, (x, y, width, height) in enumerate(bboxes):
            if mode == "hand" and index < len(hand_labels):
                color = (0, 0, 255) if "left" in hand_labels[index].lower() else (255, 0, 0)
            else:
                color = bbox_colors[index % len(bbox_colors)]
            cv2.rectangle(
                visualization,
                (round(x), round(y)),
                (round(x + width), round(y + height)),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        video_writer.write(visualization)


def main(args):
    save_vis = args.save_vis
    batch_size = args.batch_size
    project_folder = args.project_folder
    school_name = args.school
    participant_id = args.participant_id
    start_frame = args.start_frame
    end_frame = args.end_frame
    mode = args.model
    force = args.force

    video_folder = Path(project_folder) / school_name / participant_id
    runs = [i for i in video_folder.iterdir() if i.is_dir() and i.name.startswith(f"recording_") and i.name.endswith(".mp4")]
    for run in runs:
        try:
            output_dir = run
            if os.path.exists(output_dir / f"results_{run.name}_{mode}.csv") and not force:
                print(f"Predictions for {run.name} already exist at {output_dir / f'results_{run.name}_{mode}.csv'}. Skipping processing for this camera.")
                continue

            check_cuda()

            if mode == "body":
                model_path = Path(MODEL_PATH) / "rtm_body"
                path_model_config = model_path / "rtmpose-x_simcc-body7_pytorch_config.yaml"
                path_snapshot = model_path / "rtmpose-x_simcc-body7.pt"
            elif mode == "full_body":
                model_path = Path(MODEL_PATH) / "rtm_full_body"
                path_model_config = model_path / "rtmpose-x_simcc-coco-wholebody_pt-body7_pytorch_config.yaml"
                path_snapshot = model_path / "rtmpose-x_simcc-coco-wholebody.pt"
            elif mode == "hand":
                model_path = Path(MODEL_PATH)
                path_model_config = model_path / "rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_config.yaml"
                path_snapshot = model_path / "rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco.pt"
            else:
                raise NotImplementedError(f"Model {mode} not implemented")

            cmap_skeleton = plt.get_cmap("rainbow_r")

            _, skeleton = get_skeleton(mode=mode)

            full_body_predictions = load_full_body_predictions(run) if mode == "hand" else None

            video_path = [i for i in run.iterdir() if i.is_file() and i.suffix in [".mp4", ".avi"] and run.name not in i.name]
            if not video_path:
                raise FileNotFoundError(f"No video file found in: {run}")
            video_path = video_path[0]  # Take the first video file found

            print(f"Reading video from: {video_path}")

            # Open video with optimizations
            cap = cv2.VideoCapture(str(video_path))
            # Try to enable hardware acceleration
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            # Increase buffer size for faster reading
            cap.set(cv2.CAP_PROP_BUFFERSIZE, batch_size * 2)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"Total frames in video: {total_frames}")

            # Set end_frame to total_frames if not specified
            if end_frame == -1 or end_frame > total_frames:
                end_frame = total_frames

            # Validate frame range
            if start_frame < 0:
                start_frame = 0
            if start_frame >= end_frame:
                raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")

            frames_to_process = end_frame - start_frame
            print(f"Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")

            # Skip to start_frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                print(f"Skipping to frame {start_frame}")

            try:
                if torch.cuda.is_available():
                    _ = torch.zeros(1).to("cuda")
                    device = "cuda"
                    print("Using CUDA (GPU)")
                else:
                    device = "cpu"
                    print("CUDA not available, using CPU")
            except RuntimeError as e:
                print(f"CUDA error detected: {e}")
                print("Falling back to CPU")
                device = "cpu"


            max_detections = 2 if mode == "hand" else 1
            pose_cfg, runner = load_pose_runner(
                path_model_config,
                path_snapshot,
                max_detections=max_detections,
                batch_size=batch_size,
            )

            # Process video in batches
            all_predictions = []
            all_frame_indices = []
            any_detections_found = False
            visualization_writer = None
            visualization_path = output_dir / f"visualization_{run.name}_{mode}.mp4"

            print(f"\nProcessing video in batches of {batch_size}...")
            frame_idx = start_frame

            with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
                while True:
                    # Read batch of frames
                    frames_batch = []
                    frame_indices_batch = []

                    for _ in range(batch_size):
                        # Stop if we've reached end_frame
                        if frame_idx >= end_frame:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break
                        # # Increase brightness and contrast by 20%
                        # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
                        frames_batch.append(frame)
                        frame_indices_batch.append(frame_idx)
                        frame_idx += 1

                    # Exit if no frames read
                    if len(frames_batch) == 0:
                        break

                    if mode == "hand":
                        context_batch = []
                        for frame, current_frame_idx in zip(frames_batch, frame_indices_batch):
                            height, width = frame.shape[:2]
                            bboxes, hand_labels = get_hand_bboxes(
                                full_body_predictions,
                                current_frame_idx,
                                width,
                                height,
                            )
                            context_batch.append({"bboxes": bboxes, "hand_labels": hand_labels})
                    else:
                        context_batch = [
                            {"bboxes": np.array([[0, 0, frame.shape[1], frame.shape[0]]], dtype=np.float32)}
                            for frame in frames_batch
                        ]

                    # Check if any detections were found in this batch
                    for context in context_batch:
                        if len(context["bboxes"]) > 0:
                            any_detections_found = True
                            break

                    predictions_batch = run_pose_estimation_batch(frames_batch, context_batch, runner, mode=mode)
                    all_predictions.extend(predictions_batch)
                    all_frame_indices.extend(frame_indices_batch)

                    if save_vis:
                        if visualization_writer is None:
                            height, width = frames_batch[0].shape[:2]
                            visualization_writer = cv2.VideoWriter(
                                str(visualization_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                frame_rate,
                                (width, height),
                            )
                            if not visualization_writer.isOpened():
                                raise RuntimeError(f"Could not create visualization video: {visualization_path}")
                        create_visualization_video(
                            frames_batch,
                            predictions_batch,
                            visualization_writer,
                            skeleton,
                            cmap_skeleton,
                            mode=mode,
                        )
                    pbar.update(len(frames_batch))

            cap.release()
            if visualization_writer is not None:
                visualization_writer.release()
                print(f"Saved visualization video to: {visualization_path}")

            # Check if any detections were found
            if not any_detections_found:
                print(f"\nNo detections found in any frame. Exiting without saving predictions.")
                print(f"Processed {len(all_predictions)} frames but found no objects to track.")
                return

            print(f"\nSuccessfully processed {len(all_predictions)} frames")

            # Save predictions
            save_predictions(
                all_predictions,
                all_frame_indices,
                pose_cfg,
                output_dir,
                mode,
                run.name,
                max_detections=max_detections,
            )
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            continue

if __name__ == "__main__":
    args = parse_args()
    main(args)
