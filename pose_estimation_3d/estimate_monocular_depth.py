import os

# Must be set before `import torch` - avoids allocator fragmentation OOMs on long videos,
# matching what the upstream `da3` CLI does (depth_anything_3/cli.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_io import save_depth_npz
from project_config import PROJECT_FOLDER

try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    raise ImportError(
        "Could not import depth_anything_3. Run "
        "`bash pose_estimation_3d/install_depth_anything_3.sh` and activate the resulting "
        "'da3' conda environment before running this script."
    ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 monocular depth estimation on the same videos used "
        "for 2D pose estimation (participant recordings or reference videos)."
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
        "--process_reference_videos",
        action="store_true",
        help="Process the reference videos in the project_folder instead of participant recordings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="DA3 checkpoint: HuggingFace repo id or local directory. Monocular models "
        "(DA3METRIC-LARGE, DA3MONO-LARGE) treat every frame independently, which is what "
        "this script assumes. Any-view/nested models (da3-*, DA3NESTED-*) instead jointly "
        "reconstruct all frames passed in one batch as multiple views of the same scene.",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="Resolution DA3 resizes frames to before inference (see DA3 docs/API.md).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Frames per inference() call.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Frame index to stop processing (default: -1 for all frames)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Only process every Nth frame (default: 1, i.e. every frame).",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save a colorized depth visualization video alongside the raw predictions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing and overwrite existing predictions.",
    )
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA not available, using CPU (this will be very slow).")
    return torch.device("cpu")


def is_metric_only_model(model_name):
    """DA3METRIC-LARGE outputs relative depth that needs manual scaling; DA3NESTED-* is
    already metric out of the box (see DA3 README FAQ)."""
    short_name = model_name.rstrip("/").split("/")[-1].lower()
    return "metric" in short_name and "nested" not in short_name


def uses_cross_view_attention(model_name):
    """Any-view / nested DA3 models jointly reconstruct every frame passed to a single
    inference() call as multiple views of one static scene, instead of treating each frame
    independently (see `alt_start` in the model configs)."""
    short_name = model_name.rstrip("/").split("/")[-1].lower()
    return not ("mono" in short_name or ("metric" in short_name and "nested" not in short_name))


def to_metric_depth(depth, intrinsics):
    """metric_depth = focal * net_output / 300, per the DA3METRIC-LARGE FAQ entry."""
    focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
    return depth * (focal[:, None, None] / 300.0)


def colorize_depth(depth, cmap_name="turbo"):
    depth = depth.astype(np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    hi = max(hi, lo + 1e-6)
    normalized = np.clip((depth - lo) / (hi - lo), 0, 1)
    colored = (plt.get_cmap(cmap_name)(normalized)[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def load_model(model_name, device):
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()
    return model


def find_runs(project_folder, school, participant_id, process_reference_videos):
    if process_reference_videos:
        video_folders = [i for i in Path(project_folder).iterdir() if i.is_dir() and i.name.startswith("EPF_PL_")]
        runs = sorted(i for folder in video_folders for i in folder.iterdir() if i.name.endswith(".mp4"))
        if not runs:
            raise FileNotFoundError(f"No reference videos found in: {video_folders}")
    else:
        video_folder = Path(project_folder) / school / participant_id
        runs = sorted(
            i for i in video_folder.iterdir() if i.is_dir() and i.name.startswith("recording_") and i.name.endswith(".mp4")
        )
        if not runs:
            raise FileNotFoundError(f"No recording folders found in: {video_folder}")
    return runs


def resolve_run_io(run, process_reference_videos):
    if process_reference_videos:
        output_dir = run.parent / "pose_estimation" / "depth_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = run
    else:
        output_dir = run
        candidates = [
            i for i in run.iterdir() if i.is_file() and i.suffix.lower() in (".mp4", ".avi") and run.name not in i.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No video file found in: {run}")
        video_path = candidates[0]
    return video_path, output_dir


def run_depth_estimation(video_path, output_dir, run_name, model, model_name, args):
    output_npz = output_dir / f"depth_{run_name}.npz"
    output_vis = output_dir / f"visualization_{run_name}_depth.mp4"
    if output_npz.exists() and not args.force:
        print(f"Depth predictions already exist for {run_name} at {output_npz}. Skipping.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_frame = max(0, args.start_frame)
    end_frame = total_frames if args.end_frame == -1 else min(args.end_frame, total_frames)
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    metric_only = is_metric_only_model(model_name)
    already_metric = "nested" in model_name.rstrip("/").split("/")[-1].lower()

    all_depth, all_conf, all_sky, all_intrinsics = [], [], [], []
    processed_frame_indices = []
    frame_buffer, frame_indices_buffer = [], []
    visualization_writer = [None]  # mutable box so flush_batch can set it
    metric_conversion_applied = [False]
    warned_no_intrinsics = [False]

    def flush_batch():
        if not frame_buffer:
            return
        rgb_batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_buffer]
        prediction = model.inference(
            image=rgb_batch,
            process_res=args.process_res,
            export_dir=None,
        )
        depth = prediction.depth.astype(np.float32)
        if metric_only:
            if prediction.intrinsics is not None:
                depth = to_metric_depth(depth, prediction.intrinsics)
                metric_conversion_applied[0] = True
            elif not warned_no_intrinsics[0]:
                # DA3METRIC-LARGE does not predict its own camera intrinsics (no "Pose Est."
                # capability), so the `metric_depth = focal * net_output / 300` conversion from
                # the DA3 FAQ has nothing to scale with here. Without known camera intrinsics as
                # input, the saved depth is left as relative (uncalibrated) depth.
                print(
                    f"NOTE: {model_name} did not return camera intrinsics, so depth for "
                    f"{run_name} is saved as RELATIVE (non-metric) depth. Use "
                    "depth-anything/DA3NESTED-GIANT-LARGE-1.1 (with --batch_size 1 for true "
                    "per-frame independence) if real-world metric scale is required."
                )
                warned_no_intrinsics[0] = True

        all_depth.append(depth.astype(np.float16))
        if prediction.conf is not None:
            all_conf.append(prediction.conf.astype(np.float16))
        if prediction.sky is not None:
            all_sky.append(prediction.sky.astype(bool))
        if prediction.intrinsics is not None:
            all_intrinsics.append(prediction.intrinsics.astype(np.float32))
        processed_frame_indices.extend(frame_indices_buffer)

        if args.save_vis:
            if visualization_writer[0] is None:
                height, width = depth.shape[1:]
                visualization_writer[0] = cv2.VideoWriter(
                    str(output_vis), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height)
                )
                if not visualization_writer[0].isOpened():
                    raise RuntimeError(f"Could not create visualization video: {output_vis}")
            for frame_depth in depth:
                visualization_writer[0].write(colorize_depth(frame_depth))

        frame_buffer.clear()
        frame_indices_buffer.clear()

    frame_idx = start_frame
    frames_to_process = end_frame - start_frame
    with tqdm(total=frames_to_process, desc=f"Depth: {run_name}") as pbar:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % args.frame_stride == 0:
                frame_buffer.append(frame)
                frame_indices_buffer.append(frame_idx)
                if len(frame_buffer) >= args.batch_size:
                    flush_batch()
            frame_idx += 1
            pbar.update(1)
        flush_batch()

    cap.release()
    if visualization_writer[0] is not None:
        visualization_writer[0].release()
        print(f"Saved depth visualization to: {output_vis}")

    if not processed_frame_indices:
        print(f"No frames processed for {run_name}.")
        return

    save_kwargs = {
        "frame_indices": np.array(processed_frame_indices, dtype=np.int32),
        "model_name": model_name,
        "is_metric": already_metric or metric_conversion_applied[0],
        "process_res": args.process_res,
    }
    if all_conf:
        save_kwargs["conf"] = np.concatenate(all_conf, axis=0)
    if all_intrinsics:
        save_kwargs["intrinsics"] = np.concatenate(all_intrinsics, axis=0)

    save_depth_npz(
        output_npz,
        depth=np.concatenate(all_depth, axis=0),
        sky=np.concatenate(all_sky, axis=0) if all_sky else None,
        **save_kwargs,
    )
    print(f"Saved depth predictions to: {output_npz}")


def main(args):
    device = get_device()

    if uses_cross_view_attention(args.model_name) and args.batch_size > 4:
        print(
            f"WARNING: '{args.model_name}' uses cross-view attention, so every frame in a "
            f"batch of {args.batch_size} will be jointly reconstructed as multiple views of "
            "one static scene rather than processed independently. Consider a smaller "
            "--batch_size, or a monocular model (DA3METRIC-LARGE / DA3MONO-LARGE) for "
            "true per-frame depth."
        )

    print(f"Loading DA3 model: {args.model_name}")
    model = load_model(args.model_name, device)

    runs = find_runs(args.project_folder, args.school, args.participant_id, args.process_reference_videos)
    for run in runs:
        try:
            video_path, output_dir = resolve_run_io(run, args.process_reference_videos)
            print(f"\nProcessing {run.name}")
            print(f"Reading video from: {video_path}")
            run_depth_estimation(video_path, output_dir, run.name, model, args.model_name, args)
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
