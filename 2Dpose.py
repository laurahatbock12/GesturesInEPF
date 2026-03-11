"""
MMPose Wholebody Pose Estimation on Video
==========================================
Detects people and estimates wholebody keypoints (body, hands, face, feet)
on every frame of a video. Outputs an annotated video and a JSON/CSV data file.

Uses MMPose's PoseInferencer which handles detection + pose internally,
avoiding registry conflicts between mmdet and mmpose.

Usage:
    1. Set VIDEO_PATH to your video file (e.g. on external drive)
    2. Set OUTPUT_DIR to where you want results saved
    3. Set PROCESS_EVERY_N_FRAMES to control speed vs. detail tradeoff
    4. Run: python wholebody_pose.py

Speed guide (PROCESS_EVERY_N_FRAMES):
    1  = every frame        — slowest, most detail  (~2-5s/frame)
    3  = every 3rd frame    — 3x faster, smooth enough for most cases
    5  = every 5th frame    — 5x faster, good for slow/moderate movement
    10 = every 10th frame   — 10x faster, best for testing or slow movement
"""

import os
import cv2
import json
import csv
import time
from pathlib import Path

# ─────────────────────────────────────────
# ✏️  CONFIGURE THESE PATHS
# ─────────────────────────────────────────
VIDEO_PATH = r"D:\ExpertNovice\MMpose\ExNo_PL_03_Explain_Test.mp4"   # ← Change to your video path
OUTPUT_DIR = r"D:\ExpertNovice\MMpose"                  # ← Change to your desired output folder

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
POSE_THRESHOLD         = 0.3   # Minimum confidence to draw a keypoint
DETECTION_THRESHOLD    = 0.3   # Minimum confidence to detect a person
DEVICE                 = "cpu" # No GPU available

# ✏️  SPEED CONTROL
# Process every Nth frame. Skipped frames copy the last known pose overlay.
#   1  = every frame (slowest, most detail)
#   3  = 3x faster — recommended starting point
#   5  = 5x faster — good for slow/moderate movement
#   10 = 10x faster — best for testing or slow movement
PROCESS_EVERY_N_FRAMES = 1


# ════════════════════════════════════════════════════════
# MAIN SCRIPT — no need to edit below this line
# ════════════════════════════════════════════════════════

def setup_output_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}")


def load_inferencer():
    """Load MMPose PoseInferencer — handles detection + pose internally."""
    from mmpose.apis import MMPoseInferencer

    print("⏳ Loading wholebody pose inferencer (detector + pose model)...")
    inferencer = MMPoseInferencer(
  
    ## For 2D pose estimation (2D Wholebody, 2D Body only)
    pose2d='human',
    det_model=r'C:\Users\labock\mmpose\demo\mmdetection_cfg\rtmdet_nano_320-8xb32_coco-person.py',
    det_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
    device=DEVICE,
    )
    print("✅ Inferencer loaded")
    return inferencer


def process_frame(inferencer, frame_rgb):
    """Run wholebody pose inference on a single frame."""
    results = next(inferencer(
        frame_rgb,
        show=False,
        draw_bbox=True,
        kpt_thr=POSE_THRESHOLD,
        return_vis=True,
    ))
    return results


def extract_keypoints(results):
    """Extract keypoint data from inferencer results."""
    persons = []
    if results and 'predictions' in results:
        for pred in results['predictions'][0]:
            keypoints = pred.get('keypoints', [])
            scores    = pred.get('keypoint_scores', [])
            person_data = {"keypoints": []}
            for (x, y), s in zip(keypoints, scores):
                person_data["keypoints"].append([float(x), float(y), float(s)])
            persons.append(person_data)
    return persons


def save_keypoints_json(all_keypoints, output_path):
    with open(output_path, "w") as f:
        json.dump(all_keypoints, f, indent=2)
    print(f"✅ Keypoints JSON saved: {output_path}")


def save_keypoints_csv(all_keypoints, output_path):
    # Find first frame with persons to build headers
    num_kpts = 0
    for frame_data in all_keypoints:
        if frame_data["persons"]:
            num_kpts = len(frame_data["persons"][0]["keypoints"])
            break

    if num_kpts == 0:
        print("⚠️  No persons detected in any frame, skipping CSV.")
        return

    headers = ["frame", "person_id"]
    for i in range(num_kpts):
        headers += [f"kpt{i}_x", f"kpt{i}_y", f"kpt{i}_score"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for frame_data in all_keypoints:
            for p_idx, person in enumerate(frame_data["persons"]):
                row = [frame_data["frame"], p_idx]
                for kpt in person["keypoints"]:
                    row += [round(kpt[0], 2), round(kpt[1], 2), round(kpt[2], 3)]
                writer.writerow(row)

    print(f"✅ Keypoints CSV saved: {output_path}")


def process_video():
    # ── Validate input ──────────────────────────────────
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        print("   Please update VIDEO_PATH at the top of this script.")
        return

    setup_output_dir(OUTPUT_DIR)

    # ── Load inferencer ─────────────────────────────────
    inferencer = load_inferencer()

    # ── Open video ──────────────────────────────────────
    cap          = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_to_process = (total_frames + PROCESS_EVERY_N_FRAMES - 1) // PROCESS_EVERY_N_FRAMES
    print(f"\n📹 Video info   : {width}x{height} @ {fps:.1f}fps | {total_frames} total frames")
    print(f"⚡ Processing   : every {PROCESS_EVERY_N_FRAMES} frame(s) → {frames_to_process} frames to process")
    print(f"⚠️  CPU mode     : expect ~2-5 seconds per processed frame\n")

    # ── Output paths ────────────────────────────────────
    video_name = Path(VIDEO_PATH).stem
    out_video  = os.path.join(OUTPUT_DIR, f"{video_name}_2dbody.mp4")
    out_json   = os.path.join(OUTPUT_DIR, f"{video_name}_keypoints_2dbody.json")
    out_csv    = os.path.join(OUTPUT_DIR, f"{video_name}_keypoints_2dbody.csv")

    # ── Video writer ─────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    # ── Process frames ───────────────────────────────────
    all_keypoints   = []
    frame_idx       = 0
    processed_count = 0
    last_annotated  = None
    last_persons    = []
    start_time      = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx     += 1
        should_process = (frame_idx % PROCESS_EVERY_N_FRAMES == 1) or (PROCESS_EVERY_N_FRAMES == 1)

        if should_process:
            processed_count += 1

            # ETA
            elapsed   = time.time() - start_time
            avg_time  = elapsed / processed_count if processed_count > 1 else 0
            remaining = avg_time * (frames_to_process - processed_count)
            eta_str   = f"{int(remaining // 60)}m {int(remaining % 60)}s" if avg_time > 0 else "calculating..."
            print(f"  Frame {frame_idx}/{total_frames} | Processed {processed_count}/{frames_to_process} | ETA: {eta_str}    ", end="\r")

            # Convert BGR→RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                results      = process_frame(inferencer, frame_rgb)
                last_persons = extract_keypoints(results)

                # Get annotated frame from inferencer visualisation
                if results and results.get('visualization') and len(results['visualization']) > 0:
                    vis            = results['visualization'][0]
                    last_annotated = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                else:
                    last_annotated = frame

            except Exception as e:
                print(f"\n⚠️  Frame {frame_idx} error: {e}")
                last_annotated = frame
                last_persons   = []

        # Write frame and record keypoints
        frame_data = {"frame": frame_idx, "persons": last_persons}
        all_keypoints.append(frame_data)
        writer.write(last_annotated if last_annotated is not None else frame)

    # ── Cleanup ──────────────────────────────────────────
    cap.release()
    writer.release()

    elapsed_total = time.time() - start_time
    print(f"\n\n✅ Done! {frame_idx} frames written | {processed_count} processed | "
          f"Total time: {int(elapsed_total // 60)}m {int(elapsed_total % 60)}s")

    save_keypoints_json(all_keypoints, out_json)
    save_keypoints_csv(all_keypoints, out_csv)

    print("\n🎉 All outputs saved successfully!")
    print(f"   📹 Video : {out_video}")
    print(f"   📄 JSON  : {out_json}")
    print(f"   📊 CSV   : {out_csv}")


if __name__ == "__main__":
    process_video()