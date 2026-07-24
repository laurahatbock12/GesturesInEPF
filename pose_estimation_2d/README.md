# pose_estimation_2d

Runs 2D pose estimation on per-camera recordings using DeepLabCut's PyTorch
backend with RTMPose models, then merges body / full-body(wholebody) / hand
predictions into a single skeleton per frame.

- `run_2d_pose_video.py` — run one model (`body`, `full_body`, or `hand`) on all
  recordings for a given school/participant.
- `run_2d_all.py` — loop over participants and run all three models plus the merge step.
- `process_reference.py` — same pipeline for standalone reference videos (`--process_reference_videos`).
- `merge_poses.py` — combine body/full_body/hand CSVs into one merged, smoothed skeleton + visualization.
- `utils.py` — skeleton (bodypart connectivity) definitions used for visualization.
- `launch_process_references.sh` — example end-to-end invocation of `process_reference.py` + `merge_poses.py`.

## Installation

Requires **DeepLabCut with the PyTorch backend** (DLC ≥ 3.0). This was
developed/tested against:

- Python 3.12
- `deeplabcut` 3.0.0rc13
- `torch` 2.6.0+cu124, `torchvision` 0.21.0+cu124 (CUDA GPU strongly recommended; CPU works but is slow)

```bash
conda create -n deeplabcut python=3.12 -y
conda activate deeplabcut
pip install "deeplabcut[pytorch]"   # or: pip install deeplabcut
```

`deeplabcut` pulls in `torch`, `torchvision`, `pandas`, `numpy`, `tqdm`, and
`matplotlib` as dependencies. The scripts here additionally need:

```bash
pip install opencv-python
```

Verify the PyTorch backend and GPU are wired up correctly:

```bash
python -c "import deeplabcut.pose_estimation_pytorch as dlc_torch; import torch; print(torch.cuda.is_available())"
```

## Required models

Three separate RTMPose snapshots (DeepLabCut PyTorch format: a `.pt` snapshot
+ a matching `_pytorch_config.yaml`) are required, one per `--model` mode.
They are loaded from `model_path` in `config.json` (repo root), imported via
`project_config.MODEL_PATH` in both `run_2d_pose_video.py` and `process_reference.py`:

```json
{
    "model_path": "/home/andy/Documents/FingerTap/models"
}
```

**Update `model_path` in `config.json` to point at your local copy of the models below
before running anything.**

Expected layout:

```
<MODEL_PATH>/
├── rtm_body/
│   ├── rtmpose-x_simcc-body7_pytorch_config.yaml
│   └── rtmpose-x_simcc-body7.pt
├── rtm_full_body/
│   ├── rtmpose-x_simcc-coco-wholebody_pt-body7_pytorch_config.yaml
│   └── rtmpose-x_simcc-coco-wholebody.pt
├── rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_config.yaml
└── rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco.pt
```

| `--model` | Keypoints | Snapshot | Source |
|---|---|---|---|
| `body` | 17 COCO body keypoints | `rtm_body/rtmpose-x_simcc-body7.pt` | DeepLabCut **SuperAnimal-HumanBody** zoo model (`superanimal_humanbody`, pose model `rtmpose_x`). Official, downloadable via `dlclibrary`/Hugging Face (`DeepLabCut/HumanBody`). See `dlclibrary/dlcmodelzoo/modelzoo_urls_pytorch.yaml` or use `deeplabcut.modelzoo` / `dlclibrary.download_huggingface_model("superanimal_humanbody", "rtmpose_x")`. |
| `full_body` | 133 COCO-WholeBody keypoints (body+feet+face+hands), reduced to the 17 body + 42 hand keypoints used here | `rtm_full_body/rtmpose-x_simcc-coco-wholebody.pt` | TODO: add source/download link. |
| `hand` | 21+21 hand keypoints (run on hand crops derived from `full_body` predictions) | `rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco.pt` | TODO: add source/download link. |

Notes:
- `hand` mode **requires `full_body` predictions to already exist** for that
  recording (it crops hand bounding boxes from the full_body keypoints), so
  always run `--model full_body` before `--model hand`, or just use
  `run_2d_all.py` / `launch_process_references.sh` which sequence this correctly.
- `merge_poses.py` expects `results_<run>_body.csv`, `results_<run>_full_body.csv`,
  and `results_<run>_hand.csv` to exist (any subset is fine; missing ones become NaN).

## Expected data layout

```
<project_folder>/<school>/<participant_id>/recording_*.mp4/<camera_video>.mp4
```

or, for reference videos (`--process_reference_videos`):

```
<project_folder>/EPF_PL_*/<video>.mp4
```

## Usage

```bash
# Single model, single participant
python run_2d_pose_video.py --project_folder <path> --school <school> --participant_id <id> --model body

# All models + merge, for every participant found under project_folder
python run_2d_all.py --project_folder <path>

# Reference videos
bash launch_process_references.sh
```
