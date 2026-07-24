# EPF Gesture Analysis

Pipeline for the **Embodied Productive Failure (EPF)** study: it estimates 2D and 3D
body/hand pose from classroom recordings of students performing gesture tasks, lifts the
2D pose into metric 3D using monocular depth, fits a fixed-bone-length skeleton to
stabilize the 3D pose, extracts kinematic (movement) features from the poses, and
compares participant gestures against a library of reference gesture clips (clustering,
t-SNE/UMAP embeddings, DTW template matching, KNN neighbor inspection). A small Flask
dashboard tracks processing progress across the whole dataset, and a separate script
renders "pose history" artwork frames for presentations/figures.

The pipeline works over two roughly-independent video sets:

- **Participant recordings** — one folder per school, containing one folder per
  participant, containing one folder per recording (task performance).
- **Reference clips** — a library of short "relevant" / "irrelevant" gesture clips used
  as templates to compare participant motion against.

Everything downstream (merging, 3D lifting, skeletal fitting, feature extraction,
clustering, DTW, KNN) runs identically over both sets — pass `--process_reference_videos`
(pose/depth/3D scripts) or point `--reference-folder` (analysis scripts) at the reference
library instead of a participant `--project_folder`.

---

## 1. Repository layout

```
EPF_Gesture_Analysis/
├── config.json                Single source of truth for every machine-specific path (see below)
├── project_config.py          Python loader for config.json, imported by every script
├── config.sh                  Bash loader for config.json, sourced by every shell script
├── pose_estimation_2d/       2D body/hand keypoint detection + merging (DeepLabCut/RTMPose)
├── pose_estimation_3d/       Monocular depth, 2D→3D lifting, rigid-skeleton fitting (Depth Anything 3)
├── analysis_2d/              Kinematic features + clustering/DTW/KNN analysis on 2D pose
├── analysis_3d/              Same analysis pipeline, on lifted 3D pose (mirrors analysis_2d)
├── visualization/            Standalone "pose history" artwork renderer
├── progress_monitor/         Flask dashboard: per-recording pipeline-stage completion
├── graphify-out/             Generated knowledge graph of this codebase (see CLAUDE.md)
└── CLAUDE.md                 Project instructions for Claude Code / graphify usage
```

### Configuration (`config.json`)

Every machine-specific path used across the pipeline -- the participant/reference data
folders, the RTMPose checkpoint directory, and the two conda environment paths -- lives in
one place: `config.json` at the repo root.

```json
{
    "project_folder": "/media1/data/andy/laura_disk/EmbodiedProductiveFailure",
    "reference_folder": "/media1/data/andy/laura_disk/EPF_PL_DA",
    "model_path": "/home/andy/Documents/FingerTap/models",
    "conda_envs": {
        "main": "/home/andy/anaconda3/envs/deeplabcut",
        "da3": "/home/andy/anaconda3/envs/da3"
    },
    "visualization_output_dir": "visualization/output"
}
```

Edit this file to point at your own data disk, RTMPose checkpoints, and conda
environments. Every script reads its path defaults from here (still overridable per-run
with `--project_folder`, `--reference-folder`, etc.):

- **Python scripts** (`pose_estimation_2d/*`, `pose_estimation_3d/*`, `analysis_2d/*`,
  `analysis_3d/*`, `progress_monitor/*`, `visualization/*`) import the values from
  `project_config.py` (repo root), which just parses `config.json`.
- **Shell scripts** (`run_full_pipeline.sh`, `pose_estimation_2d/launch_process_references.sh`)
  `source` the repo-root `config.sh` and call `config_get <key>` (e.g.
  `config_get conda_envs.main`, `config_get project_folder`).

Each of `pose_estimation_2d`, `pose_estimation_3d`, `analysis_2d`, `analysis_3d` is a
self-contained pipeline stage; scripts within a folder import from each other (and, for
the 3D/analysis code, from `pose_estimation_2d` for shared constants like the skeleton
and body-part list). None of the folders is a Python package — scripts add the repo root
to `sys.path` themselves, so everything is run as a plain script from anywhere, e.g.:

```bash
python pose_estimation_2d/run_2d_pose_video.py --school ... --participant_id ...
```

### Data folder layout (external, not part of this repo)

The video data lives outside the repo, on a separate data disk. Two independent trees:

**Participant recordings** (`--project_folder`, e.g.
`/media1/data/andy/laura_disk/EmbodiedProductiveFailure`):

```
<project_folder>/
└── <school>/                              e.g. 2026_06_23_KantiHeerbrugg
    └── <participant_id>/                  e.g. AL79NI
        └── recording_<participant>_<task>__<n>.mp4/   a DIRECTORY, despite the .mp4 suffix
            ├── <camera_name>.mp4                      source video
            ├── results_<run>_body.csv                 2D body pose_estimation_2d output
            ├── results_<run>_full_body.csv             2D full-body (incl. hands) output
            ├── results_<run>_hand.csv                  2D hand-only output
            ├── bboxes_<run>_<mode>.csv                  detection bounding boxes
            ├── visualization_<run>_<mode>.mp4          optional overlay video (--save_vis)
            ├── results_<run>_merged.csv                merge_poses.py output
            ├── visualization_<run>_merged.mp4
            ├── depth_<run>.npz                          estimate_monocular_depth.py output
            ├── visualization_<run>_depth.mp4
            ├── results_<run>_3d.csv                     estimate_3d_pose.py output
            ├── results_<run>_3d_meta.json
            ├── visualization_<run>_3d.mp4
            ├── results_<run>_3d_fitted.csv              fit_skeletal_model.py output
            ├── results_<run>_3d_fitted_meta.json
            └── visualization_<run>_3d_fitted.mp4
```

Five known schools (also the `--school` choices baked into the scripts):
`2026_06_23_KantiHeerbrugg`, `2026_06_02_GR_KantiOlten`, `2026_05_20_KantiMusegg`,
`2026_06_08_GymKirschgarten`, `2026_06_22_GymImmensee`.

**Reference clip library** (`--reference-folder` / `--reference_project_folder`, e.g.
`/media1/data/andy/laura_disk/EPF_PL_DA`), processed with `--process_reference_videos`:

```
<reference_project_folder>/
└── EPF_PL_<category>DA/                   folder name contains "irrelevant" or not -> category
    ├── <clip_name>.mp4                    source reference video
    └── pose_estimation/
        ├── body_predictions/results_<clip>.mp4_body.csv
        ├── full_body_predictions/results_<clip>.mp4_full_body.csv
        ├── hand_predictions/results_<clip>.mp4_hand.csv
        ├── merged_predictions/results_<clip>.mp4_merged.csv (+ visualization video)
        ├── depth_predictions/depth_<clip>.mp4.npz (+ visualization video)
        ├── pose_3d_predictions/results_<clip>.mp4_3d.csv (+ meta json, visualization video)
        └── skeletal_fit_predictions/results_<clip>.mp4_3d_fitted.csv (+ meta json, visualization video)
```

All of the above (videos, per-run CSV/NPZ/MP4 outputs) are gitignored (`*.mp4`, `*/output`,
`*/cache`) — this repository holds only code; the data tree lives on the data disk.

---

## 2. Requirements & installation

There is no `requirements.txt`/`environment.yml` in the repo; the project runs across
**two separate conda environments** because of conflicting dependency pins:

### 2.1 Main environment ("deeplabcut")

Used for `pose_estimation_2d/*`, `analysis_2d/*`, `analysis_3d/*`, `visualization/*`,
and `progress_monitor/*`. Needs, at minimum:

- Python 3.10+
- [`deeplabcut`](https://github.com/DeepLabCut/DeepLabCut) with the PyTorch engine
  (`deeplabcut.pose_estimation_pytorch`) — RTMPose body/hand keypoint inference
- `torch`, `torchvision` (CUDA build recommended — inference falls back to CPU but is
  much slower)
- `opencv-python` (`cv2`)
- `numpy`, `pandas`
- `matplotlib`
- `tqdm`
- `scikit-learn` (PCA, t-SNE, StandardScaler, NearestNeighbors)
- `umap-learn` (`import umap`)
- `numba` (JIT-compiled DTW cost function)
- `scipy` (`ConvexHull`)
- `zstandard` (depth `.npz` compression)
- `flask` (progress dashboard)

Install (adjust for your CUDA version):

```bash
conda create -n deeplabcut python=3.10
conda activate deeplabcut
pip install "deeplabcut[pytorch]" torch torchvision opencv-python pandas matplotlib \
            tqdm scikit-learn umap-learn numba scipy zstandard flask
```

RTMPose checkpoints are loaded from `model_path` in `config.json` (see
[Configuration](#configuration-configjson) above) with this layout — obtain/train these
checkpoints separately (not included in this repo) and update `model_path` if needed:

```
<MODEL_PATH>/
├── rtm_body/rtmpose-x_simcc-body7_pytorch_config.yaml + rtmpose-x_simcc-body7.pt
├── rtm_full_body/rtmpose-x_simcc-coco-wholebody_pt-body7_pytorch_config.yaml + rtmpose-x_simcc-coco-wholebody.pt
└── rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_config.yaml + rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco.pt
```

### 2.2 3D depth environment ("da3")

Used only for `pose_estimation_3d/estimate_monocular_depth.py` (and the
`run_depth_all.py` wrapper). Isolated in its own environment because
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) pins specific
`torch`/`torchvision`/`xformers` versions that conflict with the main environment.

Install with the provided script (creates conda env `da3`, clones/installs DA3, and
pre-downloads the default checkpoint):

```bash
bash pose_estimation_3d/install_depth_anything_3.sh [optional_clone_dir]
```

This pins `torch==2.5.1`, `torchvision==0.20.1`, `xformers==0.0.29.post1` (matched to a
CUDA 12.4 driver — adjust the pins in the script for a different driver/CUDA version),
plus `numpy<2` and `huggingface_hub` for downloading the `depth-anything/DA3METRIC-LARGE`
checkpoint. `pose_estimation_3d/estimate_3d_pose.py` and `fit_skeletal_model.py` only need
the *main* environment (they read depth already computed by the `da3` env).

> Note: because two environments are used, avoid `conda activate` inside scripts — call
> the environment's absolute `bin/python`/`bin/pip` instead (see
> `install_depth_anything_3.sh` for the pattern), since some shells put another
> environment ahead on `PATH`.

### 2.3 Progress dashboard

`progress_monitor/app.py` only needs `flask` (part of the main environment above).
`--project_folder` defaults to `project_folder` in `config.json`, so it can usually be
omitted:

```bash
python progress_monitor/app.py --port 5000
# or, to override config.json for one run:
python progress_monitor/app.py --project_folder /path/to/EmbodiedProductiveFailure --port 5000
```

Then open `http://127.0.0.1:5000`.

---

## 3. Pipeline stages

### 3.1 `pose_estimation_2d/` — 2D keypoint detection

| Script | Purpose |
|---|---|
| `run_2d_pose_video.py` | Runs RTMPose inference on every `recording_*.mp4` folder for one participant. `--model {body,full_body,hand}` selects the checkpoint; hand mode requires `full_body` results first (crops hand boxes from wholebody keypoints). |
| `process_reference.py` | Same as above, but for the reference clip library (`--process_reference_videos`), writing into `pose_estimation/<mode>_predictions/`. |
| `merge_poses.py` | Combines `body` + `full_body` + `hand` CSVs per recording into one 59-keypoint skeleton (17 body + 21+21 hand landmarks), preferring the more reliable source per body part, with temporal median smoothing (`--smoothing_window`, default 5). Also renders an overlay video. |
| `run_2d_all.py` | Batch driver: runs body → full_body → hand → merge for every (school, participant) found under `--project_folder` (or a filtered subset via `--schools`/`--participant_ids`). |
| `launch_process_references.sh` | Example shell invocation of `process_reference.py` (body/full_body/hand) + `merge_poses.py --process_reference_videos` for the reference library. |
| `utils.py` | `get_skeleton(mode)` — the fixed keypoint connectivity ("bones") for `body` (17 pts), `full_body` (59 pts, body+hands), and `hand` (21 pts) skeletons. Shared by every visualization routine in the repo. |

**Output format:** DeepLabCut-style CSV with a 4-row MultiIndex header
(`scorer, individuals, bodyparts, coords`) and one row per `frame_XXXX` label, coordinates
`x, y, likelihood` in pixels. `merge_poses.py` writes the merged 59-keypoint version under
scorer `merged-body-hand`.

### 3.2 `pose_estimation_3d/` — monocular 3D lifting

| Script | Purpose |
|---|---|
| `estimate_monocular_depth.py` | Runs Depth Anything 3 (`--model_name`, default `depth-anything/DA3METRIC-LARGE`) per-frame on the same videos, saving depth (+ optional confidence/sky mask) to a compressed `.npz`. Requires the **da3** environment. |
| `depth_io.py` | Lossless, ~2.4x-smaller-than-`np.savez_compressed` save/load for depth arrays (zstandard + temporal delta-encoding of the float16 depth). Transparently reads old plain `.npz` files too. |
| `estimate_3d_pose.py` | Lifts `merge_poses.py`'s 2D keypoints into metric 3D using the depth maps: samples depth at each keypoint, calibrates a single global scale factor per video from shoulder width (~0.40 m assumed biacromial width; hand length was tried and rejected as a scale reference — see the module docstring), backprojects with an assumed pinhole camera (`fx = fy = frame width`), fixes hand-depth outliers, and applies heavy Z-only median smoothing. |
| `fit_skeletal_model.py` | Fits a fixed-bone-length skeleton to the noisy per-frame 3D pose from `estimate_3d_pose.py` via Position-Based-Dynamics (PBD) distance-constraint relaxation — reference bone length = per-bone median length across the whole clip; each frame is solved independently (vectorized across frames), then median-smoothed. Prints before/after bone-length variation reports. |
| `run_depth_all.py`, `run_3d_pose_all.py`, `run_skeletal_fit_all.py` | Batch drivers over every participant + the reference library for the corresponding stage above. |
| `install_depth_anything_3.sh` | Sets up the isolated `da3` conda environment (see §2.2). |

**Output format:** `depth_<run>.npz` (custom zstandard container, see `depth_io.py`);
`results_<run>_3d.csv` / `results_<run>_3d_fitted.csv` (same DLC-style CSV shape as 2D, but
coordinates `x, y, z, likelihood` in meters, scorer `lifted-3d` / `skeletal-fit`); a
`_meta.json` sidecar per run recording the calibration (depth scale, assumed focal length,
number of calibration samples, PBD iterations/stiffness, per-bone reference lengths, etc.).

### 3.3 `analysis_2d/` and `analysis_3d/` — kinematics & gesture matching

Mirror pipelines — `analysis_3d` is the identical pipeline over lifted-3D pose (in a
per-frame **body-centered** coordinate frame: origin at the shoulder midpoint, so
translation/rotation relative to the camera is cancelled out) instead of 2D pixels. All
scripts below exist in both folders with the same CLI; only `analysis_2d` is described.

| Script | Purpose |
|---|---|
| `dataloader.py` | Loads every `results_*_merged.csv` (participants) / `results_*_3d.csv` (3D) into a `Clip` (keypoints + fps + participant/school/category metadata), parsed once and cached to `.npz`+`.json` under `cache/` (hashed by source path) for fast reload. |
| `features.py` | Computes 20 aggregate kinematic features per window/clip (`FEATURE_NAMES`) — wrist distance, left/right keypoint distance, hand speed/acceleration (x,y), hand opening, index-thumb pinch distance, thumb-pinky spread, vertical hand position — each as a mean+std pair, all normalized by the clip's own shoulder width (and by fps for speed/accel). `analysis_3d/features.py` adds `elbow_angle` and `hand_shoulder_distance` (only measurable with real depth) and z-axis speed/accel. |
| `clustering_analysis.py` | End-to-end: load clips → size a sliding window from the mean reference clip duration (50% overlap) → extract + per-clip-normalize features → project with **PCA**, **t-SNE**, **UMAP** → plot each projection (participant windows colored by school/participant, references marked as numbered stars). Exposes `prepare_analysis()` / `AnalysisResult`, reused by the two scripts below. |
| `knn_reference_clusters.py` | For each reference clip, finds the K participant windows nearest to it in t-SNE space (`sklearn.neighbors.NearestNeighbors`), plots a highlighted scatter + convex hull, and renders each neighbor window as a pose-overlaid `.mp4` clip for visual inspection. Also produces a combined overview plot of all references at once. |
| `dtw_reference_matching.py` | Alternative to the KNN approach: subsequence **Dynamic Time Warping** — scans each participant clip for the best-matching, time-warped (0.5x–2x length) window against a given reference's per-frame kinematic signal (Numba-JIT'd cost function, channel-weighted by how much that reference actually moves in each channel). Outputs a ranked overlay plot + matched video clip per top-K match. |
| `reference_timeseries.py` | Plots each reference clip's 10 underlying per-frame kinematic signals over time (the same signals aggregated into the mean/std features used elsewhere). |

**Outputs**, all under `analysis_2d/output/` (mirrored in `analysis_3d/output/`):

```
output/
├── reference_length_table.csv         reference clip durations vs. the mean (sets window size)
├── kinematic_features_raw.csv         one row per window/reference, 20 raw features + metadata
├── kinematic_features_normalized.csv  same, per-clip z-scored
├── pca.png / tsne.png / umap.png      2D projections, participants vs. reference gestures
├── dtw/<reference_clip_id>/
│   ├── matches.csv                    top-K matched participant windows, ranked by DTW cost
│   ├── rank##_overlay_<participant>_<clip>.png   z-scored signal overlay (reference vs. match)
│   └── windows/rank##_..._f#####-#####.mp4       pose-overlaid video of the matched window
└── knn/
    ├── overview_tsne.png              all references' KNN neighborhoods at once
    └── <reference_clip_id>/
        ├── neighbors.csv              K nearest participant windows (t-SNE distance + rank)
        ├── tsne_highlight.png         scatter with this reference's neighborhood highlighted
        ├── kinematic_timeseries.png / .csv   per-frame signal plot/table for this reference
        └── windows/rank##_..._f#####-#####.mp4
```

`cache/` (also under each `analysis_*` folder) holds the parsed-clip `.npz`/`.json` cache;
safe to delete to force a full reparse (or pass `--force-reload`).

### 3.4 `visualization/` — pose-history artwork

`generate_pose_art.py` is a standalone figure-generation script (not part of the batch
pipeline): given one recording's video + 2D merged CSV + 3D CSV, it selects a 10-frame
window at the clip's midpoint and renders "motion trail" artwork — older frames fade out,
keypoints colored by body part — in 2D, in 3D (transparent background), and as a 10-angle
azimuth sweep around the 3D pose, plus the extracted source video frame.

```bash
python visualization/generate_pose_art.py \
    --video <run>.mp4 \
    --pose-2d-csv results_<run>_merged.csv \
    --pose-3d-csv results_<run>_3d.csv \
    --output-dir visualization/output
```

**Outputs** (`visualization/output/` by default): `middle_source_frame.png`,
`pose_history_2d.png`, `pose_history_3d.png` (transparent PNG), `pose_3d_angle_01.png`
through `pose_3d_angle_10.png` (36°-apart azimuth sweep), plus `selection.json` /
`angle_sweep.json` / `selected_frame_indices.npy` recording exactly which frames/angles
were used.

### 3.5 `progress_monitor/` — pipeline status dashboard

A small Flask app (`app.py` + `scanner.py` + `templates/index.html` + `static/`) that
walks the participant recording tree and reports, per recording, which pipeline stages
have produced their expected output file yet: **2D Body → 2D Full-Body → 2D Hands →
Merged → Depth → 3D Pose → 3D Fitted**. The scan is cheap (a couple of `os.scandir()`
calls per recording) and runs once at startup plus on a background timer
(`--refresh_interval`, default 30s); the web UI itself always reads from an in-memory
cache so it never blocks on disk I/O.

```bash
python progress_monitor/app.py --port 5000   # --project_folder defaults to config.json
```

Open `http://127.0.0.1:5000`; `POST /api/refresh` forces an immediate rescan, `GET
/api/data` returns the current status snapshot as JSON.

---

## 4. Typical end-to-end run (one participant)

`run_full_pipeline.sh` (repo root) runs stages 1-4 below (2D pose → merge → depth → 3D
lift → skeletal fit) for both participants and the reference library in one command,
switching between the `deeplabcut` and `da3` conda environments automatically:

```bash
bash run_full_pipeline.sh --schools 2026_06_23_KantiHeerbrugg --participant-ids AL79NI
bash run_full_pipeline.sh --help   # all options (--force, --dry-run, --skip-participants, ...)
```

The equivalent stage-by-stage commands it runs, for reference (`--project_folder` /
`--reference_project_folder` are shown explicitly below but match `config.json`'s
defaults, so they can be omitted in practice):

```bash
# 1) 2D pose (body, full_body, hand) + merge, for every recording of one participant
python pose_estimation_2d/run_2d_all.py \
    --project_folder /media1/data/andy/laura_disk/EmbodiedProductiveFailure \
    --schools 2026_06_23_KantiHeerbrugg --participant_ids AL79NI

# 2) Monocular depth (requires the 'da3' conda env)
python pose_estimation_3d/run_depth_all.py \
    --project_folder /media1/data/andy/laura_disk/EmbodiedProductiveFailure \
    --schools 2026_06_23_KantiHeerbrugg --participant_ids AL79NI --skip_reference

# 3) Lift to 3D, then fit a rigid skeleton (back in the main env)
python pose_estimation_3d/run_3d_pose_all.py \
    --project_folder /media1/data/andy/laura_disk/EmbodiedProductiveFailure \
    --schools 2026_06_23_KantiHeerbrugg --participant_ids AL79NI --skip_reference
python pose_estimation_3d/run_skeletal_fit_all.py \
    --project_folder /media1/data/andy/laura_disk/EmbodiedProductiveFailure \
    --schools 2026_06_23_KantiHeerbrugg --participant_ids AL79NI --skip_reference

# 4) Process the reference clip library the same way (see launch_process_references.sh
#    for the 2D half); repeat steps 2-3 above with --process_reference_videos instead
#    of --skip_reference for the reference project folder.

# 5) Kinematic clustering + reference matching (2D and/or 3D)
python analysis_2d/clustering_analysis.py
python analysis_2d/knn_reference_clusters.py
python analysis_2d/dtw_reference_matching.py --references rDA1_DeviationDistance,iDA3
```

Every stage script skips work that already has its expected output file unless `--force`
is passed, so the whole pipeline can be safely re-run incrementally as new recordings
arrive — which is exactly what `progress_monitor` is for.

---

*Code written by Andy Bonnetto.*
