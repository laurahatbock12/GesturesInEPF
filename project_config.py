"""Single source of truth for every machine-specific path used across this repo.

Edit `config.json` (repo root) to point at your own data disk, RTMPose checkpoints, and
conda environments. Every script in `pose_estimation_2d`, `pose_estimation_3d`,
`analysis_2d`, `analysis_3d`, `progress_monitor`, and `visualization` reads its path
defaults from here instead of hardcoding them. `run_full_pipeline.sh` and other shell
scripts read the same `config.json` via `config.sh`.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"

with open(CONFIG_PATH) as _f:
    _CONFIG = json.load(_f)

PROJECT_FOLDER = _CONFIG["project_folder"]
REFERENCE_FOLDER = _CONFIG["reference_folder"]
MODEL_PATH = _CONFIG["model_path"]
MAIN_CONDA_ENV = _CONFIG["conda_envs"]["main"]
DA3_CONDA_ENV = _CONFIG["conda_envs"]["da3"]
VISUALIZATION_OUTPUT_DIR = REPO_ROOT / _CONFIG["visualization_output_dir"]
