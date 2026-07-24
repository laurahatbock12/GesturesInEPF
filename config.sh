#!/usr/bin/env bash
# Shared config.json reader, sourced by shell scripts (run_full_pipeline.sh,
# pose_estimation_2d/launch_process_references.sh, ...). Mirrors project_config.py's role
# for Python scripts: edit config.json, not these scripts, to change machine-specific paths.
#
# Usage:
#   source "$SCRIPT_DIR/config.sh"
#   PROJECT_FOLDER="$(config_get project_folder)"
#   MAIN_ENV="$(config_get conda_envs.main)"

CONFIG_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON_PATH="$CONFIG_SH_DIR/config.json"

config_get() {
    python3 -c "
import json
with open('$CONFIG_JSON_PATH') as f:
    value = json.load(f)
for part in '$1'.split('.'):
    value = value[part]
print(value)
"
}
