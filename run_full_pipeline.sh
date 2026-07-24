#!/usr/bin/env bash
# End-to-end EPF Gesture Analysis pipeline: 2D pose estimation -> merge -> monocular
# depth -> 3D lifting -> rigid-skeleton fitting, for both participant recordings and
# the reference clip library. See README.md for what each stage produces.
#
# Uses two conda environments (absolute interpreter paths, no `conda activate`: some
# shells put another env's bin/ ahead on PATH, which silently misroutes python/pip):
#   - MAIN_ENV : 2D pose estimation, merging, 3D lifting, skeletal fitting
#   - DA3_ENV  : monocular depth estimation only (Depth Anything 3)
#
# All default paths (project/reference folders, conda envs) come from config.json at the
# repo root -- edit that file, not this script, to point at your own machine.
#
# Usage:
#   bash run_full_pipeline.sh [options]
#
# Options:
#   --project-folder DIR        Participant recordings root (default: config.json project_folder)
#   --reference-folder DIR      Reference clip library root (default: config.json reference_folder)
#   --schools "S1 S2 ..."       Restrict participant stages to these school folders
#   --participant-ids "P1 ..."  Restrict participant stages to these participant IDs
#   --skip-participants         Only process the reference clip library
#   --skip-reference            Only process participant recordings
#   --force                     Reprocess and overwrite existing outputs
#   --dry-run                   Print the commands without running them
#
# Example:
#   bash run_full_pipeline.sh --schools 2026_06_23_KantiHeerbrugg --participant-ids AL79NI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE2D_DIR="$SCRIPT_DIR/pose_estimation_2d"
POSE3D_DIR="$SCRIPT_DIR/pose_estimation_3d"

source "$SCRIPT_DIR/config.sh"

MAIN_ENV="$(config_get conda_envs.main)"
DA3_ENV="$(config_get conda_envs.da3)"
MAIN_PY="$MAIN_ENV/bin/python"
DA3_PY="$DA3_ENV/bin/python"

PROJECT_FOLDER="$(config_get project_folder)"
REFERENCE_FOLDER="$(config_get reference_folder)"
SCHOOLS=()
PARTICIPANT_IDS=()
SKIP_PARTICIPANTS=0
SKIP_REFERENCE=0
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-folder) PROJECT_FOLDER="$2"; shift 2 ;;
        --reference-folder) REFERENCE_FOLDER="$2"; shift 2 ;;
        --schools) read -r -a SCHOOLS <<< "$2"; shift 2 ;;
        --participant-ids) read -r -a PARTICIPANT_IDS <<< "$2"; shift 2 ;;
        --skip-participants) SKIP_PARTICIPANTS=1; shift ;;
        --skip-reference) SKIP_REFERENCE=1; shift ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) grep '^#' "$0" | cut -c3-; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

for env_dir in "$MAIN_ENV" "$DA3_ENV"; do
    if [[ ! -x "$env_dir/bin/python" ]]; then
        echo "Error: conda environment not found at $env_dir (see README.md section 2 to create it)." >&2
        exit 1
    fi
done

FORCE_FLAG=(); [[ $FORCE -eq 1 ]] && FORCE_FLAG=(--force)
DRY_RUN_FLAG=(); [[ $DRY_RUN -eq 1 ]] && DRY_RUN_FLAG=(--dry_run)
SCHOOLS_FLAG=(); [[ ${#SCHOOLS[@]} -gt 0 ]] && SCHOOLS_FLAG=(--schools "${SCHOOLS[@]}")
PARTICIPANT_IDS_FLAG=(); [[ ${#PARTICIPANT_IDS[@]} -gt 0 ]] && PARTICIPANT_IDS_FLAG=(--participant_ids "${PARTICIPANT_IDS[@]}")

run() {
    echo
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

echo "=== EPF Gesture Analysis: full pipeline ==="
echo "Participant project folder : $PROJECT_FOLDER"
echo "Reference project folder   : $REFERENCE_FOLDER"
echo "Skip participants          : $SKIP_PARTICIPANTS"
echo "Skip reference             : $SKIP_REFERENCE"

# ---------------------------------------------------------------------------
# Stage 1: 2D pose estimation (body, full_body, hand) + merge -- MAIN_ENV
# ---------------------------------------------------------------------------
if [[ $SKIP_PARTICIPANTS -eq 0 ]]; then
    echo
    echo "--- Stage 1a: 2D pose estimation + merge (participants) ---"
    run "$MAIN_PY" "$POSE2D_DIR/run_2d_all.py" \
        --project_folder "$PROJECT_FOLDER" \
        "${SCHOOLS_FLAG[@]}" "${PARTICIPANT_IDS_FLAG[@]}" \
        "${FORCE_FLAG[@]}" "${DRY_RUN_FLAG[@]}"
fi

if [[ $SKIP_REFERENCE -eq 0 ]]; then
    echo
    echo "--- Stage 1b: 2D pose estimation + merge (reference clips) ---"
    for model in body full_body hand; do
        run "$MAIN_PY" "$POSE2D_DIR/process_reference.py" \
            --project_folder "$REFERENCE_FOLDER" \
            --model "$model" \
            --process_reference_videos \
            "${FORCE_FLAG[@]}"
    done
    run "$MAIN_PY" "$POSE2D_DIR/merge_poses.py" \
        --project_folder "$REFERENCE_FOLDER" \
        --process_reference_videos \
        "${FORCE_FLAG[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 2: monocular depth estimation -- DA3_ENV
# ---------------------------------------------------------------------------
echo
echo "--- Stage 2: monocular depth estimation (Depth Anything 3) ---"
SKIP_STAGE_FLAGS=()
[[ $SKIP_PARTICIPANTS -eq 1 ]] && SKIP_STAGE_FLAGS+=(--skip_participants)
[[ $SKIP_REFERENCE -eq 1 ]] && SKIP_STAGE_FLAGS+=(--skip_reference)
run "$DA3_PY" "$POSE3D_DIR/run_depth_all.py" \
    --project_folder "$PROJECT_FOLDER" \
    --reference_project_folder "$REFERENCE_FOLDER" \
    "${SCHOOLS_FLAG[@]}" "${PARTICIPANT_IDS_FLAG[@]}" \
    "${SKIP_STAGE_FLAGS[@]}" "${FORCE_FLAG[@]}" "${DRY_RUN_FLAG[@]}"

# ---------------------------------------------------------------------------
# Stage 3: lift merged 2D pose + depth into 3D -- MAIN_ENV
# ---------------------------------------------------------------------------
echo
echo "--- Stage 3: 2D + depth -> 3D pose lifting ---"
run "$MAIN_PY" "$POSE3D_DIR/run_3d_pose_all.py" \
    --project_folder "$PROJECT_FOLDER" \
    --reference_project_folder "$REFERENCE_FOLDER" \
    "${SCHOOLS_FLAG[@]}" "${PARTICIPANT_IDS_FLAG[@]}" \
    "${SKIP_STAGE_FLAGS[@]}" "${FORCE_FLAG[@]}" "${DRY_RUN_FLAG[@]}"

# ---------------------------------------------------------------------------
# Stage 4: fixed-bone-length skeletal fitting -- MAIN_ENV
# ---------------------------------------------------------------------------
echo
echo "--- Stage 4: rigid-skeleton fitting ---"
run "$MAIN_PY" "$POSE3D_DIR/run_skeletal_fit_all.py" \
    --project_folder "$PROJECT_FOLDER" \
    --reference_project_folder "$REFERENCE_FOLDER" \
    "${SCHOOLS_FLAG[@]}" "${PARTICIPANT_IDS_FLAG[@]}" \
    "${SKIP_STAGE_FLAGS[@]}" "${FORCE_FLAG[@]}" "${DRY_RUN_FLAG[@]}"

echo
echo "=== Pipeline complete ==="
echo "Run analysis_2d/clustering_analysis.py or analysis_3d/clustering_analysis.py next"
echo "(see README.md section 3.3), or check progress_monitor/app.py for a live status dashboard."
