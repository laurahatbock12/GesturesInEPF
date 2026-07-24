SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
REFERENCE_FOLDER="$(config_get reference_folder)"

python process_reference.py --save_vis \
     --mode body \
     --process_reference_videos \
     --project_folder "$REFERENCE_FOLDER"

python process_reference.py --save_vis \
     --mode full_body \
     --process_reference_videos \
     --project_folder "$REFERENCE_FOLDER"

python process_reference.py --save_vis \
     --mode hands \
     --process_reference_videos \
     --project_folder "$REFERENCE_FOLDER"

python merge_poses.py \
     --process_reference_videos \
     --project_folder "$REFERENCE_FOLDER" \
     --visualization_frames -1
