#!/bin/bash

error() {
    echo "Usage: test_batch.sh -i <input root directory> -o <output root directory> -m <model directory>"
    echo "-c flag can be included to run on CPU"
    exit 1
}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
APP_DIR=$SCRIPT_DIR/../prostate_mri_lesion_seg_app

# Set defaults for optional arguments
INPUT_ROOT_DIR=""
OUTPUT_ROOT_DIR=$SCRIPT_DIR/../output
MODEL_DIR=$APP_DIR/models/

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
    case $1 in
        -i | --input)
            if ! has_argument $@; then
                echo "Error: No input root directory specified" && error
            fi
            INPUT_ROOT_DIR=$(extract_argument $@)
            shift
            ;;
        -o | --output)
            if ! has_argument $@; then
                echo "Error: No output root directory specified" && error
            fi
            OUTPUT_ROOT_DIR=$(extract_argument $@)
            shift
            ;;
        -m | --model)
            if ! has_argument $@; then
                echo "Error: No model directory specified" && error
            fi
            MODEL_DIR=$(extract_argument $@)
            shift
            ;;
        -c | --cpu) CPU_ARG=1 ;;
      *) echo "Invalid option: $1" >&2 && error ;;
    esac
    shift
  done
}
handle_options "$@"

# Validate input root directory
if [ -z "$INPUT_ROOT_DIR" ]; then
    echo "Error: Input root directory must be specified with -i or --input" && error
fi

if [ ! -d "$INPUT_ROOT_DIR" ]; then
    echo "Error: Input root directory does not exist: $INPUT_ROOT_DIR" && error
fi

# Prepare output root directory (clean like test_local.sh)
mkdir -p "$OUTPUT_ROOT_DIR"
rm -rf "$OUTPUT_ROOT_DIR"/*

echo "Batch processing patients under: $INPUT_ROOT_DIR"
echo "Writing per-patient outputs under: $OUTPUT_ROOT_DIR"
echo "Using model directory: $MODEL_DIR"

# Loop over patient subdirectories
for PATIENT_DIR in "$INPUT_ROOT_DIR"/*/; do
    # Skip if no subdirectories match
    [ -d "$PATIENT_DIR" ] || continue

    PATIENT_ID=$(basename "$PATIENT_DIR")
    CASE_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$PATIENT_ID"

    echo "----------------------------------------"
    echo "Processing patient: $PATIENT_ID"
    echo "  Input:  $PATIENT_DIR"
    echo "  Output: $CASE_OUTPUT_DIR"

    mkdir -p "$CASE_OUTPUT_DIR"

    if [ $CPU_ARG ]; then
        echo "  Running on CPU..."
        time CUDA_VISIBLE_DEVICES='' python "$APP_DIR" -i "$PATIENT_DIR" -o "$CASE_OUTPUT_DIR" -m "$MODEL_DIR"
    else
        echo "  Running on GPU..."
        time python "$APP_DIR" -i "$PATIENT_DIR" -o "$CASE_OUTPUT_DIR" -m "$MODEL_DIR"
    fi
done

echo "Batch processing complete."


