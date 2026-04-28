#!/bin/bash

error() {
    echo "Usage: test_batch.sh -i <input root directory> -o <output root directory> -m <model directory>"
    echo "       [-c] run on CPU  [-r] resume (skip already-processed cases)"
    echo "       [--cooldown SECS] pause between cases (default 30, prevents thermal BSOD)"
    echo "       [--audit-csv PATH] per-case status CSV (default <output>/batch_audit_log.csv)"
    exit 1
}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
APP_DIR=$SCRIPT_DIR/../prostate_mri_lesion_seg_app
CASE_STATUS_PY=$SCRIPT_DIR/case_status.py

# Set defaults for optional arguments
INPUT_ROOT_DIR=""
OUTPUT_ROOT_DIR=$SCRIPT_DIR/../output
MODEL_DIR=$APP_DIR/models/
COOLDOWN_SECS=30
RESUME=0
AUDIT_CSV=""

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
        -r | --resume) RESUME=1 ;;
        --cooldown)
            if ! has_argument $@; then
                echo "Error: No cooldown seconds specified" && error
            fi
            COOLDOWN_SECS=$(extract_argument $@)
            shift
            ;;
        --audit-csv)
            if ! has_argument $@; then
                echo "Error: No audit CSV path specified" && error
            fi
            AUDIT_CSV=$(extract_argument $@)
            shift
            ;;
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

# Prepare output root directory
mkdir -p "$OUTPUT_ROOT_DIR"

# Audit CSV: single source of truth for per-case outcomes.  Lives under
# the output root by default so it is carried back from HIVE alongside
# the result NIfTIs.
if [ -z "$AUDIT_CSV" ]; then
    AUDIT_CSV="$OUTPUT_ROOT_DIR/batch_audit_log.csv"
fi

if [ "$RESUME" -eq 0 ]; then
    # Fresh run: wipe outputs AND the audit log so the CSV stays a
    # faithful record of this batch (1 row per case).
    rm -rf "$OUTPUT_ROOT_DIR"/*
    rm -f "$AUDIT_CSV"
else
    echo "*** RESUME mode: skipping cases that already have output files ***"
    echo "*** RESUME mode: existing audit CSV will be appended to       ***"
fi

TOTAL=$(find "$INPUT_ROOT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
CURRENT=0
SKIPPED=0
FAILED=0
SUCCEEDED=0

echo "Batch processing $TOTAL patients under: $INPUT_ROOT_DIR"
echo "Writing per-patient outputs under: $OUTPUT_ROOT_DIR"
echo "Using model directory: $MODEL_DIR"
echo "Cooldown between cases: ${COOLDOWN_SECS}s"
echo "Audit CSV: $AUDIT_CSV"
echo ""

# Resolve the python interpreter to use for case_status.py.  Prefer the
# same one that runs the app.  Fall back to 'python' or 'python3'.
if command -v python >/dev/null 2>&1; then
    AUDIT_PY=python
elif command -v python3 >/dev/null 2>&1; then
    AUDIT_PY=python3
else
    AUDIT_PY=""
    echo "WARNING: no 'python' interpreter on PATH; audit CSV will be skipped."
fi

log_case_status() {
    # log_case_status <patient_id> <case_output_dir> <exit_code> <elapsed_secs> <resume_skipped 0|1> [<input_dir>]
    local pid="$1"
    local case_dir="$2"
    local ec="$3"
    local elapsed="$4"
    local resumed="$5"
    local input_dir="${6:-}"
    if [ -z "$AUDIT_PY" ]; then
        return 0
    fi
    local args=(
        "$CASE_STATUS_PY"
        --case-output-dir "$case_dir"
        --patient-id "$pid"
        --csv "$AUDIT_CSV"
    )
    if [ -n "$input_dir" ]; then
        args+=(--input-dir "$input_dir")
    fi
    if [ -n "$ec" ]; then
        args+=(--exit-code "$ec")
    fi
    if [ -n "$elapsed" ]; then
        args+=(--elapsed-seconds "$elapsed")
    fi
    if [ "$resumed" = "1" ]; then
        args+=(--resume-skipped)
    fi
    # Never let an audit-logging failure abort the batch.
    "$AUDIT_PY" "${args[@]}" || echo "  WARNING: audit log append failed for $pid"
}

# Loop over patient subdirectories
for PATIENT_DIR in "$INPUT_ROOT_DIR"/*/; do
    [ -d "$PATIENT_DIR" ] || continue

    PATIENT_ID=$(basename "$PATIENT_DIR")
    CASE_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$PATIENT_ID"
    CURRENT=$((CURRENT + 1))

    echo "========================================"
    echo "[$CURRENT/$TOTAL] Patient: $PATIENT_ID"
    echo "  Input:  $PATIENT_DIR"
    echo "  Output: $CASE_OUTPUT_DIR"

    # Resume: skip if output already has NIfTI or RTSTRUCT files
    if [ "$RESUME" -eq 1 ] && [ -d "$CASE_OUTPUT_DIR" ]; then
        FILE_COUNT=$(find "$CASE_OUTPUT_DIR" -type f \( -name "*.nii.gz" -o -name "*.dcm" \) 2>/dev/null | wc -l)
        if [ "$FILE_COUNT" -gt 0 ]; then
            echo "  SKIPPING (resume mode, $FILE_COUNT output files already exist)"
            SKIPPED=$((SKIPPED + 1))
            # Log the skipped case too -- but only if it is not already
            # recorded in the CSV, so repeat resumes do not duplicate rows.
            if [ -n "$AUDIT_PY" ]; then
                "$AUDIT_PY" "$CASE_STATUS_PY" \
                    --case-output-dir "$CASE_OUTPUT_DIR" \
                    --patient-id "$PATIENT_ID" \
                    --input-dir "$PATIENT_DIR" \
                    --csv "$AUDIT_CSV" \
                    --resume-skipped \
                    --skip-if-logged \
                    || echo "  WARNING: audit log append failed for $PATIENT_ID"
            fi
            continue
        fi
    fi

    mkdir -p "$CASE_OUTPUT_DIR"

    CASE_START=$SECONDS
    if [ $CPU_ARG ]; then
        echo "  Running on CPU..."
        CUDA_VISIBLE_DEVICES='' python "$APP_DIR" -i "$PATIENT_DIR" -o "$CASE_OUTPUT_DIR" -m "$MODEL_DIR"
    else
        echo "  Running on GPU..."
        python "$APP_DIR" -i "$PATIENT_DIR" -o "$CASE_OUTPUT_DIR" -m "$MODEL_DIR"
    fi
    EXIT_CODE=$?
    ELAPSED=$((SECONDS - CASE_START))

    # Book-keeping counters preserve existing semantics: SUCCEEDED is
    # incremented on exit-code 0 (matches pre-audit behaviour).  The
    # audit CSV tells you *per-case* whether that exit-0 was genuinely
    # complete, partial, or silently missing modalities.
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  *** FAILED (exit code $EXIT_CODE) ***"
        FAILED=$((FAILED + 1))
    else
        echo "  DONE (exit 0)"
        SUCCEEDED=$((SUCCEEDED + 1))
    fi

    log_case_status "$PATIENT_ID" "$CASE_OUTPUT_DIR" "$EXIT_CODE" "$ELAPSED" 0 "$PATIENT_DIR"

    # Cooldown: let CPU/GPU cool down and release memory between cases
    if [ "$CURRENT" -lt "$TOTAL" ]; then
        sync
        echo "  Cooldown ${COOLDOWN_SECS}s before next case..."
        sleep "$COOLDOWN_SECS"
    fi
done

echo ""
echo "========================================"
echo "Batch processing complete."
echo "  Total: $TOTAL  Succeeded: $SUCCEEDED  Failed: $FAILED  Skipped: $SKIPPED"
echo "  Exit-code counters above reflect the app's raw return status."
echo "  For the full per-case outcome (including silent empty outputs)"
echo "  inspect the audit CSV: $AUDIT_CSV"

# Per-status breakdown from the audit CSV (best-effort).
if [ -n "$AUDIT_PY" ] && [ -f "$AUDIT_CSV" ]; then
    echo ""
    echo "----- Per-status breakdown from audit CSV -----"
    "$AUDIT_PY" "$CASE_STATUS_PY" summary "$AUDIT_CSV" \
        || echo "  (summary generation failed)"
fi
