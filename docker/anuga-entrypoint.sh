#!/usr/bin/env bash
#
# ANUGA container entrypoint — headless/batch friendly with optional S3 I/O.
#
# Behaviour:
#   * If INPUT_S3 is set, its contents are synced into $ANUGA_WORKDIR/input.
#   * The command passed to the container ("$@") is run from $ANUGA_WORKDIR.
#     Write your results into $ANUGA_WORKDIR/output.
#   * If OUTPUT_S3 is set, $ANUGA_WORKDIR/output is synced up to it when the
#     command finishes (on success; also on failure if STAGE_OUT_ON_FAILURE=1).
#   * For AWS Batch array jobs, AWS_BATCH_JOB_ARRAY_INDEX is appended to
#     OUTPUT_S3 so each index lands in its own prefix (.../0/, .../1/, ...).
#   * With no command, drops to an interactive bash shell (local debugging).
#
# Credentials for aws come from the environment / instance role (e.g. the
# AWS Batch job IAM role) — none are baked into the image.
set -euo pipefail

WORKDIR="${ANUGA_WORKDIR:-/work}"
mkdir -p "$WORKDIR/output"
cd "$WORKDIR"

log() { echo "[anuga-entrypoint] $*" >&2; }

if [ -n "${INPUT_S3:-}" ]; then
    log "Staging input: ${INPUT_S3} -> ${WORKDIR}/input"
    mkdir -p input
    aws s3 cp "${INPUT_S3}" input --recursive
fi

OUT="${OUTPUT_S3:-}"
if [ -n "$OUT" ] && [ -n "${AWS_BATCH_JOB_ARRAY_INDEX:-}" ]; then
    OUT="${OUT%/}/${AWS_BATCH_JOB_ARRAY_INDEX}"
fi

stage_out() {
    local rc=$1
    [ -z "$OUT" ] && return 0
    if [ "$rc" -eq 0 ] || [ "${STAGE_OUT_ON_FAILURE:-0}" = "1" ]; then
        log "Uploading results: ${WORKDIR}/output -> ${OUT}"
        aws s3 cp output "${OUT}" --recursive || log "WARNING: upload to ${OUT} failed"
    else
        log "Command failed (rc=$rc); skipping upload (set STAGE_OUT_ON_FAILURE=1 to override)"
    fi
}

# No command -> interactive shell for debugging on a local/EC2 box.
if [ "$#" -eq 0 ]; then
    log "No command given; starting interactive shell in ${WORKDIR}"
    exec /bin/bash
fi

log "Running: $*"
set +e
"$@"
rc=$?
set -e
stage_out "$rc"
exit "$rc"
