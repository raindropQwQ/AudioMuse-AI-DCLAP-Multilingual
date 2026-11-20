#!/usr/bin/env bash
# Apply the GTE-to-DCLAP multilingual CLAP text patch to AudioMuse-AI.
#
# The patch is a standard unified diff (a/ b/ prefixes, applied with -p1).
# It adds:
#   config.py               -> CLAP_GTE_PROJ_PATH
#   tasks/clap_analyzer.py  -> GTE + projection path, with RoBERTa fallback
#
# Usage
#   ./apply_patch.sh                          # apply inside the flask container
#   ./apply_patch.sh --container NAME         # different container
#   ./apply_patch.sh --dir /path/to/src       # apply to a source tree on the host
#   ./apply_patch.sh --check                  # dry-run only, exit 0 if applicable
#   ./apply_patch.sh --check --dir /path/src  # dry-run against a source tree
#
# Restarting (needed for the patch to take effect in a running container):
#   docker exec <container> supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart flask
# The script does this automatically unless --no-restart is given.
#
# Exit codes
#   0  patch applied (or already applied with --check)
#   1  patch failed to apply
#   2  bad usage / missing file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/audiomuse-gte-dclap.patch"
MODEL_HOST_DEFAULT="$SCRIPT_DIR/../gte_to_dclap_proj.onnx"
MODEL_CONTAINER_PATH="/app/model/gte_to_dclap_proj.onnx"

CONTAINER="audiomuse-ai-flask-app"
TARGET_DIR=""
CHECK_ONLY=0
NO_RESTART=0

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) CONTAINER="${2:?}"; shift 2 ;;
        --dir)       TARGET_DIR="${2:?}"; shift 2 ;;
        --check)     CHECK_ONLY=1; shift ;;
        --no-restart) NO_RESTART=1; shift ;;
        -h|--help)   usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -f "$PATCH_FILE" ]] || { echo "patch file not found: $PATCH_FILE" >&2; exit 2; }

# --- run against a source tree on the host -----------------------------------
if [[ -n "$TARGET_DIR" ]]; then
    [[ -d "$TARGET_DIR" ]] || { echo "not a directory: $TARGET_DIR" >&2; exit 2; }
    cd "$TARGET_DIR"

    # Already applied? A reverse dry-run succeeds only when all hunks match.
    # Do NOT combine with --batch: it silently flips direction and always exits 0.
    if patch --dry-run --reverse -p1 --forward < "$PATCH_FILE" >/dev/null 2>&1; then
        echo "already applied: $TARGET_DIR"
        exit 0
    fi

    if [[ "$CHECK_ONLY" == 1 ]]; then
        patch --dry-run --forward -p1 < "$PATCH_FILE"
        echo "check ok: $TARGET_DIR"
        exit 0
    fi

    patch --forward -p1 < "$PATCH_FILE"
    echo "patch applied to $TARGET_DIR"
    echo "note: the model file must be at $MODEL_CONTAINER_PATH (see README)"
    exit 0
fi

# --- run against a live container --------------------------------------------
command -v docker >/dev/null || { echo "docker not found" >&2; exit 2; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || {
    echo "container not found: $CONTAINER" >&2; exit 2; }

WORK="/tmp/gte-dclap-patch.$$"
docker exec "$CONTAINER" mkdir -p "$WORK"
trap 'docker exec "$CONTAINER" rm -rf "$WORK" >/dev/null 2>&1 || true' EXIT

docker cp "$PATCH_FILE" "$CONTAINER:$WORK/p.patch" >/dev/null

if docker exec "$CONTAINER" sh -c "cd /app && patch --dry-run --reverse -p1 --forward < $WORK/p.patch >/dev/null 2>&1"; then
    echo "already applied in $CONTAINER"
    BECAME_APPLIED=0
else
    if ! docker exec "$CONTAINER" sh -c "cd /app && patch --dry-run --forward -p1 < $WORK/p.patch"; then
        echo "patch does not apply cleanly - the upstream code changed." >&2
        echo "Check the anchors listed in README (Upgrade notes) and regenerate the diff." >&2
        exit 1
    fi
    if [[ "$CHECK_ONLY" == 1 ]]; then
        echo "check ok: patch applies cleanly to $CONTAINER"
        exit 0
    fi
    docker exec "$CONTAINER" sh -c "cd /app && patch --forward -p1 < $WORK/p.patch"
    BECAME_APPLIED=1
fi

# Projection model: warn only, it is mounted separately in most deployments.
if docker exec "$CONTAINER" test -f "$MODEL_CONTAINER_PATH"; then
    echo "projection model present: $MODEL_CONTAINER_PATH"
else
    echo "WARNING: projection model missing at $MODEL_CONTAINER_PATH" >&2
    echo "         copy it with: docker cp <repo>/gte_to_dclap_proj.onnx $CONTAINER:$MODEL_CONTAINER_PATH" >&2
    echo "         without it the patch falls back to RoBERTa (English only)." >&2
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
    exit 0
fi

if [[ "$BECAME_APPLIED" == 1 && "$NO_RESTART" == 0 ]]; then
    echo "restarting flask inside $CONTAINER..."
    docker exec "$CONTAINER" supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart flask \
        || echo "WARNING: automatic restart failed - restart the container manually." >&2
fi

echo "done"
