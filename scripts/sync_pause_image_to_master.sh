#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SYNC_PAUSE=1
export SYNC_CALICO=0
export SYNC_LOCAL_TARS=0
export ARCHIVE_REMOTE="${ARCHIVE_REMOTE:-/home/zcx/pause-aarch64-3.8.tar}"
export ARCHIVE_LOCAL="${ARCHIVE_LOCAL:-/home/zcx/pause-aarch64-3.8.tar}"

exec "${SCRIPT_DIR}/sync_master_runtime_images.sh"
