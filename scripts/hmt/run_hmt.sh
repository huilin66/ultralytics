#!/usr/bin/env bash
set -euo pipefail

# One-click data preparation followed by training. Set SKIP_PREPARE=1 to
# train an already generated bdd_hmt_update tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    bash "${SCRIPT_DIR}/prepare_hmt.sh"
fi
bash "${SCRIPT_DIR}/train_hmt.sh" "$@"
