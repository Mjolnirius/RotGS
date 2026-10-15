#!/usr/bin/env bash

# Sequential, disconnect-safe Session_04 30-degree preprocessing queue.
# R082's SAM 3.1 preprocessing requires the separate Python 3.12 environment.
# Launch with nohup; use --dry-run to inspect commands and completed datasets.
# uv run --no-project --python .venv-sam31/bin/python bash jobs/preprocess_queue.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec "${REPO_ROOT}/.venv-sam31/bin/python" \
    "${REPO_ROOT}/tools/preprocessing/run_preprocessing_queue.py" "$@"
