#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train_v3_easy.py --install --recipe smoke "$@"
