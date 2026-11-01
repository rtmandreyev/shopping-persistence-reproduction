#!/usr/bin/env sh
# Thin wrapper: the logic lives in reproduce.py. Requires uv (https://docs.astral.sh/uv/).
set -e
cd "$(dirname "$0")"
exec uv run --frozen python reproduce.py "$@"
