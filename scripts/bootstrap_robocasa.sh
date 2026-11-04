#!/usr/bin/env bash
set -euo pipefail

# RoboCasa client environment.
#
# Deliberately tiny. RoboCasa itself is NOT installed here: the simulator runs in its own
# process (robocasa/capx_sim_server) with its own mujoco/robosuite/robocasa, and is reached over
# the CaP-X ZeroMQ wire protocol. This venv carries cap-harness, the transport, and the shared
# closed-loop driver -- nothing else. That is what keeps ASPIRE's vendored robosuite fork and
# the simulator's upstream robosuite from ever having to agree.
#
# There is no submodule-commit check here (unlike bootstrap_robosuite.sh) because this
# environment builds from no submodule.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
VENV_ROOT="${CAP_HARNESS_ROBOCASA_VENV:-$(resolve_canonical_venv "$REPO_ROOT" ".venv-robocasa")}"
UV_CACHE_ROOT="${CAP_HARNESS_ROBOCASA_CACHE:-/tmp/${USER:-root}/uv-cache-robocasa}"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"

if [[ "$(basename -- "$VENV_ROOT")" != ".venv-robocasa" ]]; then
  printf 'error: RoboCasa environment basename must be exactly .venv-robocasa: %s\n' \
    "$VENV_ROOT" >&2
  exit 2
fi
command -v uv >/dev/null || { printf 'error: uv is required\n' >&2; exit 1; }
[[ -f "$REPO_ROOT/uv.lock" ]] || { printf 'error: uv.lock is missing\n' >&2; exit 1; }
[[ -f "$REPO_ROOT/../robot-driver/pyproject.toml" ]] || {
  printf 'error: the shared driver is missing at %s\n' "$REPO_ROOT/../robot-driver" >&2
  exit 1
}

mkdir -p "$(dirname -- "$VENV_ROOT")" "$UV_CACHE_ROOT"
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "${CAP_HARNESS_PROFILE:-rtx5090}" "sim:robocasa:robocasa")"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf '%s is up to date (fingerprint %s); skipping rebuild\n' "sim:robocasa:robocasa" "${FINGERPRINT:0:12}"
  exit 0
fi
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
VIRTUAL_ENV="$VENV_ROOT" UV_PROJECT_ENVIRONMENT="$VENV_ROOT" UV_CACHE_DIR="$UV_CACHE_ROOT" \
  uv sync --frozen --active --project "$REPO_ROOT" \
    --extra robocasa --extra dev

"$VENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
import importlib.util
import sys

import cap_harness
import robot_driver
from cap_harness.robocasa import RobocasaAdapter, RobocasaTaskRegistry

root = Path(sys.prefix).resolve()
assert root.name == ".venv-robocasa", root
# The boundary is structural, not a promise: if any of these were importable here, generated
# code could reach simulator state directly and sim results would stop transferring.
leaked = [n for n in ("mujoco", "robosuite", "robocasa") if importlib.util.find_spec(n)]
assert not leaked, f"simulator packages leaked into the client venv: {leaked}"
tasks = ", ".join(RobocasaTaskRegistry().available_tasks)
print(f"Verified RoboCasa client runtime: {root}")
print(f"  embodiment={RobocasaAdapter.embodiment.fget(RobocasaAdapter)!r} tasks: {tasks}")
PY

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "sim:robocasa:robocasa" "${CAP_HARNESS_PROFILE:-rtx5090}"
cat <<EOS
RoboCasa bootstrap complete.
  source:       $REPO_ROOT
  environment:  $VENV_ROOT
  cache:        $UV_CACHE_ROOT
  driver:       $REPO_ROOT/../robot-driver

Start the simulator first (separate process, its own venv):
  cd $REPO_ROOT/.. && MUJOCO_GL=egl \\
    robocasa/capx_sim_server/.venv/bin/python \\
    robocasa/capx_sim_server/server.py --task OpenDrawer

Then:
  cap-harness run --benchmark robocasa --suite OpenDrawer --task-id 0 --seed 101 \\
    --program <program>.py --output-root outputs
EOS
