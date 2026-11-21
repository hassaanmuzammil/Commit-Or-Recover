#!/usr/bin/env bash
# Oracle speed-cap test on one Bench2Drive route (closed loop, TF++ frozen).
# Requires a CARLA server on port 2000. Runs in the garage_2 environment.
#
#   scripts/run_oracle_speed_cap.sh <variant> [cap_mps start_step end_step]
#
# Without a cap this is the raw (unmodified) TF++ run. With a cap, the agent brakes whenever the ego is faster
# than cap_mps between start_step and end_step (the window is set from the raw run's ground-truth collision step).
# Chase and bird's-eye camera frames are saved for video; logs are schema v2 plus the oracle fields.
set -u

ROUTE="${ROUTE:-1790}"
VARIANT="$1"
BASE_PATH="$HOME/Desktop/Commit-Or-Recover"
export SDL_AUDIODRIVER=dummy
export CARLA_ROOT="${BASE_PATH}/carla_garage/carla"
export WORK_DIR="${BASE_PATH}/carla_garage"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${WORK_DIR}:${PYTHONPATH:-}"
PYTHON="$HOME/anaconda3/envs/garage_2/bin/python"
cd "$WORK_DIR"

OUT="${BASE_PATH}/results/oracle_speed_cap/route_${ROUTE}/${VARIANT}"
rm -rf "${OUT}/frames"
export VIDEO_DIR="${OUT}/frames"
if [[ $# -ge 4 ]]; then
  export ORACLE_CAP_SPEED="$2" ORACLE_CAP_START_STEP="$3" ORACLE_CAP_END_STEP="$4"
fi

"$PYTHON" tools/run_bench2drive_routes.py --suite all --only "$ROUTE" --force \
  --traffic-manager-seed 100 --output "$OUT"
