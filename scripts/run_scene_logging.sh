#!/usr/bin/env bash
# Schema-v2 scene logging runs (pred boxes + ground truth + collision ids).
# Requires a CARLA server on port 2000. Runs in the garage_2 environment.
# Completed routes are skipped, so the script can be re-launched after a crash.
set -u

BASE_PATH="~/Desktop/Commit-Or-Recover"
export SDL_AUDIODRIVER=dummy
export CARLA_ROOT="${BASE_PATH}/carla_garage/carla"
export WORK_DIR="${BASE_PATH}/carla_garage"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${WORK_DIR}:${PYTHONPATH:-}"
PYTHON="~/anaconda3/envs/garage_2/bin/python"
cd "$WORK_DIR"

PHASE="${1:-focus}"   # focus | all | longest6

if [[ "$PHASE" == "focus" ]]; then
  for seed in 100 200 300; do
    "$PYTHON" tools/run_bench2drive_routes.py --suite selected \
      --traffic-manager-seed "$seed" --output "results/scene_focus_seed${seed}"
  done
elif [[ "$PHASE" == "all" ]]; then
  "$PYTHON" tools/run_bench2drive_routes.py --suite all \
    --traffic-manager-seed 100 --output results/scene_all_seed100
elif [[ "$PHASE" == "longest6" ]]; then
  # Longest6 v2: 36 routes (Towns 01-06), CARLA 0.9.15 / leaderboard 2.0 scenarios.
  "$PYTHON" tools/run_bench2drive_routes.py --suite longest6 \
    --traffic-manager-seed 100 --output results/longest6_v2
fi
