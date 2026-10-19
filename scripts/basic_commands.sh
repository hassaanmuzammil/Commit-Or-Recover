export BASE_PATH="/home/018768243@SJSUAD/Desktop/Commit-Or-Recover"

export SDL_AUDIODRIVER=dummy
export CARLA_ROOT="${BASE_PATH}/carla_garage/carla"
export WORK_DIR="${BASE_PATH}/carla_garage"
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${WORK_DIR}:${PYTHONPATH}"

conda activate garage_2
cd "$CARLA_ROOT"
./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -carla-rpc-port=2000

conda activate garage_2
cd $WORK_DIR
mkdir -p results
TEMPORAL_LOG_DIR=results/temporal_logs \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes leaderboard/data/debug.xml \
  --agent team_code/sensor_agent.py \
  --agent-config pretrained_models/project_tfpp \
  --checkpoint results/debug_tfpp.json \
  --debug 1

python tools/validate_temporal_logs.py results/temporal_logs

python tools/run_bench2drive_routes.py --output results/bench2drive_selected --validate-only
  