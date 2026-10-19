# Setup

## 1. Use the repository checkout

`carla_garage` is already included in this repository as a tracked folder.
Start from the repository root:

```bash
cd /path/to/Commit-Or-Recover
```

## 2. Install CARLA locally

The CARLA executable and other large runtime assets are kept locally and are
not committed to Git. From the repository root:

```bash
cd carla_garage
chmod +x setup_carla.sh
./setup_carla.sh
```

CARLA should be installed at:

```text
carla_garage/carla/
```

## 3. Create Python Environment

```bash
cd carla_garage
conda env create -f environment.yml
conda activate garage_2
```

## 4. Set Environment Variables

```bash
export SDL_AUDIODRIVER=dummy

export CARLA_ROOT=~/Desktop/Commit-Or-Recover/carla_garage/carla
export WORK_DIR=~/Desktop/Commit-Or-Recover/carla_garage

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${WORK_DIR}:${PYTHONPATH}"
```

Verify:

```bash
echo $CARLA_ROOT
echo $WORK_DIR
echo $PYTHONPATH
```

`SDL_AUDIODRIVER=dummy` prevents ALSA audio warnings on machines without a configured audio device.

## 5. Download Pretrained TransFuser++ Models

Download the pretrained models provided by CARLA Garage.

The downloaded structure should contain:

```text
pretrained_models/
├── all_towns/
│   ├── config.json
│   ├── model_0030_0.pth
│   ├── model_0030_1.pth
│   └── model_0030_2.pth
└── town13_withheld/
    └── ...
```

For this project, use one frozen checkpoint instead of the 3-model ensemble:

```bash
mkdir -p pretrained_models/project_tfpp

cp pretrained_models/all_towns/config.json \
   pretrained_models/project_tfpp/

cp pretrained_models/all_towns/model_0030_0.pth \
   pretrained_models/project_tfpp/
```

Final checkpoint:

```text
pretrained_models/project_tfpp/
├── config.json
└── model_0030_0.pth
```

## 6. RTX 5080 / PyTorch Fix

The original environment used:

```text
PyTorch 2.5.0+cu124
CUDA 12.4
```

This did not support the RTX 5080 (`sm_120`) and produced:

```text
CUDA error: no kernel image is available for execution on the device
```

Upgrade PyTorch to a build that supports the RTX 5080.

Verify afterward:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_arch_list())"
```

Confirm that:

```text
sm_120
```

appears in the architecture list.

## 7. Start CARLA

Terminal 1:

```bash
cd /path/to/Commit-Or-Recover
conda activate garage_2
cd "$CARLA_ROOT"
./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  -carla-rpc-port=2000
```

A CARLA city window should open. Leave it running.

## 8. Run TransFuser++

Open Terminal 2:

```bash
cd /path/to/Commit-Or-Recover
conda activate garage_2
cd "$WORK_DIR"
mkdir -p results
```

Set the environment variables from Step 4 if they are not already set.

Run the debug route:

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes leaderboard/data/debug.xml \
  --agent team_code/sensor_agent.py \
  --agent-config pretrained_models/project_tfpp \
  --checkpoint results/debug_tfpp.json \
  --debug 1
```

## 9. Expected Result

The pipeline should now be:

```text
CARLA
  ↓ sensor data
Frozen TransFuser++
  ↓
steering / throttle / brake
  ↓
CARLA vehicle
```

TF++ should autonomously drive the `debug.xml` route.

The evaluator runs CARLA at a simulation timestep of approximately:

```text
0.05 s = 20 Hz
```

At this point the basic **CARLA + frozen TransFuser++ setup is working**.

## 10. Log temporal TF++ outputs

The instrumented `team_code/sensor_agent.py` writes one JSON Lines record per
inference/control step. Each record contains the full predicted path, the full
target-speed probability distribution, the scalar target speed, the control
returned to CARLA, ego position/speed/yaw, acceleration, yaw rate, target point,
command, route ID, and timestep.

Run the instrumented evaluator with a dedicated output directory:

```bash
TEMPORAL_LOG_DIR=results/temporal_logs \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes leaderboard/data/debug.xml \
  --agent team_code/sensor_agent.py \
  --agent-config pretrained_models/project_tfpp \
  --checkpoint results/debug_tfpp_instrumented.json \
  --debug 1
```

Validate the resulting files:

```bash
python tools/validate_temporal_logs.py results/temporal_logs
```

The validator checks that every record has the required fields, predicted
paths have shape `[N, 2]`, speed probabilities sum to one, steps are
consecutive, and timestamps match the configured `0.05` second timestep.

## 11. Define scenario families

| Family | Bench2Drive types |
|---|---|
| Roundabout/junction | `T_Junction`, `SignalizedJunctionLeftTurn`, `SignalizedJunctionRightTurn`, `NonSignalizedJunctionLeftTurn` |
| Lane split/merge | `SequentialLaneChange`, `MergerIntoSlowTrafficV2`, `HighwayExit`, `EnterActorFlow` |
| Obstacle detour | `ParkedObstacle`, `ConstructionObstacle`, `HazardAtSideLane`, `AccidentTwoWays` |

Validate the selected Bench2Drive routes:

```bash
cd "$WORK_DIR"
python tools/run_bench2drive_routes.py \
  --output results/bench2drive_selected \
  --validate-only
```
