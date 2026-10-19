# Setup

## 1. Clone CARLA Garage

```bash
mkdir -p ~/Desktop/Commit-Or-Recover
cd ~/Desktop/Commit-Or-Recover

git clone https://github.com/autonomousvision/carla_garage.git
cd carla_garage

git checkout leaderboard_2
```

Verify:

```bash
git branch --show-current
```

Expected:

```text
leaderboard_2
```

## 2. Install CARLA

From `carla_garage`:

```bash
chmod +x setup_carla.sh
./setup_carla.sh
```

CARLA should be installed at:

```text
carla_garage/carla/
```

## 3. Create Python Environment

```bash
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
conda activate garage_2
cd $CARLA_ROOT

./CarlaUE4.sh
```

A CARLA city window should open. Leave it running.

## 8. Run TransFuser++

Open Terminal 2:

```bash
conda activate garage_2
cd $WORK_DIR
```

Set the environment variables from Step 4 if they are not already set.

Create the results directory:

```bash
mkdir -p results
```

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

Run the evaluator with a dedicated output directory:

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
