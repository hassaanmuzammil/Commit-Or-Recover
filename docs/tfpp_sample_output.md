# TF++ JSONL Log Entry

One line per step at 20 Hz (`dt = 0.05` s), written by `_log_temporal_step` in [sensor_agent.py](../carla_garage/team_code/sensor_agent.py).

```json
{
   "schema_version":1,
   "step":5,
   "timestamp":0.30000000447034836,
   "route_id":"bench2drive_00_route0_10_24_20_39_25",
   "dt":0.05,
   "predicted_path":[
      [
         2.507176160812378,
         -0.012013006955385208
      ],
      [
         3.5159189701080322,
         -0.0047033801674842834
      ],
      [
         4.509253025054932,
         -0.013251926749944687
      ],
      [
         5.516578674316406,
         -0.016832783818244934
      ],
      [
         6.521641731262207,
         -0.020993713289499283
      ],
      [
         7.524615287780762,
         -0.026766590774059296
      ],
      [
         8.52682113647461,
         -0.03232250735163689
      ],
      [
         9.52950382232666,
         -0.03893342614173889
      ],
      [
         10.529214859008789,
         -0.043406207114458084
      ],
      [
         11.528230667114258,
         -0.045110948383808136
      ]
   ],
   "target_speed_probabilities":[
      0.6832213401794434,
      0.311982125043869,
      4.607406935974723e-06,
      0.004764632321894169,
      6.413836217689095e-06,
      1.184195411951805e-06,
      4.968580469721928e-06,
      1.4713669770571869e-05
   ],
   "target_speed_scalar":1.2961023145498531,
   "steer":-0.010999999940395355,
   "throttle":1.0,
   "brake":0.0,
   "ego_speed":4.8023568524513394e-05,
   "ego_position":[
      0.1562549867599252,
      3673.311590072929
   ],
   "ego_yaw":-1.574793001009395,
   "acceleration":0.0,
   "yaw_rate":0.0,
   "target_point":[
      133.3874969482422,
      -0.1551337093114853
   ],
   "command":3
}
```

## Fields

| Field | Source | Meaning | Example |
|---|---|---|---|
| `predicted_path` | TF++ network | 10 `[x, y]` points in the ego frame, about 1 m apart (sampled by distance, not time) | Straight, 2.5–11.5 m ahead |
| `target_speed_probabilities` | TF++ network | Probabilities over 8 speed classes `[0, 4, 8, 10, 13.89, 16, 17.78, 20]` m/s (`config.target_speeds`) | 68% on 0 m/s, 31% on 4 m/s |
| `target_speed_scalar` | TF++ speed selection | Expected speed `Σ pᵢ·vᵢ` (m/s) | 1.296 m/s |
| `steer` | TF++ controller | Final steering command in [-1, 1] after all overrides | -0.011 |
| `throttle` | TF++ controller | Final throttle command in [0, 1] | 1.0 (car is stopped and below target speed) |
| `brake` | TF++ controller | Final brake command in [0, 1] | 0.0 |
| `ego_speed` | Speedometer | Current speed (m/s) | ~0 |
| `ego_position` | Agent state (GPS/IMU filter) | `[x, y]` in CARLA world coordinates (m) | — |
| `ego_yaw` | Agent state | Heading (rad) | -1.575 rad ≈ -90° |
| `acceleration` | Derived | Change in `ego_speed` between steps divided by `dt` (m/s²); 0 on the first logged step | 0.0 |
| `yaw_rate` | Derived | Change in `ego_yaw` between steps divided by `dt` (rad/s); 0 on the first logged step | 0.0 |
| `target_point` | Route planner | Next route target in the ego frame (m) | 133 m ahead |
| `command` | Route planner | Route command, **0-indexed** (see below) | 3 = follow lane |
| `step`, `timestamp`, `dt`, `route_id`, `schema_version` | Logger | Metadata | — |

## `command` values

The logged value is the index of TF++'s one-hot command (`command_to_one_hot` in `transfuser_utils.py`), which is CARLA's `RoadOption` minus 1:

| Logged | Meaning |
|---|---|
| 0 | Turn left |
| 1 | Turn right |
| 2 | Go straight (through a junction) |
| 3 | Follow lane |
| 4 | Change lane left |
| 5 | Change lane right |

## Not logged

- TF++'s predicted bounding boxes of other actors
