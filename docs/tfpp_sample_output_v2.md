# TF++ JSONL Log Entry (schema v2)

One line per step at 20 Hz (`dt = 0.05` s), written by `_log_temporal_step` in [sensor_agent.py](../carla_garage/team_code/sensor_agent.py). Results are in `results/bench2drive_v2/logs/<route>/*.jsonl`.

v2 keeps every v1 field (see [tfpp_sample_output.md](tfpp_sample_output.md)) and adds 8 fields:

- `pred_boxes`: TF++'s detected boxes
- `gt_ego`, `gt_actors`: simulator ground truth
- `collisions`: collision events
- 4 `tfpp_*` override flags

The ground-truth fields come from privileged simulator access. They are evaluation labels only and are never fed to the model or the controller.

Sample: route `23687`, step 318, the step where a car side-swipes the ego. Path points and box rows are shown on one line each to save space.

```json
{
   "schema_version":2,
   "step":318,
   "timestamp":15.950000237673521,
   "route_id":"bench2drive_131_route0_11_09_21_05_43",
   "dt":0.05,
   "predicted_path":[
      [2.4814562797546387, 0.2708056569099426],
      [3.4823315143585205, 0.29711323976516724],
      [4.4827375411987305, 0.3319241404533386],
      [5.48305606842041, 0.36217933893203735],
      [6.482493877410889, 0.3895930051803589],
      [7.480310916900635, 0.4153669476509094],
      [8.478625297546387, 0.4442591071128845],
      [9.474950790405273, 0.4730881452560425],
      [10.471170425415039, 0.5019254684448242],
      [11.466374397277832, 0.528577983379364]
   ],
   "target_speed_probabilities":[
      1.0887168855333584e-06,
      7.608928626723355e-06,
      0.0006362113635987043,
      0.3551620543003082,
      0.2873515784740448,
      0.354500412940979,
      0.0002366397820878774,
      0.0021044258028268814
   ],
   "target_speed_scalar":13.266036865114689,
   "steer":0.20399999618530273,
   "throttle":1.0,
   "brake":0.0,
   "ego_speed":10.2951078414917,
   "ego_position":[41.97177305741683, 4125.6101854633],
   "ego_yaw":-1.0341832200672685,
   "acceleration":2.6714706420898438,
   "yaw_rate":-0.021708011627188384,
   "target_point":[26.01401138305664, 0.4648861587047577],
   "command":3,
   "pred_boxes":[
      [31.927173614501953, 3.218524932861328, 2.3945202827453613, 0.9593045115470886, 0.1963871568441391, 0.0, 0.0, 0.0, 0.5113409161567688]
   ],
   "gt_ego":{
      "frame":23829,
      "x":-2460.584716796875,
      "y":4124.900390625,
      "z":363.6722106933594,
      "yaw":-1.0341829492654895,
      "vx":5.18220329284668,
      "vy":-8.896769523620605,
      "speed":10.296005969540586
   },
   "gt_actors":[
      {
         "id":84156,
         "type":"vehicle.ford.mustang",
         "class":"vehicle",
         "x":31.96772943302267,
         "y":3.2863395712600507,
         "yaw":0.2581243586528945,
         "ext_x":2.358762502670288,
         "ext_y":0.947413444519043,
         "speed":1.779279515702545,
         "vx":1.7146393599992282,
         "vy":0.47523411087601886,
         "world":[-2441.41748046875, 4099.10595703125, -0.776058590612595]
      },
      {
         "id":84152,
         "type":"vehicle.mercedes.coupe_2020",
         "class":"vehicle",
         "x":0.45509088537447756,
         "y":1.8137852477063658,
         "yaw":-0.038983019141443746,
         "ext_x":2.3368194103240967,
         "ext_y":0.9059062600135803,
         "speed":10.34881254371453,
         "vx":10.348349341109618,
         "vy":-0.0979131211805706,
         "world":[-2458.793212890625, 4125.4365234375, -1.0731659684069332]
      }
   ],
   "collisions":[
      {
         "frame":23829,
         "other_id":84152,
         "other_type":"vehicle.mercedes.coupe_2020",
         "impulse":212.09567523608627
      }
   ],
   "tfpp_emergency_stop":false,
   "tfpp_force_move":false,
   "tfpp_stop_sign_stop":false,
   "tfpp_bb_in_front":false
}
```

**What this step shows:** the car that hits the ego (`84152`) is right beside it, at 0.46 m ahead and 1.81 m to the right. TF++ detected only the far car (`84156`, about 32 m ahead, where its box matches the ground truth to within about 0.1 m). It detected nothing next to the ego, so `tfpp_bb_in_front` is false, and the ego is still on full throttle at 10.3 m/s. This kind of missed detection is something you can only label using the v2 fields.

## New fields

All "ego frame" values use the ego vehicle's frame: **x forward, y right** (m), with yaw in radians relative to the ego heading.

| Field | Source | Meaning | Example |
|---|---|---|---|
| `pred_boxes` | TF++ detection head, after NMS across the ensemble | List of detected boxes, each `[x, y, extent_x, extent_y, yaw, speed, brake, class, score]`, in the ego frame. Extents are half-lengths (m). Speed is in m/s. See class codes below | 1 car, 31.9 m ahead, score 0.51 |
| `gt_ego` | Simulator (hero actor) | True ego state: `frame`, `x`, `y`, `z` (CARLA world, m), `yaw` (rad), `vx`, `vy` (world, m/s), `speed` | 10.30 m/s |
| `gt_actors` | Simulator | Every vehicle, walker and `static.prop*` whose box centre is within **60 m** (set by the env var `GT_LOG_RADIUS`). The ego itself is excluded. Traffic lights and stop signs are not included | 2 vehicles |
| `gt_actors[].id`, `type`, `class` | Simulator | CARLA actor id, blueprint id, and a coarse class (`vehicle` / `walker` / `static`) | `84152`, `vehicle.mercedes.coupe_2020` |
| `gt_actors[].x`, `y`, `yaw` | Simulator | Box centre and heading in the ego frame. These are directly comparable to `pred_boxes` | 0.46 m ahead, 1.81 m right |
| `gt_actors[].ext_x`, `ext_y` | Simulator | Box half-extents (m) | 2.34 × 0.91 |
| `gt_actors[].speed`, `vx`, `vy` | Simulator | Speed (m/s), and velocity rotated into the ego frame | 10.35 m/s |
| `gt_actors[].world` | Simulator | `[x, y, yaw]` of the box centre in CARLA world coordinates (yaw in rad) | — |
| `collisions` | CARLA `sensor.other.collision` on the ego | Collision events since the previous logged step: `frame`, `other_id` (matches `gt_actors[].id`; -1 if unknown), `other_type`, and `impulse` (magnitude of the normal impulse, N·s). Usually empty | Hit by `84152`, impulse 212 |
| `tfpp_force_move` | TF++ stuck recovery | True while the creep override is active. The override starts after the ego has been stopped for more than 1100 frames (55 s), then forces throttle ≥ 0.4 for 20 frames | false |
| `tfpp_emergency_stop` | TF++ stuck recovery | True when creeping is blocked because LiDAR points are inside the safety box, which forces a brake. **It can only be true while `tfpp_force_move` is true.** It is not a general emergency brake | false |
| `tfpp_stop_sign_stop` | TF++ stop-sign controller | True when TF++ forces a brake for a detected stop sign | false |
| `tfpp_bb_in_front` | TF++'s built-in box check, run on `pred_boxes` | True if a detected box overlaps a safety box 2–4 m ahead of the ego (the length grows with speed). **This flag is logged only as a baseline. The driving stack does not use it** | false |

## `pred_boxes` class codes

Index 7 of each box (from `data.py`):

| Value | Meaning |
|---|---|
| 0 | Car |
| 1 | Walker |
| 2 | Traffic light |
| 3 | Stop sign |
| 4 | Emergency vehicle (police, ambulance, fire truck) |

The lowest box score seen in the logs is about 0.30, so lower-scoring detections are not logged.

## Caveats

- **`ego_position` and `gt_ego` are in different frames.** `ego_position` comes from TF++'s GPS/IMU filter, which uses an estimated GPS reference point. `gt_ego` is the true CARLA world position. In the sample their x values differ by about 2500 m (41.97 vs -2460.58), while y is close. To relate ground truth to the ego, use the ego-frame `gt_actors[].x/y`. Don't compute it from `ego_position`.
- **A single crash usually spans several steps.** Contact produces collision events on consecutive steps: 14,293 steps across the run have a non-empty `collisions` list. Group them by `other_id` and gaps in `frame` to count distinct collisions.
- **Control is logged after all overrides.** `steer`, `throttle` and `brake` are the values sent to CARLA after stuck recovery and the stop-sign override, as in v1. The model's raw control before overrides is not logged.

## Still not logged

- Traffic light states and ground-truth stop sign positions
- Raw sensor data (camera, LiDAR)
- The network's control before overrides
