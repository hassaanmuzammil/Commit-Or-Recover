#!/usr/bin/env python3
"""Build a sliding-window failure-prediction dataset from Bench2Drive TF++ runs.

Inputs (per run directory, as written by carla_garage/tools/run_bench2drive_routes.py):
  <run>/checkpoints/<route_id>.json   leaderboard record with infraction strings
  <run>/logs/<route_id>/<name>.jsonl  per-step TF++ log (see docs/tfpp_sample_output.md)

Collision timing. Checkpoints store each collision's world location, not its time.
The TF++ ego_position has a constant per-route x offset from CARLA world coordinates
(GPS reference estimate), so positions are shifted to world frame using the route's
first waypoint from the routes XML, and the collision step is the first step at which
the ego comes within `min distance + 0.5 m` of the collision location.

Outputs (in --out):
  train.npz / val.npz / test.npz  X [N, H, F] float32, y [N] int8, window_id [N]
  windows.csv                     one row per window (route, seed, times, label, TTC, ...)
  collisions.csv                  every collision event and how it was matched
  pre_collision.npz               the H steps before each matched collision
  meta.json                       feature names, config, splits, train normalization stats

Only numpy is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [REPO / "results" / "bench2drive_selected", REPO / "results" / "bench2drive_all"]
DEFAULT_ROUTES_XML = REPO / "carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"
DEFAULT_SEED = 100  # run_bench2drive_routes.py --traffic-manager-seed default

COLLISION_KEYS = {
    "collisions_vehicle": "vehicle",
    "collisions_pedestrian": "pedestrian",
    "collisions_layout": "static",
}
# Failures without a recorded time or place: negatives from these runs could be mislabeled.
UNTIMED_FAILURE_KEYS = ("outside_route_lanes", "vehicle_blocked")
LOCATION_RE = re.compile(r"type=(?P<actor>[^\s]+) and id=\d+ at \(x=(?P<x>[-\d.]+), y=(?P<y>[-\d.]+)")

N_PATH, N_SPEED, N_COMMAND = 10, 8, 6
FEATURE_NAMES = (
    [f"path_{i}_{a}" for i in range(N_PATH) for a in ("x", "y")]
    + [f"speed_prob_{i}" for i in range(N_SPEED)]
    + ["target_speed_scalar", "steer", "throttle", "brake",
       "ego_speed", "acceleration", "yaw_rate",
       "target_point_x", "target_point_y"]
    + [f"command_{i}" for i in range(N_COMMAND)]
)
TURN_COMMANDS, LANE_CHANGE_COMMANDS = {0, 1}, {4, 5}


@dataclass
class Run:
    run_dir: Path
    run_name: str
    seed: int
    route_id: str
    town: str
    scenarios: str
    log_path: Path
    status: str
    untimed_failures: list[str]
    features: np.ndarray        # [T, F]
    timestamps: np.ndarray      # [T]
    steps: np.ndarray           # [T]
    pose: np.ndarray            # [T, 3] world x, y, yaw
    commands: np.ndarray        # [T]
    collisions: list[dict] = field(default_factory=list)


def parse_run_spec(spec: str) -> tuple[Path, int]:
    path, _, seed = spec.partition(":")
    if seed:
        return Path(path), int(seed)
    match = re.search(r"seed(\d+)", Path(path).name)
    return Path(path), int(match.group(1)) if match else DEFAULT_SEED


def load_routes(xml_path: Path) -> dict[str, dict]:
    routes = {}
    for route in ET.parse(xml_path).getroot().iter("route"):
        first = route.find("waypoints")[0]
        scenarios = route.find("scenarios")
        routes[route.get("id")] = {
            "town": route.get("town", ""),
            "start": (float(first.get("x")), float(first.get("y"))),
            "scenarios": "|".join(sorted({s.get("type") for s in scenarios})) if scenarios is not None else "",
        }
    return routes


def step_features(record: dict) -> list[float]:
    path = record["predicted_path"]
    probs = record["target_speed_probabilities"]
    if len(path) != N_PATH or len(probs) != N_SPEED:
        raise ValueError(f"unexpected shapes path={len(path)} probs={len(probs)}")
    command = [0.0] * N_COMMAND
    command[int(record["command"])] = 1.0
    return ([v for point in path for v in point] + list(probs)
            + [record["target_speed_scalar"], record["steer"], record["throttle"], record["brake"],
               record["ego_speed"], record["acceleration"], record["yaw_rate"]]
            + list(record["target_point"]) + command)


def pick_log(log_dir: Path, record_name: str) -> Path | None:
    exact = log_dir / f"{record_name}.jsonl"
    if exact.exists():
        return exact
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def load_run(run_dir: Path, seed: int, checkpoint: Path, routes: dict, max_match_dist: float,
             skipped: list) -> Run | None:
    route_id = checkpoint.stem
    where = f"{run_dir.name}/{route_id}"
    try:
        records = json.loads(checkpoint.read_text())["_checkpoint"]["records"]
    except (json.JSONDecodeError, KeyError) as exc:
        skipped.append((where, f"unreadable checkpoint: {exc}"))
        return None
    if not records:
        skipped.append((where, "no route record: run did not finish"))
        return None
    record = records[0]
    if route_id not in routes:
        skipped.append((where, "route not in routes XML"))
        return None
    log_path = pick_log(run_dir / "logs" / route_id, record["timestamp"])
    if log_path is None:
        skipped.append((where, "no JSONL log"))
        return None

    rows = [json.loads(line) for line in log_path.open() if line.strip()]
    rows = rows[1:]  # first record has acceleration = yaw_rate = 0 by construction
    if not rows:
        skipped.append((where, "empty log"))
        return None
    steps = np.array([r["step"] for r in rows])
    if np.any(np.diff(steps) != 1):
        skipped.append((where, "non-contiguous steps"))
        return None

    route = routes[route_id]
    offset = np.array(route["start"]) - np.array(rows[0]["ego_position"])
    pose = np.array([[*(np.array(r["ego_position"]) + offset), r["ego_yaw"]] for r in rows])
    run = Run(
        run_dir=run_dir, run_name=run_dir.name, seed=seed, route_id=route_id,
        town=route["town"], scenarios=route["scenarios"], log_path=log_path,
        status=record.get("status", ""),
        untimed_failures=[k for k in UNTIMED_FAILURE_KEYS if record["infractions"].get(k)],
        features=np.array([step_features(r) for r in rows], dtype=np.float32),
        timestamps=np.array([r["timestamp"] for r in rows]),
        steps=steps, pose=pose,
        commands=np.array([int(r["command"]) for r in rows]),
    )

    for key, failure_type in COLLISION_KEYS.items():
        for text in record["infractions"].get(key, []):
            match = LOCATION_RE.search(text)
            event = {"failure_type": failure_type, "actor": match.group("actor") if match else "",
                     "x": float(match.group("x")) if match else math.nan,
                     "y": float(match.group("y")) if match else math.nan,
                     "index": -1, "match_dist": math.nan}
            if match:
                dist = np.hypot(pose[:, 0] - event["x"], pose[:, 1] - event["y"])
                event["match_dist"] = float(dist.min())
                if event["match_dist"] <= max_match_dist:
                    event["index"] = int(np.argmax(dist <= dist.min() + 0.5))
            run.collisions.append(event)
    run.collisions.sort(key=lambda e: (e["index"] < 0, e["index"]))
    return run


def maneuver_tag(run: Run, start: int, end: int, curve_yaw_rate: float) -> str:
    feats = run.features[start:end + 1]
    speed = feats[:, FEATURE_NAMES.index("ego_speed")]
    if speed.mean() < 0.5:
        return "stopped"
    commands = set(run.commands[start:end + 1].tolist())
    if commands & TURN_COMMANDS:
        return "turn"
    if commands & LANE_CHANGE_COMMANDS:
        return "lane_change"
    if np.abs(feats[:, FEATURE_NAMES.index("yaw_rate")]).max() > curve_yaw_rate:
        return "curve"
    return "straight"


def split_routes(runs: list[Run], fractions: list[float], seed: int) -> dict[str, str]:
    failing = defaultdict(bool)
    for run in runs:
        failing[run.route_id] |= any(e["index"] >= 0 for e in run.collisions)
    rng = random.Random(seed)
    assignment = {}
    for group in (sorted(r for r, f in failing.items() if f), sorted(r for r, f in failing.items() if not f)):
        rng.shuffle(group)
        n_train = round(fractions[0] * len(group))
        n_val = round(fractions[1] * len(group))
        for i, route in enumerate(group):
            assignment[route] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=None,
                        help="Run directory, optionally DIR:SEED (repeatable). Seed defaults to 'seedN' in the "
                             f"directory name, else {DEFAULT_SEED}. Default: {', '.join(str(p) for p in DEFAULT_RUNS)}")
    parser.add_argument("--routes-xml", type=Path, default=DEFAULT_ROUTES_XML)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "monitor_dataset")
    parser.add_argument("--history", type=int, default=60, help="Input steps per window (60 = 3 s at 20 Hz).")
    parser.add_argument("--horizon", type=float, default=3.0, help="Label = 1 if a collision occurs within this many seconds.")
    parser.add_argument("--pos-stride", type=int, default=1, help="Stride for positive windows.")
    parser.add_argument("--hard-neg-stride", type=int, default=5,
                        help="Stride for negatives during turns, lane changes, curves, or up to 2x horizon before a collision.")
    parser.add_argument("--neg-stride", type=int, default=20, help="Stride for other moving negatives.")
    parser.add_argument("--stopped-stride", type=int, default=100, help="Stride for negatives where the car is stopped.")
    parser.add_argument("--sparse-eval", action="store_true",
                        help="Apply the strides to val/test too. By default val/test keep every window so alarm "
                             "rates match real driving.")
    parser.add_argument("--curve-yaw-rate", type=float, default=0.15, help="|yaw_rate| (rad/s) that tags a window as a curve.")
    parser.add_argument("--max-match-dist", type=float, default=3.0,
                        help="Max distance (m) between ego path and collision location to accept a collision time.")
    parser.add_argument("--after-first-collision", choices=["drop", "keep"], default="drop",
                        help="drop: windows ending at or after a run's first collision are discarded. "
                             "keep: only windows whose history contains a collision are discarded.")
    parser.add_argument("--keep-untimed-failure-runs", action="store_true",
                        help=f"Keep runs with {', '.join(UNTIMED_FAILURE_KEYS)} (failure time unknown, so negatives may be wrong).")
    parser.add_argument("--keep-duplicate-runs", action="store_true",
                        help="Keep repeated (route, seed) runs instead of only the first given.")
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    run_specs = [parse_run_spec(s) for s in args.run] if args.run else [(p, parse_run_spec(str(p))[1]) for p in DEFAULT_RUNS]
    routes = load_routes(args.routes_xml)
    H = args.history

    runs: list[Run] = []
    skipped: list[tuple[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for run_dir, seed in run_specs:
        checkpoints = sorted(p for p in (run_dir / "checkpoints").glob("*.json"))
        if not checkpoints:
            print(f"warning: no checkpoints in {run_dir}", file=sys.stderr)
        for checkpoint in checkpoints:
            key = (checkpoint.stem, seed)
            if key in seen and not args.keep_duplicate_runs:
                skipped.append((f"{run_dir.name}/{checkpoint.stem}", f"duplicate (route, seed): {key[0]}, {seed}"))
                continue
            run = load_run(run_dir, seed, checkpoint, routes, args.max_match_dist, skipped)
            if run is None:
                continue
            if run.untimed_failures and not args.keep_untimed_failure_runs:
                skipped.append((f"{run_dir.name}/{run.route_id}", f"untimed failure: {','.join(run.untimed_failures)}"))
                continue
            if any(e["index"] < 0 for e in run.collisions):
                skipped.append((f"{run_dir.name}/{run.route_id}", "unmatched collision: no step within --max-match-dist"))
                continue
            seen.add(key)
            runs.append(run)

    assignment = split_routes(runs, args.split, args.split_seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Collisions and pre-collision segments.
    collision_rows, pre_X, pre_pose, pre_ids = [], [], [], []
    for run in runs:
        for k, event in enumerate(run.collisions):
            i = event["index"]
            event_id = f"{run.run_name}/{run.route_id}/s{run.seed}/c{k}"
            has_history = i >= H
            collision_rows.append({
                "event_id": event_id, "run": run.run_name, "route_id": run.route_id, "seed": run.seed,
                "split": assignment[run.route_id], "town": run.town, "scenarios": run.scenarios,
                "collision_index_in_run": k, "is_first_collision": k == 0,
                "failure_type": event["failure_type"], "actor": event["actor"],
                "x": event["x"], "y": event["y"], "match_dist_m": round(event["match_dist"], 3),
                "step": int(run.steps[i]), "timestamp": round(float(run.timestamps[i]), 3),
                "ego_speed": round(float(run.features[i, FEATURE_NAMES.index("ego_speed")]), 3),
                "has_full_history": has_history,
            })
            if has_history:
                pre_X.append(run.features[i - H:i])
                pre_pose.append(run.pose[i - H:i])
                pre_ids.append(event_id)

    # Sliding windows.
    windows = {"train": [], "val": [], "test": []}
    window_rows = []
    hard_tags = {"turn", "lane_change", "curve"}
    for run in runs:
        split = assignment[run.route_id]
        coll_idx = [e["index"] for e in run.collisions]
        coll_t = [run.timestamps[i] for i in coll_idx]
        first_coll = coll_idx[0] if coll_idx else len(run.steps)
        for end in range(H - 1, len(run.steps)):
            start = end - H + 1
            if args.after_first_collision == "drop" and end >= first_coll:
                break
            if args.after_first_collision == "keep" and any(start <= c <= end for c in coll_idx):
                continue
            t_end = run.timestamps[end]
            nxt = next((k for k, t in enumerate(coll_t) if t > t_end), None)
            ttc = float(coll_t[nxt] - t_end) if nxt is not None else math.nan
            label = int(nxt is not None and ttc <= args.horizon + 1e-6)
            tag = maneuver_tag(run, start, end, args.curve_yaw_rate)
            pre_buffer = not label and ttc <= 2 * args.horizon
            if label:
                stride = args.pos_stride
            elif tag in hard_tags or pre_buffer:
                stride = args.hard_neg_stride
            elif tag == "stopped":
                stride = args.stopped_stride
            else:
                stride = args.neg_stride
            if split != "train" and not args.sparse_eval:
                stride = 1
            if (end - (H - 1)) % stride:
                continue
            window_id = len(window_rows)
            windows[split].append((window_id, run.features[start:end + 1], label))
            window_rows.append({
                "window_id": window_id, "split": split, "run": run.run_name, "route_id": run.route_id,
                "seed": run.seed, "town": run.town, "scenarios": run.scenarios, "log_file": run.log_path.name,
                "start_step": int(run.steps[start]), "end_step": int(run.steps[end]),
                "start_time": round(float(run.timestamps[start]), 3), "end_time": round(float(t_end), 3),
                "label": label, "time_to_collision": round(ttc, 3) if not math.isnan(ttc) else "",
                "failure_type": run.collisions[nxt]["failure_type"] if label else "",
                "maneuver": tag, "pre_failure_buffer": pre_buffer,
                "end_ego_speed": round(float(run.features[end, FEATURE_NAMES.index("ego_speed")]), 3),
                "moving": bool(run.features[end, FEATURE_NAMES.index("ego_speed")] > 2.0),
                "run_status": run.status,
            })

    F = len(FEATURE_NAMES)
    counts = {}
    for split, items in windows.items():
        X = np.stack([w[1] for w in items]).astype(np.float32) if items else np.zeros((0, H, F), np.float32)
        y = np.array([w[2] for w in items], dtype=np.int8)
        ids = np.array([w[0] for w in items], dtype=np.int64)
        np.savez_compressed(args.out / f"{split}.npz", X=X, y=y, window_id=ids)
        counts[split] = {"windows": int(len(y)), "positives": int(y.sum()),
                         "routes": len({r for r, s in assignment.items() if s == split})}
        if split == "train" and len(y):
            flat = X.reshape(-1, F)
            mean, std = flat.mean(0), flat.std(0)
            std[std < 1e-6] = 1.0

    np.savez_compressed(
        args.out / "pre_collision.npz",
        X=np.stack(pre_X).astype(np.float32) if pre_X else np.zeros((0, H, F), np.float32),
        ego_pose=np.stack(pre_pose).astype(np.float32) if pre_pose else np.zeros((0, H, 3), np.float32),
        event_id=np.array(pre_ids))

    for name, rows in (("windows.csv", window_rows), ("collisions.csv", collision_rows)):
        with (args.out / name).open("w", newline="") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    meta = {
        "feature_names": FEATURE_NAMES,
        "history_steps": H, "dt": 0.05, "horizon_s": args.horizon,
        "label": f"1 if a collision (vehicle, pedestrian, static) occurs within {args.horizon} s after the last step",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "runs": [f"{r.run_name}/{r.route_id}/seed{r.seed}" for r in runs],
        "skipped_runs": [{"run": w, "reason": why} for w, why in skipped],
        "splits": {s: sorted(r for r, a in assignment.items() if a == s) for s in ("train", "val", "test")},
        "counts": counts,
        "normalization_train": {"mean": mean.tolist(), "std": std.tolist()} if counts.get("train", {}).get("windows") else None,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))

    print(f"runs used: {len(runs)}  skipped: {len(skipped)}")
    for why, n in Counter(why.split(":")[0] for _, why in skipped).items():
        print(f"  skipped ({why}): {n}")
    matched = collision_rows
    print(f"collisions: {len(matched)}  with full {H}-step history: {len(pre_ids)}  "
          f"max match dist: {max((r['match_dist_m'] for r in matched), default=0):.2f} m")
    for split, c in counts.items():
        print(f"  {split}: {c['routes']} routes, {c['windows']} windows, {c['positives']} positive")
    print(f"maneuver tags (negatives): {dict(Counter(r['maneuver'] for r in window_rows if not r['label']))}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
