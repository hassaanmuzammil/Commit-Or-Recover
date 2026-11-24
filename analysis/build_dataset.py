#!/usr/bin/env python3
"""Turn TF++ temporal logs + infraction events into per-step tables.

For every results directory produced by tools/run_bench2drive_routes.py
(<run>/logs/<route_id>/<x>.jsonl, events_<x>.json, <run>/checkpoints/<id>.json)
this writes:

  <out>/steps.parquet   one row per logged step: raw outputs, temporal features,
                        privileged context, time-to-failure and 2 s / 3 s labels
  <out>/events.parquet  one row per infraction event, aligned to a log step
  <out>/routes.parquet  one row per route run: outcome, scores, scenario metadata

Usage:
  python analysis/build_dataset.py carla_garage/results/selected_seed100 [...] --out analysis/data
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / 'carla_garage' / 'leaderboard' / 'data' / 'bench2drive_split'

# Events that count as a "failure" for the monitor (proposal Sec. 3.3).
FAILURE_EVENTS = {
    'COLLISION_STATIC': 'collision_static',
    'COLLISION_VEHICLE': 'collision_vehicle',
    'COLLISION_PEDESTRIAN': 'collision_pedestrian',
    'OUTSIDE_ROUTE_LANES_INFRACTION': 'off_route_lanes',
    'ON_SIDEWALK_INFRACTION': 'off_road',
    'ROUTE_DEVIATION': 'route_deviation',
    'VEHICLE_BLOCKED': 'blocked',
}
# Recorded for analysis but not part of the failure label.
OTHER_EVENTS = {
    'TRAFFIC_LIGHT_INFRACTION': 'red_light',
    'STOP_INFRACTION': 'stop_sign',
    'WRONG_WAY_INFRACTION': 'wrong_way',
    'MIN_SPEED_INFRACTION': 'min_speed',
    'YIELD_TO_EMERGENCY_VEHICLE': 'yield_emergency',
    'SCENARIO_TIMEOUT': 'scenario_timeout',
}
COMMANDS = {0: 'left', 1: 'right', 2: 'straight', 3: 'lanefollow', 4: 'changelaneleft', 5: 'changelaneright'}


def route_metadata(route_id: str) -> dict:
  """Town, weather and scenario list from the one-route split XML."""
  for xml_path in SPLIT_DIR.glob('*.xml'):
    root = ET.parse(xml_path).getroot()
    node = root.find('route')
    if node is not None and node.attrib.get('id') == str(route_id):
      scen = node.findall('./scenarios/scenario')
      weather = node.find('./weathers/weather')
      triggers = []
      for s in scen:
        tp = s.find('trigger_point')
        if tp is not None:
          triggers.append((s.attrib.get('type'), float(tp.attrib['x']), float(tp.attrib['y'])))
      return {
          'split_file': xml_path.name,
          'town': node.attrib.get('town'),
          'scenario_types': ','.join(sorted({s.attrib.get('type', '?') for s in scen})),
          'weather_cloudiness': float(weather.attrib.get('cloudiness', 'nan')) if weather is not None else np.nan,
          'weather_precip': float(weather.attrib.get('precipitation', 'nan')) if weather is not None else np.nan,
          'weather_sun_alt': float(weather.attrib.get('sun_altitude_angle', 'nan')) if weather is not None else np.nan,
          'triggers': triggers,
      }
  return {'split_file': None, 'town': None, 'scenario_types': None, 'triggers': []}


# ---------------------------------------------------------------- features
def _entropy(p):
  p = np.clip(p, 1e-12, 1.0)
  return float(-(p * np.log(p)).sum())


def _js(p, q):
  p = np.clip(p, 1e-12, 1.0); q = np.clip(q, 1e-12, 1.0)
  m = 0.5 * (p + q)
  return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def _to_prev_frame(path_prev, dx, dy, dyaw):
  """Express the previous step's ego-frame path in the current ego frame.

  dx, dy: ego displacement (previous ego frame); dyaw: heading change. TF++ paths
  are in the ego frame with x forward, y right.
  """
  shifted = path_prev - np.array([dx, dy])
  c, s = np.cos(-dyaw), np.sin(-dyaw)
  rot = np.array([[c, -s], [s, c]])
  return shifted @ rot.T


def _lateral_disagreement(prev_aligned, path):
  """|lateral(prev) - lateral(cur)| at the current path's forward distances (overlap only)."""
  order = np.argsort(prev_aligned[:, 0])
  px, py = prev_aligned[order, 0], prev_aligned[order, 1]
  keep = (path[:, 0] >= px[0]) & (path[:, 0] <= px[-1])
  if not keep.any():
    return np.array([])
  return np.abs(np.interp(path[keep, 0], px, py) - path[keep, 1])


def step_features(records: list[dict]) -> pd.DataFrame:
  rows = []
  prev = None
  for rec in records:
    path = np.asarray(rec['predicted_path'], dtype=float)
    probs = np.asarray(rec['target_speed_probabilities'], dtype=float)
    priv = rec.get('privileged') or {}
    row = {
        'step': rec['step'], 'timestamp': rec['timestamp'], 'carla_frame': rec.get('carla_frame'),
        'steer': rec['steer'], 'throttle': rec['throttle'], 'brake': rec['brake'],
        'ego_speed': rec['ego_speed'], 'acceleration': rec['acceleration'], 'yaw_rate': rec['yaw_rate'],
        'target_speed_scalar': rec['target_speed_scalar'], 'command': COMMANDS.get(rec['command'], rec['command']),
        'tp_x': rec['target_point'][0], 'tp_y': rec['target_point'][1],
        'tp_dist': float(np.hypot(*rec['target_point'][:2])),
        'tp_angle': float(np.degrees(np.arctan2(rec['target_point'][1], rec['target_point'][0]))),
        # single-frame confidence (baseline signals)
        'speed_entropy': _entropy(probs),
        'speed_top_prob': float(probs.max()),
        'speed_margin': float(np.sort(probs)[-1] - np.sort(probs)[-2]) if probs.size > 1 else 1.0,
        'speed_argmax': int(probs.argmax()),
        # path geometry
        'path_end_lat': float(path[-1, 1]),
        'path_end_heading': float(np.degrees(np.arctan2(path[-1, 1] - path[-2, 1], path[-1, 0] - path[-2, 0]))),
        'path_mid_lat': float(path[len(path) // 2, 1]),
        'path_curvature': float(np.abs(np.diff(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])))).sum()),
    }
    for key in ('x', 'y', 'speed', 'is_junction', 'road_id', 'lane_id', 'lane_width', 'lateral_offset',
                'heading_error_deg', 'nearest_any_dist', 'nearest_any_type', 'nearest_front_dist',
                'nearest_front_type', 'actors_within_20m'):
      row[f'gt_{key}'] = priv.get(key)
    if prev is not None:
      p_rec, p_path, p_probs = prev
      # ego motion between steps from privileged GT if available, else from logged state
      yaw0, yaw1 = p_rec['ego_yaw'], rec['ego_yaw']
      dyaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
      wx = rec['ego_position'][0] - p_rec['ego_position'][0]
      wy = rec['ego_position'][1] - p_rec['ego_position'][1]
      c, s = np.cos(-yaw0), np.sin(-yaw0)
      dx, dy = c * wx - s * wy, s * wx + c * wy
      # TF++ paths are resampled at fixed distances ahead of the ego, so compare
      # lateral offsets at matching forward distances after the rigid transform.
      diff = _lateral_disagreement(_to_prev_frame(p_path, dx, dy, dyaw), path)
      row['path_disagree_mean'] = float(diff.mean()) if diff.size else np.nan
      row['path_disagree_end'] = float(diff[-1]) if diff.size else np.nan
      row['speed_js'] = _js(p_probs, probs) if p_probs.shape == probs.shape else np.nan
      row['steer_rate'] = (rec['steer'] - p_rec['steer']) / rec['dt']
      row['throttle_brake_switch'] = float((rec['brake'] > 0.1) != (p_rec['brake'] > 0.1))
    rows.append(row)
    prev = (rec, path, probs)
  df = pd.DataFrame(rows)
  if df.empty:
    return df
  dt = records[0]['dt']
  # rolling temporal-instability features over 1 s and 2 s windows (causal)
  for secs in (1, 2):
    w = max(2, int(round(secs / dt)))
    sign = np.sign(df['steer'].where(df['steer'].abs() > 0.02, 0.0))
    reversals = (sign.diff().abs() == 2).astype(float)
    df[f'steer_reversals_{secs}s'] = reversals.rolling(w, min_periods=1).sum()
    df[f'steer_tv_{secs}s'] = df['steer'].diff().abs().rolling(w, min_periods=1).sum()
    df[f'path_disagree_mean_{secs}s'] = df['path_disagree_mean'].rolling(w, min_periods=1).mean()
    df[f'path_disagree_max_{secs}s'] = df['path_disagree_mean'].rolling(w, min_periods=1).max()
    df[f'path_end_lat_std_{secs}s'] = df['path_end_lat'].rolling(w, min_periods=2).std()
    df[f'speed_js_mean_{secs}s'] = df['speed_js'].rolling(w, min_periods=1).mean()
    df[f'speed_entropy_mean_{secs}s'] = df['speed_entropy'].rolling(w, min_periods=1).mean()
    df[f'speed_argmax_changes_{secs}s'] = (df['speed_argmax'].diff().fillna(0) != 0).astype(float).rolling(
        w, min_periods=1).sum()
    df[f'tb_switches_{secs}s'] = df['throttle_brake_switch'].rolling(w, min_periods=1).sum()
    df[f'jerk_abs_{secs}s'] = (df['acceleration'].diff().abs() / dt).rolling(w, min_periods=1).mean()
  return df


# ---------------------------------------------------------------- per route
def load_route(run_dir: Path, route_dir: Path):
  route_id = route_dir.name
  logs = sorted(route_dir.glob('*.jsonl'))
  if len(logs) != 1:
    return None
  records = [json.loads(line) for line in logs[0].open() if line.strip()]
  if not records:
    return None
  events_path = next(iter(route_dir.glob('events_*.json')), None)
  ckpt = run_dir / 'checkpoints' / f'{route_id}.json'
  recs = json.loads(ckpt.read_text()).get('_checkpoint', {}).get('records', []) if ckpt.exists() else []
  if events_path is None or not recs:
    return None  # route still running or crashed before statistics were written
  events = json.loads(events_path.read_text())['events']
  record = recs[-1]

  df = step_features(records)
  meta = route_metadata(route_id)
  run_name = run_dir.name
  df.insert(0, 'run', run_name)
  df.insert(1, 'route_id', route_id)

  # align events to steps by CARLA frame
  ev_rows = []
  frames = df['carla_frame'].to_numpy(dtype=float) if df['carla_frame'].notna().all() else None
  for ev in events:
    etype = ev['type']
    if etype not in FAILURE_EVENTS and etype not in OTHER_EVENTS:
      continue
    idx = None
    if frames is not None and ev.get('frame') is not None:
      idx = int(np.clip(np.searchsorted(frames, ev['frame']), 0, len(df) - 1))
    loc = (ev.get('dict') or {}).get('location') or {}
    other = (ev.get('dict') or {}).get('other_actor') or {}
    ev_rows.append({
        'run': run_name, 'route_id': route_id, 'type': etype,
        'kind': FAILURE_EVENTS.get(etype) or OTHER_EVENTS.get(etype),
        'is_failure': etype in FAILURE_EVENTS, 'frame': ev.get('frame'),
        'step_index': idx, 'timestamp': df['timestamp'].iloc[idx] if idx is not None else np.nan,
        'x': loc.get('x') if isinstance(loc, dict) else None, 'y': loc.get('y') if isinstance(loc, dict) else None,
        'other_actor_type': other.get('type_id') if isinstance(other, dict) else None,
        'message': ev.get('message'),
    })
  ev_df = pd.DataFrame(ev_rows)

  # time-to-first-failure labels
  fail_times = sorted(ev_df.loc[ev_df['is_failure'], 'timestamp'].dropna()) if not ev_df.empty else []
  first_fail = fail_times[0] if fail_times else np.nan
  df['time_to_failure'] = first_fail - df['timestamp'] if fail_times else np.nan
  for horizon in (2, 3):
    ttf = df['time_to_failure']
    df[f'label_{horizon}s'] = ((ttf >= 0) & (ttf <= horizon)).astype(int)
  df['post_failure'] = (df['time_to_failure'] < 0).astype(int) if fail_times else 0

  # distance to nearest scenario trigger point (GT position)
  if meta['triggers'] and df['gt_x'].notna().any():
    tx = np.array([[t[1], t[2]] for t in meta['triggers']])
    pos = df[['gt_x', 'gt_y']].to_numpy(dtype=float)
    d = np.linalg.norm(pos[:, None, :] - tx[None, :, :], axis=2)
    df['dist_to_trigger'] = d.min(axis=1)

  # Route ended by a time limit with no failure event (e.g. Bench2Drive "TickRuntime"):
  # the policy got stuck. Kept as a route-level outcome; there is no step to label.
  status = str(record.get('status', ''))
  stuck = (not fail_times) and any(k in status for k in ('TickRuntime', 'timed out', 'Route timeout'))

  route_row = {
      'run': run_name, 'route_id': route_id, 'status': record.get('status'),
      'stuck_timeout': stuck,
      'score_route': record.get('scores', {}).get('score_route'),
      'score_penalty': record.get('scores', {}).get('score_penalty'),
      'score_composed': record.get('scores', {}).get('score_composed'),
      'duration_game': record.get('meta', {}).get('duration_game'),
      'route_length': record.get('meta', {}).get('route_length'),
      'n_steps': len(df), 'n_failure_events': int(ev_df['is_failure'].sum()) if not ev_df.empty else 0,
      'first_failure_kind': (ev_df.loc[ev_df['is_failure']].sort_values('timestamp')['kind'].iloc[0]
                             if not ev_df.empty and ev_df['is_failure'].any() else
                             ('stuck_timeout' if stuck else None)),
      'first_failure_time': first_fail,
      **{k: v for k, v in meta.items() if k != 'triggers'},
  }
  return df, ev_df, route_row


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('runs', nargs='+', type=Path)
  parser.add_argument('--out', type=Path, default=REPO / 'analysis' / 'data')
  parser.add_argument('--families', type=Path, default=None,
                      help='Optional JSON {route_id: family}; defaults to the selected-route manifest.')
  args = parser.parse_args()

  import sys
  sys.path.insert(0, str(REPO / 'carla_garage' / 'tools'))
  from run_bench2drive_routes import ROUTES  # pylint: disable=import-outside-toplevel
  family = {r.route_id: r.family for r in ROUTES}
  focus = {r.route_id: r.scenario for r in ROUTES}

  steps, events, routes = [], [], []
  for run_dir in args.runs:
    for route_dir in sorted((run_dir / 'logs').iterdir()):
      out = load_route(run_dir, route_dir)
      if out is None:
        print(f'skip {route_dir} (no log)')
        continue
      df, ev_df, row = out
      steps.append(df); events.append(ev_df); routes.append(row)
  routes_df = pd.DataFrame(routes)
  # A route at a given traffic seed may exist in several runs (the 'all' suite repeats the
  # selected routes). Keep the first run listed so each (route, seed) is counted once.
  routes_df['seed'] = routes_df['run'].str.extract(r'seed(\d+)', expand=False).astype(float)
  dup = routes_df.duplicated(['route_id', 'seed'], keep='first')
  if dup.any():
    print(f'dropping {int(dup.sum())} duplicate (route, seed) runs:',
          ', '.join(routes_df.loc[dup, 'run'] + '/' + routes_df.loc[dup, 'route_id']))
    drop_keys = set(zip(routes_df.loc[dup, 'run'], routes_df.loc[dup, 'route_id']))
    keep = lambda d: d[[(r, i) not in drop_keys for r, i in zip(d['run'], d['route_id'])]]
    steps = [keep(d) for d in steps]
    events = [keep(e) if not e.empty else e for e in events]
    routes_df = routes_df[~dup].reset_index(drop=True)
  routes_df['family'] = routes_df['route_id'].map(family).fillna('other')
  routes_df['focus_scenario'] = routes_df['route_id'].map(focus)
  args.out.mkdir(parents=True, exist_ok=True)
  pd.concat(steps, ignore_index=True).to_parquet(args.out / 'steps.parquet')
  (pd.concat(events, ignore_index=True) if events else pd.DataFrame()).to_parquet(args.out / 'events.parquet')
  routes_df.to_parquet(args.out / 'routes.parquet')
  print(routes_df[['run', 'route_id', 'family', 'status', 'score_route', 'first_failure_kind']].to_string())


if __name__ == '__main__':
  main()
