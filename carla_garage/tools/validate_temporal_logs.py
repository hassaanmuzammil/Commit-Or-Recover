#!/usr/bin/env python3
"""Validate temporal SensorAgent JSONL logs produced during evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    'schema_version', 'step', 'timestamp', 'route_id', 'dt',
    'predicted_path', 'target_speed_probabilities', 'target_speed_scalar',
    'steer', 'throttle', 'brake', 'ego_speed', 'ego_position', 'ego_yaw',
    'acceleration', 'yaw_rate', 'target_point', 'command',
}


def validate_file(path: Path) -> int:
  records = []
  with path.open(encoding='utf-8') as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as error:
        raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
      missing = REQUIRED_KEYS - record.keys()
      if missing:
        raise ValueError(f'{path}:{line_number}: missing keys: {sorted(missing)}')
      records.append(record)

  if not records:
    raise ValueError(f'{path}: no records found')

  steps = np.asarray([record['step'] for record in records])
  timestamps = np.asarray([record['timestamp'] for record in records], dtype=float)
  dt = float(records[0]['dt'])
  for record in records:
    path_array = np.asarray(record['predicted_path'], dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] != 2:
      raise ValueError(f'{path}: invalid predicted_path shape {path_array.shape}')
    probabilities = np.asarray(record['target_speed_probabilities'], dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
      raise ValueError(f'{path}: invalid target-speed distribution shape {probabilities.shape}')
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
      raise ValueError(f'{path}: target-speed probabilities sum to {probabilities.sum()}')

  if len(records) > 1:
    step_diffs = np.diff(steps)
    if not np.all(step_diffs == 1):
      raise ValueError(f'{path}: non-consecutive steps: {step_diffs.tolist()}')
    time_diffs = np.diff(timestamps)
    if not np.allclose(time_diffs, dt, atol=1e-3):
      raise ValueError(f'{path}: timestamp spacing differs from dt={dt}: {time_diffs}')

  print(f'{path}: OK ({len(records)} records, '
        f'steps {int(steps[0])}-{int(steps[-1])}, dt={dt}, '
        f'path_shape={np.asarray(records[0]["predicted_path"]).shape}, '
        f'speed_bins={len(records[0]["target_speed_probabilities"])})')
  return len(records)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=Path, help='JSONL file or directory of JSONL files')
  args = parser.parse_args()

  paths = sorted(args.path.glob('*.jsonl')) if args.path.is_dir() else [args.path]
  if not paths:
    parser.error(f'No JSONL files found in {args.path}')

  total = sum(validate_file(path) for path in paths)
  print(f'Validated {len(paths)} file(s), {total} total records.')


if __name__ == '__main__':
  main()
