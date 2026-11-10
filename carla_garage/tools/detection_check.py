#!/usr/bin/env python3
"""Gate 2 detection check: did TF++ have a predicted box on the actor it later crashed into?

Reads schema-v2 temporal logs (pred_boxes + gt_actors + collisions). For every run with a
collision, takes the FIRST collision, finds the hit actor's ground-truth box at 3, 2 and 1 s
before impact and checks whether a TF++ box of a compatible class lies on it.

    python tools/detection_check.py results/scene_focus_seed100 results/scene_focus_seed200 ... \
        --out results/detection_check
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BEV_HALF_RANGE = 32.0  # TF++ BEV is [-32, 32] m in x and y (config.min_x/max_x/min_y/max_y).
VEHICLE_CLASSES = (0, 4)  # car, emergency vehicle
WALKER_CLASSES = (1,)


def load_records(path):
  with path.open(encoding='utf-8') as stream:
    return [json.loads(line) for line in stream if line.strip()]


def compatible(box_class, gt_class):
  box_class = int(round(box_class))
  if gt_class == 'vehicle':
    return box_class in VEHICLE_CLASSES
  if gt_class == 'walker':
    return box_class in WALKER_CLASSES
  return box_class in VEHICLE_CLASSES + WALKER_CLASSES  # static props: any dynamic-object box counts


def angle_diff_mod_pi(a, b):
  """Heading error ignoring front/back flips (box yaw is ambiguous by pi)."""
  d = (a - b + np.pi / 2) % np.pi - np.pi / 2
  return abs(d)


def match_at(record, actor_id, radius):
  """Return (gt_actor or None, best matching box or None, distance to best box)."""
  gt = next((a for a in record['gt_actors'] if a['id'] == actor_id), None)
  if gt is None:
    return None, None, np.inf
  best, best_d = None, np.inf
  for box in record['pred_boxes']:
    if not compatible(box[7], gt['class']):
      continue
    d = float(np.hypot(box[0] - gt['x'], box[1] - gt['y']))
    if d < best_d:
      best, best_d = box, d
  if best_d > radius:
    return gt, None, best_d
  return gt, best, best_d


def analyse_run(path, radius, horizons, half_window):
  records = load_records(path)
  if not records or records[0].get('schema_version', 1) < 2:
    return None
  crash_index = next((i for i, r in enumerate(records) if r['collisions']), None)
  if crash_index is None:
    return {'run': str(path), 'crashed': False}

  collision = records[crash_index]['collisions'][0]
  actor_id = collision['other_id']
  dt = float(records[0]['dt'])
  row = {
      'run': str(path.parent.parent.parent.name) + '/' + path.parent.name,
      'crashed': True,
      'crash_step': records[crash_index]['step'],
      'crash_time_s': round(crash_index * dt, 2),
      'hit_id': actor_id,
      'hit_type': collision['other_type'],
      'ego_speed_at_crash': round(records[crash_index]['ego_speed'], 2),
  }
  # Class of the hit actor as seen in GT (static props etc.).
  gt_classes = [a['class'] for r in records[max(0, crash_index - 5):crash_index + 1]
                for a in r['gt_actors'] if a['id'] == actor_id]
  row['hit_class'] = gt_classes[-1] if gt_classes else 'unknown'

  # Per-frame detection flag for the hit actor over the whole run.
  detected = np.zeros(len(records), dtype=bool)
  present = np.zeros(len(records), dtype=bool)
  for i in range(crash_index + 1):
    gt, box, _ = match_at(records[i], actor_id, radius)
    present[i] = gt is not None
    detected[i] = box is not None

  for h in horizons:
    key = f'{h:g}s'
    i = crash_index - int(round(h / dt))
    if i < 0:
      row[f'det_{key}'] = 'n/a'
      continue
    gt, box, d = match_at(records[i], actor_id, radius)
    if gt is None:
      row[f'det_{key}'] = 'not_in_gt'
      continue
    lo, hi = max(0, i - half_window), min(crash_index, i + half_window)
    window_fraction = float(detected[lo:hi + 1].mean())
    in_range = abs(gt['x']) <= BEV_HALF_RANGE and abs(gt['y']) <= BEV_HALF_RANGE
    if window_fraction >= 0.5:
      status = 'hit'
    elif not in_range:
      status = 'out_of_range'
    else:
      status = 'miss'
    row[f'det_{key}'] = status
    row[f'frac_{key}'] = round(window_fraction, 2)
    row[f'range_{key}'] = round(float(np.hypot(gt['x'], gt['y'])), 1)
    row[f'gt_speed_{key}'] = round(gt['speed'], 2)
    row[f'dist_{key}'] = round(d, 2) if np.isfinite(d) else None
    if box is not None:
      row[f'score_{key}'] = round(box[8], 2)
      row[f'yaw_err_deg_{key}'] = round(float(np.degrees(angle_diff_mod_pi(box[4], gt['yaw']))), 1)

  # Continuous track lead: how long before impact the actor has been (mostly) detected,
  # allowing gaps up to half_window frames.
  lead_index, gap = crash_index, 0
  for i in range(crash_index, -1, -1):
    if detected[i]:
      lead_index, gap = i, 0
    else:
      gap += 1
      if gap > half_window:
        break
  row['track_lead_s'] = round((crash_index - lead_index) * dt, 2) if detected[lead_index] else 0.0
  return row


def summarise(rows, horizons):
  crashes = [r for r in rows if r.get('crashed')]
  lines = [f'runs: {len(rows)}   crashes: {len(crashes)}', '']
  lines.append('| horizon | hit | miss (in range) | out of BEV range | actor not in GT radius | n/a | hit rate (all crashes) |')
  lines.append('|---|---|---|---|---|---|---|')
  summary = {}
  for h in horizons:
    key = f'det_{h:g}s'
    counts = {s: sum(r.get(key) == s for r in crashes) for s in ('hit', 'miss', 'out_of_range', 'not_in_gt', 'n/a')}
    rate = counts['hit'] / len(crashes) if crashes else float('nan')
    summary[f'{h:g}s'] = dict(counts, hit_rate=rate)
    lines.append(f"| {h:g} s | {counts['hit']} | {counts['miss']} | {counts['out_of_range']} | "
                 f"{counts['not_in_gt']} | {counts['n/a']} | {rate:.0%} |")
  leads = [r['track_lead_s'] for r in crashes]
  if leads:
    lines.append('')
    lines.append(f'continuous track lead before impact: median {np.median(leads):.2f} s, '
                 f'>=2 s for {np.mean(np.asarray(leads) >= 2):.0%} of crashes')
  by_class = {}
  for r in crashes:
    by_class.setdefault(r['hit_class'], []).append(r)
  lines.append('')
  lines.append('hit rate at 2 s by hit-actor class: ' + ', '.join(
      f"{c}: {sum(x.get('det_2s') == 'hit' for x in v)}/{len(v)}" for c, v in sorted(by_class.items())))
  return '\n'.join(lines), summary


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('roots', nargs='+', type=Path, help='Result dirs containing logs/<route>/*.jsonl')
  parser.add_argument('--radius', type=float, default=2.0, help='Max centre distance (m) for a match.')
  parser.add_argument('--horizons', type=float, nargs='+', default=[3.0, 2.0, 1.0])
  parser.add_argument('--half-window', type=int, default=5, help='Frames either side for the majority vote.')
  parser.add_argument('--out', type=Path, default=None)
  args = parser.parse_args()

  rows = []
  for root in args.roots:
    for path in sorted(root.glob('logs/*/*.jsonl')):
      row = analyse_run(path, args.radius, args.horizons, args.half_window)
      if row is not None:
        rows.append(row)

  text, summary = summarise(rows, args.horizons)
  print(text)
  crashes = [r for r in rows if r.get('crashed')]
  for r in crashes:
    print(r['run'], r['hit_type'], ' '.join(f"{h:g}s={r.get(f'det_{h:g}s')}" for h in args.horizons),
          f"lead={r['track_lead_s']}s")

  if args.out:
    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in crashes for k in r})
    with (args.out / 'per_crash.csv').open('w', newline='', encoding='utf-8') as stream:
      writer = csv.DictWriter(stream, fieldnames=keys)
      writer.writeheader()
      writer.writerows(crashes)
    (args.out / 'summary.json').write_text(json.dumps(
        {'radius_m': args.radius, 'runs': len(rows), 'crashes': len(crashes), 'by_horizon': summary}, indent=2))
    (args.out / 'summary.md').write_text(text + '\n')


if __name__ == '__main__':
  main()
