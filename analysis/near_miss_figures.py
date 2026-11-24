#!/usr/bin/env python3
"""Figures for Finding 7 (near-misses), in the style of docs/figures/f*.png.

f7a: high-speed false alarms look like pre-crash moments (ground-truth hazard shares, route-bootstrap CI)
f7b: warned >=1 s per monitor, crash-only vs near-miss counting (matched 2 alarms/min, route-bootstrap CI)
f7c: one near-miss case: monitor score, ground-truth TTC on TF++'s plan, speeds

Usage: python analysis/near_miss_figures.py [--data analysis/data/boxes_3seeds]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from near_miss_eval import load, per_route, summarize, thr_for_false_rate  # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BG, INK, MUTED, GRID, SPINE = '#fcfcfb', '#0b0b0b', '#8a8984', '#dcdbd6', '#bdbcb6'
BEST = 'tfpp+intent+map+boxes'
RNG = np.random.default_rng(0)


def style(ax):
  ax.set_facecolor(BG)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color(SPINE)
  ax.tick_params(colors=MUTED, labelsize=10)


def ratio_ci(num, den, routes, B=2000):
  """Route-bootstrap CI of sum(num)/sum(den); num/den are per-route arrays."""
  n = len(routes)
  vals = []
  for _ in range(B):
    w = np.bincount(RNG.integers(0, n, n), minlength=n)
    vals.append((w * num).sum() / max((w * den).sum(), 1e-9))
  return np.quantile(vals, [0.025, 0.975])


def thr_for_total_rate(o, score, rate):
  normal = o['normal'].to_numpy()
  lo, hi = float(np.nanmin(score[normal])), float(np.nanmax(score))
  for _ in range(40):
    mid = 0.5 * (lo + hi)
    s = summarize(per_route(o, score, mid))
    tot = s['false_onsets_per_min'] + s['justified_onsets_per_min']
    lo, hi = (mid, hi) if tot > rate else (lo, mid)
  return 0.5 * (lo + hi)


def hbars(ax, labels, series, xmax, xlabel):
  """series: list of (name, color, values, lo, hi). Grouped horizontal bars with whiskers + value labels."""
  n = len(labels)
  h = 0.36
  y = np.arange(n)[::-1] * 1.0
  for k, (name, col, v, lo, hi) in enumerate(series):
    yy = y + (0.5 - k) * h * 1.05
    ax.barh(yy, 100 * np.asarray(v), height=h, color=col, label=name, zorder=2)
    ax.errorbar(100 * np.asarray(v), yy, xerr=[100 * (np.asarray(v) - lo), 100 * (hi - np.asarray(v))],
                fmt='none', ecolor=INK, elinewidth=1.2, capsize=4, zorder=3)
    for vi, hii, yi in zip(v, hi, yy):
      ax.text(100 * hii + 1.0, yi, f'{100 * vi:.0f}%', va='center', fontsize=11, color=INK)
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=12)
  ax.set_xlim(0, xmax)
  ax.grid(axis='x', color=GRID, lw=1); ax.set_axisbelow(True)
  ax.set_xlabel(xlabel, color=MUTED, fontsize=12)
  style(ax)
  ax.legend(frameon=False, fontsize=12, loc='lower center', bbox_to_anchor=(0.55, 1.0), ncol=len(series))


def fig_a(o, thr, out):
  sc = o[f'ema:{BEST}'].to_numpy()
  st = o['speed'].to_numpy()
  ttf = o['time_to_failure'].to_numpy()
  hs = st > 10
  alarm = sc >= thr
  groups = [('Normal driving, no alarm', o['normal'].to_numpy() & hs & ~alarm),
            ('"False" alarms', o['normal'].to_numpy() & hs & alarm),
            ('Last 3 s before a crash', (ttf >= 0) & (ttf <= 3) & hs)]
  ttc = o['gt_ttc_plan'].to_numpy()
  clr = o['gt_clear_plan'].to_numpy()
  routes = np.array(sorted(o['route_id'].unique()))
  ridx = pd.Index(routes).get_indexer(o['route_id'])
  res = {}
  s1, s2 = [], []
  for name, m in groups:
    den = np.bincount(ridx[m], minlength=len(routes)).astype(float)
    n1 = np.bincount(ridx[m & (ttc < 1.5)], minlength=len(routes)).astype(float)
    n2 = np.bincount(ridx[m & (clr < 1.0)], minlength=len(routes)).astype(float)
    v1, v2 = n1.sum() / den.sum(), n2.sum() / den.sum()
    c1, c2 = ratio_ci(n1, den, routes), ratio_ci(n2, den, routes)
    s1.append((v1, *c1)); s2.append((v2, *c2))
    res[name] = {'n_steps': int(m.sum()), 'gt_ttc_lt_1.5': v1, 'ci_ttc': list(c1), 'gt_clear_lt_1m': v2,
                 'ci_clear': list(c2), 'median_clear': float(np.nanmedian(clr[m]))}
  fig, ax = plt.subplots(figsize=(13.5, 4.6), dpi=100)
  fig.patch.set_facecolor(BG)
  hbars(ax, [g[0] for g in groups],
        [('real collision course within 1.5 s', BLUE, np.array([a[0] for a in s1]), np.array([a[1] for a in s1]),
          np.array([a[2] for a in s1])),
         ('real clearance under 1 m', ORANGE, np.array([a[0] for a in s2]), np.array([a[1] for a in s2]),
          np.array([a[2] for a in s2]))],
        100, 'share of moments at > 10 m/s (%), checked against ground truth\n'
             "TF++'s planned path rolled forward against the real actors; whiskers = 95% route-bootstrap CI")
  ax.get_yticklabels()[1].set_fontweight('bold')
  fig.tight_layout()
  fig.savefig(out, facecolor=BG)
  plt.close(fig)
  return res


def fig_b(o, out):
  models = [('Speed only', 'speed_only'), ('TF++ outputs (confidence, path,\ncontrols, history, map)', 'tfpp+intent+map'),
            ('+ intent-aware geometry\n(TF++ boxes)', BEST), ('Ground-truth scene\n(reference only)', 'oracle: +GT scene (privileged ref)')]
  routes = np.array(sorted(o['route_id'].unique()))
  crash, near, res = [], [], {}
  for label, m in models:
    sc = o[f'ema:{m}'].to_numpy()
    ra = per_route(o, sc, thr_for_total_rate(o, sc, 2.0)).reindex(routes)
    rb = per_route(o, sc, thr_for_false_rate(o, sc, 2.0)).reindex(routes)
    va, vb = ra['w1'].sum() / ra['fail'].sum(), rb['w1'].sum() / rb['fail'].sum()
    ca = ratio_ci(ra['w1'].to_numpy(float), ra['fail'].to_numpy(float), routes)
    cb = ratio_ci(rb['w1'].to_numpy(float), rb['fail'].to_numpy(float), routes)
    crash.append((va, *ca)); near.append((vb, *cb))
    sb = summarize(rb)
    res[m] = {'crash_only': va, 'crash_only_ci': list(ca), 'near_miss': vb, 'near_miss_ci': list(cb),
              'near_miss_share_of_alarms': sb['justified_share']}
  fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=100)
  fig.patch.set_facecolor(BG)
  hbars(ax, [m[0] for m in models],
        [('crash-only counting', BLUE, np.array([c[0] for c in crash]), np.array([c[1] for c in crash]),
          np.array([c[2] for c in crash])),
         ('near-misses counted as justified', ORANGE, np.array([c[0] for c in near]), np.array([c[1] for c in near]),
          np.array([c[2] for c in near]))],
        100, 'failures warned ≥ 1 s ahead (%), at 2 false alarms per minute of normal driving\n'
             '197 failures, 3 seeds × 220 routes; whiskers = 95% route-bootstrap CI')
  ax.get_yticklabels()[2].set_fontweight('bold')
  fig.tight_layout()
  fig.savefig(out, facecolor=BG)
  plt.close(fig)
  return res


def fig_c(o, thr, routes_meta, out):
  """Pick a clean near-miss: route run with no failure, alarm at > 10 m/s, real TTC on plan < 1 s."""
  sc = o[f'ema:{BEST}'].to_numpy()
  o = o.assign(alarm=sc >= thr, score=sc)
  ok_runs = set(map(tuple, routes_meta.loc[routes_meta['first_failure_kind'].isna(), ['run', 'route_id']].astype(str).values))
  cands = []
  for (run, rid), g in o.groupby(['run', 'route_id'], sort=False):
    if (run, rid) not in ok_runs:
      continue
    a = g['alarm'].to_numpy()
    on = np.flatnonzero(a & ~np.r_[False, a[:-1]])
    for i in on:
      j = i
      while j + 1 < len(a) and a[j + 1]:
        j += 1
      seg = g.iloc[max(i - 20, 0):j + 21]
      mn = seg['gt_ttc_plan'].min()
      if g['speed'].iloc[i] > 10 and 0.4 <= mn < 1.0 and (j - i) >= 10:
        cands.append((-(j - i), mn, run, rid, g.index[i], g.index[j]))
  cands.sort()
  _, _, run, rid, i0, i1 = cands[0]
  g = o[(o['run'] == run) & (o['route_id'] == rid)]
  t = (g['step'] - g.loc[i0, 'step']) * 0.05
  w = (t > -4) & (t < 5)
  g, t = g[w], t[w]
  meta = routes_meta[(routes_meta['run'] == run) & (routes_meta['route_id'].astype(str) == rid)].iloc[0]
  fig, axs = plt.subplots(3, 1, figsize=(12, 7.2), dpi=100, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
  fig.patch.set_facecolor(BG)
  for ax in axs:
    style(ax)
    ax.grid(axis='y', color=GRID, lw=1); ax.set_axisbelow(True)
    on = g['alarm'].to_numpy()
    tt = t.to_numpy()
    ax.fill_between(tt, 0, 1, where=on, transform=ax.get_xaxis_transform(), color=ORANGE, alpha=0.12, lw=0)
  axs[0].plot(t, g['score'], color=ORANGE, lw=2)
  axs[0].axhline(thr, color=INK, lw=1, ls='--')
  axs[0].text(t.iloc[0], thr, ' alarm threshold', va='bottom', fontsize=10, color=INK)
  axs[0].set_ylabel('monitor risk', color=MUTED, fontsize=11)
  axs[1].plot(t, g['gt_ttc_plan'].clip(upper=5), color=BLUE, lw=2)
  axs[1].axhline(1.5, color=INK, lw=1, ls='--')
  axs[1].text(t.iloc[0], 1.5, ' near-miss line (1.5 s)', va='bottom', fontsize=10, color=INK)
  axs[1].set_ylim(0, 5.2)
  axs[1].set_ylabel('real time to collision\non TF++ plan (s)', color=MUTED, fontsize=11)
  axs[2].plot(t, g['speed'], color=INK, lw=2, label='actual speed')
  axs[2].plot(t, g['target_speed'], color=AQUA, lw=2, label='TF++ target speed')
  axs[2].set_ylabel('m/s', color=MUTED, fontsize=11)
  axs[2].legend(frameon=False, fontsize=10, loc='lower left')
  axs[2].set_xlabel('time from alarm onset (s)   ·   shaded = alarm on', color=MUTED, fontsize=11)
  fig.suptitle(f'Near-miss on route {rid} ({meta["scenario_types"]}, {run.replace("all_boxes_", "")}): '
               'alarm raised, no crash, route completed', fontsize=12, color=INK, x=0.01, ha='left')
  fig.tight_layout()
  fig.savefig(out, facecolor=BG)
  plt.close(fig)
  return {'run': run, 'route_id': rid, 'scenario': str(meta['scenario_types']),
          'min_gt_ttc': float(g['gt_ttc_plan'].min()), 'alarm_s': float((i1 - i0 + 1) * 0.05)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', type=Path, default=Path('analysis/data/boxes_3seeds'))
  args = ap.parse_args()
  o = load(args.data, 1.5)
  st = pd.read_parquet(args.data / 'steps.parquet', columns=['run', 'route_id', 'step', 'ego_speed', 'target_speed_scalar'])
  st['route_id'] = st['route_id'].astype(str)
  bx = pd.read_parquet(args.data / 'box_steps.parquet', columns=['run', 'route_id', 'step', 'gt_clear_plan'])
  bx['route_id'] = bx['route_id'].astype(str)
  o = o.merge(st, on=['run', 'route_id', 'step'], how='left').merge(bx, on=['run', 'route_id', 'step'], how='left')
  o = o.rename(columns={'ego_speed': 'speed', 'target_speed_scalar': 'target_speed'})
  o['gt_ttc_plan'] = o['gt_ttc_plan'].replace(np.inf, 5.0).clip(upper=5.0)
  routes = pd.read_parquet(args.data / 'routes.parquet')
  routes['route_id'] = routes['route_id'].astype(str)
  fig_dir = args.data / 'figures'
  fig_dir.mkdir(exist_ok=True)
  thr2 = thr_for_total_rate(o, o[f'ema:{BEST}'].to_numpy(), 2.0)  # crash-only counting: 2 alarms/min
  out = {'f7a': fig_a(o, thr2, fig_dir / 'f7a_false_alarms_vs_crashes.png'),
         'f7b': fig_b(o, fig_dir / 'f7b_near_miss_counting.png'),
         'f7c': fig_c(o, thr2, routes, fig_dir / 'f7c_near_miss_example.png')}
  (fig_dir / 'f7_numbers.json').write_text(json.dumps(out, indent=1, default=float))
  print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
  main()
