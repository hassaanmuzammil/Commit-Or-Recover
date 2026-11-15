"""Planning / control / attribution analysis of frozen TF++ (Phases C, D, E). Schema-v2 logs only.

Used by notebooks/planning_analysis.ipynb, control_analysis.ipynb and failure_attribution.ipynb.
Builds on src/ew_analysis.py (collision events, struck-actor detection tracks, step features).

Extra data extracted here, per route:
  * TF++'s full planned path (10 points, ego frame) and target speed at every step;
  * the ego's true pose from CARLA (gt_ego x, y, yaw), used to measure how well the car followed its plan;
  * for every collision, the struck actor's *actual* world trajectory (ground truth), used to check
    retrospectively whether TF++'s plan led into the actor's real path. Ground truth is diagnosis only.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np

import ew_analysis as ew

DT, HZ = ew.DT, ew.HZ
# TF++'s longitudinal control (model.control_pid_direct -> nav_planner.get_throttle, config values)
BRAKE_RATIO = 1.1          # config.brake_ratio
MIN_TARGET = 0.01
LONG_PARAMS = np.array([1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335,
                        1.556497522998393, -0.7013479734904027, 1.031266635497984])   # config.longitudinal_params
MAX_ACC_KMH = 1.89         # config.longitudinal_max_acceleration
CONFLICT_MARGIN = 0.5      # m: plan counted as conflicting if it passes within this of the actor (disc model)


def extract_route(log_path, ck):
    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows], float)
    d = {"n": n, "speed": g("ego_speed"), "steer": g("steer"), "thr": g("throttle"), "brk": g("brake"),
         "tss": g("target_speed_scalar"), "acc": g("acceleration"), "yaw_rate": g("yaw_rate"),
         "cmd": np.array([r["command"] for r in rows], int),
         "path": np.array([r["predicted_path"] for r in rows], np.float32),
         "tp": np.array([r["target_point"] for r in rows], np.float32),
         "force_move": np.array([r["tfpp_force_move"] for r in rows], bool),
         "emergency_stop": np.array([r["tfpp_emergency_stop"] for r in rows], bool),
         "stop_sign": np.array([r["tfpp_stop_sign_stop"] for r in rows], bool)}
    ge = [r["gt_ego"] or {} for r in rows]
    d["gx"] = np.array([e.get("x", np.nan) for e in ge])
    d["gy"] = np.array([e.get("y", np.nan) for e in ge])
    d["gyaw"] = np.array([e.get("yaw", np.nan) for e in ge])
    events = ew.group_contacts(rows, ck)
    for ev in events:
        lo, hi = max(0, ev["step"] - 120), min(n - 1, ev["step"] + 20)
        tr = {"step": [], "present": [], "wx": [], "wy": [], "wyaw": [], "ext_x": [], "ext_y": [], "cls": []}
        for s in range(lo, hi + 1):
            a = next((x for x in rows[s]["gt_actors"] if x["id"] == ev["other_id"]), None) if ev["other_id"] > 0 else None
            tr["step"].append(s)
            tr["present"].append(a is not None)
            tr["wx"].append(a["world"][0] if a else np.nan)
            tr["wy"].append(a["world"][1] if a else np.nan)
            tr["wyaw"].append(a["world"][2] if a else np.nan)
            tr["ext_x"].append(a["ext_x"] if a else np.nan)
            tr["ext_y"].append(a["ext_y"] if a else np.nan)
            tr["cls"].append(a["class"] if a else "")
        ev["world_track"] = {k: np.asarray(v) for k, v in tr.items()}
    d["events"] = events
    return d


def build_cache(results_dir, cache_path, log=print):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))
    results_dir = Path(results_dir)
    data = {}
    files = sorted((results_dir / "checkpoints").glob("*.json"), key=lambda p: int(p.stem))
    for i, p in enumerate(files):
        ck = ew.parse_checkpoint(p)
        lp = ew.pick_log(results_dir / "logs" / p.stem)
        if ck is None or lp is None:
            continue
        data[p.stem] = extract_route(lp, ck)
        if (i + 1) % 20 == 0:
            log(f"  parsed {i + 1}/{len(files)} routes")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, open(cache_path, "wb"))
    return data


# --------------------------------------------------------------------------- frames

def ego_to_world(d, t, x, y, h=None):
    c, s = math.cos(d["gyaw"][t]), math.sin(d["gyaw"][t])
    wx = d["gx"][t] + c * x - s * y
    wy = d["gy"][t] + s * x + c * y
    return (wx, wy) if h is None else (wx, wy, d["gyaw"][t] + h)


def world_to_ego(d, t, wx, wy):
    c, s = math.cos(d["gyaw"][t]), math.sin(d["gyaw"][t])
    dx, dy = wx - d["gx"][t], wy - d["gy"][t]
    return c * dx + s * dy, -s * dx + c * dy


def plan_polyline(path, extend=30.0):
    pts = np.vstack([[0.0, 0.0], path]).astype(float)
    seg = pts[-1] - pts[-2]
    L = np.hypot(*seg)
    if L > 1e-6:
        pts = np.vstack([pts, pts[-1] + seg / L * extend])
    return pts


# --------------------------------------------------------------------------- Phase C: plan vs the actor's real future

def retro_check(d, ev, t):
    """Follow TF++'s plan made at step t (timed by its target speed) and compare it with the struck actor's
    ACTUAL future positions. Returns the first time they come within CONFLICT_MARGIN (s, inf if never), the minimum
    clearance (m) and the time gap at the closest spatial crossing (s). Only actor positions up to the
    impact step are used."""
    tr = ev["world_track"]
    s_imp = ev["step"]
    ex, ey, eh = ew.plan_rollout(d["path"][t], d["speed"][t], d["tss"][t], ew.TAU)
    wx, wy, wh = ego_to_world(d, t, ex, ey, eh)
    idx = t + np.round(ew.TAU * HZ).astype(int)
    pos = {s: i for i, s in enumerate(tr["step"])}
    ok = np.array([j <= s_imp and j in pos and tr["present"][pos[j]] for j in idx])
    if ok.sum() < 3 or not np.isfinite(d["gx"][t]):
        return np.nan, np.nan, np.nan
    k = np.array([pos[j] for j in idx[ok]])
    ax, ay, ah = tr["wx"][k], tr["wy"][k], tr["wyaw"][k]
    dd, r = ew._box_discs(np.nanmax(tr["ext_x"]), np.nanmax(tr["ext_y"]))
    off = np.array([-dd, 0.0, dd])
    eo = np.array([-ew.EGO_D, 0.0, ew.EGO_D])
    EX = wx[ok][:, None] + eo * np.cos(wh[ok])[:, None]
    EY = wy[ok][:, None] + eo * np.sin(wh[ok])[:, None]
    AX = ax[:, None] + off * np.cos(ah)[:, None]
    AY = ay[:, None] + off * np.sin(ah)[:, None]
    gap = np.sqrt((EX[:, :, None] - AX[:, None, :]) ** 2 + (EY[:, :, None] - AY[:, None, :]) ** 2).min(axis=(1, 2))
    gap -= ew.EGO_R + float(r)
    tt = ew.TAU[ok]
    hit = gap < CONFLICT_MARGIN
    ttc = float(tt[np.argmax(hit)]) if hit.any() else np.inf
    # spatial crossing regardless of timing: closest approach between the plan path and the actor's real path
    G = np.sqrt((EX[:, None, 1] - AX[None, :, 1]) ** 2 + (EY[:, None, 1] - AY[None, :, 1]) ** 2) - (ew.EGO_R + float(r)) * 1.5
    if (G < 0).any():
        ie, ia = np.where(G < 0)
        tgap = float(np.min(np.abs(tt[ie] - tt[ia])))
    else:
        tgap = np.inf
    return ttc, float(gap.min()), tgap


# --------------------------------------------------------------------------- Phase D: did the car follow its plan?

def expected_control(speed, tss):
    """Re-run TF++'s longitudinal controller on the logged speed and target speed.
    Returns (expected throttle, expected brake) exactly as model.control_pid_direct + nav_planner.get_throttle
    compute them (before the creep / stop-sign / safety-box overrides)."""
    speed, tss = np.asarray(speed, float), np.asarray(tss, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        brake = (tss < MIN_TARGET) | (speed / np.maximum(tss, 1e-9) > BRAKE_RATIO)
    t = np.maximum(tss, 1 / 3.6) * 3.6
    v = speed * 3.6
    err = t - v
    full = ~brake & (err > MAX_ACC_KMH)
    brake2 = ~brake & ~full & (v / t > LONG_PARAMS[-1])
    e = np.clip(err, 0, None) / 100.0
    vv = v / 100.0
    feats = np.stack([vv, vv ** 2, 100 * e, e ** 2, vv * e, vv ** 2 * e], 1)
    thr = np.clip(feats @ LONG_PARAMS[:-1], 0, 1)
    thr = np.where(brake | brake2, 0.0, np.where(full, 1.0, thr))
    return thr, brake | brake2


def cross_track(d, horizon_steps):
    """Distance (m) between where the car actually was `horizon` later and the path TF++ planned at step t."""
    n = d["n"]
    out = np.full(n, np.nan)
    from ew_analysis import point_to_polyline
    for t in range(n - horizon_steps):
        if not np.isfinite(d["gx"][t + horizon_steps]):
            continue
        px, py = world_to_ego(d, t, d["gx"][t + horizon_steps], d["gy"][t + horizon_steps])
        if math.hypot(px, py) < 0.5:          # car barely moved: nothing to compare
            continue
        out[t] = point_to_polyline(np.array([[px, py]]), plan_polyline(d["path"][t]))[0]
    return out
