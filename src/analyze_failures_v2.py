#!/usr/bin/env python3
"""Failure analysis of frozen TF++ on Bench2Drive using the schema-v2 logs.

Inputs
  results/bench2drive_v2/checkpoints/<route>.json   leaderboard record (infractions, scores)
  results/bench2drive_v2/logs/<route>/*.jsonl       per-step log (docs/tfpp_sample_output_v2.md)
  carla_garage/leaderboard/data/bench2drive220.xml  scenario type and town per route

What it computes
  1. Run summary and failure list (first failure per route).
  2. Failure categories: scenario family, collision geometry x maneuver, and mechanism
     (was the struck actor detected by TF++'s boxes, did TF++ brake).
  3. Simple temporal-feature baseline (path disagreement, target-speed JS divergence,
     control variation) against speed-only, TF++ confidence, a geometric monitor on
     TF++'s predicted boxes, and a ground-truth geometric oracle. Grouped 5-fold CV by route.
  4. Timing: do temporal signals fire before a geometric hazard becomes apparent?
  5. Data sufficiency per family: failures vs benign instability episodes.

Privileged ground truth is used for labels, categories and the oracle row only.

Run:  uv run --no-project --with numpy --with scikit-learn python src/analyze_failures_v2.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DT = 0.05
HZ = 20
HORIZON_S = 3.0
SKIP_START = 40                      # first 2 s: CARLA holds the brake
TARGET_SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.8889, 16.0, 17.7778, 20.0])
EGO_EXT_X, EGO_EXT_Y = 2.4508, 1.0642
TAU = np.arange(0.0, 3.0 + 1e-9, 0.1)  # rollout times for TTC
BOX_RANGE = 32.0                     # TF++ detection range (config.max_x / max_y)
WINDOW_BEFORE = 8 * HZ               # steps kept before each failure for timing analysis

FAILURE_KEYS = ["collisions_vehicle", "collisions_pedestrian", "collisions_layout",
                "outside_route_lanes", "vehicle_blocked", "route_timeout"]
COLLISION_KEYS = FAILURE_KEYS[:3]
ID_RE = re.compile(r"type=(\S+) and id=(\d+)")

FAMILY = {
    # junction turns and crossings
    **{s: "junction" for s in [
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
        "NonSignalizedJunctionLeftTurnEnterFlow", "T_Junction", "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority", "VehicleTurningRoute", "BlockedIntersection",
        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign"]},
    # ego merges or changes lane into traffic
    **{s: "lane_merge" for s in [
        "EnterActorFlow", "HighwayExit", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2",
        "SequentialLaneChange", "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "ParkingExit"]},
    # ego detours around a blockage, often into the opposite lane
    **{s: "obstacle_detour" for s in [
        "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
        "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays",
        "VehicleOpensDoorTwoWays", "InvadingTurn"]},
    # another vehicle cuts in or brakes / loses control in front of the ego
    **{s: "cut_in_hazard" for s in [
        "HighwayCutIn", "ParkingCutIn", "StaticCutIn", "HardBreakRoute", "ControlLoss",
        "YieldToEmergencyVehicle"]},
    # vulnerable road users crossing
    **{s: "vru_crossing" for s in [
        "PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian",
        "VehicleTurningRoutePedestrian", "CrossingBicycleFlow"]},
}
BICYCLES = ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets")


# --------------------------------------------------------------------------- geometry

def ego_rollout(v, w):
    if abs(w) < 1e-3:
        h = np.zeros_like(TAU)
        return v * TAU, np.zeros_like(TAU), h
    h = w * TAU
    return v / w * np.sin(h), v / w * (1 - np.cos(h)), h


def discs(ex, ey):
    """Disc offsets along the heading and radius covering a box with half-extents ex, ey."""
    ex, ey = max(ex, 0.2), max(ey, 0.2)
    if ex <= ey:
        return np.array([0.0]), ey
    return np.array([-(ex - ey), 0.0, ex - ey]), ey


EGO_OFF, EGO_R = discs(EGO_EXT_X, EGO_EXT_Y)


def ttc_per_actor(v, w, cx, cy, yaw, vx, vy, ex, ey):
    """Constant-turn-rate ego vs constant-velocity actors in the ego frame. Returns TTC (s) per actor, inf if none."""
    n = len(cx)
    if n == 0:
        return np.zeros(0)
    exs, eys, eh = ego_rollout(v, w)                                   # (T,)
    e_x = exs[:, None] + EGO_OFF[None, :] * np.cos(eh)[:, None]         # (T,3)
    e_y = eys[:, None] + EGO_OFF[None, :] * np.sin(eh)[:, None]
    out = np.full(n, np.inf)
    for i in range(n):
        off, r = discs(ex[i], ey[i])
        ax = cx[i] + vx[i] * TAU                                        # (T,)
        ay = cy[i] + vy[i] * TAU
        a_x = ax[:, None] + off[None, :] * math.cos(yaw[i])             # (T,k)
        a_y = ay[:, None] + off[None, :] * math.sin(yaw[i])
        d2 = (e_x[:, :, None] - a_x[:, None, :]) ** 2 + (e_y[:, :, None] - a_y[:, None, :]) ** 2
        hit = (d2 < (EGO_R + r) ** 2).reshape(len(TAU), -1).any(axis=1)
        if hit.any():
            out[i] = TAU[np.argmax(hit)]
    return out


def transform_prev_path(path_prev, pos_prev, yaw_prev, pos, yaw):
    c, s = math.cos(yaw_prev), math.sin(yaw_prev)
    wx = c * path_prev[:, 0] - s * path_prev[:, 1] + pos_prev[0] - pos[0]
    wy = s * path_prev[:, 0] + c * path_prev[:, 1] + pos_prev[1] - pos[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * wx + s * wy, -s * wx + c * wy], axis=1)


def point_to_polyline(points, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((points[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- loading

def load_routes(xml_path):
    out = {}
    for r in ET.parse(xml_path).getroot().iter("route"):
        sc = [s.get("type") for s in r.find("scenarios")]
        out[r.get("id")] = {"town": r.get("town"), "scenario": sc[0] if sc else "none"}
    return out


def parse_checkpoint(path):
    rec = json.load(open(path))["_checkpoint"]["records"]
    if not rec:
        return None
    rec = rec[0]
    inf = rec["infractions"]
    events = []
    for k in COLLISION_KEYS:
        for s in inf[k]:
            m = ID_RE.search(s)
            events.append({"kind": k, "type": m.group(1) if m else "?", "id": int(m.group(2)) if m else -1})
    return {
        "status": rec["status"], "ds": rec["scores"]["score_composed"], "rc": rec["scores"]["score_route"],
        "infractions": {k: len(v) for k, v in inf.items()},
        "collision_events": events,
        "failure_kinds": [k for k in FAILURE_KEYS if inf[k]],
        "length": rec["meta"]["route_length"],
    }


def pick_log(route_dir):
    files = sorted(p for p in route_dir.glob("*.jsonl") if not p.name.startswith("zz_"))
    return files[-1] if files else None


def extract_route(log_path, ckpt):
    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    t = np.array([r["timestamp"] for r in rows])
    speed = np.array([r["ego_speed"] for r in rows])
    yaw = np.array([r["ego_yaw"] for r in rows])
    yaw_rate = np.array([r["yaw_rate"] for r in rows])
    acc = np.array([r["acceleration"] for r in rows])
    pos = np.array([r["ego_position"] for r in rows])
    steer = np.array([r["steer"] for r in rows])
    thr = np.array([r["throttle"] for r in rows])
    brk = np.array([r["brake"] for r in rows])
    tss = np.array([r["target_speed_scalar"] for r in rows])
    probs = np.array([r["target_speed_probabilities"] for r in rows])
    path = np.array([r["predicted_path"] for r in rows])
    cmd = np.array([r["command"] for r in rows])
    flags = np.array([[r["tfpp_emergency_stop"], r["tfpp_force_move"], r["tfpp_stop_sign_stop"],
                       r["tfpp_bb_in_front"]] for r in rows], dtype=bool)

    # --- planner / confidence features
    p = np.clip(probs, 1e-9, 1)
    entropy = -(p * np.log(p)).sum(1)
    ps = np.sort(p, axis=1)
    margin = ps[:, -1] - ps[:, -2]
    m = 0.5 * (p[1:] + p[:-1])
    js = np.r_[0.0, 0.5 * (p[1:] * np.log(p[1:] / m)).sum(1) + 0.5 * (p[:-1] * np.log(p[:-1] / m)).sum(1)]
    pd = np.zeros(n)
    for i in range(1, n):
        prev = transform_prev_path(path[i - 1], pos[i - 1], yaw[i - 1], pos[i], yaw[i])
        pd[i] = point_to_polyline(path[i, :8], prev).mean()
    end_lat = path[:, -1, 1]
    seg = np.diff(path, axis=1)
    hd = np.arctan2(seg[:, :, 1], seg[:, :, 0])
    curvature = np.abs(wrap(hd[:, -1] - hd[:, 0]))

    # --- geometry per step
    gt_ttc = np.full(n, np.inf)        # min over GT vehicles and walkers
    pred_ttc = np.full(n, np.inf)      # min over TF++ predicted boxes (car / walker / emergency)
    corr_dist = np.full(n, 40.0)       # nearest predicted box in the ego corridor ahead
    corr_close = np.zeros(n)
    n_pred = np.zeros(n, dtype=int)
    for i, r in enumerate(rows):
        v, w = speed[i], yaw_rate[i]
        # static props are excluded: several have bounding boxes that overlap the road edge
        a = [x for x in r["gt_actors"] if x["class"] != "static"]
        if a:
            gt_ttc[i] = ttc_per_actor(v, w, np.array([x["x"] for x in a]), np.array([x["y"] for x in a]),
                                      np.array([x["yaw"] for x in a]), np.array([x["vx"] for x in a]),
                                      np.array([x["vy"] for x in a]), np.array([x["ext_x"] for x in a]),
                                      np.array([x["ext_y"] for x in a])).min()
        b = np.array([bb for bb in r["pred_boxes"] if int(round(bb[7])) in (0, 1, 4)]).reshape(-1, 9)
        n_pred[i] = len(b)
        if len(b):
            pred_ttc[i] = ttc_per_actor(v, w, b[:, 0], b[:, 1], b[:, 4], b[:, 5] * np.cos(b[:, 4]),
                                        b[:, 5] * np.sin(b[:, 4]), b[:, 2], b[:, 3]).min()
            ahead = (b[:, 0] > 0) & (np.abs(b[:, 1]) < 2.5)
            if ahead.any():
                j = np.argmin(np.where(ahead, b[:, 0], np.inf))
                corr_dist[i] = min(b[j, 0], 40.0)
                corr_close[i] = v - b[j, 5] * math.cos(b[j, 4])

    # --- first failure (collision) step, matched by actor id from the checkpoint
    fail_step, struck = None, None
    ck_ids = {e["id"] for e in ckpt["collision_events"]}
    ck_types = {e["type"] for e in ckpt["collision_events"]}
    for i, r in enumerate(rows):
        for e in r["collisions"]:
            if e["other_id"] in ck_ids and (e["other_id"] != 0 or e["other_type"] in ck_types):
                fail_step, struck = i, e
                break
        if fail_step is not None:
            break

    window = None
    if fail_step is not None:
        lo = max(0, fail_step - WINDOW_BEFORE)
        sid = struck["other_id"]
        st = []
        for i in range(lo, fail_step + 1):
            r = rows[i]
            act = next((x for x in r["gt_actors"] if x["id"] == sid), None) if sid > 0 else None
            rec = {"k": i - fail_step, "present": act is not None}
            if act is not None:
                rec.update({kk: act[kk] for kk in ("x", "y", "yaw", "vx", "vy", "speed", "ext_x", "ext_y", "class", "type")})
                rec["ttc"] = float(ttc_per_actor(speed[i], yaw_rate[i], np.array([act["x"]]), np.array([act["y"]]),
                                                 np.array([act["yaw"]]), np.array([act["vx"]]), np.array([act["vy"]]),
                                                 np.array([act["ext_x"]]), np.array([act["ext_y"]]))[0])
                b = np.array([bb for bb in r["pred_boxes"] if int(round(bb[7])) in (0, 1, 4)]).reshape(-1, 9)
                tol = 2.0 if act["class"] == "walker" else 3.0
                rec["in_range"] = abs(act["x"]) <= BOX_RANGE and abs(act["y"]) <= BOX_RANGE
                if len(b):
                    dmin = np.sqrt((b[:, 0] - act["x"]) ** 2 + (b[:, 1] - act["y"]) ** 2).min()
                    rec["detected"] = bool(dmin < tol)
                    rec["det_err"] = float(dmin)
                else:
                    rec["detected"] = False
            st.append(rec)
        window = {"struck": struck, "track": st}

    return {
        "n": n, "t": t, "speed": speed, "yaw_rate": yaw_rate, "acc": acc, "steer": steer, "thr": thr,
        "brk": brk, "tss": tss, "entropy": entropy, "margin": margin, "pmax": ps[:, -1], "js": js, "pd": pd,
        "end_lat": end_lat, "curvature": curvature, "cmd": cmd, "flags": flags, "gt_ttc": gt_ttc,
        "pred_ttc": pred_ttc, "corr_dist": corr_dist, "corr_close": corr_close, "n_pred": n_pred,
        "fail_step": fail_step, "window": window,
    }


# --------------------------------------------------------------------------- features

def rolling(x, w, fn):
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - w + 1): i + 1])
    return out


def roll_mean(x, w):
    c = np.cumsum(np.r_[0.0, x])
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


def roll_max(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    xp = np.r_[np.full(w - 1, x[0]), x]
    return sliding_window_view(xp, w).max(axis=1)


def roll_std(x, w):
    m = roll_mean(x, w)
    m2 = roll_mean(x * x, w)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def step_features(d):
    s = d
    dsteer = np.abs(np.r_[0.0, np.diff(s["steer"])])
    sgn = np.sign(np.r_[0.0, np.diff(s["steer"])])
    rev = np.r_[0, (sgn[1:] * sgn[:-1] < 0).astype(float)]
    pedal = np.where(s["brk"] > 0.5, -1, np.where(s["thr"] > 0.05, 1, 0))
    switch = np.r_[0, (np.diff(pedal) != 0).astype(float)]
    jerk = np.abs(np.r_[0.0, np.diff(s["acc"])]) / DT
    risk = lambda ttc: np.clip(3.0 - ttc, 0, 3.0)
    f = {
        "ego_speed": s["speed"],
        "target_speed_scalar": s["tss"],
        "speed_entropy": s["entropy"],
        "speed_margin": s["margin"],
        "speed_pmax": s["pmax"],
        "path_disagree": s["pd"],
        "path_disagree_mean_1s": roll_mean(s["pd"], 20),
        "path_disagree_max_1s": roll_max(s["pd"], 20),
        "path_disagree_mean_2s": roll_mean(s["pd"], 40),
        "path_end_lat_std_1s": roll_std(s["end_lat"], 20),
        "path_curvature": s["curvature"],
        "speed_js_mean_1s": roll_mean(s["js"], 20),
        "speed_js_mean_2s": roll_mean(s["js"], 40),
        "steer_tv_1s": roll_mean(dsteer, 20) * 20,
        "steer_reversals_1s": roll_mean(rev, 20) * 20,
        "pedal_switches_1s": roll_mean(switch, 20) * 20,
        "jerk_mean_1s": roll_mean(np.minimum(jerk, 200), 20),
        "pred_box_risk": risk(s["pred_ttc"]),
        "pred_corridor_dist": s["corr_dist"],
        "pred_corridor_closing": s["corr_close"],
        "tfpp_bb_in_front": s["flags"][:, 3].astype(float),
        "gt_risk": risk(s["gt_ttc"]),
    }
    return f


FEATURE_SETS = {
    "speed only": ["ego_speed"],
    "TF++ confidence (single frame)": ["ego_speed", "target_speed_scalar", "speed_entropy", "speed_margin", "speed_pmax"],
    "temporal only (no speed)": ["path_disagree_mean_1s", "path_disagree_max_1s", "path_disagree_mean_2s",
                                 "path_end_lat_std_1s", "speed_js_mean_1s", "speed_js_mean_2s", "steer_tv_1s",
                                 "steer_reversals_1s", "pedal_switches_1s", "jerk_mean_1s"],
    "temporal + speed": ["ego_speed", "path_disagree_mean_1s", "path_disagree_max_1s", "path_disagree_mean_2s",
                         "path_end_lat_std_1s", "speed_js_mean_1s", "speed_js_mean_2s", "steer_tv_1s",
                         "steer_reversals_1s", "pedal_switches_1s", "jerk_mean_1s"],
    "geometric (TF++ boxes)": ["ego_speed", "pred_box_risk", "pred_corridor_dist", "pred_corridor_closing",
                               "tfpp_bb_in_front"],
    "geometric + temporal": ["ego_speed", "pred_box_risk", "pred_corridor_dist", "pred_corridor_closing",
                             "tfpp_bb_in_front", "path_disagree_mean_1s", "path_disagree_max_1s",
                             "path_disagree_mean_2s", "path_end_lat_std_1s", "speed_js_mean_1s", "speed_js_mean_2s",
                             "steer_tv_1s", "steer_reversals_1s", "pedal_switches_1s", "jerk_mean_1s"],
    "oracle geometric (GT, privileged)": ["ego_speed", "gt_risk"],
}


# --------------------------------------------------------------------------- categorisation

def categorize(route, d, ck):
    """Collision geometry, maneuver and mechanism for the first collision of a route."""
    w = d["window"]
    tr = w["track"]
    struck = w["struck"]
    fs = d["fail_step"]
    otype = struck["other_type"]
    pre = [r for r in tr if r["present"] and r["k"] <= -2]
    last = pre[-1] if pre else next((r for r in reversed(tr) if r["present"]), None)
    v_imp = float(d["speed"][max(0, fs - 2)])     # 0.1 s before contact; the contact step is already disturbed
    lo = max(0, fs - 60)
    cmds = d["cmd"][lo:fs + 1]
    if np.isin(cmds, [0, 1, 2]).any():
        c = collections.Counter(cmds[np.isin(cmds, [0, 1, 2])]).most_common(1)[0][0]
        maneuver = {0: "junction left turn", 1: "junction right turn", 2: "junction straight"}[int(c)]
    elif np.isin(cmds, [4, 5]).any():
        maneuver = "lane change"
    else:
        maneuver = "follow lane"

    rel_yaw = abs(math.degrees(last["yaw"])) if last else float("nan")
    if otype.startswith("walker"):
        geom = "pedestrian"
    elif otype.startswith(BICYCLES):
        geom = "cyclist"
    elif otype.startswith("static"):
        geom = "static object"
    elif v_imp < 1.0:
        geom = "hit while stopped/slow"
    elif last is None:
        geom = "other vehicle (no track)"
    elif rel_yaw > 135:
        geom = "head-on (oncoming)"
    elif rel_yaw > 45:
        geom = "crossing / turning conflict"
    elif last["x"] > EGO_EXT_X and abs(last["y"]) < 1.5:
        geom = "rear-end into vehicle ahead"
    elif last["x"] < -EGO_EXT_X and abs(last["y"]) < 1.5:
        geom = "rear-ended by vehicle behind"
    else:
        geom = "side-swipe (same direction)"

    # detection of the struck actor by TF++'s boxes
    def det_at(sec):
        k = -int(round(sec * HZ))
        rs = [r for r in tr if r["present"] and abs(r["k"] - k) <= 2]
        if not rs:
            return None
        if not any(r.get("in_range") for r in rs):
            return "out of range"
        return "detected" if any(r.get("detected") for r in rs) else "missed"

    det = {f"{s}s": det_at(s) for s in (3, 2, 1, 0.5)}
    seen = [r for r in tr if r["present"] and r.get("in_range") and -40 <= r["k"] <= -5]
    det_frac = float(np.mean([r.get("detected", False) for r in seen])) if seen else float("nan")
    first_det = next((r["k"] for r in tr if r["present"] and r.get("detected")), None)
    braked = bool((d["brk"][max(0, fs - 30):fs + 1] > 0.5).any())
    # onset of the final braking run before contact (gaps <= 0.25 s bridged); nan if not braking near contact
    b = np.where(d["brk"][max(0, fs - 120):fs + 1] > 0.5)[0] + max(0, fs - 120)
    brake_onset = float("nan")
    if len(b) and fs - b[-1] <= 10:
        st = b[-1]
        for k in b[::-1][1:]:
            if st - k <= 5:
                st = k
            else:
                break
        brake_onset = (fs - st) / HZ
    tss_drop = float(d["tss"][max(0, fs - 30)] - d["tss"][fs])

    if otype.startswith("static"):
        mech = "static object (TF++ has no box class)"
    elif geom in ("hit while stopped/slow", "rear-ended by vehicle behind"):
        mech = "struck by other (ego slow / from behind)"
    elif last is None or not seen:
        mech = "actor not observable"
    elif det_frac < 0.25:
        mech = "perception miss"
    elif brake_onset >= 0.5:
        mech = "detected, reacted too late"
    else:
        mech = "detected, no timely reaction (committed)"

    # geometric hazard onset for the struck actor (GT), threshold 2 s TTC
    def onset(thr):
        ks = [r["k"] for r in tr if r["present"] and r.get("ttc", np.inf) < thr]
        if not ks or ks[-1] < -5:
            return float("nan")
        ks = sorted(ks)
        start = ks[-1]
        for k in reversed(ks[:-1]):
            if start - k <= 5:
                start = k
            else:
                break
        return -start / HZ

    return {
        "route": route, "otype": otype, "geometry": geom, "maneuver": maneuver, "mechanism": mech,
        "v_impact": v_imp, "v_1p5s": float(d["speed"][max(0, fs - 30)]), "braked_1p5s": braked, "brake_onset_s": brake_onset,
        "tss_drop_1p5s": tss_drop, "impulse": struck["impulse"], "rel_yaw_deg": rel_yaw,
        "rel_x": last["x"] if last else float("nan"), "rel_y": last["y"] if last else float("nan"),
        "det": det, "det_frac_2s": det_frac,
        "first_det_s": (-first_det / HZ) if first_det is not None else float("nan"),
        "geo_onset_ttc2": onset(2.0), "geo_onset_ttc3": onset(3.0), "fail_time": float(d["t"][fs]),
    }


# --------------------------------------------------------------------------- evaluation

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.sum() else float("nan")


def oof_scores(X, y, groups, model):
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if model == "logreg":
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        else:
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                 class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def alarm_eval(score, y, route_idx, tti, fail_routes, rates=(1.0, 3.0, 10.0)):
    """Threshold for a target number of false-alarm onsets per minute, then warning share per failure."""
    neg = y == 0
    minutes = neg.sum() * DT / 60
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.9995, 400)))
    order = np.lexsort((np.arange(len(y)), route_idx))
    assert (np.diff(order) > 0).all()

    def onsets(thr):
        a = score >= thr
        prev = np.r_[False, a[:-1]] & (np.r_[-1, route_idx[:-1]] == route_idx)
        return ((a & ~prev) & neg).sum() / minutes

    rate_at = np.array([onsets(q) for q in qs])
    out = {}
    for r in rates:
        ok = np.where(rate_at <= r)[0]
        thr = qs[ok[0]] if len(ok) else qs[-1]
        a = score >= thr
        leads = {}
        for ri in fail_routes:
            m = (route_idx == ri) & (y == 1) & a
            leads[ri] = float(tti[m].max()) if m.any() else 0.0
        lv = np.array(list(leads.values()))
        out[r] = {"thr": float(thr), "alarm_time_frac": float(a[neg].mean()), "warn_1s": float((lv >= 1.0).mean()), "warn_2s": float((lv >= 2.0).mean()),
                  "median_lead": float(np.median(lv)), "leads": leads}
    # matched intervention budget: same share of normal driving time spent in alarm
    for frac in (0.01, 0.02, 0.05):
        thr = float(np.quantile(score[neg], 1 - frac))
        a = score >= thr
        leads = {}
        for ri in fail_routes:
            m = (route_idx == ri) & (y == 1) & a
            leads[ri] = float(tti[m].max()) if m.any() else 0.0
        lv = np.array(list(leads.values()))
        prev = np.r_[False, a[:-1]] & (np.r_[-1, route_idx[:-1]] == route_idx)
        out[f"time{frac}"] = {"thr": thr, "alarm_time_frac": float(a[neg].mean()),
                              "onsets_per_min": float(((a & ~prev) & neg).sum() / minutes),
                              "warn_1s": float((lv >= 1.0).mean()), "warn_2s": float((lv >= 2.0).mean()),
                              "median_lead": float(np.median(lv)), "leads": leads}
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "results/bench2drive_v2")
    ap.add_argument("--baseline", type=Path, default=REPO / "results/bench2drive_v1")
    ap.add_argument("--routes-xml", type=Path, default=REPO / "carla_garage/leaderboard/data/bench2drive220.xml")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or out / "cache.pkl"
    routes = load_routes(args.routes_xml)

    ckpts = {p.stem: parse_checkpoint(p) for p in sorted((args.results / "checkpoints").glob("*.json"))}
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        data = {}
        for i, (rid, ck) in enumerate(ckpts.items()):
            if ck is None:
                continue
            lp = pick_log(args.results / "logs" / rid)
            if lp is None:
                continue
            data[rid] = extract_route(lp, ck)
            print(f"[{i + 1}/{len(ckpts)}] {rid} steps={data[rid]['n']} fail_step={data[rid]['fail_step']}", flush=True)
        pickle.dump(data, open(cache, "wb"))

    summary = {}
    # ---------------- 1. run summary
    valid = {r: c for r, c in ckpts.items() if c is not None}
    fail_routes = {r for r, c in valid.items() if c["failure_kinds"]}
    statuses = collections.Counter(c["status"] for c in valid.values())
    inf_events = collections.Counter()
    inf_routes = collections.Counter()
    for c in valid.values():
        for k, v in c["infractions"].items():
            inf_events[k] += v
            inf_routes[k] += bool(v)
    summary["run"] = {
        "routes_total": len(ckpts), "routes_with_result": len(valid),
        "no_result": [r for r, c in ckpts.items() if c is None],
        "failure_routes": len(fail_routes), "failure_rate": len(fail_routes) / len(valid),
        "mean_ds": float(np.mean([c["ds"] for c in valid.values()])),
        "mean_rc": float(np.mean([c["rc"] for c in valid.values()])),
        "statuses": dict(statuses), "inf_events": dict(inf_events), "inf_routes": dict(inf_routes),
        "steps": int(sum(d["n"] for d in data.values())),
        "km": float(sum(c["length"] * c["rc"] / 100 for c in valid.values()) / 1000),
    }
    # v1 comparison
    v1 = {p.stem: parse_checkpoint(p) for p in sorted((args.baseline / "checkpoints").glob("*.json"))}
    v1f = {r for r, c in v1.items() if c and c["failure_kinds"]}
    both = [r for r in valid if v1.get(r)]
    summary["v1_compare"] = {
        "v1_failure_routes": len(v1f), "v1_rate": len(v1f) / len([c for c in v1.values() if c]),
        "fail_both": len([r for r in both if r in v1f and r in fail_routes]),
        "fail_v1_only": sorted(r for r in both if r in v1f and r not in fail_routes),
        "fail_v2_only": sorted(r for r in both if r not in v1f and r in fail_routes),
    }

    # ---------------- first failure per route
    first_kind = {}
    for r in fail_routes:
        d = data.get(r)
        kinds = valid[r]["failure_kinds"]
        if d is not None and d["fail_step"] is not None:
            first_kind[r] = "collision"
        elif any(k in COLLISION_KEYS for k in kinds):
            first_kind[r] = "collision (untimed)"
        elif "outside_route_lanes" in kinds:
            first_kind[r] = "off-route"
        elif "vehicle_blocked" in kinds:
            first_kind[r] = "blocked"
        else:
            first_kind[r] = "route timeout (stuck)"

    fam_of = lambda r: FAMILY.get(routes[r]["scenario"], "other")
    cats = []
    for r in sorted(fail_routes, key=int):
        base = {"route": r, "scenario": routes[r]["scenario"], "family": fam_of(r), "town": routes[r]["town"],
                "kinds": "|".join(valid[r]["failure_kinds"]), "first": first_kind[r], "ds": valid[r]["ds"]}
        if first_kind[r] == "collision":
            base.update(categorize(r, data[r], valid[r]))
        else:
            base.update({"geometry": first_kind[r], "maneuver": "-", "mechanism": "non-collision"})
        cats.append(base)
    summary["failures"] = cats

    # family table
    fam = collections.defaultdict(lambda: {"routes": 0, "fail": 0, "ds": []})
    for r, c in valid.items():
        f = fam[fam_of(r)]
        f["routes"] += 1
        f["fail"] += r in fail_routes
        f["ds"].append(c["ds"])
    summary["family"] = {k: {"routes": v["routes"], "fail": v["fail"], "ds": float(np.mean(v["ds"]))} for k, v in fam.items()}
    sc = collections.defaultdict(lambda: [0, 0])
    for r in valid:
        sc[routes[r]["scenario"]][0] += 1
        sc[routes[r]["scenario"]][1] += r in fail_routes
    summary["scenario"] = {k: v for k, v in sc.items()}

    # ---------------- 3. step-level dataset
    timed = {r for r in fail_routes if first_kind[r] == "collision"}
    untimed = fail_routes - timed
    feats, ys, gidx, tti_all, fam_idx, rnames = [], [], [], [], [], []
    rlist = sorted(r for r in data if r not in untimed)
    for ri, r in enumerate(rlist):
        d = data[r]
        f = step_features(d)
        end = d["fail_step"] if r in timed else d["n"]
        idx = np.arange(SKIP_START, end)
        if len(idx) == 0:
            continue
        tti = (end - idx) * DT if r in timed else np.full(len(idx), np.inf)
        y = (tti <= HORIZON_S).astype(int)
        feats.append({k: v[idx] for k, v in f.items()})
        ys.append(y)
        gidx.append(np.full(len(idx), ri))
        tti_all.append(tti)
        rnames.append(r)
    F = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
    y = np.concatenate(ys)
    g = np.concatenate(gidx)
    tti = np.concatenate(tti_all)
    moving = F["ego_speed"] > 2.0
    fail_idx = sorted({int(gi) for gi in np.unique(g[y == 1])})
    summary["dataset"] = {"steps": int(len(y)), "pos": int(y.sum()), "moving": int(moving.sum()),
                          "pos_moving": int(y[moving].sum()), "routes": len(rnames), "timed_failures": len(fail_idx),
                          "excluded_untimed": sorted(untimed, key=int)}

    # single-signal AUROC on moving steps
    single = {}
    for k, v in F.items():
        a = auroc(y[moving], v[moving])
        single[k] = {"auroc": max(a, 1 - a), "direction": "higher" if a >= 0.5 else "lower",
                     "auprc": auprc(y[moving], v[moving] if a >= 0.5 else -v[moving])}
    summary["single_signal"] = single

    models = {}
    for name, cols in FEATURE_SETS.items():
        X = np.stack([F[c] for c in cols], axis=1)
        for m in ("logreg", "gbm"):
            s = oof_scores(X, y, g, m)
            al = alarm_eval(s, y, g, tti, fail_idx)
            models[f"{name}|{m}"] = {
                "auroc_all": auroc(y, s), "auroc_moving": auroc(y[moving], s[moving]),
                "auprc_moving": auprc(y[moving], s[moving]),
                "alarm": {str(k): {kk: vv for kk, vv in v.items() if kk != "leads"} for k, v in al.items()},
                "leads": {str(k): {rnames[ri]: v["leads"][ri] for ri in fail_idx} for k, v in al.items()},
            }
            print(f"{name:40s} {m:6s} AUROC moving {models[f'{name}|{m}']['auroc_moving']:.3f} "
                  f"warn>=1s@1/min {al[1.0]['warn_1s']:.2f} @3/min {al[3.0]['warn_1s']:.2f}", flush=True)
    summary["models"] = models

    # ---------------- 5. benign instability episodes per family
    thr = {k: float(np.quantile(F[k][moving & (y == 0)], 0.95))
           for k in ("path_disagree_mean_1s", "speed_js_mean_1s", "steer_tv_1s")}
    inst = moving & ((F["path_disagree_mean_1s"] > thr["path_disagree_mean_1s"]) |
                     (F["speed_js_mean_1s"] > thr["speed_js_mean_1s"]) | (F["steer_tv_1s"] > thr["steer_tv_1s"]))
    episodes = collections.defaultdict(lambda: {"benign": 0, "failure_linked": 0})
    for ri, r in enumerate(rnames):
        m = np.where(g == ri)[0]
        a = inst[m]
        yy = y[m]
        i = 0
        while i < len(a):
            if not a[i]:
                i += 1
                continue
            j = i
            gap = 0
            while j + 1 < len(a) and (a[j + 1] or gap < 10):
                gap = 0 if a[j + 1] else gap + 1
                j += 1
            j -= gap
            if j - i + 1 >= 10:
                key = "failure_linked" if yy[i:j + 1].any() else "benign"
                episodes[fam_of(r)][key] += 1
            i = j + 1 + gap
    summary["instability_thresholds"] = thr
    summary["episodes"] = {k: dict(v) for k, v in episodes.items()}
    summary["timed_by_family"] = dict(collections.Counter(fam_of(rnames[ri]) for ri in fail_idx))
    summary["pos_steps_by_family"] = dict(collections.Counter(fam_of(rnames[int(gi)]) for gi in g[y == 1]))

    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    keys = ["route", "scenario", "family", "town", "first", "kinds", "geometry", "maneuver", "mechanism", "otype",
            "v_impact", "v_1p5s", "braked_1p5s", "brake_onset_s", "rel_yaw_deg", "rel_x", "rel_y", "det_frac_2s", "first_det_s",
            "geo_onset_ttc2", "geo_onset_ttc3", "ds"]
    with open(out / "failures.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["det_3s", "det_2s", "det_1s", "det_0.5s"])
        for c in cats:
            det = c.get("det", {})
            w.writerow([c.get(k, "") for k in keys] + [det.get(s, "") for s in ("3s", "2s", "1s", "0.5s")])
    print("wrote", out)


if __name__ == "__main__":
    main()
