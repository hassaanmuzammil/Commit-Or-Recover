"""Offline failure and early-warning analysis for frozen TF++ on Bench2Drive (schema-v2 logs).

Used by notebooks/early_warning_analysis.ipynb. Pure numpy + scikit-learn + matplotlib.

Conventions
  * Ego frame: x forward, y right (m); yaw in rad, positive turns toward +y. TF++'s predicted path,
    predicted boxes and the logged ground-truth actors all use this frame.
  * Monitor features use only TF++-observable signals (planner outputs, controls, ego state from
    TF++'s own sensors, TF++'s predicted boxes). Ground truth ("gt_*") is used for labels, actor
    matching and clearly-marked offline reference rows only.
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DT = 0.05
HZ = 20
HORIZON_S = 3.0
SKIP_START = 40                        # first 2 s: CARLA holds the brake
POST_CONTACT_S = 3.0                   # steps this long after any contact are excluded
TRACK_STEPS = 120                      # struck-actor track kept for 6 s before each collision
BOX_RANGE = 32.0                       # TF++ box range (config.max_x / max_y)
EGO_EXT_X, EGO_EXT_Y = 2.4508, 1.0642
TAU = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 3)   # rollout grid for TTC / clearance
TAUP = TAU[::2]                                       # coarser grid for the time-gap (PET) check
TARGET_SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.8889, 16.0, 17.7778, 20.0])
ACC_UP, ACC_DOWN = 2.0, 4.0            # speed profile used to time-parameterise TF++'s path
BICYCLES = ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets")
ID_RE = re.compile(r"type=(\S+) and id=(\d+)")
FAILURE_KEYS = ["collisions_vehicle", "collisions_pedestrian", "collisions_layout",
                "outside_route_lanes", "vehicle_blocked", "route_timeout"]
COLLISION_KEYS = FAILURE_KEYS[:3]

FAMILY = {
    **{s: "junction" for s in [
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
        "NonSignalizedJunctionLeftTurnEnterFlow", "T_Junction", "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority", "VehicleTurningRoute", "BlockedIntersection",
        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign"]},
    **{s: "lane_merge" for s in [
        "EnterActorFlow", "HighwayExit", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2",
        "SequentialLaneChange", "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "ParkingExit"]},
    **{s: "obstacle_detour" for s in [
        "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
        "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays",
        "VehicleOpensDoorTwoWays", "InvadingTurn"]},
    **{s: "cut_in_hazard" for s in [
        "HighwayCutIn", "ParkingCutIn", "StaticCutIn", "HardBreakRoute", "ControlLoss",
        "YieldToEmergencyVehicle"]},
    **{s: "vru_crossing" for s in [
        "PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian",
        "VehicleTurningRoutePedestrian", "CrossingBicycleFlow"]},
}


# =========================================================================== geometry

def _box_discs(ex, ey):
    """Three discs along the box's long axis: offsets d and radius ey cover a box with half-extents ex, ey."""
    ex = np.maximum(np.asarray(ex, float), 0.2)
    ey = np.maximum(np.asarray(ey, float), 0.2)
    return np.maximum(ex - ey, 0.0), ey


EGO_D, EGO_R = (float(v) for v in _box_discs(EGO_EXT_X, EGO_EXT_Y))


def ego_discs(xs, ys, hs):
    off = np.array([-EGO_D, 0.0, EGO_D])
    return xs[:, None] + off * np.cos(hs)[:, None], ys[:, None] + off * np.sin(hs)[:, None]


def ctrv_rollout(v, w, tau=TAU):
    if abs(w) < 1e-3:
        return v * tau, np.zeros_like(tau), np.zeros_like(tau)
    h = w * tau
    return v / w * np.sin(h), v / w * (1 - np.cos(h)), h


def plan_rollout(path, v0, vt, tau=TAU):
    """Ego centre along TF++'s distance-sampled path, timed by a speed profile from v0 toward the target speed."""
    pts = np.vstack([[0.0, 0.0], np.asarray(path, float)])
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    keep = L > 1e-3
    if not keep.any():
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        seg, L = np.diff(pts, axis=0), np.array([1.0])
    else:
        pts = np.vstack([pts[0], pts[1:][keep]])
        seg = np.diff(pts, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
    d = seg[-1] / L[-1]
    s_end = L.sum()
    pts = np.vstack([pts, pts[-1] + d * max(80.0 - s_end, 1.0)])          # straight extension
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    scum = np.r_[0.0, np.cumsum(L)]
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    v0 = max(float(v0), 0.0)
    vt = max(float(vt), 0.0)
    a = ACC_UP if vt >= v0 else -ACC_DOWN
    t1 = abs(vt - v0) / abs(a)
    s = np.where(tau < t1, v0 * tau + 0.5 * a * tau ** 2, v0 * t1 + 0.5 * a * t1 ** 2 + vt * (tau - t1))
    s = np.clip(s, 0, scum[-1])
    x = np.interp(s, scum, pts[:, 0])
    y = np.interp(s, scum, pts[:, 1])
    idx = np.clip(np.searchsorted(scum, s, side="right") - 1, 0, len(hd) - 1)
    return x, y, hd[idx]


def actor_discs(cx, cy, yaw, vx, vy, d, tau):
    ax = cx[:, None] + vx[:, None] * tau[None]
    ay = cy[:, None] + vy[:, None] * tau[None]
    off = np.stack([-d, np.zeros_like(d), d], 1)
    X = ax[:, :, None] + off[:, None, :] * np.cos(yaw)[:, None, None]
    Y = ay[:, :, None] + off[:, None, :] * np.sin(yaw)[:, None, None]
    return X, Y


def interaction(ego_xyh, ego_xyh_p, cx, cy, yaw, vx, vy, ex, ey):
    """Per actor: TTC (first time the boxes overlap), minimum clearance over the horizon (m, negative = overlap),
    spatial conflict (the two swept paths overlap at any pair of times) and the time gap at that conflict
    (min |t_ego - t_actor| over overlapping pairs; 0 = they arrive together)."""
    n = len(cx)
    if n == 0:
        z = np.zeros(0)
        return z, z, z.astype(bool), z
    d, r = _box_discs(ex, ey)
    EX, EY = ego_discs(*ego_xyh)
    X, Y = actor_discs(cx, cy, yaw, vx, vy, d, TAU)
    g = np.sqrt((EX[None, :, :, None] - X[:, :, None, :]) ** 2 + (EY[None, :, :, None] - Y[:, :, None, :]) ** 2)
    g = (g - (EGO_R + r[:, None, None, None])).min(axis=(2, 3))            # (N, T)
    hit = g < 0
    ttc = np.where(hit.any(1), TAU[np.argmax(hit, axis=1)], np.inf)
    clear = g.min(axis=1)
    EXp, EYp = ego_discs(*ego_xyh_p)
    Xp, Yp = actor_discs(cx, cy, yaw, vx, vy, d, TAUP)
    gp = np.sqrt((EXp[None, :, :, None, None] - Xp[:, None, None, :, :]) ** 2 +
                 (EYp[None, :, :, None, None] - Yp[:, None, None, :, :]) ** 2)
    ov = (gp - (EGO_R + r[:, None, None, None, None])).min(axis=(2, 4)) < 0   # (N, Te, Ta)
    gap = np.abs(TAUP[:, None] - TAUP[None, :])
    conflict = ov.any(axis=(1, 2))
    pet = np.where(conflict, np.where(ov, gap[None], np.inf).min(axis=(1, 2)), np.inf)
    return ttc, clear, conflict, pet


def ctrv_ttc(v, w, cx, cy, yaw, vx, vy, ex, ey):
    if len(cx) == 0:
        return np.zeros(0)
    d, r = _box_discs(ex, ey)
    EX, EY = ego_discs(*ctrv_rollout(v, w))
    X, Y = actor_discs(cx, cy, yaw, vx, vy, d, TAU)
    g = np.sqrt((EX[None, :, :, None] - X[:, :, None, :]) ** 2 + (EY[None, :, :, None] - Y[:, :, None, :]) ** 2)
    hit = ((g - (EGO_R + r[:, None, None, None])).min(axis=(2, 3))) < 0
    return np.where(hit.any(1), TAU[np.argmax(hit, axis=1)], np.inf)


def transform_prev_path(path_prev, pos_prev, yaw_prev, pos, yaw):
    c, s = math.cos(yaw_prev), math.sin(yaw_prev)
    wx = c * path_prev[:, 0] - s * path_prev[:, 1] + pos_prev[0] - pos[0]
    wy = s * path_prev[:, 0] + c * path_prev[:, 1] + pos_prev[1] - pos[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * wx + s * wy, -s * wx + c * wy], axis=1)


def point_to_polyline(points, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((points[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================================================================== loading

def load_routes(xml_path):
    out = {}
    for r in ET.parse(xml_path).getroot().iter("route"):
        sc = [s.get("type") for s in r.find("scenarios")]
        st = sc[0] if sc else "none"
        out[r.get("id")] = {"town": r.get("town"), "scenario": st, "family": FAMILY.get(st, "other")}
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
    return {"status": rec["status"], "ds": rec["scores"]["score_composed"], "rc": rec["scores"]["score_route"],
            "infractions": {k: len(v) for k, v in inf.items()}, "collision_events": events,
            "failure_kinds": [k for k in FAILURE_KEYS if inf[k]], "length": rec["meta"]["route_length"]}


def pick_log(route_dir):
    files = sorted(p for p in Path(route_dir).glob("*.jsonl") if not p.name.startswith("zz_"))
    return files[-1] if files else None


def _boxes(r):
    b = np.array([bb for bb in r["pred_boxes"] if int(round(bb[7])) in (0, 1, 4)], float).reshape(-1, 9)
    return b


def _dyn(r):
    return [a for a in r["gt_actors"] if a["class"] in ("vehicle", "walker")]


def _match(act, b):
    """Nearest class-compatible predicted box to a GT actor. Returns (index, distance) or (None, inf)."""
    if len(b) == 0:
        return None, np.inf
    ok = (np.round(b[:, 7]) == 1) if act["class"] == "walker" else np.isin(np.round(b[:, 7]), (0, 4))
    if act["type"].startswith(BICYCLES):
        ok = np.isin(np.round(b[:, 7]), (0, 1, 4))
    if not ok.any():
        return None, np.inf
    dist = np.where(ok, np.hypot(b[:, 0] - act["x"], b[:, 1] - act["y"]), np.inf)
    j = int(np.argmin(dist))
    return j, float(dist[j])


def match_tol(act):
    return 2.0 if act["class"] == "walker" else 3.0


class BoxTracker:
    """Frame-to-frame association of TF++'s predicted boxes in TF++'s own odometry frame (ego_position / ego_yaw,
    both TF++-observable). TF++'s box speed output is always 0 in this checkpoint and its box heading is often
    flipped by 180 deg, so actor velocity is estimated from the tracked box centres instead."""

    def __init__(self, gate=2.5, window=10, min_hist=4):
        self.tracks = []          # each: {"hist": [(step, wx, wy)], "cls": int}
        self.gate, self.window, self.min_hist = gate, window, min_hist

    def update(self, step, b, pos, yaw):
        """Returns per-box (vx, vy) in the current ego frame and a heading (rad, ego frame)."""
        c, s = math.cos(yaw), math.sin(yaw)
        n = len(b)
        wx = pos[0] + c * b[:, 0] - s * b[:, 1] if n else np.zeros(0)
        wy = pos[1] + s * b[:, 0] + c * b[:, 1] if n else np.zeros(0)
        self.tracks = [t for t in self.tracks if step - t["hist"][-1][0] <= 5]
        pred = []
        for t in self.tracks:
            h = t["hist"]
            if len(h) >= 2 and h[-1][0] > h[0][0]:
                dtv = (h[-1][0] - h[0][0]) * DT
                vx, vy = (h[-1][1] - h[0][1]) / dtv, (h[-1][2] - h[0][2]) / dtv
            else:
                vx = vy = 0.0
            lagt = (step - h[-1][0]) * DT
            pred.append((h[-1][1] + vx * lagt, h[-1][2] + vy * lagt))
        assigned, used = [-1] * n, set()
        if n and pred:
            P = np.array(pred)
            D = np.hypot(wx[:, None] - P[None, :, 0], wy[:, None] - P[None, :, 1])
            for flat in np.argsort(D, axis=None):
                i, j = divmod(int(flat), D.shape[1])
                if D[i, j] > self.gate:
                    break
                if assigned[i] < 0 and j not in used:
                    assigned[i] = j
                    used.add(j)
        vel = np.zeros((n, 2))
        head = b[:, 4].copy() if n else np.zeros(0)
        for i in range(n):
            if assigned[i] < 0:
                self.tracks.append({"hist": [(step, wx[i], wy[i])]})
                continue
            t = self.tracks[assigned[i]]
            t["hist"].append((step, wx[i], wy[i]))
            t["hist"] = [h for h in t["hist"] if step - h[0] <= self.window]
            h = t["hist"]
            if len(h) >= self.min_hist:
                dtv = (h[-1][0] - h[0][0]) * DT
                vwx, vwy = (h[-1][1] - h[0][1]) / dtv, (h[-1][2] - h[0][2]) / dtv
                if math.hypot(vwx, vwy) >= 0.5:
                    vel[i] = (c * vwx + s * vwy, -s * vwx + c * vwy)
                    if math.hypot(vwx, vwy) >= 1.0:
                        head[i] = math.atan2(vel[i, 1], vel[i, 0])
        return vel, head


def group_contacts(rows, ck):
    """Distinct collision events from the per-step `collisions` field (exact CARLA frames).
    Contacts with the same actor less than 1 s apart are one event."""
    ck_ids = {e["id"] for e in ck["collision_events"]} if ck else set()
    ck_types = {e["type"] for e in ck["collision_events"]} if ck else set()
    events, last = [], {}
    for i, r in enumerate(rows):
        for e in r["collisions"]:
            key = (e["other_id"], e["other_type"])
            if key in last and i - last[key][1] <= HZ:
                ev = events[last[key][0]]
                ev["last_step"] = i
                ev["n_contacts"] += 1
                ev["impulse_max"] = max(ev["impulse_max"], e["impulse"])
                last[key] = (last[key][0], i)
                continue
            counted = e["other_id"] in ck_ids and (e["other_id"] != 0 or e["other_type"] in ck_types)
            events.append({"step": i, "last_step": i, "frame": e["frame"], "time": r["timestamp"],
                           "other_id": e["other_id"], "other_type": e["other_type"], "impulse": e["impulse"],
                           "impulse_max": e["impulse"], "n_contacts": 1, "counted": bool(counted)})
            last[key] = (len(events) - 1, i)
    return events


def extract_route(log_path, ck):
    rows = [json.loads(l) for l in open(log_path)]
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows], float)
    t, speed, yaw, yaw_rate, acc = g("timestamp"), g("ego_speed"), g("ego_yaw"), g("yaw_rate"), g("acceleration")
    steer, thr, brk, tss = g("steer"), g("throttle"), g("brake"), g("target_speed_scalar")
    pos = np.array([r["ego_position"] for r in rows], float)
    probs = np.array([r["target_speed_probabilities"] for r in rows], float)
    path = np.array([r["predicted_path"] for r in rows], float)
    tp = np.array([r["target_point"] for r in rows], float)
    cmd = np.array([r["command"] for r in rows], int)
    frame = np.array([(r["gt_ego"] or {}).get("frame", -1) for r in rows], int)
    flags = np.array([[r["tfpp_emergency_stop"], r["tfpp_force_move"], r["tfpp_stop_sign_stop"],
                       r["tfpp_bb_in_front"]] for r in rows], bool)

    p = np.clip(probs, 1e-9, 1.0)
    entropy = -(p * np.log(p)).sum(1)
    ps = np.sort(p, axis=1)
    m = 0.5 * (p[1:] + p[:-1])
    js = np.r_[0.0, 0.5 * (p[1:] * np.log(p[1:] / m)).sum(1) + 0.5 * (p[:-1] * np.log(p[:-1] / m)).sum(1)]
    pd = np.zeros(n)
    for i in range(1, n):
        prev = transform_prev_path(path[i - 1], pos[i - 1], yaw[i - 1], pos[i], yaw[i])
        pd[i] = point_to_polyline(path[i, :8], prev).mean()
    seg = np.diff(path, axis=1)
    hd = np.arctan2(seg[:, :, 1], seg[:, :, 0])

    events = group_contacts(rows, ck)
    track_steps = {}
    for k, ev in enumerate(events):
        for s in range(max(0, ev["step"] - TRACK_STEPS), ev["step"] + 1):
            track_steps.setdefault(s, []).append(k)
    tracks = [[] for _ in events]

    G = {k: np.full(n, np.inf) for k in ("ttc_plan", "pet_plan", "ttc_ctrv", "gt_ttc_plan", "gt_ttc_ctrv", "gt_pet_plan")}
    G.update({k: np.full(n, 30.0) for k in ("clear_plan", "gt_clear_plan")})
    G["conflict_plan"] = np.zeros(n)
    G["corr_dist"] = np.full(n, 40.0)
    G["corr_close"] = np.zeros(n)
    G["n_boxes"] = np.zeros(n)
    det_sample = []
    tracker = BoxTracker()
    for i, r in enumerate(rows):
        v, w = speed[i], yaw_rate[i]
        ego = plan_rollout(path[i], v, tss[i])
        ego_p = plan_rollout(path[i], v, tss[i], TAUP)
        b = _boxes(r)
        vel, head = tracker.update(i, b, pos[i], yaw[i])
        G["n_boxes"][i] = len(b)
        if len(b):
            bvx, bvy = vel[:, 0], vel[:, 1]
            ttc, clear, conf, pet = interaction(ego, ego_p, b[:, 0], b[:, 1], head, bvx, bvy, b[:, 2], b[:, 3])
            G["ttc_plan"][i], G["clear_plan"][i] = ttc.min(), min(clear.min(), 30.0)
            G["pet_plan"][i], G["conflict_plan"][i] = pet.min(), float(conf.any())
            G["ttc_ctrv"][i] = ctrv_ttc(v, w, b[:, 0], b[:, 1], head, bvx, bvy, b[:, 2], b[:, 3]).min()
            ahead = (b[:, 0] > 0) & (np.abs(b[:, 1]) < 2.5)
            if ahead.any():
                j = int(np.argmin(np.where(ahead, b[:, 0], np.inf)))
                G["corr_dist"][i] = min(b[j, 0], 40.0)
                G["corr_close"][i] = v - bvx[j]
        a = _dyn(r)
        if a:
            A = {k: np.array([x[k] for x in a], float) for k in ("x", "y", "yaw", "vx", "vy", "ext_x", "ext_y")}
            ttc, clear, conf, pet = interaction(ego, ego_p, A["x"], A["y"], A["yaw"], A["vx"], A["vy"], A["ext_x"], A["ext_y"])
            G["gt_ttc_plan"][i], G["gt_clear_plan"][i], G["gt_pet_plan"][i] = ttc.min(), min(clear.min(), 30.0), pet.min()
            G["gt_ttc_ctrv"][i] = ctrv_ttc(v, w, A["x"], A["y"], A["yaw"], A["vx"], A["vy"], A["ext_x"], A["ext_y"]).min()
            if i % 10 == 0:                                    # detection-recall context sample
                for x in a:
                    if abs(x["x"]) <= BOX_RANGE and abs(x["y"]) <= BOX_RANGE:
                        jj, dm = _match(x, b)
                        ok = dm < match_tol(x)
                        det_sample.append((math.hypot(x["x"], x["y"]), x["class"] == "walker", ok, x["speed"],
                                           b[jj, 5] if ok else np.nan, math.hypot(*vel[jj]) if ok else np.nan,
                                           abs(math.degrees(wrap(b[jj, 4] - x["yaw"]))) if ok else np.nan,
                                           abs(math.degrees(wrap(head[jj] - x["yaw"]))) if ok else np.nan))
        for k in track_steps.get(i, []):
            ev = events[k]
            act = next((x for x in r["gt_actors"] if x["id"] == ev["other_id"]), None) if ev["other_id"] > 0 else None
            rec = {"k": i - ev["step"], "present": act is not None}
            if act is not None and act["class"] in ("vehicle", "walker"):
                one = lambda key: np.array([act[key]], float)
                ttc, clear, conf, pet = interaction(ego, ego_p, one("x"), one("y"), one("yaw"), one("vx"), one("vy"),
                                                    one("ext_x"), one("ext_y"))
                jj, dm = _match(act, b)
                det = dm < match_tol(act)
                rec.update({"x": act["x"], "y": act["y"], "yaw": act["yaw"], "speed": act["speed"],
                            "class": act["class"], "in_range": abs(act["x"]) <= BOX_RANGE and abs(act["y"]) <= BOX_RANGE,
                            "detected": bool(det), "det_err": dm, "gt_ttc_plan": float(ttc[0]),
                            "gt_clear_plan": float(clear[0]), "gt_conflict": bool(conf[0]), "gt_pet": float(pet[0]),
                            "gt_ttc_ctrv": float(ctrv_ttc(v, w, one("x"), one("y"), one("yaw"), one("vx"), one("vy"),
                                                          one("ext_x"), one("ext_y"))[0])})
                if det:
                    bb = b[jj:jj + 1]
                    ttc_b, clear_b, _, _ = interaction(ego, ego_p, bb[:, 0], bb[:, 1], head[jj:jj + 1],
                                                       vel[jj:jj + 1, 0], vel[jj:jj + 1, 1], bb[:, 2], bb[:, 3])
                    rec.update({"pred_ttc_plan": float(ttc_b[0]), "pred_clear_plan": float(clear_b[0]),
                                "pred_speed_tracked": float(math.hypot(*vel[jj])), "pred_speed_raw": float(bb[0, 5])})
            elif act is not None:
                rec.update({"class": act["class"]})
            tracks[k].append(rec)
    for k, ev in enumerate(events):
        ev["track"] = tracks[k]
    return {"n": n, "t": t, "frame": frame, "speed": speed, "yaw": yaw, "yaw_rate": yaw_rate, "acc": acc,
            "steer": steer, "thr": thr, "brk": brk, "tss": tss, "entropy": entropy, "margin": ps[:, -1] - ps[:, -2],
            "pmax": ps[:, -1], "js": js, "pd": pd, "end_lat": path[:, -1, 1], "end_heading": hd[:, -1],
            "curvature": np.abs(wrap(hd[:, -1] - hd[:, 0])), "tp": tp, "cmd": cmd, "flags": flags,
            "events": events, "det_sample": np.array(det_sample, float).reshape(-1, 8), **G}


def build_cache(results_dir, routes_xml, cache_path, log=print):
    results_dir = Path(results_dir)
    routes = load_routes(routes_xml)
    ckpts = {p.stem: parse_checkpoint(p) for p in sorted((results_dir / "checkpoints").glob("*.json"))}
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = pickle.load(open(cache_path, "rb"))
    else:
        data = {}
        for i, (rid, ck) in enumerate(ckpts.items()):
            lp = pick_log(results_dir / "logs" / rid)
            if ck is None or lp is None:
                continue
            data[rid] = extract_route(lp, ck)
            if (i + 1) % 20 == 0:
                log(f"  parsed {i + 1}/{len(ckpts)} routes")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(data, open(cache_path, "wb"))
    return routes, ckpts, data


# =========================================================================== events and categories

def classify_event(ev, d, fam):
    """Situation (what the ego was doing) and collision geometry (how it hit) for one collision event."""
    s = ev["step"]
    otype = ev["other_type"]
    v_pre = float(d["speed"][max(0, s - 2)])
    cmds = d["cmd"][max(0, s - 60): s + 1]
    pre = [r for r in ev["track"] if "x" in r and r["k"] <= -2]
    last = pre[-1] if pre else None
    rel_yaw = abs(math.degrees(last["yaw"])) if last else float("nan")
    if otype.startswith("walker"):
        geom = "pedestrian"
    elif otype.startswith(BICYCLES):
        geom = "cyclist"
    elif otype.startswith("static"):
        geom = "static object"
    elif v_pre < 1.0:
        geom = "ego stopped"
    elif last is None:
        geom = "unknown"
    elif rel_yaw > 135:
        geom = "head-on"
    elif rel_yaw > 45:
        geom = "crossing"
    elif last["x"] > EGO_EXT_X and abs(last["y"]) < 1.5:
        geom = "rear-end (ego hits lead)"
    elif last["x"] < -EGO_EXT_X and abs(last["y"]) < 1.5:
        geom = "rear-ended"
    else:
        geom = "side-swipe"

    if geom in ("pedestrian", "cyclist"):
        sit = "pedestrian / cyclist"
    elif geom == "static object":
        sit = "static object"
    elif geom in ("ego stopped", "rear-ended"):
        sit = "struck while stopped / from behind"
    elif np.isin(cmds, [0, 1, 2]).any():
        sit = "junction turn / crossing"
    elif np.isin(cmds, [4, 5]).any():
        sit = "lane change"
    elif fam == "obstacle_detour":
        sit = "obstacle bypass"
    elif fam == "lane_merge":
        sit = "merge / exit"
    elif geom == "rear-end (ego hits lead)":
        sit = "rear-end (lead vehicle)"
    else:
        sit = "other vehicle conflict"
    return {"situation": sit, "geometry": geom, "v_pre": v_pre, "rel_yaw_deg": rel_yaw}


def event_table(routes, ckpts, data):
    rows = []
    for rid, d in data.items():
        fam = routes[rid]["family"]
        for k, ev in enumerate(d["events"]):
            if ev["other_type"].startswith("static.road"):
                continue
            c = classify_event(ev, d, fam)
            prev = [e for e in d["events"][:k] if not e["other_type"].startswith("static.road")]
            clean = ev["step"] >= SKIP_START + HZ and all(ev["step"] - e["last_step"] > HORIZON_S * HZ for e in prev)
            rows.append({"route": rid, "event": k, "scenario": routes[rid]["scenario"], "family": fam,
                         "town": routes[rid]["town"], "step": ev["step"], "frame": ev["frame"],
                         "time_s": round(ev["time"], 2), "other_type": ev["other_type"], "other_id": ev["other_id"],
                         "impulse": round(ev["impulse_max"], 1), "duration_s": (ev["last_step"] - ev["step"]) / HZ,
                         "counted": ev["counted"], "clean_history": bool(clean),
                         "first_in_route": all(not e["counted"] for e in prev), **c})
    return rows


# =========================================================================== step features

def roll_mean(x, w):
    c = np.cumsum(np.r_[0.0, x])
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


def roll_max(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(np.r_[np.full(w - 1, x[0]), x], w).max(axis=1)


def roll_min(x, w):
    return -roll_max(-x, w)


def roll_std(x, w):
    m = roll_mean(x, w)
    return np.sqrt(np.maximum(roll_mean(x * x, w) - m * m, 0))


def lag(x, k):
    return np.r_[np.full(k, x[0]), x[:-k]]


def step_features(d):
    """Per-step monitor features. Everything except the `gt_` keys is TF++-observable."""
    risk = lambda ttc: np.clip(HORIZON_S - ttc, 0, HORIZON_S)
    dsteer = np.abs(np.r_[0.0, np.diff(d["steer"])])
    sg = np.sign(np.r_[0.0, np.diff(d["steer"])])
    rev = np.r_[0.0, (sg[1:] * sg[:-1] < 0).astype(float)]
    pedal = np.where(d["brk"] > 0.5, -1, np.where(d["thr"] > 0.05, 1, 0))
    switch = np.r_[0.0, (np.diff(pedal) != 0).astype(float)]
    jerk = np.minimum(np.abs(np.r_[0.0, np.diff(d["acc"])]) / DT, 200)
    route_align = np.abs(wrap(np.arctan2(d["tp"][:, 1], d["tp"][:, 0]) - d["end_heading"]))
    f = {
        # current frame: ego state and TF++ confidence
        "ego_speed": d["speed"], "target_speed": d["tss"], "speed_gap": d["tss"] - d["speed"],
        "speed_entropy": d["entropy"], "speed_margin": d["margin"], "speed_pmax": d["pmax"],
        "path_curvature": d["curvature"], "path_end_lat_abs": np.abs(d["end_lat"]),
        "brake": d["brk"], "steer_abs": np.abs(d["steer"]),
        # current frame: geometry from TF++'s predicted boxes and its own planned path
        "risk_plan": risk(d["ttc_plan"]), "risk_ctrv": risk(d["ttc_ctrv"]),
        "clear_plan": np.clip(d["clear_plan"], -3, 15), "pet_plan": np.clip(d["pet_plan"], 0, HORIZON_S),
        "conflict_plan": d["conflict_plan"], "corr_dist": d["corr_dist"], "corr_close": d["corr_close"],
        "n_boxes": d["n_boxes"], "bb_in_front": d["flags"][:, 3].astype(float),
        # temporal (proposal section 2.2)
        "route_align": route_align,
        "risk_plan_max_1s": roll_max(risk(d["ttc_plan"]), 20),
        "clear_plan_trend_1s": np.clip(d["clear_plan"], -3, 15) - lag(np.clip(d["clear_plan"], -3, 15), 20),
    }
    for w, tag in ((10, "0.5s"), (20, "1s"), (40, "2s"), (60, "3s")):
        f[f"path_disagree_mean_{tag}"] = roll_mean(d["pd"], w)
        f[f"path_disagree_max_{tag}"] = roll_max(d["pd"], w)
        f[f"path_end_lat_std_{tag}"] = roll_std(d["end_lat"], w)
        f[f"speed_js_mean_{tag}"] = roll_mean(d["js"], w)
        f[f"steer_tv_{tag}"] = roll_mean(dsteer, w) * w
        f[f"steer_reversals_{tag}"] = roll_mean(rev, w) * w
        f[f"pedal_switches_{tag}"] = roll_mean(switch, w) * w
        f[f"jerk_mean_{tag}"] = roll_mean(jerk, w)
        f[f"entropy_change_{tag}"] = d["entropy"] - lag(d["entropy"], w)
        f[f"curvature_change_{tag}"] = np.abs(d["curvature"] - lag(d["curvature"], w))
    f["path_disagree"] = d["pd"]
    f["speed_js"] = d["js"]
    # offline reference only (privileged ground truth)
    f["gt_risk_plan"] = risk(d["gt_ttc_plan"])
    f["gt_risk_ctrv"] = risk(d["gt_ttc_ctrv"])
    f["gt_clear_plan"] = np.clip(d["gt_clear_plan"], -3, 15)
    return f


TEMPORAL = lambda tag="1s", tag2="2s": [
    f"path_disagree_mean_{tag}", f"path_disagree_max_{tag}", f"path_disagree_mean_{tag2}", f"path_end_lat_std_{tag}",
    f"curvature_change_{tag}", f"speed_js_mean_{tag}", f"speed_js_mean_{tag2}", f"entropy_change_{tag}",
    f"steer_tv_{tag}", f"steer_reversals_{tag}", f"pedal_switches_{tag}", f"jerk_mean_{tag}", "speed_gap", "route_align"]
CONFIDENCE = ["ego_speed", "target_speed", "speed_entropy", "speed_margin", "speed_pmax"]
GEOMETRY = ["ego_speed", "risk_plan", "risk_ctrv", "clear_plan", "pet_plan", "conflict_plan", "corr_dist",
            "corr_close", "n_boxes", "bb_in_front"]
CURRENT = sorted(set(CONFIDENCE + GEOMETRY + ["path_curvature", "path_end_lat_abs"]), key=(CONFIDENCE + GEOMETRY + ["path_curvature", "path_end_lat_abs"]).index)
TEMPORAL_GEOM = ["risk_plan_max_1s", "clear_plan_trend_1s"]

METHODS = {
    # name: (kind, features or single feature)
    "Rule: speed": ("rule", "ego_speed"),
    "Rule: TF++ entropy": ("rule", "speed_entropy"),
    "Rule: path disagreement 1 s": ("rule", "path_disagree_mean_1s"),
    "Rule: speed-distribution JS 1 s": ("rule", "speed_js_mean_1s"),
    "Rule: steering variation 1 s": ("rule", "steer_tv_1s"),
    "Rule: TTC on TF++ path (boxes)": ("rule", "risk_plan"),
    "LR: speed": ("lr", ["ego_speed"]),
    "LR: TF++ confidence": ("lr", CONFIDENCE),
    "LR: temporal instability": ("lr", ["ego_speed"] + TEMPORAL()),
    "LR: current-frame geometry": ("lr", GEOMETRY),
    "LR: current frame (geometry + confidence)": ("lr", CURRENT),
    "LR: current frame + temporal": ("lr", CURRENT + TEMPORAL() + TEMPORAL_GEOM),
    "Offline ref: GT geometry on TF++ path": ("lr", ["ego_speed", "gt_risk_plan", "gt_risk_ctrv", "gt_clear_plan"]),
}


# =========================================================================== dataset

def build_dataset(routes, ckpts, data, events):
    """Step-level dataset. Label = a counted collision within 3 s. Excludes the first 2 s of each route,
    the 3 s after any contact, the 3 s before uncounted contacts, and routes whose failure has no timestamp."""
    ev_by_route = collections.defaultdict(list)
    for e in events:
        ev_by_route[e["route"]].append(e)
    untimed = {r for r, c in ckpts.items() if c and c["failure_kinds"] and not any(e["counted"] for e in ev_by_route[r])}
    feats, cols = None, collections.defaultdict(list)
    meta = collections.defaultdict(list)
    eid = 0
    event_index = []
    for rid in sorted(data, key=int):
        if rid in untimed:
            continue
        d = data[rid]
        f = step_features(d)
        n = d["n"]
        keep = np.ones(n, bool)
        keep[:SKIP_START] = False
        tti = np.full(n, np.inf)
        ev_id = np.full(n, -1)
        for e in sorted(ev_by_route[rid], key=lambda e: e["step"]):
            s = e["step"]
            last = d["events"][e["event"]]["last_step"]
            keep[s: min(n, last + int(POST_CONTACT_S * HZ) + 1)] = False
            lo = max(0, s - int(HORIZON_S * HZ))
            if not e["counted"]:
                keep[lo:s] = False
                continue
            if not e["clean_history"]:
                keep[lo:s] = False
                continue
            idx = np.arange(lo, s)
            closer = (s - idx) * DT < tti[idx]
            tti[idx[closer]] = (s - idx[closer]) * DT
            ev_id[idx[closer]] = eid
            event_index.append({**{k: v for k, v in e.items()}, "eid": eid})
            eid += 1
        # an event that overlaps a later exclusion keeps whatever remains
        idx = np.where(keep)[0]
        for k, v in f.items():
            cols[k].append(v[idx])
        meta["route"].append(np.full(len(idx), int(rid)))
        meta["step"].append(idx)
        meta["tti"].append(tti[idx])
        meta["eid"].append(ev_id[idx])
        meta["scenario"].append(np.full(len(idx), routes[rid]["scenario"], dtype=object))
        meta["family"].append(np.full(len(idx), routes[rid]["family"], dtype=object))
    F = {k: np.concatenate(v) for k, v in cols.items()}
    M = {k: np.concatenate(v) for k, v in meta.items()}
    M["y"] = (M["tti"] <= HORIZON_S).astype(int)
    M["moving"] = F["ego_speed"] > 2.0
    present = set(np.unique(M["eid"][M["y"] == 1]))
    event_index = [e for e in event_index if e["eid"] in present]
    return F, M, event_index, sorted(untimed, key=int)


def design(F, cols):
    X = np.stack([F[c] for c in cols], 1).astype(float)
    for j, c in enumerate(cols):
        if c.startswith(("path_disagree", "speed_js", "steer_tv", "jerk", "path_end_lat_std", "curvature_change")):
            X[:, j] = np.log1p(np.maximum(X[:, j], 0) * 10)
    return X


# =========================================================================== alarms and metrics

def alarm_episodes(alarm, route, step, merge_gap=HZ):
    """Debounced alarm episodes: consecutive alarm steps (gaps <= merge_gap steps, same route) form one episode.
    Returns the index of each episode's onset."""
    idx = np.where(alarm)[0]
    if len(idx) == 0:
        return idx
    new = np.r_[True, (route[idx[1:]] != route[idx[:-1]]) | (step[idx[1:]] - step[idx[:-1]] > merge_gap)]
    return idx[new]


FA_REPEAT_S = 5.0     # a false alarm still active after this long counts again (an always-on alarm = 12/min)


def false_alarms(a, M):
    """Per-route false-alarm counts. An episode is false if it starts outside the 3 s before a collision; it
    counts once per started FA_REPEAT_S of alarm time on normal (label 0) steps, so long or always-on alarms
    are not cheaper than many short ones."""
    idx = np.where(a)[0]
    out = collections.Counter()
    if len(idx) == 0:
        return out
    route, step, y = M["route"], M["step"], M["y"]
    new = np.r_[True, (route[idx[1:]] != route[idx[:-1]]) | (step[idx[1:]] - step[idx[:-1]] > HZ)]
    ep = np.cumsum(new) - 1
    starts = idx[new]
    neg_steps = np.bincount(ep, weights=(y[idx] == 0).astype(float))
    for e, st in enumerate(starts):
        if y[st] == 0:
            out[route[st]] += int(math.ceil(max(neg_steps[e], 1) / (FA_REPEAT_S * HZ)))
    return out


def alarm_stats(score, thr, M, events_eids):
    a = score >= thr
    neg = M["y"] == 0
    minutes = neg.sum() * DT / 60
    fa = int(sum(false_alarms(a, M).values()))
    lead = {}
    pos = np.where((M["y"] == 1) & a)[0]
    best = collections.defaultdict(float)
    for i in pos:
        e = M["eid"][i]
        best[e] = max(best[e], M["tti"][i])
    for e in events_eids:
        lead[e] = best.get(e, 0.0)
    lv = np.array([lead[e] for e in events_eids])
    return {"fa_per_min": fa / minutes, "n_false_alarms": fa, "alarm_time_frac": float(a[neg].mean()),
            "recall_1s": float((lv >= 1.0).mean()), "recall_2s": float((lv >= 2.0).mean()),
            "recall_any": float((lv > 0).mean()), "lead": lead}


def pick_threshold(score, M, target_fa, events_eids):
    """Lowest threshold whose debounced false-alarm rate on this (validation) data is <= target."""
    qs = np.unique(np.quantile(score, np.r_[np.linspace(0.80, 0.999, 120), np.linspace(0.999, 0.99999, 40)]))
    neg = M["y"] == 0
    minutes = neg.sum() * DT / 60
    best = qs[-1]
    for q in qs:
        if sum(false_alarms(score >= q, M).values()) / minutes <= target_fa:
            best = q
            break
    return float(best)


def subset(M, mask):
    return {k: v[mask] for k, v in M.items()}


def run_method(kind, spec, F, M, events, groups, targets=(1.0, 3.0), n_outer=5, n_inner=4, C=1.0, seed=0):
    """Outer GroupKFold for test scores; thresholds chosen on inner out-of-fold validation scores only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = M["y"]
    X = design(F, [spec] if kind == "rule" else spec)
    score = np.zeros(len(y))
    thr = {t: np.zeros(len(y)) for t in targets}
    fold_of = np.zeros(len(y), int)

    def fit_predict(tr, te):
        if kind == "rule":
            from sklearn.metrics import roc_auc_score
            sign = 1.0 if roc_auc_score(y[tr], X[tr, 0]) >= 0.5 else -1.0
            return sign * X[te, 0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        clf.fit(X[tr], y[tr])
        return clf.decision_function(X[te])

    for fo, (tr, te) in enumerate(GroupKFold(n_splits=n_outer).split(X, y, groups)):
        fold_of[te] = fo
        score[te] = fit_predict(tr, te)
        val = np.zeros(len(tr))
        for itr, ite in GroupKFold(n_splits=n_inner).split(X[tr], y[tr], groups[tr]):
            val[ite] = fit_predict(tr[itr], tr[ite])
        Mtr = subset(M, tr)
        ev_tr = sorted(set(Mtr["eid"][Mtr["y"] == 1]))
        for t in targets:
            thr[t][te] = pick_threshold(val, Mtr, t, ev_tr)
    return score, thr, fold_of


def evaluate(score, thr, M, events_eids):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = M["y"]
    mv = M["moving"]
    out = {"auroc": roc_auc_score(y, score), "auprc": average_precision_score(y, score),
           "auroc_moving": roc_auc_score(y[mv], score[mv]), "auprc_moving": average_precision_score(y[mv], score[mv])}
    for t, th in thr.items():
        out[t] = alarm_stats(score - th, 0.0, M, events_eids)  # per-step threshold (one per outer fold)
    return out


def bootstrap(score, thr, M, events_eids, B=300, seed=0, targets=(1.0,)):
    """Route-level bootstrap of AUROC/AUPRC (moving steps) and event recall / false alarms at each target."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    routes = np.unique(M["route"])
    by_route = {r: np.where(M["route"] == r)[0] for r in routes}
    ev_route = {e: M["route"][np.where(M["eid"] == e)[0][0]] for e in events_eids}
    stats = {t: alarm_stats(score - thr[t], 0.0, M, events_eids) for t in targets}
    # per-route false alarms and negative minutes for each target
    per_route = {}
    for t in targets:
        fa = false_alarms(score >= thr[t], M)
        neg = collections.Counter(M["route"][M["y"] == 0])
        per_route[t] = (fa, neg)
    res = collections.defaultdict(list)
    for _ in range(B):
        samp = rng.choice(routes, len(routes), replace=True)
        idx = np.concatenate([by_route[r] for r in samp])
        mv = idx[M["moving"][idx]]
        yy = M["y"][mv]
        if 0 < yy.sum() < len(yy):
            res["auroc_moving"].append(roc_auc_score(yy, score[mv]))
            res["auprc_moving"].append(average_precision_score(yy, score[mv]))
        cnt = collections.Counter(samp)
        for t in targets:
            lead = stats[t]["lead"]
            ev = [e for e in events_eids for _ in range(cnt.get(ev_route[e], 0))]
            if ev:
                lv = np.array([lead[e] for e in ev])
                res[f"recall_1s@{t}"].append((lv >= 1).mean())
                res[f"recall_2s@{t}"].append((lv >= 2).mean())
            fa, neg = per_route[t]
            nm = sum(neg[r] for r in samp) * DT / 60
            res[f"fa_per_min@{t}"].append(sum(fa[r] for r in samp) / nm if nm else np.nan)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in res.items()}


def recall_fa_curve(score, M, events_eids, n=60):
    qs = np.unique(np.quantile(score, np.r_[np.linspace(0.5, 0.999, n), np.linspace(0.999, 0.99995, 20)]))
    out = []
    for q in qs:
        s = alarm_stats(score, q, M, events_eids)
        out.append((s["fa_per_min"], s["recall_1s"], s["recall_2s"]))
    return np.array(out)


# =========================================================================== maneuver windows (safe vs failure)

def segments(mask, min_len):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def maneuver_anchors(routes, data, events, seed=0):
    """Anchor steps of successful manoeuvres, far from any contact: junction turns, lane changes,
    obstacle bypasses, and normal lane following."""
    rng = np.random.default_rng(seed)
    contact = collections.defaultdict(list)
    for rid, d in data.items():
        for ev in d["events"]:
            contact[rid].append((ev["step"], ev["last_step"]))
    out = []
    for rid, d in data.items():
        n = d["n"]
        bad = np.zeros(n, bool)
        bad[:SKIP_START + 60] = True
        for s, e in contact[rid]:
            bad[max(0, s - 8 * HZ): min(n, e + 5 * HZ)] = True
        mv = d["speed"] > 2.0
        cands = []
        for lab, mask, key in (("junction turn", np.isin(d["cmd"], [0, 1]), np.abs(d["yaw_rate"])),
                               ("lane change", np.isin(d["cmd"], [4, 5]), np.abs(d["steer"]))):
            for i, j in segments(mask, 20):
                k = i + int(np.argmax(np.where(mv[i:j + 1], key[i:j + 1], -1)))
                if mv[k] and not bad[k]:
                    cands.append((lab, k))
        if FAMILY.get(routes[rid]["scenario"]) == "obstacle_detour":
            for i, j in segments((np.abs(d["end_lat"]) > 1.0) & (d["cmd"] == 3), 20):
                k = i + int(np.argmax(np.abs(d["end_lat"][i:j + 1])))
                if mv[k] and not bad[k]:
                    cands.append(("obstacle bypass", k))
        normal = np.where(mv & (d["cmd"] == 3) & (np.abs(d["yaw_rate"]) < 0.05) & ~bad & (np.abs(d["end_lat"]) < 0.5))[0]
        if len(normal):
            for k in rng.choice(normal, min(3, len(normal)), replace=False):
                cands.append(("normal driving", int(k)))
        out += [{"route": rid, "label": lab, "step": k} for lab, k in cands]
    return out
