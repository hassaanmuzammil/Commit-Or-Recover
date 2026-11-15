"""Perception analysis of TF++'s predicted bounding boxes against CARLA ground truth (schema-v2 logs).

Used by notebooks/perception_analysis.ipynb. numpy only.

TF++ perception output available in the logs: `pred_boxes`, after NMS across the ensemble and the
score >= 0.3 cut-off (config.bb_confidence_threshold), each [x, y, extent_x, extent_y, yaw, speed, brake,
class, score] in the ego frame (x forward, y right). Classes: 0 car, 1 walker, 2 traffic light, 3 stop sign,
4 emergency vehicle. TF++'s other perception heads (BEV semantics, depth, semantic segmentation) are not logged.

Ground truth: `gt_actors` (vehicles, walkers and static props within 60 m, ego frame). It has no traffic
lights or stop signs, so those two box classes are inventoried but not scored.
"""

from __future__ import annotations

import collections
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DT, HZ = 0.05, 20
RANGE = 32.0                                  # TF++ box range: |x|, |y| <= 32 m
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)             # centre-distance match thresholds (nuScenes style)
PRIMARY = 2                                   # index of the primary threshold (2.0 m)
IOU_EVERY = 10                                # BEV IoU on every 10th step (polygon clipping is slow)
BOX_CLASSES = {0: "car", 1: "walker", 2: "traffic light", 3: "stop sign", 4: "emergency vehicle"}
GT_CATS = ["car", "large vehicle", "two-wheeler", "emergency vehicle", "pedestrian"]
TWO_WHEELERS = ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets",
                "vehicle.harley-davidson", "vehicle.kawasaki", "vehicle.yamaha", "vehicle.vespa")
EMERGENCY = ("police", "ambulance", "firetruck")
FP_TYPES = ["true positive", "static prop", "duplicate", "poorly localised (2-4 m)", "ghost"]

FAMILY = {}
for fam, names in {
    "junction": ["SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurn",
                 "NonSignalizedJunctionRightTurn", "SignalizedJunctionLeftTurnEnterFlow",
                 "NonSignalizedJunctionLeftTurnEnterFlow", "T_Junction", "OppositeVehicleRunningRedLight",
                 "OppositeVehicleTakingPriority", "VehicleTurningRoute", "BlockedIntersection",
                 "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
                 "VanillaNonSignalizedTurn", "VanillaNonSignalizedTurnEncounterStopsign"],
    "lane_merge": ["EnterActorFlow", "HighwayExit", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2",
                   "SequentialLaneChange", "InterurbanActorFlow", "InterurbanAdvancedActorFlow", "ParkingExit"],
    "obstacle_detour": ["Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
                        "ParkedObstacle", "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays",
                        "VehicleOpensDoorTwoWays", "InvadingTurn"],
    "cut_in_hazard": ["HighwayCutIn", "ParkingCutIn", "StaticCutIn", "HardBreakRoute", "ControlLoss",
                      "YieldToEmergencyVehicle"],
    "vru_crossing": ["PedestrianCrossing", "DynamicObjectCrossing", "ParkingCrossingPedestrian",
                     "VehicleTurningRoutePedestrian", "CrossingBicycleFlow"],
}.items():
    for n in names:
        FAMILY[n] = fam


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gt_category(a):
    t = a["type"]
    if a["class"] == "walker":
        return 4
    if t.startswith(TWO_WHEELERS):
        return 2
    if any(k in t for k in EMERGENCY):
        return 3
    return 1 if a["ext_x"] > 3.0 else 0


def compatible(cat, cls):
    """Which box classes may match a GT category. Two-wheelers are accepted as car or walker."""
    if cat == 4:
        return cls == 1
    if cat == 2:
        return cls in (0, 1, 4)
    return cls in (0, 4)


def greedy_match(gx, gy, gcat, bx, by, bcls, bscore, gate):
    """nuScenes-style: boxes in descending score order take the nearest unmatched compatible GT within `gate`."""
    nb, ng = len(bx), len(gx)
    box_gt = np.full(nb, -1)
    gt_box = np.full(ng, -1)
    if nb == 0 or ng == 0:
        return box_gt, gt_box
    D = np.hypot(bx[:, None] - gx[None, :], by[:, None] - gy[None, :])
    ok = np.array([[compatible(int(gcat[j]), int(bcls[i])) for j in range(ng)] for i in range(nb)])
    D = np.where(ok, D, np.inf)
    for i in np.argsort(-bscore, kind="stable"):
        d = np.where(gt_box < 0, D[i], np.inf)
        j = int(np.argmin(d))
        if d[j] <= gate:
            box_gt[i], gt_box[j] = j, i
    return box_gt, gt_box


def occlusion(gx, gy, ex, ey, cat):
    """2-D line-of-sight proxy: share of each actor's angular extent (seen from the ego) covered by closer
    vehicles. 0 = fully visible, 1 = fully hidden. Pedestrians do not occlude."""
    n = len(gx)
    if n == 0:
        return np.zeros(0)
    dist = np.hypot(gx, gy)
    bear = np.arctan2(gy, gx)
    r = np.sqrt(ex ** 2 + ey ** 2) * 0.8
    hw = np.arctan2(r, np.maximum(dist, 0.5))
    samp = bear[:, None] + np.linspace(-1, 1, 15)[None, :] * hw[:, None]           # (n, 15)
    cover = np.zeros((n, 15), bool)
    for j in range(n):
        if cat[j] == 4:
            continue
        closer = dist[j] + r[j] * 0.5 < dist
        if not closer.any():
            continue
        inside = np.abs(wrap(samp[closer] - bear[j])) <= hw[j]
        cover[closer] |= inside
    return cover.mean(1)


def rect(x, y, ex, ey, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(ex, ey), (-ex, ey), (-ex, -ey), (ex, -ey)]
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in pts]


def poly_area(p):
    return 0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p))))


def clip(subject, clipper):
    """Sutherland-Hodgman polygon clipping (both convex, counter-clockwise or clockwise consistently)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def inter(p1, p2, a, b):
        x1, y1, x2, y2 = *p1, *p2
        x3, y3, x4, y4 = *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = subject
    for k in range(len(clipper)):
        a, b = clipper[k], clipper[(k + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(inter(s, e, a, b))
            s = e
    return out


def bev_iou(b1, b2):
    p1, p2 = rect(*b1), rect(*b2)
    inter = clip(p1, p2)
    ia = poly_area(inter) if len(inter) >= 3 else 0.0
    u = poly_area(p1) + poly_area(p2) - ia
    return ia / u if u > 0 else 0.0


# =========================================================================== extraction

def pick_log(route_dir):
    files = sorted(p for p in Path(route_dir).glob("*.jsonl") if not p.name.startswith("zz_"))
    return files[-1] if files else None


def extract_route(log_path):
    G = collections.defaultdict(list)       # one row per GT dynamic actor in box range per step
    B = collections.defaultdict(list)       # one row per predicted dynamic box (car / walker / emergency)
    T = collections.defaultdict(list)       # traffic-light and stop-sign boxes
    S = collections.defaultdict(list)       # one row per step
    contacts = {}
    for step, line in enumerate(open(log_path)):
        r = json.loads(line)
        v = r["ego_speed"]
        boxes = np.array(r["pred_boxes"], float).reshape(-1, 9)
        dynb = boxes[np.isin(np.round(boxes[:, 7]), (0, 1, 4))] if len(boxes) else boxes
        tl = boxes[np.isin(np.round(boxes[:, 7]), (2, 3))] if len(boxes) else boxes
        acts = [a for a in r["gt_actors"] if a["class"] in ("vehicle", "walker")
                and abs(a["x"]) <= RANGE and abs(a["y"]) <= RANGE]
        props = np.array([[a["x"], a["y"]] for a in r["gt_actors"] if a["class"] == "static"], float).reshape(-1, 2)
        n_all_dyn = sum(a["class"] in ("vehicle", "walker") for a in r["gt_actors"])
        for e in r["collisions"]:
            contacts.setdefault(e["other_id"], (step, e["other_type"]))

        gx = np.array([a["x"] for a in acts], float)
        gy = np.array([a["y"] for a in acts], float)
        gyaw = np.array([a["yaw"] for a in acts], float)
        gex = np.array([a["ext_x"] for a in acts], float)
        gey = np.array([a["ext_y"] for a in acts], float)
        gcat = np.array([gt_category(a) for a in acts], int)
        gspd = np.array([a["speed"] for a in acts], float)
        occ = occlusion(gx, gy, gex, gey, gcat)
        bcls = np.round(dynb[:, 7]).astype(int) if len(dynb) else np.zeros(0, int)
        matches = [greedy_match(gx, gy, gcat, dynb[:, 0], dynb[:, 1], bcls, dynb[:, 8], t) if len(dynb) and len(acts)
                   else (np.full(len(dynb), -1), np.full(len(acts), -1)) for t in THRESHOLDS]
        box_gt, gt_box = matches[PRIMARY]
        b_any = matches[-1][0]               # match at 4 m, used to label poorly localised boxes

        for j, a in enumerate(acts):
            i = gt_box[j]
            G["step"].append(step)
            G["id"].append(a["id"])
            G["cat"].append(gcat[j])
            G["x"].append(gx[j]); G["y"].append(gy[j]); G["speed"].append(gspd[j])
            G["ext_x"].append(gex[j]); G["ext_y"].append(gey[j])
            G["occ"].append(occ[j])
            G["ego_speed"].append(v)
            for k, t in enumerate(THRESHOLDS):
                G[f"det_{t}"].append(matches[k][1][j] >= 0)
            if i >= 0:
                b = dynb[i]
                G["err_x"].append(b[0] - gx[j]); G["err_y"].append(b[1] - gy[j])
                G["head_err"].append(math.degrees(abs(wrap(b[4] - gyaw[j]))))
                G["ext_err_x"].append(b[2] - gex[j]); G["ext_err_y"].append(b[3] - gey[j])
                G["box_cls"].append(int(round(b[7]))); G["score"].append(b[8])
                G["iou"].append(bev_iou((b[0], b[1], b[2], b[3], b[4]), (gx[j], gy[j], gex[j], gey[j], gyaw[j]))
                                if step % IOU_EVERY == 0 else np.nan)
            else:
                for k in ("err_x", "err_y", "head_err", "ext_err_x", "ext_err_y", "score", "iou"):
                    G[k].append(np.nan)
                G["box_cls"].append(-1)

        for i, b in enumerate(dynb):
            B["step"].append(step)
            B["cls"].append(int(round(b[7])))
            B["score"].append(b[8])
            B["x"].append(b[0]); B["y"].append(b[1])
            B["ext_x"].append(b[2]); B["ext_y"].append(b[3])
            B["speed_field"].append(b[5]); B["brake_field"].append(b[6])
            B["ego_speed"].append(v)
            for k, t in enumerate(THRESHOLDS):
                B[f"tp_{t}"].append(matches[k][0][i] >= 0)
            if box_gt[i] >= 0:
                B["gt_cat"].append(gcat[box_gt[i]])
                fp = 0
            else:
                B["gt_cat"].append(-1)
                dprop = np.hypot(props[:, 0] - b[0], props[:, 1] - b[1]).min() if len(props) else np.inf
                dgt = np.hypot(gx - b[0], gy - b[1]) if len(acts) else np.zeros(0)
                near_matched = len(acts) and ((dgt <= THRESHOLDS[PRIMARY] * 1.5) & (gt_box >= 0)).any()
                if dprop <= 2.5:
                    fp = 1
                elif near_matched:
                    fp = 2
                elif b_any[i] >= 0:
                    fp = 3
                else:
                    fp = 4
            B["fp_type"].append(fp)

        for b in tl:
            T["step"].append(step); T["cls"].append(int(round(b[7]))); T["score"].append(b[8])
            T["x"].append(b[0]); T["y"].append(b[1])

        S["ego_speed"].append(v)
        S["n_gt"].append(len(acts))
        S["n_gt_all"].append(n_all_dyn)
        S["n_box"].append(len(dynb))
        S["n_tl"].append(int((np.round(tl[:, 7]) == 2).sum()) if len(tl) else 0)
        S["n_stop"].append(int((np.round(tl[:, 7]) == 3).sum()) if len(tl) else 0)
        S["n_props"].append(len(props))

    arr = lambda D: {k: np.asarray(v) for k, v in D.items()}
    return {"gt": arr(G), "box": arr(B), "tl": arr(T), "steps": arr(S), "contacts": contacts}


def load_routes(xml_path):
    out = {}
    for r in ET.parse(xml_path).getroot().iter("route"):
        sc = [s.get("type") for s in r.find("scenarios")]
        st = sc[0] if sc else "none"
        w = r.find("weathers")
        w0 = w.find("weather").attrib if w is not None and w.find("weather") is not None else {}
        out[r.get("id")] = {"town": r.get("town"), "scenario": st, "family": FAMILY.get(st, "other"),
                            "precipitation": float(w0.get("precipitation", 0)), "fog": float(w0.get("fog_density", 0)),
                            "sun_altitude": float(w0.get("sun_altitude_angle", 45)),
                            "wetness": float(w0.get("wetness", 0))}
    return out


def load_outcomes(ckpt_dir):
    out = {}
    for p in Path(ckpt_dir).glob("*.json"):
        rec = json.load(open(p))["_checkpoint"]["records"]
        if rec:
            inf = rec[0]["infractions"]
            out[p.stem] = {"failed": any(inf[k] for k in ("collisions_vehicle", "collisions_pedestrian",
                                                          "collisions_layout", "outside_route_lanes",
                                                          "vehicle_blocked", "route_timeout")),
                           "ds": rec[0]["scores"]["score_composed"],
                           "collided_ids": {int(s.split("id=")[1].split(" ")[0]) for k in
                                            ("collisions_vehicle", "collisions_pedestrian")
                                            for s in inf[k] if "id=" in s}}
    return out


def build_cache(results_dir, routes_xml, cache_path, log=print):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))
    results_dir = Path(results_dir)
    routes = load_routes(routes_xml)
    data = {}
    dirs = sorted((results_dir / "logs").iterdir(), key=lambda p: int(p.name))
    for i, d in enumerate(dirs):
        lp = pick_log(d)
        if lp is None:
            continue
        data[d.name] = extract_route(lp)
        if (i + 1) % 20 == 0:
            log(f"  parsed {i + 1}/{len(dirs)} routes")
    out = {"routes": routes, "outcomes": load_outcomes(results_dir / "checkpoints"), "data": data}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache_path, "wb"))
    return out


def stack(data, part):
    """Concatenate one table across routes, adding a `route` column."""
    keys = None
    cols = collections.defaultdict(list)
    for rid, d in data.items():
        t = d[part]
        if not t or len(next(iter(t.values()))) == 0:
            continue
        keys = keys or list(t)
        n = len(next(iter(t.values())))
        cols["route"].append(np.full(n, int(rid)))
        for k in keys:
            cols[k].append(t[k])
    return {k: np.concatenate(v) for k, v in cols.items()}


# =========================================================================== metrics

def ratio_ci(num, den, route, B=1000, seed=0):
    """Pooled ratio sum(num)/sum(den) with a route-level bootstrap 95% CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    routes, inv = np.unique(route, return_inverse=True)
    rn = np.bincount(inv, weights=num, minlength=len(routes))
    rd = np.bincount(inv, weights=den, minlength=len(routes))
    point = rn.sum() / rd.sum() if rd.sum() else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(routes), (B, len(routes)))
    bs = rn[idx].sum(1) / np.maximum(rd[idx].sum(1), 1e-12)
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def average_precision(score, tp, n_gt):
    """AP over the logged boxes (all have score >= 0.3, so recall is capped where the cut-off falls)."""
    if n_gt == 0 or len(score) == 0:
        return np.nan, np.zeros(0), np.zeros(0)
    o = np.argsort(-score, kind="stable")
    t = tp[o].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.r_[0.0, rec]
    return float(np.sum((r[1:] - r[:-1]) * env)), rec, prec


def binned(x, y, bins, min_n=50):
    idx = np.digitize(x, bins) - 1
    out = []
    for k in range(len(bins) - 1):
        m = idx == k
        out.append((0.5 * (bins[k] + bins[k + 1]), y[m].mean() if m.sum() >= min_n else np.nan, int(m.sum())))
    return np.array(out)


def tracks(gt, thr_key):
    """Per (route, actor id) track statistics while the actor is in box range."""
    route = gt["route"] if "route" in gt else np.zeros(len(gt["step"]), int)
    order = np.lexsort((gt["step"], gt["id"], route))
    r, i, s, det, dist, occ = (route[order], gt["id"][order], gt["step"][order], gt[thr_key][order],
                               np.hypot(gt["x"], gt["y"])[order], gt["occ"][order])
    cat = gt["cat"][order]
    new = np.r_[True, (r[1:] != r[:-1]) | (i[1:] != i[:-1])]
    starts = np.where(new)[0]
    ends = np.r_[starts[1:], len(r)]
    out = []
    for a, b in zip(starts, ends):
        st, dt, ds = s[a:b], det[a:b], dist[a:b]
        contiguous = np.r_[True, np.diff(st) == 1]
        seg = np.cumsum(~contiguous)                   # separate visits to the range
        drop = 0
        for g in np.unique(seg):
            d = dt[seg == g]
            on = np.where(d)[0]
            if len(on) >= 2:
                drop += int((np.diff(on) > 1).sum())     # box disappears then reappears within one visit
        first = np.where(dt)[0]
        out.append({"route": int(r[a]), "id": int(i[a]), "cat": int(cat[a]), "steps": b - a,
                    "det_rate": float(dt.mean()), "dropouts": drop,
                    "entered_in_range": bool(st[0] > 0),
                    "latency_s": float((st[first[0]] - st[0]) * DT) if len(first) else np.nan,
                    "first_det_dist": float(ds[first[0]]) if len(first) else np.nan,
                    "entry_dist": float(ds[0]), "mean_occ": float(occ[a:b].mean())})
    return out
