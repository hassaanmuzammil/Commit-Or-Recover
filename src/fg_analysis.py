"""Phase F (collision-prediction baselines) and Phase G (ablations) for frozen TF++. Schema-v2 logs only.

Used by notebooks/prediction_baselines.ipynb and notebooks/ablations.ipynb. Reuses the dataset and features
of src/ew_analysis.py without modifying it.

Evaluation protocol
  * Outer 5-fold cross-validation grouped by route (a route's steps never appear in both train and test).
  * Inside every outer training fold, a 4-fold route-grouped inner split gives out-of-fold *validation*
    scores. Alarm thresholds are chosen on those validation scores only, so that a fixed share of normal
    driving time is spent in alarm (the "alarm-time budget"). The model is then refit on the whole training
    fold and applied to the untouched test routes with that threshold.
  * Event metrics count each collision once. Confidence intervals resample routes (route-level bootstrap).
"""

from __future__ import annotations

import collections

import numpy as np

import ew_analysis as ew

DT, HZ = ew.DT, ew.HZ
BUDGETS = (0.01, 0.02, 0.05)          # share of normal driving time in alarm

# ----------------------------------------------------------------------------- feature groups (all TF++-observable)
SPEED = ["ego_speed"]
CONFIDENCE = ["ego_speed", "target_speed", "speed_entropy", "speed_margin", "speed_pmax"]
GEOMETRY = ["ego_speed", "risk_plan", "clear_plan", "pet_plan", "conflict_plan", "corr_dist", "corr_close",
            "n_boxes", "bb_in_front"]          # intent-aware: TF++'s planned path vs its tracked boxes


def temporal(tag="1s", tag2="2s"):
    """Handcrafted temporal instability (proposal section 2.2): path, speed distribution, controls, context."""
    return ["ego_speed"] + ew.TEMPORAL(tag, tag2)


GEOMETRY_TEMPORAL = GEOMETRY + temporal()[1:] + ew.TEMPORAL_GEOM

GROUPS = {   # for the feature-group ablation (each also gets ego speed)
    "path": ["path_disagree_mean_1s", "path_disagree_max_1s", "path_disagree_mean_2s", "path_end_lat_std_1s",
             "curvature_change_1s", "path_curvature", "path_end_lat_abs"],
    "speed distribution": ["target_speed", "speed_entropy", "speed_margin", "speed_pmax", "speed_js_mean_1s",
                           "speed_js_mean_2s", "entropy_change_1s"],
    "controls": ["steer_tv_1s", "steer_reversals_1s", "pedal_switches_1s", "jerk_mean_1s", "brake", "steer_abs"],
    "actor geometry": GEOMETRY[1:] + ew.TEMPORAL_GEOM,
}


def design(F, cols):
    return ew.design(F, list(dict.fromkeys(cols)))


# ----------------------------------------------------------------------------- model fitting with validation thresholds

def _model(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=0)


def _score(clf, X):
    return clf.decision_function(X) if hasattr(clf, "decision_function") else np.log(
        np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1) / np.clip(clf.predict_proba(X)[:, 0], 1e-9, 1))


def cross_validate(X, y, groups, budgets=BUDGETS, kind="lr", train_mask=None, n_outer=5, n_inner=4):
    """Returns out-of-fold test scores and, for every budget, a per-step threshold chosen on validation data.
    `train_mask` (optional) restricts which steps are used to FIT models; validation and test use all steps."""
    from sklearn.model_selection import GroupKFold
    tm = np.ones(len(y), bool) if train_mask is None else train_mask
    score = np.zeros(len(y))
    thr = {b: np.zeros(len(y)) for b in budgets}
    for tr, te in GroupKFold(n_splits=n_outer).split(X, y, groups):
        val = np.zeros(len(tr))
        for itr, ite in GroupKFold(n_splits=n_inner).split(X[tr], y[tr], groups[tr]):
            fit = tr[itr][tm[tr[itr]]]
            clf = _model(kind).fit(X[fit], y[fit])
            val[ite] = _score(clf, X[tr[ite]])
        neg = y[tr] == 0
        for b in budgets:
            thr[b][te] = np.quantile(val[neg], 1 - b)
        fit = tr[tm[tr]]
        score[te] = _score(_model(kind).fit(X[fit], y[fit]), X[te])
    return score, thr


# ----------------------------------------------------------------------------- metrics

def episodes(alarm, M, merge_gap=HZ):
    """Alarm episodes (gaps <= 1 s merged, within a route): start index and length in steps."""
    idx = np.where(alarm)[0]
    if len(idx) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    new = np.r_[True, (M["route"][idx[1:]] != M["route"][idx[:-1]]) | (M["step"][idx[1:]] - M["step"][idx[:-1]] > merge_gap)]
    starts = idx[new]
    ep = np.cumsum(new) - 1
    lengths = np.bincount(ep)
    return starts, lengths


def alarm_metrics(score, thr, M, eids, mask=None):
    """Event recall, lead times and false-alarm statistics. `mask` restricts the evaluation to some steps
    (e.g. one scenario family); events are kept if their pre-impact steps fall inside the mask."""
    mask = np.ones(len(score), bool) if mask is None else mask
    a = (score >= thr) & mask
    neg = (M["y"] == 0) & mask
    minutes = neg.sum() * DT / 60
    starts, lengths = episodes(a, M)
    false = M["y"][starts] == 0
    best = collections.defaultdict(float)
    for i in np.where(a & (M["y"] == 1))[0]:
        best[M["eid"][i]] = max(best[M["eid"][i]], M["tti"][i])
    ev = [e for e in eids if mask[M["eid"] == e].any()]
    lead = {e: best.get(e, 0.0) for e in ev}
    lv = np.array(list(lead.values())) if lead else np.zeros(0)
    return {"events": len(ev), "recall_1s": float((lv >= 1).mean()) if len(lv) else np.nan,
            "recall_2s": float((lv >= 2).mean()) if len(lv) else np.nan,
            "median_lead": float(np.median(lv)) if len(lv) else np.nan,
            "alarm_time": float(a[neg].mean()) if neg.any() else np.nan,
            "onsets_per_min": float(false.sum() / minutes) if minutes else np.nan,
            "mean_false_alarm_s": float(lengths[false].mean() * DT) if false.any() else 0.0,
            "lead": lead}


def ranking(score, M, mask=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = M["moving"] if mask is None else (M["moving"] & mask)
    y = M["y"][m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    return float(roc_auc_score(y, score[m])), float(average_precision_score(y, score[m]))


def bootstrap(score, thr, M, eids, budgets=BUDGETS, B=300, seed=0):
    """Route-level 95% intervals for moving-step AUROC / AUPRC, and recall and alarm statistics per budget."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    routes = np.unique(M["route"])
    by_route = {r: np.where(M["route"] == r)[0] for r in routes}
    ev_route = {e: M["route"][np.where(M["eid"] == e)[0][0]] for e in eids}
    base = {b: alarm_metrics(score, thr[b], M, eids) for b in budgets}
    per_route = {}
    for b in budgets:
        a = score >= thr[b]
        starts, _ = episodes(a, M)
        fa = collections.Counter(M["route"][starts][M["y"][starts] == 0])
        neg = collections.Counter(M["route"][M["y"] == 0])
        at = collections.Counter(M["route"][a & (M["y"] == 0)])
        per_route[b] = (fa, neg, at)
    res = collections.defaultdict(list)
    for _ in range(B):
        samp = rng.choice(routes, len(routes))
        idx = np.concatenate([by_route[r] for r in samp])
        mv = idx[M["moving"][idx]]
        if 0 < M["y"][mv].sum() < len(mv):
            res["auroc"].append(roc_auc_score(M["y"][mv], score[mv]))
            res["auprc"].append(average_precision_score(M["y"][mv], score[mv]))
        cnt = collections.Counter(samp)
        evs = [e for e in eids for _ in range(cnt.get(ev_route[e], 0))]
        for b in budgets:
            lv = np.array([base[b]["lead"][e] for e in evs])
            if len(lv):
                res[f"recall_1s@{b}"].append((lv >= 1).mean())
                res[f"recall_2s@{b}"].append((lv >= 2).mean())
            fa, neg, at = per_route[b]
            nm = sum(neg[r] for r in samp)
            res[f"onsets@{b}"].append(sum(fa[r] for r in samp) / (nm * DT / 60))
            res[f"alarm_time@{b}"].append(sum(at[r] for r in samp) / nm)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}


def paired_difference(sa, ta, sb, tb, M, eids, budgets=BUDGETS, B=500, seed=1):
    """Route-bootstrap of (A - B) for moving-step AUPRC and event recall at each budget."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    routes = np.unique(M["route"])
    by_route = {r: np.where(M["route"] == r)[0] for r in routes}
    ev_route = {e: M["route"][np.where(M["eid"] == e)[0][0]] for e in eids}
    la = {b: alarm_metrics(sa, ta[b], M, eids)["lead"] for b in budgets}
    lb = {b: alarm_metrics(sb, tb[b], M, eids)["lead"] for b in budgets}
    out = collections.defaultdict(list)
    for _ in range(B):
        samp = rng.choice(routes, len(routes))
        idx = np.concatenate([by_route[r] for r in samp]); idx = idx[M["moving"][idx]]
        out["AUPRC (moving)"].append(average_precision_score(M["y"][idx], sa[idx]) - average_precision_score(M["y"][idx], sb[idx]))
        cnt = collections.Counter(samp)
        evs = [e for e in eids for _ in range(cnt.get(ev_route[e], 0))]
        for b in budgets:
            out[f"recall >=1 s @ {b:.0%}"].append(np.mean([la[b][e] >= 1 for e in evs]) - np.mean([lb[b][e] >= 1 for e in evs]))
            out[f"recall >=2 s @ {b:.0%}"].append(np.mean([la[b][e] >= 2 for e in evs]) - np.mean([lb[b][e] >= 2 for e in evs]))
    _, pa = ranking(sa, M); _, pb = ranking(sb, M)
    point = {"AUPRC (moving)": pa - pb}
    for b in budgets:
        point[f"recall >=1 s @ {b:.0%}"] = np.mean([la[b][e] >= 1 for e in eids]) - np.mean([lb[b][e] >= 1 for e in eids])
        point[f"recall >=2 s @ {b:.0%}"] = np.mean([la[b][e] >= 2 for e in eids]) - np.mean([lb[b][e] >= 2 for e in eids])
    return {k: (point[k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def hard_negatives(F, M, anchors, route_ids):
    """Benign 'hard negative' steps: label-0 steps inside successful manoeuvre windows (junction turns, lane
    changes, bypasses: 3 s before to 1 s after the peak) or with planner instability above the 95th percentile
    of normal moving driving."""
    hn = np.zeros(len(M["y"]), bool)
    key = {(int(a["route"]), int(s)) for a in anchors if a["label"] != "normal driving"
           for s in range(a["step"] - 60, a["step"] + 21)}
    hn |= np.array([(int(r), int(s)) in key for r, s in zip(M["route"], M["step"])])
    neg_mv = (M["y"] == 0) & M["moving"]
    for k in ("path_disagree_mean_1s", "speed_js_mean_1s", "steer_tv_1s"):
        hn |= neg_mv & (F[k] > np.quantile(F[k][neg_mv], 0.95))
    return hn & (M["y"] == 0)
