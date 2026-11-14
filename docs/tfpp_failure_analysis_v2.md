# TF++ Failure Analysis v2: categories, detection and timing

Commit or Recover · CMPE249 · Bench2Drive run with v2 logging (`results/bench2drive_v2`), analysed 2026-11-14

This report follows [tfpp_failure_analysis.pdf](tfpp_failure_analysis.pdf) (2026-11-06). The new logs add TF++'s predicted boxes, ground-truth actors and collision events with exact timing (see [tfpp_sample_output_v2.md](tfpp_sample_output_v2.md)). With them, this report can answer three questions the earlier analysis could not:

1. **What kind of failure is it?** Each failure is categorised three ways: scenario family, collision geometry, and failure mechanism (did TF++ see the actor it hit, and did it react?).
2. **Does the feedback's suggested baseline work?** The feedback on the proposal asked for this baseline first: path disagreement, target-speed distribution divergence and control variation. This report compares it with speed alone, TF++'s own confidence and geometric monitors, all at the same intervention budget.
3. **Do temporal signals fire before a geometric hazard becomes apparent?**

It also checks whether the 2–3 recommended scenario families have enough failures and benign corrections, as the feedback suggested.

## 1. Summary

- **Failure rate.** 61 of 219 routes failed (27.9%), at a single seed. Mean driving score 78.8, mean route completion 99.3%. 56 failures are collisions with an exact impact time; 5 are off-route infractions.
- **TF++ sees the actor and drives into it anyway.** 48 of the 56 collisions were with a vehicle or pedestrian that TF++'s own detector had boxed. It had been detected at 1 s before impact in 47 of 48 cases, and for a median of 2.8 s before impact. In 33 of those 48, TF++ did not brake until less than 0.5 s before contact, and it was usually speeding up (from 9.1 to 10.4 m/s over the last 1.5 s). Only 1 collision was a clear perception miss. **These are planning and commitment failures, not perception failures.** This is the regime the proposal targets.
- **Constant-velocity geometry sees the hazard late.** Even with perfect ground-truth positions, a constant-velocity / constant-turn-rate rollout flags the struck actor (TTC < 3 s) only a median of **0.55 s** before impact. It does so ≥1 s ahead in just 26% of cases. The ego's own manoeuvre (a turn or lane change) creates the conflict, and a rollout that ignores the ego's plan cannot anticipate it.
- **The feedback's temporal baseline works about as well as the alternatives, not better.** All monitors were compared at the same budget: in alarm for 1% of normal driving time. At that budget:

  | Monitor | Failures warned ≥1 s ahead |
  |---|---|
  | Speed alone | 18% |
  | Temporal features | 30% |
  | Geometric monitor on TF++'s boxes | 32% |
  | TF++ confidence | 39% |
  | **Geometric + temporal** | **48%** |

  With 56 failures, differences under about 13 points are within noise. So: temporal beats speed alone, matches confidence and geometry, and **adds to geometry when combined**.
- **The temporal signal helps most at junctions.** At a 2% alarm budget it warns ≥1 s ahead on 7 of 13 junction failures. The box-based geometric monitor warns on 0 of 13, and confidence on 3. This is the clearest support for H1, but n = 13.
- **Recommended families: obstacle detour, lane merge / exit, and junction turns.** Together they hold 48 of the 56 timed failures, with about 3 benign instability episodes for every failure-linked one. One seed gives 13–20 failures per family, which is not enough for training and held-out evaluation. Data generation is the bottleneck, as the feedback predicted.

## 2. What ran

| | |
|---|---|
| Policy | Frozen TF++ `model_0030_0`, CARLA 0.9.15, Bench2Drive leaderboard and scenario runner |
| Routes | All 220 Bench2Drive routes, one run each. Route 11715 has no result record and is excluded, leaving 219 |
| Logged | 175,264 steps at 20 Hz, about 23 km |
| Outcome statuses | 61 Perfect, 154 Completed, 3 timed out, 1 blocked |

A route is a **failure** if it has a collision, leaves its route lanes, is blocked, or times out. This is the same rule as the earlier PDF. Every route that timed out or was blocked had also collided first.

| Infraction | Events | Routes | Counted as failure |
|---|---|---|---|
| Collision with vehicle | 54 | 46 | yes |
| Collision with static object | 13 | 6 | yes |
| Collision with pedestrian | 6 | 5 | yes |
| Outside route lanes | 7 | 7 | yes |
| Route timeout | 3 | 3 | yes |
| Vehicle blocked | 1 | 1 | yes |
| Min-speed | 173 | 124 | no |
| Scenario timeout | 8 | 8 | no |
| Failed to yield to emergency vehicle | 5 | 5 | no |
| Red light | 2 | 2 | no |

**Compared with the v1 run** (same routes, basic logging): 60 routes failed in v1, 61 in v2. 52 failed in both runs, 8 failed only in v1 and 9 only in v2. The earlier PDF found 8 routes that failed on every seed (2084, 3936, 2050, 23658, 24092, 17563, 1852, 1790); 7 of them failed again here, and only 17563 passed. So failures still repeat, with roughly 15% run-to-run churn.

## 3. Failure categories

### 3.1 By scenario family

The proposal's families map to Bench2Drive scenario types as follows. Bench2Drive has no roundabout-exit scenario.

| Family | Scenario types (examples) | Routes | Failed | Rate | Mean DS |
|---|---|---|---|---|---|
| **obstacle_detour** | AccidentTwoWays, ConstructionObstacle(TwoWays), ParkedObstacle(TwoWays), HazardAtSideLane, VehicleOpensDoorTwoWays, InvadingTurn | 50 | 22 | 44% | 70.2 |
| **lane_merge** | MergerIntoSlowTraffic(V2), HighwayExit, SequentialLaneChange, EnterActorFlow, InterurbanActorFlow, ParkingExit | 39 | 18 | 46% | 74.2 |
| **junction** | Signalized/NonSignalized Junction Left/Right Turn, T_Junction, OppositeVehicle*, Vanilla* turns, BlockedIntersection | 75 | 13 | 17% | 84.3 |
| vru_crossing | PedestrianCrossing, DynamicObjectCrossing, ParkingCrossingPedestrian, CrossingBicycleFlow | 25 | 5 | 20% | 80.4 |
| cut_in_hazard | HighwayCutIn, ParkingCutIn, StaticCutIn, HardBreakRoute, ControlLoss, YieldToEmergencyVehicle | 30 | 3 | 10% | 84.2 |

The junction rate is diluted by many easy routes, such as Vanilla turns and T_Junction. The hard junction scenarios fail at rates similar to the earlier PDF: NonSignalizedJunctionLeftTurn and SignalizedJunctionLeftTurn 4/5 each, SignalizedJunctionRightTurn 3/5.

The scenario types with the most failures (5 routes each):

| ConstructionObstacleTwoWays | NonSignalizedJunctionLeftTurn | SignalizedJunctionLeftTurn | MergerIntoSlowTraffic | HighwayExit | InterurbanActorFlow | HazardAtSideLane | AccidentTwoWays | SignalizedJunctionRightTurn | InvadingTurn |
|---|---|---|---|---|---|---|---|---|---|
| 5/5 | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 3/5 | 3/5 | 3/5 | 3/5 |

### 3.2 By collision geometry and manoeuvre

**Geometry** is read from the ground truth just before contact: the struck actor's position and heading relative to the ego, and the ego's speed. Rules are applied in order: pedestrian, static, ego stopped (< 1 m/s), head-on (relative heading > 135°), crossing (45–135°), rear-end into lead, rear-ended, side-swipe.

**Manoeuvre** is the route command over the last 3 s: any junction command (left, right, straight), else lane change, else follow lane.

| Collision geometry | Count | Median speed before contact (m/s) | Where it happens |
|---|---|---|---|
| Side-swipe (same direction) | 20 | 11.1 | lane_merge 9, obstacle_detour 9, junction 2 |
| Head-on (oncoming) | 12 | 10.5 | junction 8 (left turns), obstacle_detour 4 (bypass in opposite lane) |
| Rear-end into vehicle ahead | 8 | 9.8 | obstacle_detour 3, cut_in 3, lane_merge 2 |
| Static object | 5 | 11.5 | obstacle_detour 4 (cones, debris, mesh), lane_merge 1 |
| Pedestrian | 5 | 10.9 | vru_crossing 5 |
| Crossing / turning conflict | 4 | 9.1 | junction 3, lane_merge 1 |
| Rear-ended by vehicle behind | 1 | — | lane_merge |
| Hit while stopped | 1 | — | lane_merge |
| Off-route (no collision) | 5 | — | lane_merge 3 (HighwayExit, InterurbanActorFlow ×2), obstacle_detour 2 (InvadingTurn, ParkedObstacleTwoWays) |

| Manoeuvre in the last 3 s | Count |
|---|---|
| Follow lane | 29 |
| Junction left turn | 10 |
| Junction straight | 7 |
| Lane change | 6 |
| Junction right turn | 4 |

What the categories show:

- **Junction failures are unprotected left turns into oncoming traffic.** 8 of 13 are head-on with an oncoming car, at about 10 m/s.
- **Lane-merge failures are side-swipes.** TF++ moves into an occupied lane at speed.
- **Obstacle-detour failures are mixed.** They include side-swipes while squeezing past, head-on hits in the opposite lane during a bypass, and hits on the obstacle itself (cones and debris have no TF++ box class).
- **The route command misses detours.** It stays at "follow lane" during a bypass, so the command alone does not capture these manoeuvres. That explains why "follow lane" is the largest group.

### 3.3 By failure mechanism (the new category)

For each collision, the struck actor was matched to TF++'s predicted boxes: same class, centre within 3 m (2 m for pedestrians). The median match error was 0.18 m. The table then records whether and when TF++ braked.

| Mechanism | Count | Definition | Can a TF++-output monitor help? |
|---|---|---|---|
| **Detected, no timely reaction ("committed")** | **33** | Actor boxed at least 25% of the time in the last 2 s; braking started < 0.5 s before impact, or not at all | **Yes: the target case.** TF++ sees the actor and commits anyway |
| Detected, reacted too late | 15 | As above, but braking started ≥ 0.5 s before impact | Yes, if it warns earlier than TF++'s own reaction |
| Static object | 5 | Cones, debris, pole. TF++ has no box class for these | Only through planner signals |
| Off-route | 5 | No collision | Planner signals only |
| Struck by another vehicle | 2 | Ego slow, or hit from behind | No |
| Perception miss | 1 | Actor in range but boxed < 25% of the time (DynamicObjectCrossing) | No (needs better perception) |

| Family | Committed | Reacted too late | Static | Off-route | Struck by other | Perception miss |
|---|---|---|---|---|---|---|
| junction | **12** | 1 | – | – | – | – |
| lane_merge | **9** | 3 | 1 | 3 | 2 | – |
| obstacle_detour | **9** | 7 | 4 | 2 | – | – |
| vru_crossing | 2 | 2 | – | – | – | 1 |
| cut_in_hazard | 1 | 2 | – | – | – | – |

**When TF++ first saw the struck actor** (48 detected-actor collisions):

| Time before impact | Detected | Missed | Beyond the 32 m box range | Not in the ground-truth log (> 60 m away or not yet spawned) |
|---|---|---|---|---|
| 3 s | 21 | 2 | 19 | 6 |
| 2 s | 35 | 2 | 8 | 3 |
| 1 s | 47 | 1 | 0 | 0 |

The median first detection is 2.8 s before impact, and 71% of these actors were first detected at least 2 s ahead.

This passes step 2 of the plan in `extra/task.md`: "most crash actors detected ≥2 s ahead". When the actor is within box range at 2 s, TF++ detects it 35 times out of 37. The usual limit is range, not missed detection.

**Reaction.** Where TF++ braked at all, braking started a median of 0.23 s before impact. Speed rose from 9.1 m/s 1.5 s before impact to 10.4 m/s just before contact. TF++ is confidently wrong: its speed-head entropy is lower, not higher, before failures (single-signal AUROC 0.62 in the "lower" direction). This repeats the earlier PDF's finding.

## 4. The feedback's baseline: temporal features vs confidence vs geometry

### 4.1 Setup

| | |
|---|---|
| Label | A step is positive if a collision happens within the next 3 s |
| Steps used | Steps before the route's first collision. The first 2 s of each route and the 5 off-route routes (no impact time) are excluded |
| Data | 214 routes, 104,863 steps, 3,126 positive. On moving steps (> 2 m/s): 39,286 steps, 2,662 positive. 56 failures |
| Model | Gradient-boosted trees, 5-fold cross-validation grouped by route |
| Intervention budget | All monitors are compared at the same share of normal driving time in alarm (1%, 2%, 5%) |
| Warned ≥ X s | The monitor alarms at some step between 3 s and X s before impact |

The earlier "false-alarm onsets per minute" metric is replaced by alarm time. Onset counting rewards monitors that stay on for long stretches. Speed-only and the linear models did exactly that: at "10 onsets/min" they were in alarm for 40–49% of driving. Share of time in alarm is the fair version of the proposal's matched intervention budget.

| Monitor | Inputs |
|---|---|
| Speed only | Ego speed |
| TF++ confidence (single frame) | Speed, target speed, speed-head entropy, top-class margin, top probability |
| **Temporal (feedback baseline)** | Path disagreement after ego-motion alignment (1 s and 2 s mean, 1 s max), path-end lateral jitter, JS divergence between consecutive speed distributions (1 s, 2 s), steering total variation, steering reversals, pedal switches, jerk. Run once without speed and once with it |
| Geometric (TF++ boxes) | TTC from rolling out the ego (constant turn rate) against TF++'s predicted boxes (constant velocity), nearest box in the ego's lane corridor, closing speed, TF++'s own `bb_in_front` flag |
| Geometric + temporal | Union of the two sets above |
| Oracle geometric (privileged) | Same TTC, but against ground-truth vehicles and pedestrians. An upper reference, not a deployable monitor |

### 4.2 Results

| Monitor | AUROC, moving | AUPRC, moving | Warned ≥1 s @1% | ≥2 s @1% | ≥1 s @2% | ≥2 s @2% | ≥1 s @5% |
|---|---|---|---|---|---|---|---|
| Speed only | 0.637 | 0.107 | 18% | 7% | 30% | 12% | 45% |
| TF++ confidence | 0.717 | 0.199 | 39% | **30%** | 55% | **39%** | 73% |
| Temporal only (no speed) | 0.702 | 0.146 | 30% | 21% | 55% | 36% | 71% |
| Temporal + speed | 0.707 | 0.153 | 30% | 20% | 52% | 27% | **79%** |
| Geometric (TF++ boxes) | 0.729 | 0.205 | 32% | 12% | 43% | 20% | 64% |
| **Geometric + temporal** | 0.729 | 0.213 | **48%** | 21% | **61%** | 32% | 75% |
| Oracle geometric (GT) | 0.733 | **0.315** | 27% | 12% | 41% | 21% | 55% |

At the 1% budget the monitors raise about 2.5–3.5 alarm onsets per minute; at 2%, about 4.5–6.5. Linear (logistic regression) versions of every monitor were clearly worse (moving AUROC 0.62–0.68) and are omitted. The base rate is 6.8% of moving steps. One failure is 1.8 points, and the 95% interval on any warning share is about ±13 points.

The single signals that best separate pre-failure steps (moving-step AUROC) are all from the planner:

| Signal | AUROC |
|---|---|
| Path disagreement, 1 s mean | 0.680 |
| Path-end lateral jitter, 1 s | 0.677 |
| Path disagreement, 1 s max | 0.675 |
| Target speed | 0.673 |
| Path curvature | 0.667 |
| JS divergence, 2 s mean | 0.665 |
| Steering total variation, 1 s | 0.662 |

Ego speed scores 0.624, TF++'s `bb_in_front` flag 0.548 and the predicted-box TTC 0.544.

What this means:

- **The temporal features do carry signal.** They warn ≥1 s ahead on 30% of failures at 1% alarm time, against 18% for speed alone. This is a different conclusion from `extra/task.md`, where a GRU on TF++ outputs "did not beat speed". The difference is mostly the metric: under onset counting, speed-only looked artificially good.
- **Temporal does not beat confidence or geometry on its own.** Temporal, TF++ confidence and the box-based geometric monitor are within noise of each other. Confidence is best at ≥2 s.
- **Temporal and geometric catch different failures.** At 2% alarm time, temporal warns ≥1 s ahead on 13 failures that geometry misses. Geometry catches 8 that temporal misses, both catch 16, and neither catches 19. Combined, they are the best monitor at 1% and 2% budgets. Temporal and confidence overlap much more: 5 failures are caught only by temporal and 7 only by confidence.
- **Even the ground-truth oracle is not better at warning.** It ranks best on AUPRC, because it is very sure in the last second. But it warns ≥1 s ahead no more often than the others. With perfect perception, a constant-velocity rollout is still too late, because the hazard comes from what the ego is about to do.

### 4.3 By family (2% alarm time, warned ≥1 s ahead)

| Family (timed failures) | Speed | Confidence | Temporal + speed | Geometric (boxes) | Geometric + temporal |
|---|---|---|---|---|---|
| junction (13) | 1 | 3 | **7** | **0** | 8 |
| lane_merge (15) | 5 | **8** | 4 | 6 | 5 |
| obstacle_detour (20) | 9 | 16 | 15 | 15 | **17** |
| vru_crossing (5) | 1 | 3 | 2 | 2 | 3 |
| cut_in_hazard (3) | 1 | 1 | 1 | 1 | 1 |

- **Junction left turns are where the temporal signal matters.** The geometric monitor warns on none of the 13: oncoming cars are not on a collision course until the ego commits to the turn. The planner's path and speed history start moving earlier.
- **Lane merges are hard for every monitor.** At most 8 of 15 are warned, which matches the earlier PDF's ~1 s lead time for side-swipes.
- **Obstacle detours are easy for everything.** The ego spends seconds in an abnormal state while bypassing.

For the 33 "committed" failures, the warning rates at 2% are: confidence 48%, temporal 48%, geometric 42%, geometric + temporal 61%.

## 5. Do temporal signals fire before the geometric hazard is apparent?

"Geometric hazard apparent" is defined as the start of the final stretch in which ground-truth TTC to the struck actor stays below 3 s. That is a median of 0.55 s before impact, and ≥1 s before impact in only 12 of 47 cases.

The table shows how often each monitor alarmed at least 0.5 s before that point, across the 47 failures where TF++ had detected the struck actor:

| Monitor | 1% budget | 2% budget | 5% budget |
|---|---|---|---|
| Temporal + speed | 28% | 47% | 64% |
| Temporal only | 28% | 45% | 66% |
| TF++ confidence | 36% | 51% | 68% |
| Geometric (TF++ boxes) | 34% | 47% | 57% |
| Geometric + temporal | 40% | 53% | 70% |

**Answer to the feedback:** yes, partly. At a 2% budget, the temporal signals fire before a constant-velocity geometric hazard appears in about half of the failures they could matter for. Two caveats limit this claim:

1. **The geometric baseline is weak.** It rolls the ego forward at constant turn rate and ignores TF++'s planned path. Section 4.3 shows that this is exactly why it misses junction turns. A reviewer will rightly say that an *intent-aware* geometric check would also fire earlier: one that runs TF++'s own path, time-parameterised by its target speed, against its predicted boxes. That baseline must be built before claiming that temporal history beats geometry (H1).
2. **The budgets are not yet closed-loop budgets.** A 2% alarm budget means about 5 alarm onsets per minute. Without debouncing and hysteresis, that is too many stops for closed-loop use.

## 6. Data sufficiency for 2–3 families (feedback point)

**Instability episode:** at least 0.5 s on moving steps where path disagreement (1 s mean > 0.13 m), speed JS divergence (1 s mean > 0.083) or steering variation (1 s > 1.16) exceeds the 95th percentile of normal driving. An episode is *failure-linked* if it overlaps the 3 s before a collision, and *benign* otherwise. Benign episodes are the hard negatives the proposal asks for.

| Family | Routes | Failures (timed) | Positive steps | Benign episodes | Failure-linked episodes | Benign : failure-linked |
|---|---|---|---|---|---|---|
| obstacle_detour | 50 | 22 (20) | 1,191 | 42 | 14 | 3.0 |
| lane_merge | 39 | 18 (15) | 706 | 29 | 9 | 3.2 |
| junction | 75 | 13 (13) | 774 | 27 | 3 | 9.0 |
| cut_in_hazard | 30 | 3 (3) | 155 | 24 | 1 | 24 |
| vru_crossing | 25 | 5 (5) | 300 | 8 | 2 | 4.0 |

- **Recommended core families: obstacle_detour, lane_merge and junction turns (left turns first).** They hold 48 of the 56 timed failures. They show three different temporal profiles: long abnormal bypasses, late side-swipes, and turn commitments that geometry misses. They also map onto the proposal's "obstacle detour" and "lane split / merge" families; junction left turns replace "roundabout exit", which Bench2Drive does not have.
- **Benign corrections are plentiful; failures are not.** There are about 3 benign episodes for every failure-linked one, so hard negatives are not the constraint. A single seed gives 13–20 failures per family. A held-out evaluation with route-grouped splits and 3 GRU seeds needs roughly 50 or more per family.
- **More seeds alone will not fix this.** The earlier PDF showed failures repeat across seeds (8 of 20 focus routes failed on all three), so extra seeds mostly duplicate the same failures. More useful options:
  - parameterised variants of the roughly 10 failing scenario types per family (approach speed, gap size, oncoming-traffic timing, obstacle position);
  - more Bench2Drive routes of those types;
  - 3–5 seeds on top for within-scenario variation.
- **Drop cut_in_hazard and vru_crossing from the core set.** They have too few failures (3 and 5).

## 7. Implications for the proposal

| Item | Status after this analysis |
|---|---|
| Framing ("commit or recover") | **Strengthened.** 33 of 56 collisions are TF++ committing to a manoeuvre while its own detector sees the conflict. This is a planning-level failure, so monitoring the planner is the right level |
| H1: temporal history beats geometry and confidence at the same budget | **Not shown overall.** Temporal ≈ confidence ≈ box geometry (30–39% at 1%). Supported only for junction turns against constant-velocity geometry (7 vs 0 of 13), and not yet against an intent-aware geometric baseline |
| H3: signals are complementary | **Supported.** Geometric + temporal is best at 1% (48%) and 2% (61%), with 13 failures caught only by temporal and 8 only by geometry |
| Feedback: start with simple temporal features, then test whether a GRU adds value | Done for the first half: the handcrafted temporal baseline beats speed alone and matches confidence. The GRU should now be judged against the **geometric + temporal GBM** (48% / 61%), not against speed alone |
| Feedback: 2–3 families, verify enough failures and benign corrections | Families chosen (Section 6). Benign corrections are sufficient; failures are about 3× short per family |
| Earlier recommendation: give the monitor scene inputs | Validated. TF++'s boxes are accurate (median error 0.18 m) and usually catch the struck actor ≥2 s ahead. The weak part is the motion model, not the perception |

## 8. Next steps (in order)

1. **Build the intent-aware geometric baseline.** Time-parameterise TF++'s predicted path with its target speed, roll it out against tracked predicted boxes, and compute TTC and minimum separation. This is the "current-frame geometry" baseline H1 must beat, and it may close the junction gap by itself.
2. **Run the oracle stop test** (`extra/task.md` step 1) on the 33 committed failures. Force a safe stop 1, 2 and 3 s before impact to find how much lead time is actually needed. This tells us which warning horizon (≥1 s or ≥2 s) matters.
3. **Generate data for the three families.** Parameterised variants, target ≥50 timed failures per family. Keep the benign-episode logging.
4. **Then compare the GRU** against intent-aware geometry, TF++ confidence, the handcrafted temporal GBM and the geometric + temporal GBM, at matched alarm time. Use debouncing and hysteresis so a 1–2% budget corresponds to a tolerable number of stops.

## 9. Caveats

- **Single seed and small sample.** 56 timed failures means per-family results rest on 3–20 cases, and per-family warning counts should be read as directional only.
- **Categories come from rules, not video review.** The geometry, manoeuvre and mechanism categories use simple rules on ground truth. "Committed" uses a 0.5 s braking cut-off and a 25% detection cut-off; borderline cases can move between categories. Mechanism uses TF++'s box detections, which do not include static props or traffic lights.
- **The TTC rollout is coarse.** Constant velocity for actors and constant turn rate for the ego, with disc approximations of the boxes. Static props are left out of the ground-truth oracle because several have bounding boxes that overlap the road edge and would raise constant false hazards.
- **Off-route failures have no timestamp.** The checkpoint gives no time for off-route infractions, so those 5 routes are excluded from the step-level evaluation.
- **Monitors are offline and not tuned.** All results are offline classifiers on logged signals; nothing here is closed-loop. Hyperparameters were not tuned.

## 10. Reproduce

```bash
uv run --no-project --with numpy --with scikit-learn python src/analyze_failures_v2.py
```

Script: [src/analyze_failures_v2.py](../src/analyze_failures_v2.py). Outputs are written to `results/bench2drive_v2/analysis/`:

- `summary.json`: all tables
- `failures.csv`: one row per failure
- `cache.pkl`: per-step features, about 31 MB; delete it to force a full re-parse

The first run parses the 1.3 GB of logs in a few minutes; later runs take about 1 minute.

## Appendix: all 61 failures

Speed is measured 0.1 s before contact. "First detected" is the earliest time in the logged 8 s window at which a TF++ box was on the struck actor.

| Route | Scenario | Family | Collision geometry | Maneuver | Mechanism | Speed before contact (m/s) | First detected (s before) | Brake onset (s before) |
|---|---|---|---|---|---|---|---|---|
| 2084 | NonSignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 10.5 | 1.70 | 0.20 |
| 2091 | NonSignalizedJunctionLeftTurn | junction | crossing / turning conflict | junction left turn | detected, reacted too late | 5.1 | 2.00 | 0.75 |
| 2881 | NonSignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 10.4 | 1.75 | 0.10 |
| 26990 | NonSignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 10.2 | 1.95 | 0.00 |
| 3670 | NonSignalizedJunctionRightTurn | junction | side-swipe (same direction) | junction right turn | detected, no timely reaction (committed) | 9.7 | 3.45 | 0.30 |
| 26950 | OppositeVehicleRunningRedLight | junction | head-on (oncoming) | junction right turn | detected, no timely reaction (committed) | 11.3 | 1.10 | 0.00 |
| 3936 | SignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 8.9 | 1.65 | 0.00 |
| 4468 | SignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 10.5 | 1.60 | 0.20 |
| 4669 | SignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 10.4 | 1.80 | 0.15 |
| 4683 | SignalizedJunctionLeftTurn | junction | head-on (oncoming) | junction left turn | detected, no timely reaction (committed) | 9.8 | 1.70 | — |
| 2050 | SignalizedJunctionRightTurn | junction | side-swipe (same direction) | junction right turn | detected, no timely reaction (committed) | 10.2 | 2.90 | 0.20 |
| 26956 | SignalizedJunctionRightTurn | junction | crossing / turning conflict | junction right turn | detected, no timely reaction (committed) | 12.0 | 2.80 | 0.00 |
| 27018 | SignalizedJunctionRightTurn | junction | crossing / turning conflict | follow lane | detected, no timely reaction (committed) | 8.5 | 2.80 | 0.10 |
| 2201 | EnterActorFlow | lane_merge | rear-ended by vehicle behind | junction left turn | struck by other (ego slow / from behind) | 10.3 | 1.85 | 0.10 |
| 11755 | EnterActorFlow | lane_merge | crossing / turning conflict | junction straight | detected, no timely reaction (committed) | 9.7 | 1.40 | 0.00 |
| 23658 | HighwayExit | lane_merge | off-route | - | non-collision | — | — | — |
| 23659 | HighwayExit | lane_merge | side-swipe (same direction) | follow lane | detected, no timely reaction (committed) | 4.3 | 7.40 | 0.25 |
| 23670 | HighwayExit | lane_merge | side-swipe (same direction) | junction straight | detected, reacted too late | 11.4 | 1.85 | 0.50 |
| 23687 | HighwayExit | lane_merge | side-swipe (same direction) | junction straight | detected, reacted too late | 11.7 | 3.45 | 0.65 |
| 23901 | InterurbanActorFlow | lane_merge | static object | junction straight | static object (TF++ has no box class) | 11.5 | — | — |
| 23910 | InterurbanActorFlow | lane_merge | off-route | - | non-collision | — | — | — |
| 23918 | InterurbanActorFlow | lane_merge | rear-end into vehicle ahead | junction straight | detected, no timely reaction (committed) | 13.5 | 4.30 | 0.00 |
| 24098 | InterurbanActorFlow | lane_merge | off-route | - | non-collision | — | — | — |
| 2283 | MergerIntoSlowTraffic | lane_merge | side-swipe (same direction) | lane change | detected, no timely reaction (committed) | 15.6 | 2.05 | 0.35 |
| 3048 | MergerIntoSlowTraffic | lane_merge | side-swipe (same direction) | lane change | detected, reacted too late | 8.4 | 2.00 | 1.10 |
| 3785 | MergerIntoSlowTraffic | lane_merge | side-swipe (same direction) | lane change | detected, no timely reaction (committed) | 13.6 | 2.10 | — |
| 3800 | MergerIntoSlowTraffic | lane_merge | side-swipe (same direction) | lane change | detected, no timely reaction (committed) | 14.9 | 1.70 | 0.10 |
| 23771 | MergerIntoSlowTrafficV2 | lane_merge | hit while stopped/slow | junction straight | struck by other (ego slow / from behind) | 0.6 | 7.95 | 2.55 |
| 24092 | MergerIntoSlowTrafficV2 | lane_merge | side-swipe (same direction) | junction straight | detected, no timely reaction (committed) | 17.8 | 4.35 | 0.05 |
| 17569 | SequentialLaneChange | lane_merge | rear-end into vehicle ahead | lane change | detected, no timely reaction (committed) | 16.5 | 7.80 | 0.15 |
| 17655 | SequentialLaneChange | lane_merge | side-swipe (same direction) | lane change | detected, no timely reaction (committed) | 17.5 | 8.00 | 0.20 |
| 2534 | Accident | obstacle_detour | side-swipe (same direction) | follow lane | detected, no timely reaction (committed) | 16.1 | 4.70 | 0.30 |
| 1852 | AccidentTwoWays | obstacle_detour | side-swipe (same direction) | follow lane | detected, reacted too late | 12.3 | 2.05 | 0.50 |
| 2643 | AccidentTwoWays | obstacle_detour | head-on (oncoming) | follow lane | detected, no timely reaction (committed) | 11.0 | 2.15 | 0.05 |
| 25857 | AccidentTwoWays | obstacle_detour | head-on (oncoming) | follow lane | detected, no timely reaction (committed) | 12.7 | 2.10 | 0.00 |
| 2509 | ConstructionObstacle | obstacle_detour | side-swipe (same direction) | follow lane | detected, reacted too late | 10.8 | 4.85 | 1.00 |
| 24795 | ConstructionObstacle | obstacle_detour | side-swipe (same direction) | follow lane | detected, reacted too late | 8.6 | 5.15 | 1.20 |
| 1825 | ConstructionObstacleTwoWays | obstacle_detour | head-on (oncoming) | follow lane | detected, no timely reaction (committed) | 11.1 | 2.00 | 0.00 |
| 1833 | ConstructionObstacleTwoWays | obstacle_detour | static object | follow lane | static object (TF++ has no box class) | 12.0 | 4.90 | 0.20 |
| 2606 | ConstructionObstacleTwoWays | obstacle_detour | static object | follow lane | static object (TF++ has no box class) | 11.2 | — | 0.25 |
| 20920 | ConstructionObstacleTwoWays | obstacle_detour | static object | follow lane | static object (TF++ has no box class) | 13.3 | 5.95 | 0.10 |
| 25424 | ConstructionObstacleTwoWays | obstacle_detour | head-on (oncoming) | follow lane | detected, no timely reaction (committed) | 13.5 | 1.45 | 0.45 |
| 1790 | HazardAtSideLane | obstacle_detour | rear-end into vehicle ahead | follow lane | detected, no timely reaction (committed) | 9.8 | 8.00 | — |
| 1792 | HazardAtSideLane | obstacle_detour | rear-end into vehicle ahead | follow lane | detected, reacted too late | 18.7 | 5.10 | 0.55 |
| 25439 | HazardAtSideLane | obstacle_detour | side-swipe (same direction) | follow lane | detected, reacted too late | 7.2 | 6.00 | 1.30 |
| 2790 | InvadingTurn | obstacle_detour | off-route | - | non-collision | — | — | — |
| 2802 | InvadingTurn | obstacle_detour | rear-end into vehicle ahead | follow lane | detected, reacted too late | 9.7 | 6.05 | 0.50 |
| 3564 | InvadingTurn | obstacle_detour | static object | follow lane | static object (TF++ has no box class) | 10.8 | — | 0.15 |
| 25383 | ParkedObstacle | obstacle_detour | side-swipe (same direction) | follow lane | detected, reacted too late | 10.1 | 5.25 | 1.20 |
| 2664 | ParkedObstacleTwoWays | obstacle_detour | side-swipe (same direction) | follow lane | detected, no timely reaction (committed) | 9.8 | 8.00 | — |
| 25896 | ParkedObstacleTwoWays | obstacle_detour | off-route | - | non-collision | — | — | — |
| 3472 | VehicleOpensDoorTwoWays | obstacle_detour | side-swipe (same direction) | follow lane | detected, no timely reaction (committed) | 7.2 | 2.80 | — |
| 3482 | VehicleOpensDoorTwoWays | obstacle_detour | side-swipe (same direction) | follow lane | detected, no timely reaction (committed) | 19.1 | 7.50 | — |
| 24781 | HardBreakRoute | cut_in_hazard | rear-end into vehicle ahead | follow lane | detected, reacted too late | 5.9 | 3.75 | 0.90 |
| 25358 | StaticCutIn | cut_in_hazard | rear-end into vehicle ahead | follow lane | detected, no timely reaction (committed) | 3.2 | 8.00 | — |
| 26396 | StaticCutIn | cut_in_hazard | rear-end into vehicle ahead | follow lane | detected, reacted too late | 9.4 | 3.45 | 1.05 |
| 24211 | DynamicObjectCrossing | vru_crossing | pedestrian | follow lane | perception miss | 14.0 | 0.90 | 0.50 |
| 24224 | DynamicObjectCrossing | vru_crossing | pedestrian | follow lane | detected, no timely reaction (committed) | 16.0 | 0.75 | — |
| 3255 | ParkingCrossingPedestrian | vru_crossing | pedestrian | follow lane | detected, reacted too late | 0.4 | 4.00 | 0.60 |
| 24206 | ParkingCrossingPedestrian | vru_crossing | pedestrian | follow lane | detected, reacted too late | 10.9 | 1.10 | 1.30 |
| 3731 | VehicleTurningRoutePedestrian | vru_crossing | pedestrian | junction left turn | detected, no timely reaction (committed) | 7.6 | 4.20 | 0.00 |
