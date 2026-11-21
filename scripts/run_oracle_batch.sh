#!/usr/bin/env bash
# Oracle speed-cap test on several Bench2Drive routes, same recipe as route 3936.
# Requires a CARLA server on port 2000. Runs in the garage_2 environment.
#
#   scripts/run_oracle_batch.sh <n_wanted> <route>:<dataset_collision_step> [...]
#
# For each route, in order, until n_wanted routes have finished:
#   1. raw TF++ run, stopped 20 s after its first collision and never before dataset step + 20 s (no collision by
#      then: the crash did not replay locally, skip the route);
#   2. capped run: 5 m/s cap from 3 s before to 3 s after the raw collision step H, stopped 10 s after its own
#      collision and never before H + 10 s;
#   3. videos for steps H - 10 s .. H + 10 s (later if the capped run crashed later): per run and raw vs capped.
set -u

N_WANTED="$1"; shift
CAP="${CAP:-5}"
FPS=20
BASE="$HOME/Desktop/Commit-Or-Recover"
PYTHON="$HOME/anaconda3/envs/garage_2/bin/python"
done_count=0

first_hit() {  # first step with a ground-truth collision in a run folder, or empty
  "$PYTHON" - "$1" <<'EOF'
import glob, json, sys
logs = glob.glob(sys.argv[1] + "/logs/*/*.jsonl")
for line in (open(logs[0]) if logs else []):
    r = json.loads(line)
    if r["collisions"]:
        print(r["step"]); break
EOF
}

for PAIR in "$@"; do
  [[ $done_count -ge $N_WANTED ]] && break
  ROUTE="${PAIR%%:*}"; D="${PAIR##*:}"
  OUT="$BASE/results/oracle_speed_cap/route_${ROUTE}"
  echo "=== route $ROUTE: raw"
  ROUTE="$ROUTE" ORACLE_STOP_AFTER_COLLISION_STEPS=$((20 * FPS)) ORACLE_STOP_MIN_STEP=$((D + 20 * FPS)) \
    "$BASE/scripts/run_oracle_speed_cap.sh" raw > /dev/null 2>&1
  H=$(first_hit "$OUT/raw")
  if [[ -z "$H" ]]; then
    echo "route $ROUTE: no collision in the local raw run, skipping"
    continue
  fi
  START=$((H - 3 * FPS)); [[ $START -lt 0 ]] && START=0
  END=$((H + 3 * FPS))
  echo "=== route $ROUTE: raw collision at step $H; cap ${CAP} m/s on steps $START-$END"
  ROUTE="$ROUTE" ORACLE_STOP_AFTER_COLLISION_STEPS=$((10 * FPS)) ORACLE_STOP_MIN_STEP=$((H + 10 * FPS)) \
    "$BASE/scripts/run_oracle_speed_cap.sh" "cap_${CAP}mps" "$CAP" "$START" "$END" > /dev/null 2>&1
  C=$(first_hit "$OUT/cap_${CAP}mps")
  LO=$((H - 10 * FPS)); [[ $LO -lt 0 ]] && LO=0
  HI=$((H + 10 * FPS))
  [[ -n "$C" && $((C + 10 * FPS)) -gt $HI ]] && HI=$((C + 10 * FPS))
  echo "=== route $ROUTE: capped run first collision: ${C:-none}; video steps $LO-$HI"
  "$PYTHON" "$BASE/scripts/make_oracle_video.py" "$OUT" --steps "$LO" "$HI"
  done_count=$((done_count + 1))
done
echo "finished $done_count routes"
