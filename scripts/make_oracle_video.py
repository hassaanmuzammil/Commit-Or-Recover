"""Videos for the oracle speed-cap test.

    python scripts/make_oracle_video.py results/oracle_speed_cap/route_3936 [--steps 100 260]

For every run folder (raw, cap_*) writes <run>/video.mp4: chase + bird's-eye camera, labelled from the run's
log. For every cap_* run also writes compare_raw_vs_<cap>.mp4: raw on top, capped below, aligned by step.
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

FPS = 20


def load_run(run):
    log = glob.glob(str(run / "logs" / "*" / "*.jsonl"))[0]
    recs = [json.loads(line) for line in open(log)]
    by_step = {r["step"]: r for r in recs}
    hit_steps = [r["step"] for r in recs if r["collisions"]]
    return by_step, (hit_steps[0] if hit_steps else None)


def frame_image(run, view, frame):
    path = run / "frames" / view / f"{frame:08d}.jpg"
    img = cv2.imread(str(path)) if path.exists() else None
    return img if img is not None else np.zeros((540, 960, 3), np.uint8)


def panel(run, rec, first_hit, label):
    frame = rec["gt_ego"]["frame"]
    img = np.hstack([frame_image(run, "chase", frame), frame_image(run, "bev", frame)])
    img = cv2.resize(img, (1440, 405))
    cap = rec.get("oracle_cap_active", False)
    lines = [f"{label} | step {rec['step']}  t={rec['timestamp']:.2f}s",
             f"speed {rec['gt_ego']['speed']:.1f} m/s  throttle {rec['throttle']:.2f}  brake {rec['brake']:.0f}"]
    if "oracle_cap_speed" in rec:
        lines.append(f"oracle cap {rec['oracle_cap_speed']:.0f} m/s: {'ACTIVE' if cap else 'off'}"
                     f"  (TF++ wanted throttle {rec['tfpp_throttle']:.2f}, brake {rec['tfpp_brake']:.0f})")
    for i, text in enumerate(lines):
        cv2.putText(img, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 200, 255) if cap else (255, 255, 255), 2, cv2.LINE_AA)
    if first_hit is not None and first_hit <= rec["step"] < first_hit + 2 * FPS:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 10)
        cv2.putText(img, "COLLISION", (1180, 390), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return img


def write(path, images):
    h, w = images[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for img in images:
        out.write(img)
    out.release()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route_dir", type=Path)
    ap.add_argument("--steps", nargs=2, type=int, default=None)
    args = ap.parse_args()
    runs = {p.name: p for p in sorted(args.route_dir.iterdir()) if (p / "logs").exists() and (p / "frames").exists()}
    data = {name: load_run(p) for name, p in runs.items()}
    for name, run in runs.items():
        by_step, hit = data[name]
        lo, hi = args.steps or (min(by_step), max(by_step))
        write(run / "video.mp4", [panel(run, by_step[s], hit, name) for s in range(lo, hi + 1) if s in by_step])
    if "raw" in runs:
        raw_steps, raw_hit = data["raw"]
        for name, run in runs.items():
            if not name.startswith("cap"):
                continue
            by_step, hit = data[name]
            lo, hi = args.steps or (min(by_step), max(by_step))
            # Once the raw run has ended (route finished or stopped), hold its last frame so the capped run continues.
            raw_last = max(raw_steps)
            imgs = [np.vstack([panel(runs["raw"], raw_steps[min(s, raw_last)], raw_hit,
                                     "raw TF++" if s <= raw_last else "raw TF++ (run ended)"),
                               panel(run, by_step[s], hit, name)])
                    for s in range(lo, hi + 1) if s in by_step and min(s, raw_last) in raw_steps]
            write(args.route_dir / f"compare_raw_vs_{name}.mp4", imgs)


if __name__ == "__main__":
    main()
