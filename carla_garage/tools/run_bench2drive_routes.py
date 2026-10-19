#!/usr/bin/env python3
"""Run and validate selected or all Bench2Drive routes.

Run this from the carla_garage directory while CARLA is running:

    python tools/run_bench2drive_routes.py

By default, the script runs the 20 selected routes. Use ``--suite all`` to
run all 220 split Bench2Drive routes.

The script runs routes sequentially because they share one CARLA server and
one Traffic Manager port. Each route receives its own checkpoint, temporal
log directory, and stdout log. Completed routes are skipped automatically;
use ``--force`` to run them again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Route:
    split_file: str
    route_id: str
    family: str
    scenario: str


ROUTES = (
    Route("bench2drive_08.xml", "2050", "junction", "SignalizedJunctionRightTurn"),
    Route("bench2drive_10.xml", "2084", "junction", "NonSignalizedJunctionLeftTurn"),
    Route("bench2drive_103.xml", "3936", "junction", "SignalizedJunctionLeftTurn"),
    Route("bench2drive_104.xml", "4183", "junction", "SignalizedJunctionLeftTurn"),
    Route("bench2drive_193.xml", "26458", "junction", "T_Junction"),
    Route("bench2drive_196.xml", "26956", "junction", "SignalizedJunctionRightTurn"),
    Route("bench2drive_206.xml", "28035", "junction", "T_Junction"),
    Route("bench2drive_111.xml", "11715", "lane_merge", "EnterActorFlow"),
    Route("bench2drive_117.xml", "17563", "lane_merge", "SequentialLaneChange"),
    Route("bench2drive_128.xml", "23658", "lane_merge", "HighwayExit"),
    Route("bench2drive_130.xml", "23670", "lane_merge", "HighwayExit"),
    Route("bench2drive_135.xml", "23771", "lane_merge", "MergerIntoSlowTrafficV2"),
    Route("bench2drive_143.xml", "24092", "lane_merge", "MergerIntoSlowTrafficV2"),
    Route("bench2drive_201.xml", "27506", "lane_merge", "EnterActorFlow"),
    Route("bench2drive_01.xml", "1773", "obstacle", "ParkedObstacle"),
    Route("bench2drive_02.xml", "1790", "obstacle", "HazardAtSideLane"),
    Route("bench2drive_06.xml", "1852", "obstacle", "AccidentTwoWays"),
    Route("bench2drive_30.xml", "2509", "obstacle", "ConstructionObstacle"),
    Route("bench2drive_155.xml", "24367", "obstacle", "ConstructionObstacle"),
    Route("bench2drive_156.xml", "24757", "obstacle", "ParkedObstacle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("selected", "all"),
        default="selected",
        help="Run the selected 20 routes or all 220 Bench2Drive routes.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_models/project_tfpp"))
    parser.add_argument("--traffic-manager-seed", type=int, default=100)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun routes even when a valid checkpoint and temporal log already exist.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first evaluator or validation failure.",
    )
    return parser.parse_args()


def discover_all_routes(root: Path) -> tuple[Route, ...]:
    """Build a route manifest from the 220 one-route split XML files."""
    split_dir = root / "leaderboard" / "data" / "bench2drive_split"
    routes = []
    for route_xml in sorted(split_dir.glob("*.xml")):
        xml_root = ET.parse(route_xml).getroot()
        route_nodes = xml_root.findall("route")
        if len(route_nodes) != 1:
            raise ValueError(f"Expected one route in {route_xml}, found {len(route_nodes)}")
        route_node = route_nodes[0]
        scenario_types = sorted({
            node.attrib.get("type", "unknown")
            for node in route_node.findall("./scenarios/scenario")
        })
        routes.append(Route(
            route_xml.name,
            route_node.attrib["id"],
            "all",
            ",".join(scenario_types) or "none",
        ))
    if len(routes) != 220:
        raise ValueError(f"Expected 220 split routes, found {len(routes)}")
    return tuple(routes)


def load_route_record(checkpoint_path: Path, route_id: str) -> dict | None:
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = data.get("_checkpoint", {}).get("records", [])
    for record in records:
        if str(record.get("route_id")) == route_id:
            return record
    return records[-1] if len(records) == 1 else None


def validate_log(root: Path, log_dir: Path) -> tuple[bool, str]:
    logs = sorted(log_dir.glob("*.jsonl")) if log_dir.exists() else []
    if len(logs) != 1:
        return False, f"expected one JSONL file, found {len(logs)}"

    validator = root / "tools" / "validate_temporal_logs.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(log_dir)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "validator failed"
    return True, result.stdout.strip().splitlines()[-1]


def route_is_complete(root: Path, route: Route, output: Path) -> tuple[bool, str]:
    """Return true only when both the evaluator result and temporal log validate."""
    checkpoint_path = output / "checkpoints" / f"{route.route_id}.json"
    record = load_route_record(checkpoint_path, route.route_id)
    if record is None:
        return False, "checkpoint missing or route record missing"
    log_valid, log_message = validate_log(root, output / "logs" / route.route_id)
    if not log_valid:
        return False, f"temporal log incomplete: {log_message}"
    return True, "existing checkpoint and temporal log are valid"


def run_route(root: Path, route: Route, args: argparse.Namespace) -> tuple[int, dict | None, bool, str]:
    output = args.output
    checkpoint_path = output / "checkpoints" / f"{route.route_id}.json"
    debug_checkpoint = output / "checkpoints" / f"{route.route_id}_live.txt"
    log_dir = output / "logs" / route.route_id
    stdout_path = output / "stdout" / f"{route.route_id}.log"
    route_xml = root / "leaderboard" / "data" / "bench2drive_split" / route.split_file

    command = [
        sys.executable,
        str(root / "leaderboard" / "leaderboard" / "leaderboard_evaluator_local.py"),
        "--routes", str(route_xml),
        "--agent", str(root / "team_code" / "sensor_agent.py"),
        "--agent-config", str(root / args.checkpoint),
        "--checkpoint", str(checkpoint_path),
        "--debug-checkpoint", str(debug_checkpoint),
        "--traffic-manager-seed", str(args.traffic_manager_seed),
        "--debug", str(args.debug),
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TEMPORAL_LOG_DIR"] = str(log_dir.resolve())

    if args.dry_run:
        print("DRY RUN:", " ".join(command))
        return 0, None, False, "dry run"

    with stdout_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)

    record = load_route_record(checkpoint_path, route.route_id)
    valid, validation_message = validate_log(root, log_dir)
    return result.returncode, record, valid, validation_message


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output is None:
        args.output = Path("results/bench2drive_all" if args.suite == "all" else "results/bench2drive_selected")
    args.output = args.output if args.output.is_absolute() else root / args.output

    routes = discover_all_routes(root) if args.suite == "all" else ROUTES

    for subdir in ("checkpoints", "logs", "stdout"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)

    failures = 0
    print(f"Running {len(routes)} routes ({args.suite} suite); output: {args.output}")
    print("route_id\tfamily\tscenario\tstatus\tcompletion\tlog")

    for route in routes:
        checkpoint_path = args.output / "checkpoints" / f"{route.route_id}.json"
        log_dir = args.output / "logs" / route.route_id

        if not args.force and not args.dry_run and not args.validate_only:
            complete, existing_message = route_is_complete(root, route, args.output)
            if complete:
                record = load_route_record(checkpoint_path, route.route_id)
                status = record.get("status", "NO_RESULT") if record else "NO_RESULT"
                completion = record.get("scores", {}).get("score_route", "-") if record else "-"
                print(f"{route.route_id}\t{route.family}\t{route.scenario}\tSKIPPED ({status})\t{completion}\t{existing_message}")
                continue

        if args.validate_only:
            return_code = 0
            record = load_route_record(checkpoint_path, route.route_id)
            log_valid, log_message = validate_log(root, log_dir)
        else:
            return_code, record, log_valid, log_message = run_route(root, route, args)

        status = record.get("status", "NO_RESULT") if record else "NO_RESULT"
        completion = record.get("scores", {}).get("score_route", "-") if record else "-"
        if return_code != 0 or record is None or not log_valid:
            failures += 1
        print(f"{route.route_id}\t{route.family}\t{route.scenario}\t{status}\t{completion}\t{log_message}")

        if failures and args.stop_on_error:
            break

    print(f"\nFinished with {failures} route(s) needing attention.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
