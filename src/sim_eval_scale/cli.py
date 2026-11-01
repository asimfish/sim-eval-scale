"""Public CLI; planning and synthetic examples never create GPUs."""

import argparse
import json
from pathlib import Path

from .profiling import diagnose, sensor_budget
from .runner import digest, run, validate_policy
from .topology import make_plan


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(prog="sim-eval-scale")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan", help="Generate bounded topology candidates; no simulator launched")
    q.add_argument(
        "--profile", choices=["rigid-state", "rigid-vision", "liquid-vision"], required=True
    )
    q.add_argument("--gpus", type=int, default=1)
    q.add_argument("--max-envs-per-gpu", type=int, required=True)
    q.add_argument("--max-processes-per-gpu", type=int, default=4)
    q.add_argument("--output", required=True)
    q = sub.add_parser("policy", help="Bind a reviewed real adapter and frozen scene manifest")
    q.add_argument("--plan", required=True)
    q.add_argument("--cases", required=True)
    q.add_argument("--runtime-manifest", required=True)
    q.add_argument("--adapter-json", required=True)
    q.add_argument("--mode", choices=["evaluation", "collection"], default="evaluation")
    q.add_argument("--trial-timeout", type=int, default=1800)
    q.add_argument("--total-timeout", type=int, default=14400)
    q.add_argument("--authorized-allocation", action="store_true")
    q.add_argument("--output", required=True)
    q.add_argument("--min-free-vram-mib", type=int, default=2048)
    q.add_argument("--min-waves", type=int, default=2)
    q = sub.add_parser(
        "sweep", help="Dry plan by default; --execute uses only the existing owned allocation"
    )
    q.add_argument("--policy", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("diagnose", help="Rank exclusive synchronized profiling costs")
    q.add_argument("profile")
    q = sub.add_parser("sensor-budget", help="Raw RGB byte budget, not measured capacity")
    q.add_argument("--envs", type=int, required=True)
    q.add_argument("--width", type=int, default=640)
    q.add_argument("--height", type=int, default=480)
    q.add_argument("--cameras", type=int, default=1)
    q.add_argument("--hz", type=float, default=15)
    args = p.parse_args()
    if args.command == "plan":
        value = make_plan(
            args.profile, args.gpus, args.max_envs_per_gpu, args.max_processes_per_gpu
        )
        write_new(args.output, value)
    elif args.command == "policy":
        plan = json.loads(Path(args.plan).read_text())
        cases = Path(args.cases).resolve()
        rows = json.loads(cases.read_text())
        if not rows or not all("case_id" in r and "seed" in r and "model_seed" in r for r in rows):
            raise ValueError("full case and model seed roster required")
        tops = [
            {k: r[k] for k in ["processes_per_gpu", "envs_per_process"]} for r in plan["candidates"]
        ]
        value = {
            "mode": args.mode,
            "allocated_gpus": plan["allocated_gpus"],
            "topologies": tops,
            "max_envs_per_gpu": plan["max_envs_per_gpu"],
            "max_processes_per_gpu": plan["max_processes_per_gpu"],
            "case_ids": [r["case_id"] for r in rows],
            "case_manifest": str(cases),
            "case_manifest_sha256": digest(cases),
            "runtime_fingerprint": digest(Path(args.runtime_manifest).resolve()),
            "adapter_argv": json.loads(args.adapter_json),
            "resource_authorized": args.authorized_allocation,
            "trial_timeout_s": args.trial_timeout,
            "total_timeout_s": args.total_timeout,
            "max_trials": len(tops) + 1,
            "min_free_vram_mib": args.min_free_vram_mib,
            "min_gain_fraction": 0.05,
            "plateau_limit": 2,
            "max_extra_disagreement_fraction": 0.05,
            "max_success_drop_pp": 5,
            "min_waves": args.min_waves,
        }
        validate_policy(value)
        write_new(args.output, value)
    elif args.command == "sweep":
        value = run(args.policy, args.output, args.execute)
    elif args.command == "diagnose":
        value = diagnose(json.loads(Path(args.profile).read_text()))
    else:
        value = sensor_budget(args.envs, args.width, args.height, args.cameras, args.hz)
    print(json.dumps(value, indent=2))
