#!/usr/bin/env python3
"""Bounded, evidence-gated simulator sweeps. No simulator or cloud dependency."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .topology import Topology, from_dict

FLAGS = (
    "image_input_contract",
    "controller_contract",
    "state_isolation",
    "trajectory_against_N1_repeat",
    "original_oracle",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    p = Path(path)
    t = p.with_suffix(".pending")
    t.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    t.replace(p)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_policy(p):
    require(p["mode"] in ("evaluation", "collection"), "mode")
    require(type(p["allocated_gpus"]) is int and p["allocated_gpus"] > 0, "allocated_gpus")
    levels = p.get("levels")
    if "topologies" in p:
        tops = [from_dict(t) for t in p["topologies"]]
        require(
            tops and tops[0] == Topology(1, 1) and len(tops) == len(set(tops)),
            "unique topologies beginning with N1",
        )
        require(
            all(
                t.envs_per_gpu <= p["max_envs_per_gpu"]
                and t.processes_per_gpu <= p["max_processes_per_gpu"]
                for t in tops
            ),
            "topology resource bounds",
        )
    else:
        require(levels[0] == 1 and levels == sorted(set(levels)), "ascending levels beginning at1")
        require(
            all(type(n) is int and 1 <= n <= p["max_envs_per_gpu"] for n in levels),
            "parallel bound",
        )
        if "max_processes_per_gpu" in p:
            cap = p["max_processes_per_gpu"]
            require(
                type(cap) is int and cap >= 1 and all(n <= cap for n in levels),
                "legacy level process bound",
            )
    ids = p["case_ids"]
    require(ids and len(set(ids)) == len(ids), "fixed unique case roster")
    require(
        type(p.get("min_waves", 1)) is int and p.get("min_waves", 1) >= 1,
        "positive pilot wave count",
    )
    for k in ["trial_timeout_s", "total_timeout_s", "max_trials"]:
        require(type(p[k]) is int and p[k] > 0, k)
    for k in [
        "min_free_vram_mib",
        "min_gain_fraction",
        "max_extra_disagreement_fraction",
        "max_success_drop_pp",
    ]:
        require(finite(p[k]) and p[k] >= 0, k)
    require(type(p["plateau_limit"]) is int and p["plateau_limit"] >= 1, "plateau limit")
    require(
        isinstance(p["runtime_fingerprint"], str) and len(p["runtime_fingerprint"]) == 64,
        "frozen runtime SHA256 required",
    )
    require(
        isinstance(p["adapter_argv"], list)
        and p["adapter_argv"]
        and all(isinstance(x, str) for x in p["adapter_argv"]),
        "adapter argv array, never shell",
    )
    return p


def validate_report(report, request, policy, allocation, trial_dir):
    require(report["trial_id"] == request["trial_id"], "trial identity")
    require(report["evidence_kind"] == "actual_workload", "synthetic evidence cannot qualify")
    require(
        report["allocation_id"] == allocation["allocation_id"]
        and report["gpu_uuids"] == allocation["gpu_uuids"],
        "same physical allocation/order",
    )
    require(
        report["runtime_fingerprint"] == policy["runtime_fingerprint"],
        "science/model/controller drift",
    )
    require(
        report["case_manifest_sha256"] == request["case_manifest_sha256"], "case/seed roster drift"
    )
    require(
        report["envs_per_gpu"] == request["envs_per_gpu"] and report["mode"] == policy["mode"],
        "configuration binding",
    )
    if "topology" in request:
        require(
            report.get("topology") == request["topology"], "process/vectorized topology binding"
        )
    require(
        report.get("owned_workers_closed") is True,
        "UNKNOWN owned worker closure: do not launch another trial",
    )
    require(set(report["correctness"]) == set(FLAGS), "all correctness checks must be reported")
    for flag in FLAGS:
        evidence = report["correctness_evidence"][flag]
        path = (trial_dir / evidence["file"]).resolve()
        require(
            path.is_relative_to(trial_dir.resolve()) and digest(path) == evidence["sha256"],
            "correctness evidence scope/SHA",
        )
        raw = json.loads(path.read_text())
        require(
            raw.get("check") == flag
            and raw.get("trial_id") == request["trial_id"]
            and raw.get("runtime_fingerprint") == policy["runtime_fingerprint"]
            and type(raw.get("passed")) is bool
            and raw["passed"] is report["correctness"][flag],
            "correctness evidence binding",
        )
    rows = report["cases"]
    require(len(rows) == len(policy["case_ids"]), "complete attempted pilot roster")
    require(
        {r["case_id"] for r in rows} == set(policy["case_ids"]),
        "no duplicate/selected/retried cases",
    )
    for row in rows:
        require(
            row["status"] in ("success", "failure", "infrastructure", "unknown"),
            "explicit outcome category",
        )
        if row["status"] in ("success", "failure"):
            require(row["audit_passed"] is True, "original independent audit required")
            path = (trial_dir / row["audit_file"]).resolve()
            require(path.is_relative_to(trial_dir.resolve()), "audit must be inside this trial")
            require(digest(path) == row["audit_sha256"], "audit evidence SHA mismatch")
            raw = json.loads(path.read_text())
            require(
                raw["case_id"] == row["case_id"]
                and raw["status"] == row["status"]
                and raw["audit_passed"] is True,
                "audit row disagrees with evidence",
            )
            if policy["mode"] == "collection" and row["status"] == "success":
                require(
                    raw.get("training_eligible") is True,
                    "original dataset gate must admit successful collection",
                )
    metrics = report["metrics"]
    n = policy["allocated_gpus"]
    require(
        len(metrics["min_free_vram_mib"]) == n
        and all(finite(v) and v >= 0 for v in metrics["min_free_vram_mib"]),
        "per-GPU free VRAM measured",
    )
    require(
        finite(metrics["gpu_util_median_pct"]) and 0 <= metrics["gpu_util_median_pct"] <= 100,
        "measured utilization",
    )
    require(finite(report["wall_seconds"]) and report["wall_seconds"] > 0, "end-to-end wall time")
    require(
        report["wall_seconds"] <= policy["trial_timeout_s"] + 1, "trial exceeds declared wall bound"
    )
    return report


def counts(report):
    rows = report["cases"]
    valid = [r for r in rows if r["status"] in ("success", "failure")]
    return {
        "attempted": len(rows),
        "audited": len(valid),
        "success": sum(r["status"] == "success" for r in valid),
        "infrastructure_or_unknown": len(rows) - len(valid),
    }


def disagreement(a, b):
    aa = {r["case_id"]: r["status"] for r in a["cases"]}
    bb = {r["case_id"]: r["status"] for r in b["cases"]}
    require(aa.keys() == bb.keys(), "matched roster required")
    return sum(aa[k] != bb[k] for k in aa) / len(aa)


def candidate_gate(report, baseline, repeat, policy):
    c = counts(report)
    reasons = []
    if not all(report["correctness"][k] is True for k in FLAGS):
        reasons.append("correctness_invariant_failed")
    if c["infrastructure_or_unknown"]:
        reasons.append("infrastructure_or_unknown")
    if min(report["metrics"]["min_free_vram_mib"]) < policy["min_free_vram_mib"]:
        reasons.append("VRAM_headroom")
    if baseline and repeat:
        floor = disagreement(baseline, repeat)
        if disagreement(report, baseline) > floor + policy["max_extra_disagreement_fraction"]:
            reasons.append("extra_disagreement_vs_N1_repeat")
        # Finite pilot guardrail, NOT a statistical equivalence test.
        n = len(report["cases"])
        reference = (counts(baseline)["success"] + counts(repeat)["success"]) / (2 * n)
        if (reference - c["success"] / n) * 100 > policy["max_success_drop_pp"]:
            reasons.append("pilot_success_drop")
    return reasons


def score(report, policy):
    c = counts(report)
    accepted = c["audited"] if policy["mode"] == "evaluation" else c["success"]
    return accepted * 3600 / report["wall_seconds"] / policy["allocated_gpus"]


def adapter_environment(extra_names=()):
    # Preserve CUDA/graphics/runtime setup, but do not implicitly forward API credentials.
    names = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TZ",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "DISPLAY",
        "EGL_PLATFORM",
        "VK_ICD_FILENAMES",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_MODULE_LOADING",
        "XDG_CACHE_HOME",
        "VK_LAYER_PATH",
    }
    require(
        isinstance(extra_names, (list, tuple))
        and all(isinstance(n, str) and n.isidentifier() for n in extra_names),
        "explicit environment variable names only",
    )
    return {name: os.environ[name] for name in names.union(extra_names) if name in os.environ}


def invoke(argv, request, directory, timeout_s):
    """Bounded foreground adapter; uncertainty stops the entire sweep."""
    request_path = directory / "REQUEST.json"
    save(request_path, request)
    started = time.monotonic()
    with (directory / "adapter.log").open("wb") as log:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=directory,
            env=adapter_environment(request.get("environment_allowlist", [])),
        )
        try:
            child.communicate(json.dumps(request).encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # This process group was created here. Escaped Native workers remain
            # UNKNOWN; no new trial is scheduled without positive adapter closure.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Leader exit does not imply group exit: workers can ignore TERM.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=5)
            raise RuntimeError("adapter_timeout_OWNED_NATIVE_CLOSURE_UNKNOWN_stop_sweep")
    require(child.returncode == 0, "adapter_failed; closure uncertain, stop sweep")
    report = json.loads((directory / "REPORT.json").read_text())
    return report, time.monotonic() - started


def run(policy_path, output, execute=False):
    policy = validate_policy(json.loads(Path(policy_path).read_text()))
    roster = Path(policy["case_manifest"])
    require(
        roster.is_absolute() and roster.is_file(), "absolute complete case/seed manifest required"
    )
    manifest_sha = digest(roster)
    require(manifest_sha == policy["case_manifest_sha256"], "case manifest changed")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    common = {
        "mode": policy["mode"],
        "case_ids": policy["case_ids"],
        "case_manifest": str(roster),
        "case_manifest_sha256": manifest_sha,
        "runtime_fingerprint": policy["runtime_fingerprint"],
        "allocated_gpus": policy["allocated_gpus"],
        "no_cloud_creation": True,
        "environment_allowlist": policy.get("environment_allowlist", []),
    }
    # Two N1 repeats distinguish ordinary repeat variation from concurrency.
    topologies = policy.get(
        "topologies", [Topology(n, 1).to_dict() for n in policy.get("levels", [])]
    )
    schedule = [topologies[0], topologies[0]] + topologies[1:]
    require(len(schedule) <= policy["max_trials"], "max_trials must include two N1 repeats")
    save(
        root / "PLAN.json",
        {
            "policy": policy,
            "schedule_topologies": schedule,
            "execute": execute,
            "recommendation_is_pilot_only": True,
            "formal_equivalence_requires_separate_cohort": True,
        },
    )
    if not execute:
        return {
            "status": "dry_plan_only",
            "output": str(root),
            "schedule": [from_dict(t).envs_per_gpu for t in schedule],
            "topologies": schedule,
        }
    require(
        policy.get("resource_authorized") is True,
        "execution must stay within existing session resource authorization",
    )
    deadline = time.monotonic() + policy["total_timeout_s"]
    summary = {"status": "running", "trials": [], "recommendation": None}
    save(root / "SUMMARY.json", summary)
    baseline = repeat = best = None
    plateau = 0
    try:
        directory = root / "inspect"
        directory.mkdir()
        allocation, _ = invoke(
            policy["adapter_argv"],
            {**common, "op": "inspect", "output_dir": str(directory)},
            directory,
            min(60, policy["total_timeout_s"]),
        )
        require(
            allocation.get("ownership_verified") is True
            and not allocation.get("foreign_gpu_processes"),
            "owned exclusive allocation required",
        )
        require(
            "independent" in allocation.get("supported_topologies", ["independent"]),
            "N1 repeat controls must be supported",
        )
        require(
            len(allocation["gpu_uuids"]) == policy["allocated_gpus"]
            and len(set(allocation["gpu_uuids"])) == policy["allocated_gpus"],
            "exact distinct physical GPU identity",
        )
        for index, topology in enumerate(schedule):
            shape = from_dict(topology)
            level = shape.envs_per_gpu
            family = shape.family
            supported = allocation.get("supported_topologies", ["independent"])
            if family not in supported:
                summary["trials"].append(
                    {"topology": topology, "status": "unsupported_not_launched"}
                )
                continue
            if len(policy["case_ids"]) < policy["allocated_gpus"] * level * policy.get(
                "min_waves", 1
            ):
                summary["trials"].append(
                    {
                        "topology": topology,
                        "status": "insufficient_predeclared_pilot_cases_not_launched",
                    }
                )
                continue
            remaining = deadline - time.monotonic()
            if remaining < policy["trial_timeout_s"] + 25:
                summary["status"] = "budget_bound"
                break
            directory = root / ("trial_%02d_n%d" % (index, level))
            directory.mkdir()
            request = {
                **common,
                "op": "run_trial",
                "trial_id": str(uuid.uuid4()),
                "output_dir": str(directory),
                "envs_per_gpu": level,
                "topology": topology,
                "allocation": allocation,
                "trial_timeout_s": policy["trial_timeout_s"],
            }
            report, measured = invoke(
                policy["adapter_argv"], request, directory, policy["trial_timeout_s"]
            )
            validate_report(report, request, policy, allocation, directory)
            require(
                report["wall_seconds"] <= measured + 2,
                "adapter cannot claim more completed workload time than observed",
            )
            reasons = candidate_gate(report, baseline, repeat, policy)
            row = {
                "envs_per_gpu": level,
                "topology": topology,
                "counts": counts(report),
                "eligible": not reasons,
                "reasons": reasons,
                "audited_or_accepted_per_gpu_hour": score(report, policy),
                "elapsed_adapter_seconds": measured,
                "measured_end_to_end_per_gpu_hour": (
                    counts(report)["audited"]
                    if policy["mode"] == "evaluation"
                    else counts(report)["success"]
                )
                * 3600
                / measured
                / policy["allocated_gpus"],
            }
            # Rank actual external wall time, including startup and audit drain.
            row["ranking_score"] = row["measured_end_to_end_per_gpu_hour"]
            summary["trials"].append(row)
            if reasons:
                summary["status"] = "stopped_candidate_rejected"
                save(root / "SUMMARY.json", summary)
                break
            if index == 0:
                baseline = report
            elif index == 1:
                repeat = report
            if index >= 1:
                reference = row["ranking_score"] if best is None else best["ranking_score"]
                if best is None or row["ranking_score"] > reference:
                    gain = (
                        1
                        if best is None or reference == 0
                        else row["ranking_score"] / reference - 1
                    )
                    best = row
                    plateau = 0 if gain >= policy["min_gain_fraction"] else plateau + 1
                else:
                    plateau += 1
                summary["recommendation"] = {
                    "envs_per_gpu": best["envs_per_gpu"],
                    "topology": best["topology"],
                    "pilot_only": True,
                    "scope": "this workload/model/hardware only; not formal equivalence",
                }
            save(root / "SUMMARY.json", summary)
            if plateau >= policy["plateau_limit"]:
                summary["status"] = "throughput_plateau"
                break
        else:
            summary["status"] = "completed_bounded_sweep"
    except BaseException as exc:
        summary.update(status="needs_diagnosis_no_further_trials", error=repr(exc))
        save(root / "SUMMARY.json", summary)
        raise
    save(root / "SUMMARY.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.policy, args.output, args.execute), indent=2))
