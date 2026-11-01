import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_eval_scale import runner as a
from sim_eval_scale.artifacts import publish


class KitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audit = self.root / "audit.json"
        self.audit.write_text(
            json.dumps(
                {
                    "case_id": "c1",
                    "status": "success",
                    "audit_passed": True,
                    "training_eligible": True,
                }
            )
        )
        self.policy = {
            "mode": "evaluation",
            "allocated_gpus": 1,
            "levels": [1, 2, 4],
            "max_envs_per_gpu": 4,
            "case_ids": ["c1"],
            "trial_timeout_s": 10,
            "total_timeout_s": 300,
            "max_trials": 4,
            "min_free_vram_mib": 8192,
            "min_gain_fraction": 0.05,
            "max_extra_disagreement_fraction": 0.05,
            "max_success_drop_pp": 5,
            "plateau_limit": 2,
            "runtime_fingerprint": "a" * 64,
            "adapter_argv": ["python3", "adapter.py"],
        }
        self.allocation = {"allocation_id": "owned", "gpu_uuids": ["gpu-1"]}
        self.request = {"trial_id": "t1", "case_manifest_sha256": "b" * 64, "envs_per_gpu": 2}
        self.report = {
            "trial_id": "t1",
            "evidence_kind": "actual_workload",
            "allocation_id": "owned",
            "gpu_uuids": ["gpu-1"],
            "runtime_fingerprint": "a" * 64,
            "case_manifest_sha256": "b" * 64,
            "envs_per_gpu": 2,
            "mode": "evaluation",
            "owned_workers_closed": True,
            "correctness": dict.fromkeys(a.FLAGS, True),
            "cases": [
                {
                    "case_id": "c1",
                    "status": "success",
                    "audit_passed": True,
                    "audit_file": "audit.json",
                    "audit_sha256": a.digest(self.audit),
                }
            ],
            "metrics": {"min_free_vram_mib": [12000], "gpu_util_median_pct": 96},
            "wall_seconds": 2,
        }
        self.report["correctness_evidence"] = {}
        for flag in a.FLAGS:
            f = self.root / (flag + ".json")
            f.write_text(
                json.dumps(
                    {
                        "check": flag,
                        "trial_id": "t1",
                        "runtime_fingerprint": "a" * 64,
                        "passed": True,
                    }
                )
            )
            self.report["correctness_evidence"][flag] = {"file": f.name, "sha256": a.digest(f)}

    def tearDown(self):
        self.temp.cleanup()

    def test_credentials_are_not_implicitly_forwarded(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-only-not-a-real-key",
                "ZENITH_API_TOKEN": "test-only-token",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        ):
            env = a.adapter_environment()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("ZENITH_API_TOKEN", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertIn("OPENAI_API_KEY", a.adapter_environment(["OPENAI_API_KEY"]))

    def test_real_subprocess_protocol_and_private_cwd(self):
        code = "import json,sys,os;from pathlib import Path;x=json.load(sys.stdin);Path('REPORT.json').write_text(json.dumps({'cwd':os.getcwd(),'has_secret':'ZENITH_API_TOKEN' in os.environ}))"
        with patch.dict("os.environ", {"ZENITH_API_TOKEN": "test-only-token"}):
            report, elapsed = a.invoke(
                [sys.executable, "-c", code], {"op": "inspect"}, self.root, 3
            )
        self.assertEqual(report["cwd"], str(self.root))
        self.assertFalse(report["has_secret"])
        self.assertGreater(elapsed, 0)

    def test_real_adapter_timeout_is_not_reported_as_closed(self):
        with self.assertRaisesRegex(RuntimeError, "CLOSURE_UNKNOWN"):
            a.invoke(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                {"op": "inspect"},
                self.root,
                0.05,
            )

    def test_legacy_levels_honor_process_cap(self):
        self.policy["max_processes_per_gpu"] = 1
        with self.assertRaisesRegex(ValueError, "process bound"):
            a.validate_policy(self.policy)

    def test_timeout_kills_group_even_after_leader_exits(self):
        import signal
        import subprocess
        from unittest.mock import Mock

        child = Mock(pid=12345)
        child.communicate.side_effect = subprocess.TimeoutExpired("adapter", 1)
        child.poll.return_value = None
        child.wait.return_value = 0
        with (
            patch.object(a.subprocess, "Popen", return_value=child),
            patch.object(a.os, "killpg") as killpg,
        ):
            with self.assertRaisesRegex(RuntimeError, "CLOSURE_UNKNOWN"):
                a.invoke(["adapter"], {}, self.root, 1)
        self.assertEqual(
            killpg.call_args_list,
            [unittest.mock.call(12345, signal.SIGTERM), unittest.mock.call(12345, signal.SIGKILL)],
        )

    def test_real_timeout_removes_sigterm_ignoring_worker(self):
        import os
        import signal
        import time

        code = """import os, signal, time
from pathlib import Path
worker = os.fork()
if worker == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('worker.pid').write_text(str(os.getpid()))
    while True:
        time.sleep(1)
while True:
    time.sleep(1)
"""
        worker = None
        try:
            with self.assertRaisesRegex(RuntimeError, "CLOSURE_UNKNOWN"):
                a.invoke([sys.executable, "-c", code], {}, self.root, 1)
            worker = int((self.root / "worker.pid").read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{worker}/stat")
                if not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z"):
                    break
                time.sleep(0.01)
            else:
                self.fail("SIGTERM-ignoring worker still running after adapter timeout")
        finally:
            pid_file = self.root / "worker.pid"
            if worker is None and pid_file.exists():
                worker = int(pid_file.read_text())
            if worker is not None:
                try:
                    os.kill(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_capacity_probe_records_actual_sdk_versions(self):
        import runpy
        import types

        probe = runpy.run_path(
            str(Path(__file__).parents[1] / "examples/isaaclab_state_benchmark.py")
        )
        modules = {
            "isaaclab": types.SimpleNamespace(__version__="lab-test-version"),
            "isaacsim.core.version": types.SimpleNamespace(
                get_version=lambda: ("sim-test", "build")
            ),
        }
        with patch.dict(sys.modules, modules):
            versions = probe["sdk_versions"]()
        self.assertEqual(versions["isaaclab_version"], "lab-test-version")
        self.assertEqual(versions["isaacsim_version"], ("sim-test", "build"))

    def test_valid_evidence_and_bounds(self):
        a.validate_policy(self.policy)
        a.validate_report(self.report, self.request, self.policy, self.allocation, self.root)
        for k, v in [
            ("runtime_fingerprint", "c" * 64),
            ("owned_workers_closed", False),
            ("gpu_uuids", ["foreign"]),
            ("evidence_kind", "synthetic"),
        ]:
            row = copy.deepcopy(self.report)
            row[k] = v
            with self.assertRaises(ValueError):
                a.validate_report(row, self.request, self.policy, self.allocation, self.root)

    def test_audit_corruption_rejected(self):
        self.audit.write_text("{}")
        with self.assertRaises(ValueError):
            a.validate_report(self.report, self.request, self.policy, self.allocation, self.root)

    def test_roster_and_scope_rejected(self):
        for key, value in [("case_id", "other"), ("audit_file", "../foreign.json")]:
            row = copy.deepcopy(self.report)
            row["cases"][0][key] = value
            with self.assertRaises(ValueError):
                a.validate_report(row, self.request, self.policy, self.allocation, self.root)

    def test_collection_requires_original_data_gate(self):
        policy = copy.deepcopy(self.policy)
        policy["mode"] = "collection"
        report = copy.deepcopy(self.report)
        report["mode"] = "collection"
        raw = json.loads(self.audit.read_text())
        raw["training_eligible"] = False
        self.audit.write_text(json.dumps(raw))
        report["cases"][0]["audit_sha256"] = a.digest(self.audit)
        with self.assertRaises(ValueError):
            a.validate_report(report, self.request, policy, self.allocation, self.root)

    def test_quality_and_resource_stop(self):
        baseline = copy.deepcopy(self.report)
        candidate = copy.deepcopy(self.report)
        candidate["metrics"]["min_free_vram_mib"] = [100]
        self.assertIn("VRAM_headroom", a.candidate_gate(candidate, baseline, baseline, self.policy))
        candidate = copy.deepcopy(self.report)
        candidate["cases"][0]["status"] = "unknown"
        self.assertIn(
            "infrastructure_or_unknown",
            a.candidate_gate(candidate, baseline, baseline, self.policy),
        )
        candidate = copy.deepcopy(self.report)
        candidate["cases"][0]["status"] = "failure"
        reasons = a.candidate_gate(candidate, baseline, baseline, self.policy)
        self.assertIn("extra_disagreement_vs_N1_repeat", reasons)
        self.assertIn("pilot_success_drop", reasons)

    def test_dry_plan_never_invokes_adapter_and_fresh_root_required(self):
        manifest = self.root / "cases.json"
        manifest.write_text('[{"case_id":"c1","seed":1}]')
        p = {
            **self.policy,
            "case_manifest": str(manifest),
            "case_manifest_sha256": a.digest(manifest),
        }
        path = self.root / "policy.json"
        path.write_text(json.dumps(p))
        output = self.root / "dry"
        with patch.object(a, "invoke", side_effect=AssertionError("must not execute")):
            result = a.run(path, output)
        self.assertEqual(result["status"], "dry_plan_only")
        self.assertEqual(result["schedule"], [1, 1, 2, 4])
        with self.assertRaises(FileExistsError):
            a.run(path, output)

    def test_stop_on_adapter_exception_no_second_trial(self):
        manifest = self.root / "cases.json"
        manifest.write_text("[]")
        p = {
            **self.policy,
            "case_manifest": str(manifest),
            "case_manifest_sha256": a.digest(manifest),
            "resource_authorized": True,
        }
        path = self.root / "policy.json"
        path.write_text(json.dumps(p))
        inspect = {**self.allocation, "ownership_verified": True, "foreign_gpu_processes": []}
        with patch.object(
            a, "invoke", side_effect=[(inspect, 0.1), RuntimeError("timeout_unknown")]
        ) as call:
            with self.assertRaises(RuntimeError):
                a.run(path, self.root / "run", True)
        self.assertEqual(call.call_count, 2)
        summary = json.loads((self.root / "run/SUMMARY.json").read_text())
        self.assertEqual(summary["status"], "needs_diagnosis_no_further_trials")

    def test_sweep_prefers_useful_throughput_not_largest_parallelism(self):
        manifest = self.root / "cases.json"
        manifest.write_text('[{"case_id":"c1","seed":1}]')
        policy = {
            **self.policy,
            "case_ids": ["c" + str(i) for i in range(8)],
            "levels": [1, 2, 4, 6, 8],
            "max_envs_per_gpu": 8,
            "max_trials": 6,
            "case_manifest": str(manifest),
            "case_manifest_sha256": a.digest(manifest),
            "resource_authorized": True,
        }
        path = self.root / "policy.json"
        path.write_text(json.dumps(policy))

        def adapter(argv, request, directory, timeout):
            if request["op"] == "inspect":
                return {
                    **self.allocation,
                    "ownership_verified": True,
                    "foreign_gpu_processes": [],
                }, 0.1
            row = copy.deepcopy(self.report)
            row.update(
                trial_id=request["trial_id"],
                case_manifest_sha256=request["case_manifest_sha256"],
                envs_per_gpu=request["envs_per_gpu"],
                topology=request["topology"],
                wall_seconds=0.1,
            )
            row["cases"] = []
            for cid in policy["case_ids"]:
                f = directory / (cid + ".json")
                f.write_text(
                    json.dumps({"case_id": cid, "status": "success", "audit_passed": True})
                )
                row["cases"].append(
                    {
                        "case_id": cid,
                        "status": "success",
                        "audit_passed": True,
                        "audit_file": f.name,
                        "audit_sha256": a.digest(f),
                    }
                )
            for flag in a.FLAGS:
                f = directory / (flag + ".json")
                f.write_text(
                    json.dumps(
                        {
                            "check": flag,
                            "trial_id": request["trial_id"],
                            "runtime_fingerprint": "a" * 64,
                            "passed": True,
                        }
                    )
                )
                row["correctness_evidence"][flag] = {"file": f.name, "sha256": a.digest(f)}
            return row, {1: 8, 2: 4, 4: 2, 6: 2.1, 8: 2.2}[request["envs_per_gpu"]]

        with patch.object(a, "invoke", side_effect=adapter):
            summary = a.run(path, self.root / "sweep", True)
        self.assertEqual(summary["recommendation"]["envs_per_gpu"], 4)
        self.assertEqual(summary["status"], "throughput_plateau")
        self.assertTrue(summary["recommendation"]["pilot_only"])
        self.assertEqual(len(summary["trials"]), 6)

    def test_topology_binding_cannot_be_hidden_by_same_total_envs(self):
        request = {**self.request, "topology": {"processes_per_gpu": 1, "envs_per_process": 2}}
        report = {**self.report, "topology": {"processes_per_gpu": 2, "envs_per_process": 1}}
        with self.assertRaises(ValueError):
            a.validate_report(report, request, self.policy, self.allocation, self.root)

    def test_N1_controls_cannot_be_skipped(self):
        manifest = self.root / "cases.json"
        manifest.write_text("[]")
        policy = {
            **self.policy,
            "resource_authorized": True,
            "case_manifest": str(manifest),
            "case_manifest_sha256": a.digest(manifest),
        }
        path = self.root / "policy.json"
        path.write_text(json.dumps(policy))
        inspect = {
            **self.allocation,
            "ownership_verified": True,
            "foreign_gpu_processes": [],
            "supported_topologies": ["vectorized"],
        }
        with patch.object(a, "invoke", return_value=(inspect, 0.1)) as call:
            with self.assertRaisesRegex(ValueError, "N1 repeat controls"):
                a.run(path, self.root / "sweep", True)
        self.assertEqual(call.call_count, 1)

    def test_large_hot_json_publishes_atomic_snapshot(self):
        src = self.root / "source.json"
        dst = self.root / "destination.json"
        src.write_text(json.dumps({"payload": "x" * (2 << 20)}))
        dst.write_text('{"old":true}')
        result = publish(src, dst)
        self.assertTrue(result["published"])
        self.assertEqual(a.digest(src), a.digest(dst))
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_zero_accepted_collection_has_zero_score(self):
        report = copy.deepcopy(self.report)
        report["cases"][0]["status"] = "failure"
        self.assertEqual(a.score(report, {**self.policy, "mode": "collection"}), 0)


if __name__ == "__main__":
    unittest.main()
