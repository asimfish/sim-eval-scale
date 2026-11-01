# Task adapter protocol

The core schedules bounded trials; a reviewed task adapter runs real simulation/policy/collection and the original independent validator. The core has no universal Isaac task adapter. Neither a healthy service nor a CPU mock is actual physical acceptance.

## Process boundary

`adapter_argv` is a JSON argv array. Each invocation reads one JSON request from stdin and writes `output_dir/REPORT.json`, then exits. stdout/stderr stay in the trial log. The working directory is the newly created trial directory. Use absolute code/config paths; no shell expansion is performed. Do not create cloud resources inside an adapter.

The adapter is trusted executable task code, not sandboxed untrusted input. Explicitly bound time, GPU count, processes and environments. On timeout, the core stops its owned adapter process group and marks descendant closure unknown; it never assumes escaped workers were stopped and never escalates another trial automatically.

## Inspection request/report

An `op=inspect` request includes the owned GPU count, runtime/scene-manifest fingerprints and output directory. Verify the actual allocation identity, physical CUDA/NVML PCI order and process ownership. Return:

```json
{
  "allocation_id": "actual-own-job-id",
  "ownership_verified": true,
  "gpu_uuids": ["actual-physical-GPU-UUID-in-logical-order"],
  "foreign_gpu_processes": [],
  "supported_topologies": ["independent", "vectorized", "hybrid"]
}
```

Declare only actually supported families. Without `supported_topologies`, only independent workers are assumed. Existing unrelated GPU processes make inspection fail. An environment variable or device name alone is not proof of ownership.

## Trial request

```json
{
  "op": "run_trial",
  "trial_id": "unique-id",
  "output_dir": "/absolute/new-trial",
  "mode": "evaluation",
  "allocated_gpus": 1,
  "topology": {"processes_per_gpu": 2, "envs_per_process": 8},
  "envs_per_gpu": 16,
  "case_ids": ["fixed-case-1"],
  "case_manifest": "/absolute/full-scene-and-model-seeds.json",
  "case_manifest_sha256": "manifest-sha256",
  "runtime_fingerprint": "scientific-runtime-manifest-sha256",
  "allocation": {"allocation_id": "actual-own-job-id", "gpu_uuids": ["actual-uuid"]},
  "trial_timeout_s": 1800,
  "no_cloud_creation": true,
  "environment_allowlist": []
}
```

This shortened schema example is not a sufficient16-environment pilot. Predeclare enough unique cases or explicitly labelled independent repeats to fill all slots across multiple waves. The full manifest includes scene/model seeds, task variation, model/ckpt/horizon, camera and original acceptance conditions. It is fixed across candidates. The scientific fingerprint excludes scheduling topology but includes every scientific setting and executable adapter source.

## Trial report

```json
{
  "trial_id": "same-request-id",
  "evidence_kind": "actual_workload",
  "allocation_id": "same-own-allocation",
  "gpu_uuids": ["same-physical-uuid-list"],
  "runtime_fingerprint": "same-scientific-fingerprint",
  "case_manifest_sha256": "same-manifest-sha256",
  "topology": {"processes_per_gpu": 2, "envs_per_process": 8},
  "envs_per_gpu": 16,
  "mode": "evaluation",
  "owned_workers_closed": true,
  "wall_seconds": 723.2,
  "correctness": {
    "image_input_contract": true,
    "controller_contract": true,
    "state_isolation": true,
    "trajectory_against_N1_repeat": true,
    "original_oracle": true
  },
  "correctness_evidence": {
    "image_input_contract": {"file": "checks/image.json", "sha256": "..."},
    "controller_contract": {"file": "checks/controller.json", "sha256": "..."},
    "state_isolation": {"file": "checks/state.json", "sha256": "..."},
    "trajectory_against_N1_repeat": {"file": "checks/trajectory.json", "sha256": "..."},
    "original_oracle": {"file": "checks/oracle.json", "sha256": "..."}
  },
  "metrics": {"gpu_util_median_pct": 96, "min_free_vram_mib": [8192]},
  "cases": [{
    "case_id": "fixed-case-1", "status": "success", "audit_passed": true,
    "audit_file": "cases/fixed-case-1/audit-wrapper.json", "audit_sha256": "..."
  }]
}
```

Return every attempted case exactly once. Status is `success`, `failure`, `infrastructure` or `unknown`; an original physically failed episode can still be a valid audited evaluation. Infrastructure/unknown is never silently counted as a model failure. A failed quality/resource gate stops escalation. The current implementation stops the sweep on a rejected candidate; use a separately frozen family plan after positive worker closure if trying another architecture.

Every correctness evidence file is SHA-bound within the trial and contains `check`, `trial_id`, `runtime_fingerprint`, Boolean `passed`, and actual validator details. Never fill a check with true without measuring it. Keep full physical-trajectory, per-lane/RNG/reset, camera and original-oracle proof. The core checks identity and evidence integrity; it cannot make a fake task validator scientifically valid.

Each successful/failed case's original-audit wrapper contains matching `case_id`, `status`, `audit_passed=true`, plus original validator output paths/hashes. In collection mode successful rows also need `training_eligible=true` from the original Data Gate. Missing, overwritten or rejected data stays out of the accepted dataset.

`owned_workers_closed=true` requires positive closure receipts for the adapter's native simulation, model and CPU audit descendants. Record PID generation/starttime/namespace, not a bare PID. No foreign processes may be killed to satisfy this condition.

## Timing and recommendation

- `wall_seconds` is the complete workload interval, including startup and drain, not kernel launch latency. The core ranks **externally measured adapter wall time**.
- Save cold/warm/steady timing and GPU/CPU/RAM/readback/encoding/audit curves alongside the report. GPU utilization and VRAM are not the objective function.
- Initial twoN1 controls expose natural repeat variability. Default pilot guardrails compare success drop and extra matched disagreements against those controls. These are pre-trial configurable stopping rules, **not a formal statistical equivalence test**.
- A recommendation always has `pilot_only=true`. Formal evaluation still needs the original complete cohort, predeclared margins, uncertainty and failure analysis. Repeat/interleave candidates before claiming a speedup.
- Insufficient budget stops new trials before the next one starts. UNKNOWN must be reconciled before a fresh plan; no implicit resume/retry or outcome-selected reuse exists.

## Runtime environment

Linux/Python3.10+. Required system/CUDA/graphics/cache/thread settings are forwarded. Credentials are not inherited by default; authorized external calls can opt in to variable **names** through `environment_allowlist`. Read values only from the secure process environment; never put them in policy JSON, request data, logs or command arguments. See [security boundaries](../SECURITY.md).
