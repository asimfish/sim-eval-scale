# Contributing

Use an isolated branch (`feat/`, `fix/`, `perf/`, `docs/`, `test/`, or `ci/`). Preserve unrelated work. Commit focused changes and document actual verification. Do not force-push shared branches, bypass hooks, or merge someone else's work without authority.

Suggested commit format: `[scope/op]: concise summary`, for example `[runner/fix]: reject incomplete audit evidence`. Include the relevant issue/task reference in the body.

## Local checks

```bash
python -m pip install ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
python -m unittest discover -s tests -v
python examples/synthetic_demo.py --output /tmp/new-synthetic-run
python examples/isaaclab_state_benchmark.py --help
```

Use new output directories. CPU tests include mocked/synthetic reports to test orchestration; they do not measure actual simulator capacity. Optional SDK examples must declare the installed simulator version and actual GPU verification before claiming support.

## Performance changes

Include the frozen workload/hardware, hypothesis, changed variables, repeated baseline/candidate measurements, end-to-end accepted throughput/GPU-hour, original correctness evidence and uncertainty. Separate cold/warm/steady time. Do not claim improvements from utilization alone, a single unsynchronized kernel launch, reduced fidelity or dropped failures.

Tests should cover rejection and recovery boundaries as well as successful paths. New adapters must preserve original scene/model seeds, observation/control semantics and the original success/Data Gate, and provide positive owned-worker closure. No universal environment count should be introduced without workload-specific evidence.

## Public material

Submit only redistributable code/docs/data. Do not add models, proprietary USD assets, credentials, private endpoints, raw experiment logs, host/user directory paths or internal task records. Cite version-specific primary documentation. Runtime manifests bind private external inputs by their identifiers/hashes locally; do not require those private artifacts for CPU CI.
