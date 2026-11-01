# Sim Eval Scale

Evidence-gated throughput experiments for simulator evaluation and data collection.

**Four independent worlds per GPU is a tested fallback, not a scaling ceiling.** This project separates GPU count, simulator process count and vectorized environments per process. It helps agents explore independent, vectorized and hybrid execution while retaining original workload acceptance checks.

[中文说明](README.zh-CN.md) · [Agent workflow](docs/agent-playbook.md) · [Liquid/vision tuning](docs/liquid-and-vision.md) · [Adapter protocol](docs/adapter-protocol.md)

## What works today

- A standard-library Python core with bounded trial/time/GPU/world limits.
- Topology planning for rigid-state, rigid-vision and PBD-liquid/vision workloads, including 1024+ environment candidates.
- Two N1 repeat controls, fixed scenes/model seeds, same-allocation binding, original audit hashes and independent lane-state checks.
- Useful-throughput ranking, resource/quality rejection, plateau stopping and explicit unknown/infrastructure outcomes.
- Atomic publication of large hot JSON snapshots; private subprocess working directories and opt-in credential forwarding.
- CPU tests, a synthetic runnable example, and an experimental installed-Isaac-Lab state capacity probe.

**Not bundled:** a universal Isaac task/policy adapter, simulator assets, model weights or cloud credentials. Actual evaluation/collection needs your task adapter and original acceptance oracle. The capacity probe is not a policy success test. There is no claim that1024 fluid/RGB environments have been measured.

## Install and run without a GPU

```bash
python -m pip install .
python -m unittest discover -s tests -v
python examples/synthetic_demo.py --output /tmp/sim-scale-demo

sim-eval-scale plan --profile rigid-state --gpus 1 \
  --max-envs-per-gpu 1024 --output /tmp/state-plan.json
sim-eval-scale plan --profile liquid-vision --gpus 4 \
  --max-envs-per-gpu 64 --max-processes-per-gpu 4 \
  --output /tmp/liquid-plan.json
sim-eval-scale sensor-budget --envs 1024 --width 640 --height 480 --hz 15
sim-eval-scale diagnose examples/profile.json
```

Planning creates no simulator or GPU jobs. The synthetic example is labelled synthetic and cannot qualify as actual workload evidence.

## Real workload sweeps

Implement the [adapter protocol](docs/adapter-protocol.md), freeze a full scene/model-seed manifest and a scientific runtime manifest, then:

```bash
sim-eval-scale policy --plan /absolute/topology-plan.json \
  --cases /absolute/pilot-cases.json --runtime-manifest /absolute/runtime-manifest.json \
  --adapter-json '["python3","/absolute/task_adapter.py"]' \
  --mode evaluation --authorized-allocation --output /absolute/policy.json

# Preview. Output directories are exclusive; use a fresh path for each run.
sim-eval-scale sweep --policy /absolute/policy.json --output /absolute/preview
# Execute only inside the already authorized, inspected GPU allocation.
sim-eval-scale sweep --policy /absolute/policy.json --output /absolute/sweep --execute
```

Use `--mode collection` for Data-Gate-passed training trajectories. Do not replace original success criteria, silently resample failures or drop audit data to raise throughput. Default CLI headroom is2GiB; set an appropriate per-GPU bound before trials (for example8GiB on a large-memory workload). Pilot tolerances are configuration defaults, not proven universal equivalence margins.

The selected score is independently audited episodes/GPU-hour for evaluation, or original-Data-Gate-accepted unique trajectories/GPU-hour for collection. It includes adapter startup and audit drain. GPU utilization and allocated VRAM are supporting measurements, not the optimization objective.

## High-density state capacity probe

Inside a compatible, user-installed Isaac Lab2.x environment and an exclusive owned allocation:

```bash
# GPU UUID/allocation ID must come from your real allocation inventory.
python examples/isaaclab_state_benchmark.py \
  --expected-gpu-uuid YOUR_ACTUAL_GPU_UUID --owned-allocation-id YOUR_JOB_ID \
  --num-envs 1024 --device cuda:0 --steps 256 --output /absolute/new-capacity-run
```

This is an experimental camera-free, zero-action state task probe. Only CLI/source checks run in CPU CI; run it on your installed SDK before relying on its output. It does not validate liquid, vision, learned policies or task success. More environments require explicit bounds and measured capacity, not an extrapolation from this example.

## Why liquid/RGB needs a separate protocol

At 1024 environments, one 640×480 RGB camera at15Hz produces about14.16GB/s of raw RGB payload before accounting for renderer history, conversion, inference, extra cameras and recordings. That is not measured bus traffic or total VRAM. PBD neighbor solving, particle readback and liquid surface meshing add distinct costs. Benchmark these phases instead of attributing all GPU busy time to liquid physics.

See [the parameter matrix and diagnostic protocol](docs/liquid-and-vision.md). Changes to particle spacing/neighborhood/solver settings are fidelity changes and require separate acceptance. Camera resolution, render cadence and liquid appearance can change policy inputs even when physics stays identical.

## Evidence and limitations

A prior task-specific implementation completed independent-worker learned-policy evaluations with4×96GB GPUs and4 worlds/GPU. Sampled GPU utilization was93–98%, with one completion window around45 cases/hour. This is workload-specific evidence, not a reproducible benchmark package or a universal speedup claim. Aggregate single/multi success rates were close, but strict predeclared statistical equivalence and per-trajectory identity were not established. Private inputs/logs are not distributed.

## Development

Python3.10+, Linux for process-group lifecycle control. The core has no runtime third-party dependencies. Optional Isaac Lab/PyTorch/Gymnasium come from the user's licensed/version-pinned environment. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [design decision](docs/adr-001-topologies.md).

MIT licensed. No NVIDIA SDK source or assets are redistributed.
