# ADR-001: Separate scheduling topology from scientific workload

Status: Accepted for orchestration. New GPU capacities remain experimental.

## Context

GPU count, Kit process count and environments per process are different axes. An isolated-process fallback can preserve practical evaluation behavior but repeats Kit/model/renderer overhead. A state-only vectorized task and an RGB/PBD-liquid manipulation task have very different costs. A low process count is not a hardware limit.

## Candidates

1. Independent workers: private Stage, simulation context, renderer and model lane per world. Strong isolation; repeated startup and VRAM overhead.
2. One vectorized worker: many cloned environments in one context, tiled images and batched policy. Better amortization; collision/particle/camera/reset/state isolation requires validation.
3. Hybrid: a few vectorized workers per GPU. Contains shared-scene failure blast radius while amortizing some overhead.

## Decision

Represent topology as `processes_per_gpu × envs_per_process`, independent of the allocation's GPU count. Make all three available as **planning candidates**, and let a task adapter declare which families its installed simulator supports. Use the same immutable scientific fingerprint and scene roster across matched trials. Validate resource/quality closure before ranking audited work per GPU-hour. A task-specific adapter owns simulator API/version details; the core has no simulator, model or cloud dependency.

The boundary is a JSON request/report protocol plus original audit artifacts. The core never edits physics settings, creates cloud resources or guesses whether a GPU belongs to it. Unsupported families are recorded and skipped; a failed correctness/closure gate stops escalation. A separately frozen fallback plan is allowed only after positive worker closure.

## Consequences

- Planning can include1024/2048 vectorized state environments without creating1024 Kit processes.
- Independent process results do not validate shared-scene clones.
- A pilot recommendation is not a success-rate equivalence claim.
- Task adapters must prove camera, particle, collision and policy-state isolation, including full-trajectory checks against N1 repeats.
- More GPUs are considered after useful single-GPU density is measured; cloud allocation changes require new hardware baselines.
