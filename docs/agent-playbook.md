# Agent playbook / 可直接交给agent执行

Read the repository README, adapter protocol, tuning matrix and your task's operating rules. Work in an isolated scope. Use existing session authorization and preserve other users' processes, data and hooks.

## Outcome

Maximize original-acceptance-completed episodes/GPU-hour for evaluation, or original-Data-Gate-accepted unique trajectories/GPU-hour for collection. GPU count, simulator process count and vectorized environments per process are separate variables. A prior4-world/GPU fallback is not a limit.1024-state-world anecdotes are not liquid/RGB results.

## Autonomous loop

1. Freeze task/scenes/seeds, model/ckpt/horizon, image processing, controller/dt/observation cadence, physics/fluid, original oracle and required data. Save a scientific manifest. Inspect ownership and actual physical GPUs; use no extra allocation without session authorization.
2. Capture separate synchronized startup, physics/PBD, rendering, policy, readback, writer and audit/queue measurements. List required cameras, particles, reserved buffers and process/weight duplication. Do not guess a bottleneck from utilization alone.
3. Run twoN1 controls on the same physical allocation and identical seeded roster. Save actual first RGB/qpos/actions, full physical trajectories, model state/RNG and original audits. Run identical-input policy canaries before batching changes.
4. For camera-free rigid tasks start with one vectorized worker and bounded1,2,4,...128,256,512,1024+ candidates. For vision/liquid compare independent, vectorized and hybrid candidates; validate clone/particle/camera/collision/reset/lane isolation before large pilots. Keep a few processes instead of hundreds of Kit instances.
5. Implement a thin **real** task adapter with original validators and positive descendant closure. Use `plan`, `policy` and `sweep`; first dry-plan, then execute within existing authorization. The core does not acquire cards, implement every task or validate scientific facts from a fake adapter.
6. Use a fixed, predeclared pilot with all layouts/skills and enough waves. Only change scheduling per sweep. Repeat/interleave topologies and report distributions when selecting the best. Record unsupported families rather than inventing support.
7. Try semantics-preserving amortization first: trusted private caches, persistent workers with fully validated resets, GPU observation batches, independent lane state, batched particle readbacks, bounded CPU audits and asynchronous storage. Preserve all oracle-required samples.
8. Put particle resolution/neighborhood, solver/dt, collision simplification and observation/render changes into separately versioned fidelity experiments. Require original physical/input acceptance before adopting them; do not silently replace baseline scores or training data.
9. Stop escalating on OOM, invalid input/state, overflow, observer errors, unknown ownership/closure or worse useful throughput. Keep all attempts/unknowns. After positive closure a new separately frozen fallback plan is possible; no selected retries or duplicate datasets.
10. Formalize the pilot winner on the task's full fixed cohort. Report per-method/ckpt/horizon/layout single/concurrent rates, paired disagreements and uncertainty. Aggregate closeness can be a practical decision; lack of significance or0%/0% does not prove equivalence or correct physics.
11. After single-card density is measured, shard across the authorized GPU allocation. Compare actual end-to-end accepted throughput; do not assume one physics scene automatically spans cards. Keep metrics, checkpoint/progress custody, atomic large-ledger sync and owned-worker closure.
12. Deliver the runnable task adapter/config, versioned manifests, per-topology throughput and rejection table, full original audits, formal comparison, and exact limitations. Close only your resources. Continue useful work without repeated permission questions when already authorized.

## Maintenance output

For every optimization record: frozen workload/hardware, hypothesis, one changed knob, baseline/repeated measurements, accepted output/GPU-hour, correctness/unknown counts, raw evidence locations and rollback. A recommendation is specific to this simulator/model/hardware configuration; do not set a universal global environment count.
