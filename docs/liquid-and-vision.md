# Scaling rigid, RGB and particle-liquid workloads

A camera-free rigid task can amortize a large batch of environments in one process. A scene with RTX cameras, a large policy and PBD liquid is a different workload. Do not infer liquid capacity from a1024-environment state benchmark, or infer a four-environment limit from an isolated-process fallback.

## First capture a useful profile

Record cold startup, warm startup, steady simulated seconds, audited episode throughput, CPU cores/RAM, per-GPU utilization and memory, render/update/observation rates, active/reserved particle counts, RPC queue/latency, encoding and audit drain. Use CUDA events or synchronization for GPU timing. Do not sum overlapping asynchronous intervals; `sim-eval-scale diagnose` requires exclusive synchronized wall-time categories.

Run controlled **diagnostic-only** probes with an identical frozen scene where possible:

| Probe | What it helps separate | What it cannot establish |
|---|---|---|
| Rigid state task, no camera/policy | Vectorization and rigid physics capacity | Liquid or learned-policy success |
| Actual liquid physics, observation disabled | PBD/rigid cost and allocations | Vision policy parity |
| Same simulation with required camera | Incremental observation/render cost | That liquid is the only bottleneck |
| Fixed recorded observations through policy | Model batching, state/RNG and RPC cost | Closed-loop physical success |
| Full policy with diagnostic outputs | End-to-end baseline and audit cost | Effects of removing required evidence |

Removing liquid/cameras changes a task. Keep these probes out of formal success rates and training data. Interleaved repeated measurements and trace evidence are stronger than subtracting one noisy run from another.

## Parameter matrix

### Scheduling and data flow: preserve task semantics, then prove parity

| Axis | Candidate experiment | Required measurements/checks |
|---|---|---|
| Processes × vectorized environments | Compare1×N,2×N/2,4×N/4 at the same total worlds | Kit/model overhead, total audited throughput, same-allocation repeat controls |
| Persistent workers | Reuse compiled/runtime setup between validated episode resets | Clean per-lane queues/RNG/memory, particles and contact state; no leakage from a prior case |
| Physics replication/Fabric | Inspect installed SDK support for homogeneous assets | Startup gain, scene differences, collision filtering, particle/reset compatibility |
| GPU-side observations/inference | Batched tensors and independent policy lanes | Same-input output canary, tensor lifetime, padding/history and action order |
| CPU/Kit/BLAS thread budgets | Measure a small thread-count grid with constant task settings | Effective cgroup cores, oversubscription, latency and total throughput |
| GPU particle readback | One consistent snapshot per tick, batch object queries and outputs | Exact original timestamps/semantics; never silently omit oracle-required data |
| Writers/auditors | Bounded queues, asynchronous encoding, chunked storage and CPU audit | All attempts retained, bounded backlog, audit hashes, memory pressure and drain time |
| Shader/texture/asset cache | Source-bound immutable seed, private writable worker caches | Source identities, startup time, no physics-state cache reuse |

Thread count, inference precision and kernel/batch implementation can still affect numerical results: validate rather than assume semantic preservation.

### Rendering/observations: input changes need separate validation

- Count every camera/render product, viewport and recording stream. Remove non-required diagnostic output only after the task contract permits it and required observations/evidence remain intact.
- Try TiledCamera on the installed SDK. It consolidates cloned camera rendering/data handling; it does not by itself isolate light, collision, particle systems or policy lanes.
- Check primary input against the training view: raw RGB, dimensions, preprocessing, lens/pose, render tick, temporal history and reset freshness. Batch asynchronous observations by simulated time, not accidental wall-clock arrival.
- Measure liquid smoothing/anisotropy/isosurface generation and shading separately. They can change visible inputs even without changing particle dynamics.
- Record lower resolution, lower render rate, different antialiasing or cheaper liquid rendering as a new observation configuration. Do not silently use it for the original trained-policy benchmark.

For1024 environments, one640×480 uint8 RGB camera at15Hz has a raw payload rate of14.155776GB/s and943,718,400 bytes per observation batch. Float conversion, multiple copies, render histories, extra sensors and encoders add costs. This arithmetic is a data-volume estimate, not an assertion about actual bus traffic or total memory. Prefer GPU-side batching and only task-required data transfers before shrinking training inputs.

### PBD/liquid physics: fidelity-changing experiments, never silent optimizations

| Parameter group | Why measure it | Acceptance boundary |
|---|---|---|
| Active particles vs reserved capacity | Oversized buffers consume memory even with few active particles | Do not truncate live fluid or hide capacity overflow |
| Particle spacing/contact/rest offsets and mass | Resolution affects particle count, density and contact behavior | Revalidate volume/mass, flow, spill, contact and task outcomes |
| Neighbor limits (`maxNeighborhood` in some versions) | Neighborhood work can dominate dense fluid | Reducing neighbors changes constraints; a faster run is not equal physics |
| Position/velocity iterations, substeps, physics dt | Solver work and accuracy trade off | Preserve old benchmark or explicitly version a new physics regime |
| Particle-system partitioning | Compare shared/per-environment systems if supported | Full particle–particle and particle–rigid isolation, reset behavior and memory |
| Rigid/contact buffers and collision complexity | Under-sized buffers can drop relevant interactions; over-allocation costs memory | High-water measurements, overflow checks, penetration/contact/trajectory validation |

Approximately, filling a fixed volume at spacing `d` uses particle count proportional to `d^-3`. Doubling spacing may reduce count toward one eighth, but this is a resolution change, not a free speedup. Keep physical calibration and original acceptance conditions explicit.

Different particle phases do **not** automatically mean isolation. Particle groups/systems and rigid collision filters have distinct semantics. Version-specific Omni Physics documentation describes separate particle systems without cross-system particle collision, but that alone does not prove complete environment isolation. Check interactions with every robot, cup and room as well as fluid neighbors. Query actual backend/system limits instead of assuming thousands of systems are supported.

## Shared-scene admission gate

Before treating a vectorized/hybrid candidate as an alternative to independent workers:

1. Freeze assets, actual installed SDK/driver, physics settings, controller, seeds, model and observation contracts.
2. Run N1-A and N1-B on the same physical allocation; measure natural repeat variation.
3. Check all cloned camera prims and returned image indices. Re-render fixed states and inspect reset freshness and temporal caches.
4. Hold the inspected room fixed; change neighboring lights from off to4×. Compare against the N1 repeat floor. Verify occluding geometry/shadows and source membership; a roof alone is not proof of isolation.
5. Verify per-env robot/rigid/PBD birth positions, velocities, emitter relationships, particle buffers, resets and collision filtering. Prevent cross-env contacts and wrong global-buffer indexing.
6. Apply identical recorded commands and compare **full** physical trajectories, not just final poses. Then replay identical observations into fresh identical policy state/RNG.
7. Run the actual learned-policy or collection workload using its original oracle. Preserve all failures and uncertainty. No test-selected threshold changes or selected retries.

A failed family stops its escalation. After positive worker closure, test a separately frozen fallback family. High GPU utilization is not a reason to waive this gate.

## Scaling across cards

Shard independent cases/worlds across the explicitly owned GPU allocation. Bind CUDA/NVML PCI order and renderer devices, not guessed GPU names or potentially different UUID namespaces. A renderer's multi-GPU mode is not evidence that one PhysX scene scales physics across cards. Measure single-card density and useful throughput first, then measure cross-card scheduling and policy serving; each new physical allocation needs matched controls.

## Primary references and version scope

Consult the API in the **installed** version before setting a knob. These references describe mechanisms, not measured capacity for your task:

- [Isaac Sim performance handbook](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html): profiling and renderer/CPU configuration. The `latest` page can differ from deployed installations.
- [Isaac Lab camera documentation, v2.0.1](https://isaac-sim.github.io/IsaacLab/v2.0.1/source/overview/sensors/camera.html): tiled render products, image tensors and bandwidth considerations.
- [Isaac Lab scene API, v2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.scene.html): replication/Fabric configuration and constraints.
- [PhysX particle-system documentation,5.6.0](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/ParticleSystem.html): GPU particle buffers, active capacity, phases and neighborhoods.
- [Omni Physics particle guide,110.1](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/particles/particles.html): liquid configuration and visual post-processing.

Source/API compatibility does not imply a benchmark was run. No1024-liquid-environment result is claimed by this repository.
