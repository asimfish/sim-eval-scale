# Provenance and optional components

The orchestration, planning, artifact and example instrumentation code in this repository is newly authored for this project. It was generalized from a private local orchestration prototype; private experiment data, infrastructure identifiers, task assets, model weights and SDK implementation files are not distributed.

Isaac Sim, Isaac Lab, PhysX, PyTorch and Gymnasium are optional user-installed tools with their own licenses/terms. This repository calls documented APIs and does not redistribute their source, models or assets. The experimental state example follows the public AppLauncher/task-registry API pattern; the official version-specific reference is:
https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/scripts/environments/random_agent.py

Primary technical references are linked in docs/liquid-and-vision.md. Source documentation compatibility is not a claim that an actual GPU trial was run. Prior task-specific utilization/throughput mentioned in README is contextual, not a public reproducible benchmark dataset or universal speedup.
