"""Explain measured costs without conflating physics, rendering, and GPU busy time."""

import math


def positive(value, name, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError("invalid " + name)
    return value


def sensor_budget(envs, width=640, height=480, cameras=1, hz=15):
    for name, value in [("envs", envs), ("width", width), ("height", height), ("cameras", cameras)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    positive(hz, "hz")
    pixels = envs * width * height * cameras
    return {
        "envs": envs,
        "pixels_per_observation": pixels,
        "uint8_rgb_bytes_per_observation": pixels * 3,
        "float32_rgb_bytes_per_observation": pixels * 12,
        "raw_uint8_rgb_GB_s_decimal": pixels * 3 * hz / 1e9,
        "scope": "Raw RGB payload only; not total renderer VRAM, encoder cost or measured PCIe traffic.",
    }


def diagnose(profile):
    if profile.get("timing_method") != "synchronized_exclusive_wall":
        raise ValueError(
            "need synchronized, non-overlapping wall-time categories; asynchronous launches are not timings"
        )
    wall = positive(profile["wall_seconds"], "wall_seconds")
    times = profile["exclusive_seconds"]
    if not times:
        raise ValueError("timing categories required")
    for k, v in times.items():
        positive(v, k, zero=True)
    if sum(times.values()) > wall * 1.01:
        raise ValueError("overlapping or inconsistent timings cannot be summed")
    ranked = sorted(times, key=times.get, reverse=True)
    tips = []
    actions = {
        "startup": "Amortize trusted shader/asset warmup; compare persistent vector workers with repeated Kit births.",
        "physics": "Profile rigid and PBD kernels separately; test vectorization and collision isolation before more processes.",
        "fluid": "Measure active versus reserved particles, neighbor work and readback. Treat resolution/solver changes as new fidelity.",
        "render": "Batch required cameras with TiledCamera; measure auxiliary cameras/viewports and liquid isosurface separately.",
        "inference": "Batch observations in GPU memory with independent lane state/RNG; validate identical-input action parity.",
        "readback": "Batch GPU-to-CPU transfers once per physics step; keep required audit evidence and exact timestamps.",
        "encoding": "Use bounded asynchronous writers and audit-preserving chunked storage; avoid per-frame process launches.",
        "audit": "Run the original audit in a bounded CPU pool, overlap with simulation, and track backlog/drain.",
        "queue": "Measure CPU oversubscription, RPC serialization and I/O before increasing world count.",
    }
    for key in ranked:
        if times[key] / wall >= 0.10:
            tips.append(
                {
                    "phase": key,
                    "wall_fraction": times[key] / wall,
                    "next_probe": actions.get(key, "Capture a finer exclusive timing trace."),
                }
            )
    return {
        "ranked_phases": [
            {"phase": k, "seconds": times[k], "wall_fraction": times[k] / wall} for k in ranked
        ],
        "next_probes": tips,
        "gpu_utilization_is_not_a_bottleneck_diagnosis": True,
        "unattributed_seconds": max(0, wall - sum(times.values())),
        "claim": "Profiling guidance, not measured speedup or authorization to change task fidelity.",
    }
