"""Separate GPU count, process count, and vectorized environments per process."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Topology:
    processes_per_gpu: int
    envs_per_process: int

    def __post_init__(self):
        for value in (self.processes_per_gpu, self.envs_per_process):
            if type(value) is not int or value < 1:
                raise ValueError("topology dimensions must be positive integers")

    @property
    def envs_per_gpu(self):
        return self.processes_per_gpu * self.envs_per_process

    @property
    def family(self):
        if self.envs_per_process == 1:
            return "independent"
        return "vectorized" if self.processes_per_gpu == 1 else "hybrid"

    def to_dict(self):
        return asdict(self)


def from_dict(value):
    if set(value) != {"processes_per_gpu", "envs_per_process"}:
        raise ValueError("exact topology schema required")
    return Topology(**value)


def candidates(profile, max_envs_per_gpu, max_processes_per_gpu=4):
    if profile not in ("rigid-state", "rigid-vision", "liquid-vision"):
        raise ValueError("unknown workload profile")
    if type(max_envs_per_gpu) is not int or max_envs_per_gpu < 1:
        raise ValueError("explicit environment cap required")
    if type(max_processes_per_gpu) is not int or max_processes_per_gpu < 1:
        raise ValueError("explicit process cap required")
    # Vectorization grows density without creating hundreds of Kit/model processes.
    powers = []
    n = 1
    while n <= max_envs_per_gpu:
        powers.append(n)
        n *= 2
    values = [Topology(1, n) for n in powers]
    if profile != "rigid-state":
        for p in [2, 4, 8]:
            if p > max_processes_per_gpu:
                continue
            for e in powers:
                if p * e <= max_envs_per_gpu:
                    values.append(Topology(p, e))
    return sorted(
        set(values), key=lambda t: (t.envs_per_gpu, t.processes_per_gpu, t.envs_per_process)
    )


def make_plan(profile, gpus, max_envs_per_gpu, max_processes_per_gpu=4):
    if type(gpus) is not int or gpus < 1:
        raise ValueError("GPU cap must be positive")
    values = candidates(profile, max_envs_per_gpu, max_processes_per_gpu)
    return {
        "schema": "sim_eval_scale.topology_plan.v1",
        "profile": profile,
        "allocated_gpus": gpus,
        "max_envs_per_gpu": max_envs_per_gpu,
        "max_processes_per_gpu": max_processes_per_gpu,
        "candidates": [
            dict(
                t.to_dict(),
                family=t.family,
                total_envs=t.envs_per_gpu * gpus,
                recommended_min_pilot_cases=2 * t.envs_per_gpu * gpus,
            )
            for t in values
        ],
        "claim": "Planning candidates only, not measured capacity. No GPUs or simulator processes created.",
    }
