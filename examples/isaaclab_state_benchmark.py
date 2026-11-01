"""Experimental, bounded GPU physics capacity probe; never a learned-policy evaluation.

Run inside a compatible existing Isaac Lab environment. No SDK/assets are bundled.
Only --help and source checks are exercised by CPU CI; actual GPU capacity is unclaimed.
"""

import argparse
import ctypes
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--owned-allocation-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (
        1 <= args.num_envs <= args.max_envs
        and 1 <= args.steps <= 1000000
        and 0 <= args.warmup <= 10000
    ):
        raise ValueError("bounded probe sizes required")
    if not args.device.startswith("cuda:"):
        raise ValueError("explicit CUDA ordinal required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"], text=True
    )
    devices = {
        row.split(",")[0].strip(): row.split(",")[1].strip().lower()[-12:]
        for row in inventory.splitlines()
    }
    if args.expected_gpu_uuid not in devices:
        raise ValueError("expected physical GPU absent")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if any(row.split(",")[0].strip() == args.expected_gpu_uuid for row in active.splitlines()):
        raise ValueError("selected GPU has existing compute processes; inspect ownership first")
    cuda = ctypes.CDLL("libcuda.so.1")
    ordinal = int(args.device.split(":")[1])
    device = ctypes.c_int()
    pci = ctypes.create_string_buffer(64)
    if (
        cuda.cuInit(0)
        or cuda.cuDeviceGet(ctypes.byref(device), ordinal)
        or cuda.cuDeviceGetPCIBusId(pci, 64, device)
    ):
        raise RuntimeError("CUDA PCI query failed")
    if pci.value.decode().lower()[-12:] != devices[args.expected_gpu_uuid]:
        raise ValueError("CUDA/NVML physical GPU mismatch")
    launcher_started = time.perf_counter()
    from isaaclab.app import AppLauncher

    application = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
    env = None
    result = None
    begin = time.perf_counter()
    launcher_wall = time.perf_counter() - launcher_started
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 - registers user-installed tasks after Kit startup.
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=True)
        cfg.seed = args.seed
        env = gym.make(args.task, cfg=cfg)
        if env.unwrapped.num_envs != args.num_envs:
            raise RuntimeError("actual vectorized environment count differs")
        env.reset()
        torch.manual_seed(args.seed)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                env.step(actions)
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            for _ in range(args.steps):
                if not application.is_running():
                    raise RuntimeError("simulator stopped before complete probe")
                env.step(actions)
            torch.cuda.synchronize(args.device)
            wall = time.perf_counter() - start
        result = {
            "evidence_kind": "physics_capacity_diagnostic_not_policy_evaluation",
            "allocation_id": args.owned_allocation_id,
            "gpu_uuid": args.expected_gpu_uuid,
            "cuda_pci": pci.value.decode(),
            "task": args.task,
            "num_envs": args.num_envs,
            "processes": 1,
            "launcher_wall_seconds": launcher_wall,
            "warmup_steps": args.warmup,
            "measured_vector_steps": args.steps,
            "steady_wall_seconds": wall,
            "environment_steps_per_second": args.steps * args.num_envs / wall,
            "torch_version": torch.__version__,
            "cameras_enabled": False,
            "liquid_task_not_validated": True,
            "torch_peak_allocated_bytes_not_total_Kit_VRAM": torch.cuda.max_memory_allocated(
                args.device
            ),
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            application.close()
    if result is not None:
        result["post_AppLauncher_wall_seconds_including_cleanup"] = time.perf_counter() - begin
        with (output / "CAPACITY_DIAGNOSTIC.json").open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
