"""CPU-only API example. All timings are synthetic, never a capacity result."""

import argparse
import json
from pathlib import Path

from sim_eval_scale.profiling import diagnose, sensor_budget
from sim_eval_scale.topology import make_plan

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
r = Path(args.output)
r.mkdir(parents=True, exist_ok=False)
profile = {
    "timing_method": "synchronized_exclusive_wall",
    "wall_seconds": 100,
    "exclusive_seconds": {
        "startup": 10,
        "physics": 15,
        "render": 40,
        "inference": 15,
        "readback": 5,
        "encoding": 5,
        "audit": 5,
    },
}
result = {
    "evidence_kind": "synthetic_example_not_GPU_evidence",
    "plan": make_plan("rigid-state", 1, 1024),
    "sensor_budget": sensor_budget(1024),
    "profile_example": diagnose(profile),
}
(r / "EXAMPLE.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps({"output": str(r), "evidence_kind": result["evidence_kind"], "GPU_jobs_created": 0})
)
