import unittest

from sim_eval_scale.profiling import diagnose, sensor_budget
from sim_eval_scale.topology import Topology, candidates, from_dict, make_plan


class TopologyTests(unittest.TestCase):
    def test_1024_is_one_vector_worker_not_1024_Kits(self):
        plan = make_plan("rigid-state", 1, 1024)
        self.assertEqual(plan["candidates"][-1]["envs_per_process"], 1024)
        self.assertTrue(all(t["processes_per_gpu"] == 1 for t in plan["candidates"]))

    def test_caps_are_enforced_for_every_shape(self):
        shapes = candidates("liquid-vision", 64, 2)
        self.assertTrue(all(t.envs_per_gpu <= 64 and t.processes_per_gpu <= 2 for t in shapes))
        self.assertEqual(len(shapes), len(set(shapes)))
        self.assertIn(Topology(2, 16), shapes)
        self.assertNotIn(Topology(4, 4), shapes)

    def test_no_universal_4_environment_ceiling(self):
        self.assertEqual(candidates("rigid-state", 2048)[-1], Topology(1, 2048))

    def test_invalid_dimensions(self):
        for pair in [(True, 1), (0, 2), (2, -1), (2, 1.1)]:
            with self.assertRaises(ValueError):
                Topology(*pair)
        with self.assertRaises(ValueError):
            from_dict({"processes_per_gpu": 1, "envs_per_process": 2, "extra": 1})

    def test_density_and_gpu_count_distinct(self):
        self.assertEqual(make_plan("rigid-vision", 4, 32)["candidates"][-1]["total_envs"], 128)

    def test_sensor_budget_is_not_renderer_capacity(self):
        b = sensor_budget(1024)
        self.assertAlmostEqual(b["raw_uint8_rgb_GB_s_decimal"], 14.155776)
        self.assertEqual(b["uint8_rgb_bytes_per_observation"], 943718400)

    def test_async_and_double_counted_timings_rejected(self):
        with self.assertRaises(ValueError):
            diagnose({"timing_method": "asynchronous_launches"})
        with self.assertRaises(ValueError):
            diagnose(
                {
                    "timing_method": "synchronized_exclusive_wall",
                    "wall_seconds": 10,
                    "exclusive_seconds": {"physics": 8, "render": 8},
                }
            )

    def test_render_bottleneck_identified_from_timings(self):
        p = diagnose(
            {
                "timing_method": "synchronized_exclusive_wall",
                "wall_seconds": 100,
                "exclusive_seconds": {"physics": 10, "render": 70, "inference": 10},
            }
        )
        self.assertEqual(p["ranked_phases"][0]["phase"], "render")
        self.assertEqual(p["unattributed_seconds"], 10)

    def test_nonfinite_numbers_rejected(self):
        with self.assertRaises(ValueError):
            sensor_budget(1, hz=float("nan"))
        with self.assertRaises(ValueError):
            diagnose(
                {
                    "timing_method": "synchronized_exclusive_wall",
                    "wall_seconds": 10,
                    "exclusive_seconds": {"physics": float("inf")},
                }
            )


if __name__ == "__main__":
    unittest.main()
