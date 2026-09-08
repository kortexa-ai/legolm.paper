import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from legolm_p01.aggregate import registered_conditions, complete_conditions, trajectory_conditions
from legolm_p01.artifacts import checkpoint_gaps, PROMPTS
from legolm_p01.provenance import digest
from legolm_p01.render import Report
from legolm_p01.train import Recipe


class NumericalFailureCoverage(unittest.TestCase):
    def setUp(self):
        self.recipes = [asdict(Recipe(host="mini", cohort="P", modality="imu", steps=6000, seed=s, batch_size=4))
                        for s in (42, 1042, 2042)]
        protocol = {"identity": {"recipes": self.recipes}}
        self.planned = {k: v for k, v in registered_conditions(protocol).items()
                        if v["condition"]["measurement"] == "ranking"
                        and v["condition"]["step"] in (600, 6000)
                        and v["condition"]["settings"] == {"prompt": PROMPTS["imu"][0], "intervention": "correct"}}
        self.keys = {v["condition"]["step"]: k for k, v in self.planned.items()}
        self.groups = {k: [{"recipe": r} for r in self.recipes if step == 600 or r["seed"] != 42]
                       for step, k in self.keys.items()}
        self.failure = {"recipe": self.recipes[0], "training_id": "a" * 64, "unavailable_checkpoints": [2400, 6000]}

    def test_short_endpoint_keeps_all_seeds_and_long_endpoint_has_no_survivor_mean(self):
        groups, unavailable = complete_conditions(self.groups, self.planned, [self.failure])
        self.assertEqual(set(groups), {self.keys[600]})
        self.assertEqual(len(groups[self.keys[600]]), 3)
        self.assertEqual(unavailable[0]["missing_seeds"], [42])
        self.assertEqual(unavailable[0]["available_seeds"], [1042, 2042])
        self.assertEqual(unavailable[0]["reason"], "numerical_failure")
        self.assertEqual(len(self.groups[self.keys[6000]]), 2)

    def test_an_unattempted_or_unexplained_missing_seed_still_blocks_release(self):
        with self.assertRaisesRegex(ValueError, "not justified"):
            complete_conditions(self.groups, self.planned, [])
        wrong = {**self.failure, "unavailable_checkpoints": [2400]}
        with self.assertRaisesRegex(ValueError, "not justified"):
            complete_conditions(self.groups, self.planned, [wrong])

    def test_failure_does_not_excuse_a_missing_earlier_checkpoint(self):
        record = {"schema": "paper1-numerical-failure-v1", "status": "numerical_failure", "recipe": self.recipes[0],
                  "checkpoints": [{"step": s} for s in (0, 100, 300, 600, 1200)],
                  "failure": {"kind": "nonfinite_gradient", "stage": "before_optimizer_step", "step_range": [1301, 1350],
                              "exception": "gradient failure", "traceback": "trace", "observer_sha256": "b" * 64}}
        self.assertEqual(checkpoint_gaps(record), [2400, 6000])
        bad = copy.deepcopy(record)
        bad["checkpoints"].pop(2)
        with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
            checkpoint_gaps(bad)

    def test_final_renderer_allows_only_an_explicitly_justified_unavailable_cell(self):
        _, unavailable = complete_conditions(self.groups, self.planned, [self.failure])
        with tempfile.TemporaryDirectory() as temporary:
            report = Report({"conditions": [], "unavailable_conditions": unavailable}, Path(temporary))
            self.assertIsNone(report.find("mini", "imu", "ranking", cohort="P"))
            with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
                report.find("mini", "imu", "ranking", cohort="P", step=600)

    def test_primary_and_fixed_three_seed_curves_have_separate_failure_coverage(self):
        recipes = self.recipes + [{**self.recipes[0], "seed": s} for s in (3042, 4042)]
        key = self.keys[6000]
        declaration = registered_conditions({"identity": {"recipes": recipes}})[key]
        planned = {key: declaration}
        curves = trajectory_conditions(planned)
        self.assertEqual(set(curves[key]["recipes_by_seed"]), {42, 1042, 2042})
        for failed_seed in (42, 3042):
            items = [{"recipe": r} for r in recipes if r["seed"] != failed_seed]
            failure = {**self.failure, "recipe": next(r for r in recipes if r["seed"] == failed_seed)}
            complete, unavailable = complete_conditions({key: items}, planned, [failure])
            self.assertEqual(complete, {})
            self.assertEqual(unavailable[0]["missing_seeds"], [failed_seed])
            selected = [i for i in items if i["recipe"]["seed"] in (42, 1042, 2042)]
            estimated, absent = complete_conditions({key: selected}, curves, [failure])
            if failed_seed == 42:
                self.assertEqual(estimated, {})
                self.assertEqual(absent[0]["missing_seeds"], [42])
            else:
                self.assertEqual(len(estimated[key]), 3)
                self.assertEqual(absent, [])
                row = {"condition_id": key, "condition": declaration["condition"], "seeds": [42, 1042, 2042]}
                with tempfile.TemporaryDirectory() as temporary:
                    report = Report({"conditions": [], "unavailable_conditions": unavailable,
                                     "trajectory_conditions": [row]}, Path(temporary))
                    self.assertIsNone(report.find("mini", "imu", "ranking", cohort="P"))
                    self.assertEqual(report.find("mini", "imu", "ranking", cohort="P", trajectory=True), row)

    def test_curve_never_averages_survivors_or_extra_primary_seeds(self):
        with self.assertRaisesRegex(ValueError, "all three"):
            Report.shared_seed_mean({"seeds": [1042, 2042]}, {"per_seed": [0.2, 0.4]})
        value = Report.shared_seed_mean({"seeds": [42, 1042, 2042, 3042, 4042]},
                                        {"per_seed": [0.2, 0.4, 0.6, 1., 1.]})
        self.assertAlmostEqual(value, 0.4)


if __name__ == "__main__":
    unittest.main()
