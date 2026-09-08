from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from legolm_p01 import execution, train
from legolm_p01.outcomes import terminal_record, validate_failure
from legolm_p01.provenance import file_hash, write_json


class TrainingOutcomes(unittest.TestCase):
    def test_nonfinite_gradient_is_terminal_before_optimizer_update(self):
        recipe = train.Recipe(host="mini", cohort="P", modality="imu", kind="static", steps=6000, batch_size=4)
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        host = SimpleNamespace(device="cpu", inject=lambda: None,
                               losses=lambda x, y, prefix: parameter.sqrt().reshape(1, 1))
        inputs = (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1), None)
        original_operation = train.optimize_step
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "registered"
            write_json(directory / "config.json", {"training_id": "a" * 64, "identity": {"recipe": asdict(recipe)}})
            def run(*args, **kwargs):
                batch = train.make_batch(recipe, 17, None, None, host)
                train.optimize_step(host, None, [parameter], optimizer, recipe, batch)
                self.fail("An infinite gradient must not reach optimizer.step")
            with patch.object(train, "run", side_effect=run), patch.object(train, "make_batch", return_value=inputs):
                result = execution.run(recipe, root, root, root, root, root, "cpu")
            self.assertEqual(result["failure"]["kind"], "nonfinite_gradient")
            self.assertEqual(result["failure"]["step_range"], [18, 18])
            self.assertEqual(parameter.item(), 0.)
            self.assertFalse(optimizer.state)
            self.assertIs(train.optimize_step, original_operation)
            receipt = directory / "numerical-failure.json"
            job = {"status": "numerical_failure", "outcome": result, "outcome_sha256": file_hash(receipt)}
            self.assertEqual(terminal_record(job), result)
            with self.assertRaisesRegex(ValueError, "terminal"):
                execution.run(recipe, root, root, root, root, root, "cpu")
            changed = json.loads(receipt.read_text())
            changed["failure"]["step_range"] = [19, 19]
            write_json(receipt, changed)
            with self.assertRaisesRegex(ValueError, "changed"):
                terminal_record(job)

    def test_supervision_preserves_successful_optimizer_operations(self):
        recipe = train.Recipe(host="mini", cohort="P", modality="imu", kind="static", steps=6000, batch_size=4)
        inputs = (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1), None)
        operations = []
        def run(*args, **kwargs):
            parameter = torch.nn.Parameter(torch.tensor(2.))
            optimizer = torch.optim.AdamW([parameter], lr=.001)
            host = SimpleNamespace(device="cpu", inject=lambda: None,
                                   losses=lambda x, y, prefix: parameter.square().reshape(1, 1))
            metrics = []
            for step in range(3):
                batch = train.make_batch(recipe, step, None, None, host)
                metrics.append(train.optimize_step(host, None, [parameter], optimizer, recipe, batch))
            operations.append(optimizer.state[parameter]["step"].item())
            return {"parameter": parameter.item(), "metrics": metrics,
                    "moment": optimizer.state[parameter]["exp_avg"].item()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(train, "run", side_effect=run), patch.object(train, "make_batch", return_value=inputs):
                ordinary = train.run()
                supervised = execution.run(recipe, root, root, root, root, root, "cpu")
        self.assertEqual(ordinary, supervised)
        self.assertEqual(operations, [3, 3])

    def test_infrastructure_and_unknown_errors_are_not_scientific_failures(self):
        for error in (RuntimeError("CUDA error: launch timeout"), MemoryError("out of memory"),
                      OSError("disk full"), ValueError("Nonfinite training loss"), RuntimeError("unexpected")):
            self.assertIsNone(execution.numerical_kind(error))
        self.assertEqual(execution.numerical_kind(FloatingPointError("Nonfinite training loss")), "nonfinite_loss")
        failure = {"schema": "paper1-numerical-failure-v1", "status": "numerical_failure", "recipe": {"steps": 6000},
                   "checkpoints": [{"step": 600}], "failure": {"kind": "nonfinite_gradient", "stage": "before_optimizer_step",
                   "step_range": [600, 600], "exception": "gradient", "traceback": "trace", "observer_sha256": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "later update"):
            validate_failure(failure)


if __name__ == "__main__":
    unittest.main()
