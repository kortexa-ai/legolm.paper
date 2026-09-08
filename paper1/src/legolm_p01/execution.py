"""Observe numerical failures without altering the frozen training operations.

The registered train.py remains byte-identical. Only errors raised inside its
optimizer operation are classified; CUDA, memory, I/O and unknown errors still
stop the scheduler. Successful calls perform the original operation once.
"""
from dataclasses import asdict
import json
from pathlib import Path
import traceback

from . import train
from .outcomes import validate_failure
from .provenance import file_hash, write_json


def numerical_kind(error):
    if isinstance(error, FloatingPointError) and str(error) == "Nonfinite training loss":
        return "nonfinite_loss"
    if isinstance(error, RuntimeError) and str(error).startswith("The total norm of order 2.0 for gradients from `parameters` is non-finite"):
        return "nonfinite_gradient"
    return None


def run(recipe, asset_root, data_cache, features, text, output, device, *, stop_after=None):
    output = Path(output)
    for receipt in output.glob("*/numerical-failure.json"):
        prior = json.loads(receipt.read_text())
        if prior["recipe"] == asdict(recipe):
            raise ValueError("A recorded numerical failure is terminal; do not retry seeds until one succeeds")
    original_batch, original_operation = train.make_batch, train.optimize_step
    observed = {}

    def batch(actual_recipe, step, *args):
        observed["step"] = step + 1
        return original_batch(actual_recipe, step, *args)

    def operation(host, adaptation, params, optimizer, actual_recipe, inputs):
        try:
            return original_operation(host, adaptation, params, optimizer, actual_recipe, inputs)
        except Exception as error:
            kind = numerical_kind(error)
            if kind is not None:
                observed.update(kind=kind, exception=str(error), traceback=traceback.format_exc(),
                                trainable_parameters=sum(p.numel() for p in params))
            raise

    train.make_batch, train.optimize_step = batch, operation
    try:
        return train.run(recipe, asset_root, data_cache, features, text, output, device, stop_after=stop_after)
    except Exception as error:
        if numerical_kind(error) is None or "kind" not in observed:
            raise
        matches = []
        for path in output.glob("*/config.json"):
            config = json.loads(path.read_text())
            if config["identity"]["recipe"] == asdict(recipe):
                matches.append((path.parent, config))
        if len(matches) != 1:
            raise ValueError("Cannot uniquely attribute numerical failure to its training identity") from error
        directory, config = matches[0]
        path = directory / "checkpoints.json"
        checkpoints = json.loads(path.read_text()) if path.exists() else []
        for point in checkpoints:
            if file_hash(directory / point["path"]) != point["sha256"]:
                raise ValueError("Checkpoint changed while recording a numerical failure") from error
        result = {"schema": "paper1-numerical-failure-v1", "status": "numerical_failure",
                  "training_id": config["training_id"], "identity": config["identity"],
                  "recipe": asdict(recipe), "directory": str(directory), "checkpoints": checkpoints,
                  "trainable_parameters": observed["trainable_parameters"],
                  "failure": {"kind": observed["kind"], "stage": "before_optimizer_step",
                              "step_range": [observed["step"], observed["step"]],
                              "step_basis": "Observed call to the unchanged optimizer operation",
                              "exception": observed["exception"], "traceback": observed["traceback"],
                              "observer_sha256": file_hash(__file__)}}
        validate_failure(result)
        receipt = directory / "numerical-failure.json"
        if receipt.exists():
            raise ValueError("Refusing to replace a previously recorded numerical failure") from error
        write_json(receipt, result)
        return result
    finally:
        train.make_batch, train.optimize_step = original_batch, original_operation
