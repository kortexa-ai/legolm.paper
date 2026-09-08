"""Terminal training outcomes, distinct from interrupted or unattempted jobs."""
import json
from pathlib import Path

from .provenance import file_hash

TERMINAL = ("complete", "numerical_failure")


def validate_failure(record):
    if record.get("schema") != "paper1-numerical-failure-v1" or record.get("status") != "numerical_failure":
        raise ValueError("Missing explicit numerical-failure record")
    detail = record["failure"]
    if detail["kind"] not in ("nonfinite_loss", "nonfinite_gradient") or detail["stage"] != "before_optimizer_step":
        raise ValueError("An infrastructure or unknown error is not a numerical outcome")
    first, last = detail["step_range"]
    if not isinstance(first, int) or not isinstance(last, int) or not 1 <= first <= last <= record["recipe"]["steps"]:
        raise ValueError("Invalid numerical-failure step range")
    if not detail["exception"] or not detail["traceback"] or len(detail["observer_sha256"]) != 64:
        raise ValueError("Numerical failure lacks execution evidence")
    if any(p["step"] >= first for p in record["checkpoints"]):
        raise ValueError("A failed or later update cannot supply a scientific checkpoint")
    return first


def terminal_record(job):
    """Verify the immutable result receipt before using its saved checkpoints."""
    if job["status"] == "complete":
        record, filename, checksum = job["completed"], "complete.json", job["complete_sha256"]
    elif job["status"] == "numerical_failure":
        record, filename, checksum = job["outcome"], "numerical-failure.json", job["outcome_sha256"]
        validate_failure(record)
    else:
        raise ValueError("Training is interrupted or unattempted, not a terminal outcome")
    path = Path(record["directory"]) / filename
    if file_hash(path) != checksum or json.loads(path.read_text()) != record:
        raise ValueError("Terminal training receipt changed")
    return record
