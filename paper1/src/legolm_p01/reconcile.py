"""Re-score old evaluator versions on immutable checkpoints before publication."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .artifacts import load, checked_path, expected_measurements, measurement_key
from .checkpoint_eval import evaluator_source
from .job_evaluation import obligations
from .provenance import digest, file_hash, write_json
from .outcomes import TERMINAL, terminal_record


def select_catalog(directory, expected, source, backend):
    """Select by declared implementation, never by a score; do not mutate files."""
    directory = Path(directory)
    path = directory / "evaluations.json"
    catalog = load(path) if path.exists() else {}
    retained, retired = {}, {}
    cells = set()
    for key, item in catalog.items():
        cell = measurement_key(item["measurement"], item["settings"])
        if cell not in expected:
            raise ValueError("Unregistered measurement in a reconciliation catalog")
        record = checked_path(directory, item["path"])
        if file_hash(record) != item["sha256"]:
            raise ValueError("Evaluation integrity failure during reconciliation")
        identity = load(record)["identity"]
        if digest(identity) != key or item["evaluation_id"] != key:
            raise ValueError("Evaluation identity failure during reconciliation")
        current = (identity["evaluation_source"] == source and identity["backend"] == backend
                   and identity["sensor_batch"] == 4)
        if current:
            if cell in cells:
                raise ValueError("Repeated current-evaluator measurement for one scientific cell")
            retained[key] = item
            cells.add(cell)
        else:
            retired[key] = item
    return catalog, retained, retired, set(expected) - cells


def run(matrix, asset_root, data_cache, features, text, output, device, linear_dir):
    matrix, output = Path(matrix), Path(output)
    completed = load(matrix / "matrix-complete.json")
    if not completed["passed"] or completed["completed"] != 411 or not completed["evaluated"] or completed["excluded_from_results"]:
        raise ValueError("Final scoring requires the fully evaluated training matrix")
    jobs = [load(p) for p in sorted(matrix.glob("job-*.json"))]
    if len(jobs) != 411 or any(j["status"] not in TERMINAL for j in jobs):
        raise ValueError("Training matrix is incomplete")
    source = evaluator_source()
    version = digest(source)
    history = matrix / "evaluation-history"
    output.mkdir(parents=True, exist_ok=True)
    control = {"stopped": False, "child": None}
    def stop_owned(signum, frame):
        control["stopped"] = True
        if control["child"] is not None and control["child"].poll() is None:
            control["child"].terminate()
    handlers = {s: signal.signal(s, stop_owned) for s in (signal.SIGTERM, signal.SIGINT)}
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    results = []
    try:
        for job in jobs:
            outcome = terminal_record(job)
            for obligation in obligations(outcome):
                if control["stopped"]:
                    raise InterruptedError("Final scoring interrupted")
                if evaluator_source() != source:
                    raise ValueError("Evaluator changed during final scoring")
                name = obligation["checkpoint_sha256"] + "-" + obligation["modality"]
                directory = matrix / "evaluations" / name
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = next(c for c in outcome["checkpoints"] if c["sha256"] == obligation["checkpoint_sha256"])
                expected = expected_measurements(job["recipe"], checkpoint["step"], obligation["modality"])
                original, retained, retired, missing = select_catalog(directory, expected, source, completed["backend"])
                archived = None
                if retired:
                    archived = history / name / (digest({"catalog": original, "selected_evaluator": source}) + ".json")
                    value = {"selection_basis": "Final evaluator source/backend and declared batch; no score-based choice.",
                             "selected_evaluator": source, "retired_evaluation_ids": sorted(retired), "original_catalog": original}
                    if archived.exists() and load(archived) != value:
                        raise ValueError("Historical catalog identity collision")
                    write_json(archived, value)
                    # Old per-item files remain untouched. Only the active index
                    # changes, after its complete old state is durably archived.
                    write_json(directory / "evaluations.json", retained)
                if missing:
                    if file_hash(obligation["checkpoint"]) != obligation["checkpoint_sha256"]:
                        raise ValueError("Checkpoint changed before final scoring")
                    attempt = 1
                    while (directory / f"reconcile-{version[:12]}-{attempt}.log").exists():
                        attempt += 1
                    command = [sys.executable, "-m", "legolm_p01.cli", "evaluate", "--checkpoint", obligation["checkpoint"],
                               "--asset-root", str(asset_root), "--data-cache", str(data_cache), "--features", str(features),
                               "--text", str(text), "--output", str(directory), "--device", device, "--sensor-batch", "4",
                               "--eval-modality", obligation["modality"], "--linear-dir", str(linear_dir), "--suites", *obligation["suites"]]
                    with (directory / f"reconcile-{version[:12]}-{attempt}.log").open("w") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                        control["child"] = child
                        write_json(directory / "reconcile-process.json", {"pid": child.pid, "version": version, "attempt": attempt})
                        if control["stopped"]:
                            child.terminate()
                        code = child.wait()
                        control["child"] = None
                    if code or control["stopped"]:
                        raise RuntimeError("Final checkpoint scoring failed; original records are preserved")
                _, selected, unexpected_old, absent = select_catalog(directory, expected, source, completed["backend"])
                if unexpected_old or absent:
                    raise ValueError("Final scoring did not satisfy all current-evaluator obligations")
                histories = sorted((history / name).glob("*.json"))
                retired_ids = sorted({key for p in histories for key in load(p)["retired_evaluation_ids"]})
                result = {"catalog": name, "evaluation_ids": sorted(selected), "rescored_in_this_pass": bool(missing),
                          "retired_evaluation_ids": retired_ids,
                          "historical_catalogs": [{"name": p.name, "sha256": file_hash(p)} for p in histories]}
                results.append(result)
                write_json(output / "progress.json", {"version": version, "catalogs": len(results),
                                                     "rescored_catalogs": sum(r["rescored_in_this_pass"] for r in results)})
                print(json.dumps({"final_evaluator_catalog": len(results), "rescored": bool(missing), "catalog": name}), flush=True)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    result = {"schema": "paper1-evaluation-version-v2", "passed": True, "matrix_id": completed["matrix_id"],
              "source": source, "source_id": version, "sensor_batch": 4, "backend": completed["backend"],
              "selection_basis": "One declared evaluator source, backend, and batch. Every registered measurement is required; no outcome-based selection.",
              "catalogs": results}
    write_json(output / ("complete-" + version[:12] + ".json"), result)
    write_json(matrix / "evaluation-version.json", result)
    return {"passed": True, "version": version, "catalogs": len(results), "rescored_catalogs": sum(r["rescored_in_this_pass"] for r in results)}
