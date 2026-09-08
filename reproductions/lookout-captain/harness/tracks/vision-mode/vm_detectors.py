"""V-line rung 2y-size, part 1: the detector size axis, no host.

Registration: tracks/vision-mode/PLAN.md, "Rung 2y-size registration — the
detector size axis, for the last five misses (2026-08-23)". yolo26n / s / m
under the registered rule on rung 2's 200 eval frames and the 3,472
calibration frames; per size the confusion, the wall time per frame, and
the verdict on each of the nano's error frames. The size with the fewest
misses on the 3,472 (ties to the smaller) is the one rung 2y reruns with.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
import vm_rung2y as r2y  # noqa: E402
from vm_rung1 import atomic_json_dump, display_path, emit, git_head, release_device_memory, select_frames  # noqa: E402

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 2y-size registration — the detector size "
    "axis, for the last five misses (2026-08-23)"
)
CANONICAL_STEM = "vision-mode-r2y-detectors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    base = Path(args.output_prefix).expanduser() if args.output_prefix else r1.ROOT / "findings" / CANONICAL_STEM
    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else r1.default_coco_dir()
    frames = select_frames(coco_dir, per_class_train=r1.TRAIN_PER_CLASS, per_class_eval=r1.EVAL_PER_CLASS,
                           rung2_per_class=r2.EVAL_PER_CLASS)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    sets = {"eval": frames["rung2"], "calibration": frames["classifier_train"]}
    label_of = {int(f["image_id"]): f["label"] for rows in sets.values() for f in rows}
    sizes: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for name in r2y.DETECTOR_SIZES:
        detector, record = r2y.load_detector(r2y.DETECTOR_PATH.with_name(f"{name}.pt"), device)
        verdicts[name] = {}
        for set_name, rows in sets.items():
            started = time.perf_counter()
            verdicts[name][set_name] = r2y.verdicts_for(detector, rows, device)
            wall = time.perf_counter() - started
            record[set_name] = r2.confusion(rows, verdicts[name][set_name])
            record[set_name]["wall_seconds"] = round(wall, 3)
            record[set_name]["ms_per_frame"] = round(1000 * wall / len(rows), 3)
        record["parameters"] = int(sum(p.numel() for p in detector.model.parameters()))
        sizes[name] = record
        emit("vm_detector", name=name, parameters=record["parameters"],
             eval=record["eval"], calibration=record["calibration"])
        del detector
        release_device_memory(device)
    nano_errors = sorted(image_id for image_id, v in verdicts[r2y.DETECTOR_NAME]["eval"].items() if not v["correct"])
    error_frames = {str(image_id): {"label": label_of[image_id],
                                    **{name: verdicts[name]["eval"][image_id] for name in r2y.DETECTOR_SIZES}}
                    for image_id in nano_errors}
    by_misses = sorted(r2y.DETECTOR_SIZES, key=lambda n: (sizes[n]["calibration"]["misses"], r2y.DETECTOR_SIZES.index(n)))
    chosen = by_misses[0]
    eval_misses = {name: sizes[name]["eval"]["misses"] for name in r2y.DETECTOR_SIZES}
    if min(eval_misses[n] for n in ("yolo26s", "yolo26m")) <= 2:
        reading = "DETECTOR FLOOR"
    elif eval_misses["yolo26m"] >= 4:
        reading = "FRAME FLOOR"
    else:
        reading = "BETWEEN"
    payload = {
        "track": r1.TRACK, "experiment": CANONICAL_STEM, "registration": REGISTRATION,
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "rule": sizes[r2y.DETECTOR_NAME]["rule"], "frames": {k: len(v) for k, v in sets.items()},
        "sizes": sizes, "nano_error_frames": error_frames,
        "chosen": {"name": chosen, "rule": "fewest misses on the calibration frames, ties to the smaller",
                   "rerun_rung_2y": chosen != r2y.DETECTOR_NAME},
        "reading": {"reading": reading, "eval_misses": eval_misses,
                    "bands": {"detector_floor_misses_max": 2, "frame_floor_misses_min": 4}},
    }
    atomic_json_dump(payload, base.with_suffix(".json"))
    lines = ["# V-line rung 2y-size — the detector size axis", "", f"Registration: {REGISTRATION}.", "",
             f"Reading on the 200: **{reading}**; chosen for rung 2y by calibration misses: **{chosen}**"
             f"{'' if chosen != r2y.DETECTOR_NAME else ' (the nano — rung 2y is not rerun)'}.", "",
             "| detector | params | eval acc | eval FA | eval misses | calib acc | calib FA | calib misses | ms/frame (calib) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in r2y.DETECTOR_SIZES:
        e, c = sizes[name]["eval"], sizes[name]["calibration"]
        lines.append(f"| {name} | {sizes[name]['parameters'] / 1e6:.1f}M | {e['accuracy']:.3f} | {e['false_alarms']} | {e['misses']} | "
                     f"{c['accuracy']:.3f} | {c['false_alarms']} | {c['misses']} | {c['ms_per_frame']:.1f} |")
    lines += ["", "The nano's error frames on the 200, verdict / best qualifying confidence / largest box fraction per size:", "",
              "| image | label | " + " | ".join(r2y.DETECTOR_SIZES) + " |", "|---|---|" + "---|" * len(r2y.DETECTOR_SIZES)]
    for image_id, row in error_frames.items():
        cells = [f"{row[n]['verdict']} / {row[n]['p_present']:.2f} / {row[n]['largest_fraction']:.3f}" for n in r2y.DETECTOR_SIZES]
        lines.append(f"| {image_id} | {row['label']} | " + " | ".join(cells) + " |")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n")
    emit("vm_detectors_written", json=display_path(base.with_suffix(".json")), reading=reading, chosen=chosen)


if __name__ == "__main__":
    main()
