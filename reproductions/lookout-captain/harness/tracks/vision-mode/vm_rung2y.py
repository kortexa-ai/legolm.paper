"""V-line rung 2y: a person DETECTOR emits the mode.

Registration: tracks/vision-mode/PLAN.md, "Rung 2y registration — a person
detector in the classifier's seat". Rung 2's switch, with YOLO26n
(ultralytics, COCO-pretrained, ~2.6M params, frozen) where rung 2's
MobileNetV3 head sat: the verdict is PRESENT when any `person` box at
confidence >= 0.25 covers >= 5% of the frame — rung 1's PRESENT rule read
off a detector's boxes instead of COCO's masks — else ABSENT. Nothing
trains. The arms, policy, metrics, readings and artifacts are rung 2's
code, rebound to the r2y stem.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
from vm_rung1 import HOSTS, HOST_ORDER, SEEDS, display_path, emit, release_device_memory, select_frames, sha256_of  # noqa: E402

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 2y registration — a person detector in "
    "the classifier's seat (2026-08-23)"
)
CANONICAL_STEM = "vision-mode-r2y"
RUNG_LABEL = "rung 2y"

#: the detector, by number
DETECTOR_NAME = "yolo26n"
#: the size axis (rung 2y-size registration): `--detector yolo26s` writes the
#: `vision-mode-r2y-26s` family; the nano is the canonical `vision-mode-r2y`
DETECTOR_SIZES = ("yolo26n", "yolo26s", "yolo26m")
DETECTOR_PATH = r1.ROOT / "checkpoints" / "experiments" / f"{DETECTOR_NAME}.pt"
DETECTOR_IMAGE = 640
DETECTOR_CONF = 0.25
DETECTOR_BATCH = 32
COCO_PERSON_CLASS = 0  # ultralytics' COCO-80 index for `person`
PRESENT_MIN_BOX_FRACTION = r1.PRESENT_MIN_AREA_FRACTION


def load_detector(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    """ultralytics YOLO from `path`; a missing file downloads the release
    asset of that name into place (the canary on a fresh machine)."""

    from ultralytics import YOLO

    if not path.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        path.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(path))
        if not path.is_file():
            raise SystemExit(f"no detector weights at {display_path(path)} and the download did not land")
    detector = YOLO(str(path))
    return detector, {
        "path": display_path(path), "sha256": sha256_of(path), "trained_here": False,
        "training": {
            "backbone": f"ultralytics {path.stem}, COCO-pretrained detector, frozen",
            "head": "none: a rule over the person boxes", "parameters_trained": 0,
            "image_size": DETECTOR_IMAGE, "confidence": DETECTOR_CONF,
            "note": "COCO val2017 is the detector's own validation split; no frame here was trained on",
        },
        "rule": {
            "present": (f"any `person` box with confidence >= {DETECTOR_CONF} whose box area is >= "
                        f"{PRESENT_MIN_BOX_FRACTION:.0%} of the frame"),
            "absent": "no such box",
            "note": ("rung 1's PRESENT rule on COCO uses the annotation's mask area; a box is never "
                     "smaller than its mask, so the detector's side of the rule is the looser one"),
        },
    }


@torch.no_grad()
def person_boxes_for(detector: Any, frames: Sequence[Mapping[str, Any]],
                     device: torch.device) -> dict[int, list[dict[str, float]]]:
    """image_id -> the frame's person boxes at the registered confidence, each
    as {confidence, fraction, cx, cy, w, h} normalized to the frame."""

    out: dict[int, list[dict[str, float]]] = {}
    for start in range(0, len(frames), DETECTOR_BATCH):
        chunk = frames[start:start + DETECTOR_BATCH]
        results = detector.predict([f["file"] for f in chunk], imgsz=DETECTOR_IMAGE, conf=DETECTOR_CONF,
                                   classes=[COCO_PERSON_CLASS], device=device, verbose=False)
        for frame, result in zip(chunk, results):
            height, width = result.orig_shape
            boxes = []
            for (x1, y1, x2, y2), confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                w, h = (x2 - x1) / width, (y2 - y1) / height
                boxes.append({"confidence": confidence, "fraction": w * h,
                              "cx": (x1 + x2) / (2 * width), "cy": (y1 + y2) / (2 * height), "w": w, "h": h})
            out[int(frame["image_id"])] = boxes
    return out


def verdict_of(boxes: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    qualifying = [b["confidence"] for b in boxes if b["fraction"] >= PRESENT_MIN_BOX_FRACTION]
    return {"verdict": "present" if qualifying else "absent",
            "p_present": round(max(qualifying), 6) if qualifying else 0.0,
            "person_boxes": len(boxes),
            "largest_fraction": round(max(b["fraction"] for b in boxes), 6) if boxes else 0.0}


def verdicts_for(detector: Any, frames: Sequence[Mapping[str, Any]],
                 device: torch.device) -> dict[int, dict[str, Any]]:
    """image_id -> {verdict, p_present, correct, person_boxes, largest_fraction}
    on `frames`; p_present is the best confidence among qualifying boxes."""

    boxes = person_boxes_for(detector, frames, device)
    out: dict[int, dict[str, Any]] = {}
    for frame in frames:
        row = verdict_of(boxes[int(frame["image_id"])])
        row["correct"] = row["verdict"] == frame["label"]
        out[int(frame["image_id"])] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=(*HOSTS, "all"), default="450m")
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--detector", choices=DETECTOR_SIZES, default=DETECTOR_NAME,
                        help="weights at checkpoints/experiments/<name>.pt; non-nano sizes write their own family")
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--all-seeds runs every registered seed; drop --seed")
    seeds = list(SEEDS) if args.all_seeds else [args.seed if args.seed is not None else SEEDS[0]]
    stem = CANONICAL_STEM if args.detector == DETECTOR_NAME else f"{CANONICAL_STEM}-{args.detector.removeprefix('yolo')}"
    if args.output_prefix is None:
        if args.smoke:
            raise SystemExit("--smoke requires a non-canonical --output-prefix")
        base = r1.ROOT / "findings" / stem
    else:
        base = Path(args.output_prefix).expanduser()
        if args.smoke and base.name.startswith(r2.CANONICAL_STEM) and base.parent.resolve() == (r1.ROOT / "findings").resolve():
            raise SystemExit("a smoke may not write the canonical family")

    # rung 2's harness writes rung 2y's artifacts
    r2.REUSES = [r1.REGISTRATION, r2.REGISTRATION]
    r2.CANONICAL_STEM = stem
    r2.REGISTRATION = REGISTRATION
    r2.RUNG_LABEL = RUNG_LABEL

    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else r1.default_coco_dir()
    frames = select_frames(
        coco_dir, per_class_train=r1.TRAIN_PER_CLASS, per_class_eval=r1.EVAL_PER_CLASS,
        rung2_per_class=r2.SMOKE_EVAL_PER_CLASS if args.smoke else r2.EVAL_PER_CLASS,
    )
    emit("vm2y_frames", rung1_train=len(frames["train"]), rung1_eval=len(frames["eval"]),
         rung2_eval=len(frames["rung2"]), calibration=len(frames["classifier_train"]))

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector, record = load_detector(DETECTOR_PATH.with_name(f"{args.detector}.pt"), device)
    started = time.perf_counter()
    verdicts = verdicts_for(detector, frames["rung2"], device)
    record["eval"] = r2.confusion(frames["rung2"], verdicts)
    record["eval"]["wall_seconds"] = round(time.perf_counter() - started, 3)
    record["verdicts"] = {str(k): v for k, v in verdicts.items()}
    # the frames rung 2's head trained on: a larger check of the rule, nothing trains on them here
    calibration = frames["classifier_train"]
    if args.smoke:
        calibration = calibration[:16] + calibration[-16:]
    started = time.perf_counter()
    record["calibration"] = r2.confusion(calibration, verdicts_for(detector, calibration, device))
    record["calibration"]["wall_seconds"] = round(time.perf_counter() - started, 3)
    emit("vm2y_detector", **{k: v for k, v in record.items() if k != "verdicts"})
    del detector
    release_device_memory(device)

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        r2.run_host(args, host_key, seeds, frames, verdicts, record, base)


if __name__ == "__main__":
    main()
