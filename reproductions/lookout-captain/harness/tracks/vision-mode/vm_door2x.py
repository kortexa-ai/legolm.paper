"""V-line alternate-door trainer: stronger and mixed-domain door pairs.

Registration: tracks/vision-mode/PLAN.md, "Stronger-door registration — the
weak seed's doors at 2× steps (3B s42)" (2026-08-23). Rung 1's recipe, rng
and frames exactly, with STEPS doubled (192 → 384); the pair saves under
the `vm-r1x2` stem — the registered doors are not touched. The `mx` variant
uses the registered step count and appends rung 3d's mesh training frames.
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung3 as r3  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, MODES, SEEDS, TRAIN_DOSE, FrameBank, atomic_json_dump, bank_features, cuda_snapshot,
    display_path, emit, git_head, held_out_gate, layer_plan, load_host, parameter_samples,
    release_device_memory, samples_unchanged, seed_everything, select_frames, sha256_of,
    train_door, train_tap_stack, zero_dose_gate,
)

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Stronger-door registration — the weak seed's "
    "doors at 2× steps (3B s42) (2026-08-23)"
)
X2_STEM = "vm-r1x2"
X2_STEPS = 2 * r1.STEPS
MX_STEM = "vm-r1mx"
MX2_STEM = "vm-r1mx2"
MX_DRAW_SEED = 2028
MX_PER_CLASS = 40
MX_START_INDEX = r3.EVAL_PER_CLASS
MX_ESCAPE_STEPS = 8
MX_REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 3d registration — mixed-domain doors: "
    "renders in the training mix (2026-08-24)"
)
R3G_REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 3g registration — the seed-42 autopsy, "
    "the neutral habit door, and the small host's step count (2026-08-27)"
)


def mixed_mesh_frames(frames_dir: Path, *, regenerate: bool) -> tuple[dict, list[dict]]:
    mesh_dir = frames_dir.expanduser() / "mesh"
    manifest = r3.load_or_generate_frames(
        mesh_dir, smoke=False, regenerate=regenerate, figure="mesh",
        per_class=MX_PER_CLASS, draw_seed=MX_DRAW_SEED,
        start_index=MX_START_INDEX, registration=MX_REGISTRATION,
        mesh_escape_steps=MX_ESCAPE_STEPS,
    )
    frames = r3.level_frames(manifest, "mesh")
    if len(frames) != 2 * MX_PER_CLASS:
        raise RuntimeError("mixed-domain mesh draw did not produce 40 present + 40 absent frames")
    return manifest, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(HOSTS), default="3b")
    parser.add_argument("--seed", type=int, default=42, choices=SEEDS)
    parser.add_argument("--variant", choices=("x2", "mx", "mx2"), default="x2")
    parser.add_argument("--order-seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--frames-dir", default="/tmp/vm-r3-trainframes")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.order_seed is not None and args.variant != "mx2":
        raise SystemExit("--order-seed requires --variant mx2")
    if args.order_seed == args.seed:
        raise SystemExit("--order-seed must differ from --seed")

    mesh_manifest = None
    mesh_frames: list[dict] = []
    if args.variant in ("mx", "mx2"):
        mesh_manifest, mesh_frames = mixed_mesh_frames(
            Path(args.frames_dir), regenerate=args.regenerate)
    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else r1.default_coco_dir()
    per_class_train = r1.SMOKE_FRAMES if args.smoke else r1.TRAIN_PER_CLASS
    per_class_eval = r1.SMOKE_FRAMES if args.smoke else r1.EVAL_PER_CLASS
    frames = select_frames(coco_dir, per_class_train=per_class_train, per_class_eval=per_class_eval)
    if mesh_manifest is not None:
        frames["train"] = frames["train"] + mesh_frames
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(
        args.host, device, local_files_only=args.local_files_only
    )
    tokenizer = processor.tokenizer
    plan = layer_plan(depth)
    stem = (f"{MX2_STEM}o{args.order_seed}" if args.order_seed is not None else
            {"x2": X2_STEM, "mx": MX_STEM, "mx2": MX2_STEM}[args.variant])
    steps = r1.SMOKE_STEPS if args.smoke else (r1.STEPS if args.variant == "mx" else X2_STEPS)
    event = {"x2": "vm_x2", "mx": "vm_mx", "mx2": "vm_mx2"}[args.variant]
    emit(f"{event}_start", host=args.host, seed=args.seed, steps=steps, plan=plan,
         **cuda_snapshot(device))

    started = time.perf_counter()
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    seed_everything(args.seed, device)
    features = bank_features(model, tokenizer, device, plan["source_layer"])
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    train_taps = train_tap_stack(features)
    constant_feature = train_taps.mean(0)
    rng = random.Random(args.order_seed if args.order_seed is not None else args.seed)
    train_frames = frames["train"]
    mode_bridge, mode_history = train_door(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=train_frames, features=features, constant_feature=constant_feature,
        hidden=hidden, inject_layers=plan["inject_layers"], dose=TRAIN_DOSE, steps=steps,
        rng=rng, mode_conditioned=True, train_taps=train_taps,
    )
    constant_bridge, constant_history = train_door(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=train_frames, features=features, constant_feature=constant_feature,
        hidden=hidden, inject_layers=plan["inject_layers"], dose=TRAIN_DOSE, steps=steps,
        rng=rng, mode_conditioned=False, train_taps=train_taps,
    )
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved while the doors trained")
    emit(f"{event}_trained", host=args.host, seed=args.seed,
         mode_final=mode_history[-1], constant_final=constant_history[-1])

    gate_frames = [row for row in train_frames if row["label"] == "present"][:4] + \
                  [row for row in train_frames if row["label"] == "absent"][:4]
    held_out = held_out_gate(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=gate_frames, features=features, bridge=mode_bridge,
        inject_layers=plan["inject_layers"], dose=TRAIN_DOSE,
    )
    zero_dose = zero_dose_gate(
        model=model, tower=tower, bank=bank, frame=gate_frames[0],
        bridge=mode_bridge, feature=features[(MODES[0], r1.HELD_OUT[0])],
        inject_layers=plan["inject_layers"], device=device,
    )
    emit(f"{event}_gates", host=args.host, seed=args.seed, held_out=held_out["pooled"],
         zero_dose_exact=zero_dose["exact"])
    if not held_out["passed"] or not zero_dose["passed"]:
        raise SystemExit(f"the {args.variant} pair failed a rung 1 gate; nothing saved")

    doors = {}
    if args.smoke:
        doors = {door: {"path": None, "sha256": None, "note": "SMOKE: not saved"}
                 for door in ("mode", "constant")}
    else:
        for door, bridge, history in (("mode", mode_bridge, mode_history),
                                      ("constant", constant_bridge, constant_history)):
            path = r1.ROOT / "checkpoints" / "experiments" / \
                f"{stem}-{door}-bridge-{args.host}-s{args.seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "legolm-vision-mode-r1-door-v1", "door": door,
                "host": HOSTS[args.host]["model_id"], "seed": args.seed,
                "source_layer": plan["source_layer"], "inject_layers": list(plan["inject_layers"]),
                "train_dose": TRAIN_DOSE, "bus_dim": r1.BUS_DIM,
                "delta_positions": r1.DELTA_POSITIONS, "injection": r1.INJECTION,
                "hidden_size": hidden, "depth": depth,
                "state_dict": bridge.state_dict(), "history": history,
                "note": (f"batch-order variant: mx2 recipe, init s{args.seed}, "
                         f"batch stream s{args.order_seed}" if args.order_seed is not None else
                         (f"stronger-door variant: rung 1's recipe at {steps} steps"
                          if args.variant == "x2" else
                          "mixed-domain variant: rung 1 COCO frames plus rung 3d mesh frames")),
            }, path)
            doors[door] = {"path": display_path(path), "sha256": sha256_of(path)}
    payload = {
        "track": r1.TRACK,
        "experiment": (f"vision-mode-mx2o{args.order_seed}door-{args.host}-s{args.seed}"
                       if args.order_seed is not None else
                       f"vision-mode-{args.variant}door-{args.host}-s{args.seed}"),
        "registration": (R3G_REGISTRATION if args.order_seed is not None else
                         (REGISTRATION if args.variant == "x2" else MX_REGISTRATION)),
        "reuses": r1.REGISTRATION,
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "host": {"key": args.host, **HOSTS[args.host], "depth": depth, "hidden_size": hidden},
        "seed": args.seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "steps": steps, "registered_steps": r1.STEPS, "doors": doors,
        "gates": {"held_out": held_out, "zero_dose": zero_dose},
        "training": {"mode_final": mode_history[-1], "constant_final": constant_history[-1]},
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    if args.order_seed is not None:
        payload["order_seed"] = args.order_seed
    if mesh_manifest is not None:
        payload["training_frames"] = {
            "coco": len(frames["train"]) - len(mesh_frames),
            "mesh": len(mesh_frames), "total": len(frames["train"]),
            "mesh_manifest": display_path(Path(args.frames_dir).expanduser() /
                                          "mesh" / "manifest.json"),
            "mesh_manifest_sha256": mesh_manifest["sha256"],
        }
    finding_variant = (f"mx2o{args.order_seed}" if args.order_seed is not None else args.variant)
    base = (Path(f"/tmp/vm-smoke/vm-{finding_variant}door") if args.smoke
            else r1.ROOT / "findings" / f"vision-mode-{finding_variant}door")
    prefix = base.with_name(f"{base.name}-{args.host}-s{args.seed}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, prefix.with_suffix(".json"))
    emit(f"{event}_written", json=display_path(prefix.with_suffix(".json")))
    del model, tower, bank
    release_device_memory(device)


if __name__ == "__main__":
    main()
