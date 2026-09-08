"""V-line rung 3/3b: the rendered world and photoscan person.

Registration: tracks/vision-mode/PLAN.md, "Rung 3b registration — the
photoscan person: L2 realism as a textured human mesh, plus the capsule
host-probe (2026-08-24)". The default capsule path retains rung 3's two
registered realism levels. The mesh path replaces only the figure with the
pinned HSR0081 photoscan. Nothing trains anywhere in this harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
import vm_rung2y as r2y  # noqa: E402
from vm_rung1 import atomic_json_dump, display_path, emit, sha256_of  # noqa: E402

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 3b registration — the photoscan person: "
    "L2 realism as a textured human mesh, plus the capsule host-probe (2026-08-24)"
)
CANONICAL_DETECTOR_STEM = "vision-mode-r3-detector"
CANONICAL_STEM = "vision-mode-r3"
CANONICAL_R3B_DETECTOR_STEM = "vision-mode-r3b-detector"
CANONICAL_R3B_STEM = "vision-mode-r3b"
CANONICAL_PROBE_STEM = "vision-mode-r3-hostprobe-3b"
HUMANOID_PATH = HERE / "assets" / "humanoid.xml"
MESH_URL = (
    "https://huggingface.co/datasets/digitalrealitylab/HSRD-100/resolve/main/data/"
    "HSR0081/HSR0081-Body-004/scans/HSR0081-Body-004-Scan-LOD2.zip?download=true"
)
MESH_ZIP_SHA256 = "71afed0be980a1c9343f7ea034256f824a5a1efe6998539e98ae7922c9dca67e"
MESH_ROOT = r1.ROOT / "checkpoints" / "experiments" / "hsr0081"
MESH_ARCHIVE = MESH_ROOT / "HSR0081-Body-004-Scan-LOD2.zip"
MESH_OBJ_NAME = "HSR0081-Body-004_LOD2.obj"
MESH_TEXTURE_NAME = "HSR0081-Body-004_LOD2_u0_v0_diffuse.png"
FRAME_DRAW_SEED = 2027
WIDTH, HEIGHT = 640, 480
EVAL_PER_CLASS = 100
SMOKE_PER_CLASS = 4
MIN_MASK_FRACTION = 0.05
LEVELS = (0, 1)
DETECTORS = ("yolo26n", "yolo26s", "yolo26m")
OPERATING_DETECTOR = "yolo26s"
DETECTOR_TRANSFER = 0.90
PART2_GATE = 0.75
MAX_REDRAWS = 100
PROBE_PER_CLASS = 30
PROBE_HOST = "3b"

L1_RGBA = {
    "skin": [0.76, 0.52, 0.38, 1.0],
    "shirt": [0.12, 0.32, 0.68, 1.0],
    "pants": [0.10, 0.12, 0.18, 1.0],
    "shoes": [0.05, 0.05, 0.06, 1.0],
}
SCENE_RANGES = {
    "obstacles": {"count": [2, 4], "half_size_xy": [0.15, 0.38],
                  "half_height": [0.18, 0.62], "center_xy": [-1.65, 1.65]},
    "camera": {"azimuth_degrees": [-51.0, -39.0], "radius": [2.05, 2.35],
               "height": [1.55, 1.80], "target_xy": [-0.08, 0.08]},
    "light": {"x": [-1.2, 1.2], "y": [-1.2, 1.2], "z": [3.0, 4.2]},
    "humanoid": {"x": [-0.22, 0.22], "y": [-0.22, 0.22],
                 "yaw_degrees": [-180.0, 180.0], "joint_sigma_radians": 0.035},
}


# --------------------------------------------------------------------------
# part A: paired rendered frames
# --------------------------------------------------------------------------

def _f(values: Sequence[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _look_axes(position: Sequence[float], target: Sequence[float]) -> list[float]:
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def draw_scene(rng: random.Random, index: int) -> dict[str, Any]:
    obstacles = []
    for obstacle in range(rng.randint(2, 4)):
        while True:
            x, y = rng.uniform(-1.65, 1.65), rng.uniform(-1.65, 1.65)
            if math.hypot(x, y) >= 0.72:
                break
        sx, sy, sz = rng.uniform(0.15, 0.38), rng.uniform(0.15, 0.38), rng.uniform(0.18, 0.62)
        obstacles.append({"name": f"obstacle_{obstacle}", "pos": [x, y, sz],
                          "size": [sx, sy, sz],
                          "rgba": [rng.uniform(0.15, 0.9) for _ in range(3)] + [1.0]})
    azimuth = math.radians(rng.uniform(-51.0, -39.0))
    radius = rng.uniform(2.05, 2.35)
    target = [rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 0.88]
    camera = [target[0] + radius * math.cos(azimuth),
              target[1] + radius * math.sin(azimuth), rng.uniform(1.55, 1.80)]
    return {"index": index, "obstacles": obstacles,
            "camera": {"pos": camera, "target": target,
                       "yaw_degrees": math.degrees(azimuth), "fovy": 50.0},
            "light": {"pos": [rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2),
                                rng.uniform(3.0, 4.2)]}}


def room_xml(draw: Mapping[str, Any]) -> str:
    obstacle_xml = "".join(
        f'<geom name="{row["name"]}" type="box" pos="{_f(row["pos"])}" '
        f'size="{_f(row["size"])}" rgba="{_f(row["rgba"])}"/>'
        for row in draw["obstacles"]
    )
    camera = draw["camera"]
    return f"""<mujoco model="vm_rung3_room">
  <option timestep="0.005"/>
  <visual><map force="0.1" zfar="30"/><rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/></visual>
  <statistic center="0 0 0.7"/>
  <asset><texture name="room_grid" type="2d" builtin="checker" width="512" height="512"
      rgb1=".48 .50 .52" rgb2=".62 .64 .66"/>
    <material name="room_grid" texture="room_grid" texrepeat="8 8" texuniform="true" reflectance=".08"/></asset>
  <worldbody>
    <geom name="room_floor" type="plane" size="2.4 2.4 .05" material="room_grid"/>
    <geom name="room_back" type="box" pos="0 2.35 1.2" size="2.4 .05 1.2" rgba=".72 .75 .78 1"/>
    <geom name="room_side" type="box" pos="-2.35 0 1.2" size=".05 2.4 1.2" rgba=".67 .70 .74 1"/>
    {obstacle_xml}
    <light name="room_light" pos="{_f(draw['light']['pos'])}" diffuse=".9 .88 .84" specular=".2 .2 .2"/>
    <camera name="room_camera" pos="{_f(camera['pos'])}" xyaxes="{_f(_look_axes(camera['pos'], camera['target']))}" fovy="50"/>
  </worldbody>
</mujoco>"""


def acquire_mesh_asset() -> dict[str, Any]:
    """Download, verify, and extract the registered HSR0081 archive."""

    MESH_ROOT.mkdir(parents=True, exist_ok=True)
    if MESH_ARCHIVE.is_file():
        digest = sha256_of(MESH_ARCHIVE)
        if digest != MESH_ZIP_SHA256:
            raise SystemExit(
                f"HSR0081 archive sha256 mismatch at {display_path(MESH_ARCHIVE)}: "
                f"got {digest}, expected {MESH_ZIP_SHA256}; remove the bad archive and retry"
            )
    else:
        partial = MESH_ARCHIVE.with_suffix(MESH_ARCHIVE.suffix + ".part")
        try:
            with urllib.request.urlopen(MESH_URL) as source, partial.open("wb") as destination:
                while chunk := source.read(1 << 20):
                    destination.write(chunk)
            digest = sha256_of(partial)
            if digest != MESH_ZIP_SHA256:
                partial.unlink(missing_ok=True)
                raise SystemExit(
                    f"downloaded HSR0081 archive sha256 mismatch: got {digest}, "
                    f"expected {MESH_ZIP_SHA256}"
                )
            partial.replace(MESH_ARCHIVE)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    obj_path = MESH_ROOT / MESH_OBJ_NAME
    texture_path = MESH_ROOT / MESH_TEXTURE_NAME
    if not obj_path.is_file() or not texture_path.is_file():
        root = MESH_ROOT.resolve()
        with zipfile.ZipFile(MESH_ARCHIVE) as archive:
            for member in archive.infolist():
                target = (MESH_ROOT / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise SystemExit(f"unsafe path in HSR0081 archive: {member.filename}")
            archive.extractall(MESH_ROOT)
    if not obj_path.is_file() or not texture_path.is_file():
        raise SystemExit(f"HSR0081 archive did not contain {MESH_OBJ_NAME} and {MESH_TEXTURE_NAME}")
    return {"url": MESH_URL, "archive": display_path(MESH_ARCHIVE),
            "archive_sha256": MESH_ZIP_SHA256, "obj": str(obj_path.resolve()),
            "texture": str(texture_path.resolve())}


def mesh_room_xml(draw: Mapping[str, Any], asset: Mapping[str, Any], *, present: bool) -> str:
    xml = room_xml(draw)
    if not present:
        return xml
    mesh_assets = (
        f'<mesh name="person_mesh" file="{asset["obj"]}"/>'
        f'<texture name="person_texture" type="2d" file="{asset["texture"]}"/>'
        '<material name="person_material" texture="person_texture" specular="0" shininess="0"/>'
    )
    person = (
        '<body name="mesh_person_body" pos="0 0 .002">'
        '<freejoint name="mesh_person_root"/>'
        '<geom name="mesh_person" type="mesh" mesh="person_mesh" material="person_material"/>'
        '</body>'
    )
    return xml.replace("<asset>", f"<asset>{mesh_assets}", 1).replace(
        "  </worldbody>", f"    {person}\n  </worldbody>", 1)


def compile_scene(draw: Mapping[str, Any], *, present: bool):
    import mujoco

    scene = mujoco.MjSpec.from_string(room_xml(draw))
    if present:
        humanoid = mujoco.MjSpec.from_file(str(HUMANOID_PATH))
        # The asset stays byte-for-byte untouched. Its demo floor and lights are
        # removed from this in-memory copy before its articulated body is attached.
        for element in list(humanoid.worldbody.geoms) + list(humanoid.worldbody.lights):
            humanoid.delete(element)
        mount = scene.worldbody.add_frame(name="human_mount")
        scene.attach(humanoid, prefix="human_", frame=mount)
    return scene.compile()


def compile_mesh_scene(draw: Mapping[str, Any], asset: Mapping[str, Any], *, present: bool):
    import mujoco

    return mujoco.MjSpec.from_string(mesh_room_xml(draw, asset, present=present)).compile()


def pose_humanoid(model, data, rng: random.Random, *, toward_camera: float = 0.0,
                  camera_pos=None) -> dict[str, Any]:
    """`toward_camera` shifts the placement draw's centre toward the camera —
    the deterministic escape hatch for draws whose geometry makes the 5% mask
    unreachable by pose jitter alone (episode 21 on the smarty EGL raster)."""

    import mujoco

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "human_stand_on_left_leg")
    if key < 0:
        raise RuntimeError("the attached humanoid has no stand keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    cx = cy = 0.0
    if toward_camera and camera_pos is not None:
        norm = math.hypot(camera_pos[0], camera_pos[1]) or 1.0
        cx, cy = toward_camera * camera_pos[0] / norm, toward_camera * camera_pos[1] / norm
    x, y, yaw = cx + rng.uniform(-0.22, 0.22), cy + rng.uniform(-0.22, 0.22), rng.uniform(-math.pi, math.pi)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "human_root")
    address = int(model.jnt_qposadr[root])
    data.qpos[address:address + 2] = [x, y]
    data.qpos[address + 3:address + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    sigma = SCENE_RANGES["humanoid"]["joint_sigma_radians"]
    for joint in range(model.njnt):
        if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        qadr = int(model.jnt_qposadr[joint])
        value = float(data.qpos[qadr]) + rng.gauss(0.0, sigma)
        if model.jnt_limited[joint]:
            value = min(max(value, float(model.jnt_range[joint, 0])), float(model.jnt_range[joint, 1]))
        data.qpos[qadr] = value
    mujoco.mj_forward(model, data)
    return {"pos_xy": [x, y], "yaw_degrees": math.degrees(yaw), "keyframe": "stand_on_left_leg"}


def humanoid_geom_ids(model) -> list[int]:
    return [index for index in range(model.ngeom) if model.geom(index).name.startswith("human_")]


def recolor_l1(model) -> None:
    for geom_id in humanoid_geom_ids(model):
        name = model.geom(geom_id).name.removeprefix("human_")
        if name == "head" or "hand" in name:
            group = "skin"
        elif name.startswith(("torso", "waist", "upper_arm", "lower_arm")):
            group = "shirt"
        elif name.startswith(("butt", "thigh", "shin")):
            group = "pants"
        else:
            group = "shoes"
        model.geom_rgba[geom_id] = L1_RGBA[group]


def set_mesh_pose(model, data, *, x: float, y: float, yaw_degrees: float) -> None:
    import mujoco

    yaw = math.radians(yaw_degrees)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mesh_person_root")
    address = int(model.jnt_qposadr[root])
    data.qpos[address:address + 2] = [x, y]
    data.qpos[address + 3:address + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def render_draw(draw: Mapping[str, Any], rng: random.Random,
                frames_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import mujoco

    absent_model = compile_scene(draw, present=False)
    absent_data = mujoco.MjData(absent_model)
    mujoco.mj_forward(absent_model, absent_data)
    absent_renderer = mujoco.Renderer(absent_model, height=HEIGHT, width=WIDTH)
    absent_renderer.update_scene(absent_data, camera="room_camera")
    absent_rgb = absent_renderer.render().copy()
    absent_renderer.close()
    if not absent_rgb.any():
        raise RuntimeError(f"episode {draw['index']}: absent render is black")

    model = compile_scene(draw, present=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    geom_ids = humanoid_geom_ids(model)
    redraws = 0
    while True:
        placement = pose_humanoid(model, data, rng,
                                  toward_camera=min(0.6, 0.15 * (redraws // 25)),
                                  camera_pos=draw["camera"]["pos"])
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="room_camera")
        segmentation = renderer.render()
        fraction = float(np.isin(segmentation[..., 0], geom_ids).mean())
        if fraction >= MIN_MASK_FRACTION:
            break
        redraws += 1
        if redraws > MAX_REDRAWS:
            renderer.close()
            raise RuntimeError(f"episode {draw['index']}: mask stayed below {MIN_MASK_FRACTION:.0%}")
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera="room_camera")
    l0_rgb = renderer.render().copy()
    recolor_l1(model)
    renderer.update_scene(data, camera="room_camera")
    l1_rgb = renderer.render().copy()
    renderer.close()
    if not l0_rgb.any() or not l1_rgb.any():
        raise RuntimeError(f"episode {draw['index']}: present render is black")

    records = []
    for level, present_rgb in ((0, l0_rgb), (1, l1_rgb)):
        directory = frames_dir / f"l{level}"
        directory.mkdir(parents=True, exist_ok=True)
        for label, image_id, pixels, mask in (
            ("present", 2 * draw["index"], present_rgb, fraction),
            ("absent", 2 * draw["index"] + 1, absent_rgb, 0.0),
        ):
            path = directory / f"ep{draw['index']:03d}-{label}.png"
            Image.fromarray(pixels).save(path)
            if label == "present" and mask < MIN_MASK_FRACTION:
                raise AssertionError("PRESENT mask rule failed")
            if label == "absent" and mask != 0.0:
                raise AssertionError("ABSENT mask must be zero")
            records.append({"image_id": image_id, "level": level, "file": str(path.resolve()),
                            "label": label, "mask_fraction": round(mask, 8),
                            "redraws": redraws if label == "present" else 0,
                            "sha256": sha256_of(path)})
    return records, {**draw, "humanoid": placement, "redraws": redraws,
                     "mask_fraction": round(fraction, 8)}


def render_mesh_draw(draw: Mapping[str, Any], rng: random.Random, frames_dir: Path,
                     asset: Mapping[str, Any], *, escape_steps: int = 4
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import mujoco

    absent_model = compile_mesh_scene(draw, asset, present=False)
    absent_data = mujoco.MjData(absent_model)
    mujoco.mj_forward(absent_model, absent_data)
    absent_renderer = mujoco.Renderer(absent_model, height=HEIGHT, width=WIDTH)
    absent_renderer.update_scene(absent_data, camera="room_camera")
    absent_rgb = absent_renderer.render().copy()
    absent_renderer.close()
    if not absent_rgb.any():
        raise RuntimeError(f"episode {draw['index']}: absent render is black")

    # Run the registered capsule placement loop as the draw oracle. This both
    # consumes every pose-jitter draw and preserves capsule redraw decisions,
    # so all later room/camera/light draws remain exactly aligned with rung 3.
    reference_model = compile_scene(draw, present=True)
    reference_data = mujoco.MjData(reference_model)
    reference_renderer = mujoco.Renderer(reference_model, height=HEIGHT, width=WIDTH)
    reference_geom_ids = humanoid_geom_ids(reference_model)
    reference_redraws = 0
    while True:
        placement = pose_humanoid(
            reference_model, reference_data, rng,
            toward_camera=min(0.15 * escape_steps, 0.15 * (reference_redraws // 25)),
            camera_pos=draw["camera"]["pos"],
        )
        reference_renderer.enable_segmentation_rendering()
        reference_renderer.update_scene(reference_data, camera="room_camera")
        reference_segmentation = reference_renderer.render()
        reference_fraction = float(np.isin(reference_segmentation[..., 0], reference_geom_ids).mean())
        if reference_fraction >= MIN_MASK_FRACTION:
            break
        reference_redraws += 1
        if reference_redraws > 25 * escape_steps:
            reference_renderer.close()
            raise RuntimeError(
                f"episode {draw['index']}: capsule reference mask stayed below {MIN_MASK_FRACTION:.0%}"
            )
    reference_renderer.close()

    model = compile_mesh_scene(draw, asset, present=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_person")
    if geom_id < 0:
        raise RuntimeError("the HSR0081 mesh geom is missing")
    base_x, base_y = placement["pos_xy"]
    camera_pos = draw["camera"]["pos"]
    norm = math.hypot(camera_pos[0], camera_pos[1]) or 1.0
    mesh_escape_steps = 0
    while True:
        toward_camera = min(0.15 * escape_steps, 0.15 * mesh_escape_steps)
        x = base_x + toward_camera * camera_pos[0] / norm
        y = base_y + toward_camera * camera_pos[1] / norm
        set_mesh_pose(model, data, x=x, y=y, yaw_degrees=placement["yaw_degrees"])
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="room_camera")
        segmentation = renderer.render()
        fraction = float((segmentation[..., 0] == geom_id).mean())
        if fraction >= MIN_MASK_FRACTION:
            break
        mesh_escape_steps += 1
        if mesh_escape_steps > escape_steps:
            renderer.close()
            raise RuntimeError(f"episode {draw['index']}: mesh mask stayed below {MIN_MASK_FRACTION:.0%}")
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera="room_camera")
    present_rgb = renderer.render().copy()
    renderer.close()
    if not present_rgb.any():
        raise RuntimeError(f"episode {draw['index']}: mesh present render is black")

    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, image_id, pixels, mask in (
        ("present", 1_000_000 + 2 * draw["index"], present_rgb, fraction),
        ("absent", 1_000_000 + 2 * draw["index"] + 1, absent_rgb, 0.0),
    ):
        path = frames_dir / f"ep{draw['index']:03d}-{label}.png"
        Image.fromarray(pixels).save(path)
        if label == "present" and mask < MIN_MASK_FRACTION:
            raise AssertionError("PRESENT mesh mask rule failed")
        if label == "absent" and mask != 0.0:
            raise AssertionError("ABSENT mesh mask must be zero")
        records.append({"image_id": image_id, "level": "mesh", "file": str(path.resolve()),
                        "label": label, "mask_fraction": round(mask, 8),
                        "redraws": reference_redraws if label == "present" else 0,
                        "sha256": sha256_of(path)})
    mesh_placement = {"pos_xy": [x, y], "yaw_degrees": placement["yaw_degrees"],
                      "capsule_pose_jitter_draws_consumed": 21 * (reference_redraws + 1),
                      "escape_steps": mesh_escape_steps}
    return records, {**draw, "mesh": mesh_placement, "capsule_reference": placement,
                     "redraws": reference_redraws, "capsule_mask_fraction": round(reference_fraction, 8),
                     "mask_fraction": round(fraction, 8)}


def manifest_sha(records: Sequence[Mapping[str, Any]]) -> str:
    # PNG hashes are deterministic on one renderer/machine. Cross-GPU identity
    # is intentionally not part of the registration.
    digest = hashlib.sha256()
    for row in records:
        digest.update(row["sha256"].encode())
        digest.update(row["label"].encode())
    return digest.hexdigest()


def generate_frames(frames_dir: Path, *, smoke: bool, figure: str = "capsule",
                    per_class: int | None = None, draw_seed: int = FRAME_DRAW_SEED,
                    start_index: int = 0, registration: str = REGISTRATION,
                    mesh_escape_steps: int = 4) -> dict[str, Any]:
    per_class = per_class if per_class is not None else (SMOKE_PER_CLASS if smoke else EVAL_PER_CLASS)
    rng = random.Random(draw_seed)
    records, draws = [], []
    frames_dir.mkdir(parents=True, exist_ok=True)
    asset = acquire_mesh_asset() if figure == "mesh" else None
    levels = ["mesh"] if figure == "mesh" else list(LEVELS)
    emit("vm3_scene", seed=draw_seed, episodes=per_class, size=[WIDTH, HEIGHT],
         figure=figure, levels=levels, ranges=SCENE_RANGES, l1_rgba=L1_RGBA)
    for index in range(start_index, start_index + per_class):
        scene_draw = draw_scene(rng, index)
        if figure == "mesh":
            rows, draw = render_mesh_draw(
                scene_draw, rng, frames_dir, asset, escape_steps=mesh_escape_steps)
        else:
            rows, draw = render_draw(scene_draw, rng, frames_dir)
        records.extend(rows)
        draws.append(draw)
    manifest = {"format": "legolm-vision-mode-r3-frames-v1", "registration": registration,
                "draw_seed": draw_seed, "width": WIDTH, "height": HEIGHT,
                "per_class_per_level": per_class, "levels": levels,
                "rule": {"present": "humanoid segmentation-mask area >= 5% of the frame",
                         "absent": "the paired drawn scene with the humanoid removed"},
                "scene_ranges": SCENE_RANGES, "l1_rgba": L1_RGBA,
                "figure": figure,
                "draws": draws, "records": records}
    if start_index:
        manifest["start_index"] = start_index
    if mesh_escape_steps != 4:
        manifest["mesh_escape_steps"] = mesh_escape_steps
    if figure == "mesh":
        manifest["mesh_asset"] = asset
    else:
        manifest["humanoid"] = {"path": display_path(HUMANOID_PATH),
                                "sha256": sha256_of(HUMANOID_PATH)}
    manifest["sha256"] = manifest_sha(records)
    atomic_json_dump(manifest, frames_dir / "manifest.json")
    emit("vm3_frames", frames=len(records), per_level=per_class * 2,
         manifest=display_path(frames_dir / "manifest.json"), sha256=manifest["sha256"])
    return manifest


def load_or_generate_frames(frames_dir: Path, *, smoke: bool, regenerate: bool,
                            figure: str = "capsule", per_class: int | None = None,
                            draw_seed: int = FRAME_DRAW_SEED, start_index: int = 0,
                            registration: str = REGISTRATION, mesh_escape_steps: int = 4
                            ) -> dict[str, Any]:
    path = frames_dir / "manifest.json"
    expected = per_class if per_class is not None else (SMOKE_PER_CLASS if smoke else EVAL_PER_CLASS)
    if not regenerate and path.is_file():
        manifest = json.loads(path.read_text())
        records = manifest.get("records", [])
        expected_records = expected * (2 if figure == "mesh" else 4)
        manifest_figure = manifest.get("figure", "capsule")
        valid = (manifest.get("draw_seed") == draw_seed and
                 manifest.get("start_index", 0) == start_index and
                 manifest.get("mesh_escape_steps", 4) == mesh_escape_steps and
                 manifest_figure == figure and
                 manifest.get("per_class_per_level") == expected and
                 len(records) == expected_records and
                 all(Path(row["file"]).is_file() and sha256_of(Path(row["file"])) == row["sha256"]
                     for row in records))
        if valid and manifest_sha(records) == manifest.get("sha256"):
            emit("vm3_frames", frames=len(records), reused=True, sha256=manifest["sha256"])
            return manifest
    return generate_frames(frames_dir, smoke=smoke, figure=figure, per_class=per_class,
                           draw_seed=draw_seed, start_index=start_index,
                           registration=registration, mesh_escape_steps=mesh_escape_steps)


def level_frames(manifest: Mapping[str, Any], level: int | str) -> list[dict[str, Any]]:
    rows = [{k: row[k] for k in ("image_id", "file", "label", "mask_fraction", "redraws")}
            for row in manifest["records"] if str(row["level"]) == str(level)]
    # Match rung 1's class-blocked frame order so held_out_position alternates
    # within each class rather than confounding phrasing position with label.
    return sorted(rows, key=lambda row: (row["label"] == "absent", row["image_id"]))


# --------------------------------------------------------------------------
# part 1: detector transfer
# --------------------------------------------------------------------------

def detector_reading(accuracies: Mapping[str, float]) -> dict[str, Any]:
    if max(accuracies.values()) >= DETECTOR_TRANSFER:
        reading = "(i) DETECTOR TRANSFERS"
    elif all(value < PART2_GATE for value in accuracies.values()):
        reading = "(ii) SIM GAP"
    else:
        reading = "(iii) MIXED"
    return {"reading": reading, "operating_detector": OPERATING_DETECTOR,
            "accuracy_by_level": dict(accuracies),
            "bands": {"transfers_at_either_level": DETECTOR_TRANSFER,
                      "sim_gap_below_both_levels": PART2_GATE}}


def write_detector_markdown(payload: Mapping[str, Any], path: Path) -> None:
    title = "# V-line rung 3b — mesh detector transfer" if payload["figure"] == "mesh" \
        else "# V-line rung 3 — detector transfer"
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    lines += [f"Registration: {payload['registration']}.", "",
              f"Reading: **{payload['reading']['reading']}**.", ""]
    if payload["figure"] == "mesh":
        lines += ["| detector | mesh accuracy | false alarms | misses |",
                  "|---|---|---|---|"]
        for name in DETECTORS:
            row = payload["levels"]["mesh"]["detectors"][name]["confusion"]
            lines.append(f"| {name} | {row['accuracy']:.2f} | {row['false_alarms']} | {row['misses']} |")
        lines += ["", "The mesh is the pinned textured HSR0081-Body-004 LOD2 photoscan."]
    else:
        lines += ["| detector | L0 accuracy | L0 false alarms | L0 misses | L1 accuracy | L1 false alarms | L1 misses |",
                  "|---|---|---|---|---|---|---|"]
        for name in DETECTORS:
            a = payload["levels"]["l0"]["detectors"][name]["confusion"]
            b = payload["levels"]["l1"]["detectors"][name]["confusion"]
            lines.append(f"| {name} | {a['accuracy']:.2f} | {a['false_alarms']} | {a['misses']} | "
                         f"{b['accuracy']:.2f} | {b['false_alarms']} | {b['misses']} |")
        lines += ["", "L0 is the stock humanoid; L1 changes only humanoid geom RGBA values."]
    lines += ["", f"Frame manifest SHA-256: `{payload['manifest']['sha256']}`."]
    path.write_text("\n".join(lines) + "\n")


def run_part1(args, base: Path) -> dict[str, Any]:
    frames_dir = figure_frames_dir(args)
    manifest = load_or_generate_frames(frames_dir, smoke=args.smoke,
                                       regenerate=args.regenerate, figure=args.figure)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    level_names: tuple[int | str, ...] = ("mesh",) if args.figure == "mesh" else LEVELS
    levels: dict[str, Any] = {level_key(level): {"detectors": {}} for level in level_names}
    for name in DETECTORS:
        detector, detector_record = r2y.load_detector(r2y.DETECTOR_PATH.with_name(f"{name}.pt"), device)
        for level in level_names:
            frames = level_frames(manifest, level)
            started = time.perf_counter()
            boxes = r2y.person_boxes_for(detector, frames, device)
            verdicts = {}
            for frame in frames:
                verdict = r2y.verdict_of(boxes[int(frame["image_id"])])
                verdict["correct"] = verdict["verdict"] == frame["label"]
                verdicts[int(frame["image_id"])] = verdict
            confusion = r2.confusion(frames, verdicts)
            record = {**detector_record, "confusion": confusion,
                      "wall_seconds": round(time.perf_counter() - started, 3)}
            if name == OPERATING_DETECTOR:
                record["verdicts"] = {str(frame["image_id"]): {"file": frame["file"],
                    "label": frame["label"], **verdicts[int(frame["image_id"])]} for frame in frames}
            levels[level_key(level)]["detectors"][name] = record
            emit("vm3_detector", detector=name, level=level, **confusion)
        del detector
        r1.release_device_memory(device)
    accuracies = {key: value["detectors"][OPERATING_DETECTOR]["confusion"]["accuracy"]
                  for key, value in levels.items()}
    reading = detector_reading(accuracies)
    experiment = CANONICAL_R3B_DETECTOR_STEM if args.figure == "mesh" else CANONICAL_DETECTOR_STEM
    payload = {"track": r1.TRACK, "experiment": experiment,
               "registration": REGISTRATION, "git_head": r1.git_head(),
               "host_machine": platform.node(), "device": str(device), "smoke": bool(args.smoke),
               "figure": args.figure,
               "claim_scope": ("frozen YOLO26 size sweep on paired MuJoCo PRESENT / ABSENT "
                               + ("frames with the textured HSR0081 mesh" if args.figure == "mesh"
                                  else "frames at L0 and L1")),
               "manifest": manifest, "levels": levels, "reading": reading}
    atomic_json_dump(payload, base.with_suffix(".json"))
    write_detector_markdown(payload, base.with_suffix(".md"))
    emit("vm3_reading", part=1, **reading)
    return payload


# --------------------------------------------------------------------------
# part 2: unchanged rung 1 doors on rendered frames
# --------------------------------------------------------------------------

def coco_leans(host_key: str) -> dict[str, float]:
    path = r1.ROOT / "findings" / f"vision-mode-r1-{host_key}.json"
    if not path.is_file():
        raise SystemExit(f"rung 1 pooled artifact missing: {display_path(path)}")
    payload = json.loads(path.read_text())
    return {mode: float(payload["pooled_metrics"][mode]["lean"]) for mode in r1.MODES}


def part2_reading(metrics: Mapping[str, Mapping[str, Any]], coco: Mapping[str, float]) -> dict[str, Any]:
    leans = {mode: float(metrics[mode]["lean"]) for mode in r1.MODES}
    grounding = {mode: metrics[mode]["grounding"] for mode in r1.GROUNDED_MODES}
    lean_modes = [mode for mode, value in leans.items() if value >= r1.LEAN_BAND]
    clean = all(grounding[mode] is not None and grounding[mode] <= r1.GROUNDING_CLEAN
                for mode in r1.GROUNDED_MODES)
    coco_live = [mode for mode, value in coco.items() if value >= r1.LEAN_BAND]
    blind = all(value < r1.NULL_BAND for value in leans.values()) and len(coco_live) >= 2
    if len(lean_modes) >= 2 and clean:
        reading = "(i) DOORS TRAVEL"
    elif blind:
        reading = "(ii) HOST-BLIND"
    else:
        reading = "(iii) MIXED"
    return {"reading": reading, "leans": leans, "grounding": grounding,
            "lean_modes": lean_modes, "grounding_clean": clean,
            "coco_leans": dict(coco), "coco_live_modes": coco_live,
            "bands": {"lean": r1.LEAN_BAND, "grounding_clean": r1.GROUNDING_CLEAN,
                      "host_blind": r1.NULL_BAND}}


def run_part2_seed(args, *, host_key: str, seed: int, model, tower, processor,
                   device, dtype, depth: int, hidden: int, frames, manifest, bank,
                   base_samples, plan, coco: Mapping[str, float], selected_level) -> dict[str, Any]:
    started = time.perf_counter()
    r1.seed_everything(seed, device)
    mode_bridge, mode_record = r1.load_door(host_key, "mode", seed, hidden=hidden,
                                             inject_layers=plan["inject_layers"], device=device)
    constant_bridge, constant_record = r1.load_door(host_key, "constant", seed, hidden=hidden,
                                                     inject_layers=plan["inject_layers"], device=device)
    if mode_record["source_layer"] != plan["source_layer"]:
        raise SystemExit("the door's source layer is not this host plan's")
    emit("vm3_seed_start", host=host_key, seed=seed, level=selected_level,
         doors={"mode": mode_record["sha256"], "constant": constant_record["sha256"]})
    features = r1.bank_features(model, processor.tokenizer, device, plan["source_layer"])
    constant_feature = r1.train_tap_stack(features).mean(0)
    cells = {mode: {} for mode in r1.MODES}
    cache: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    for mode in r1.MODES:
        for arm in r1.ARMS:
            key = r1.cell_key(arm, mode)
            if key not in cache:
                cache[key] = r1.run_cell(
                    model=model, tower=tower, tokenizer=processor.tokenizer, device=device,
                    bank=bank, frames=frames, key=key, features=features,
                    mode_bridge=mode_bridge, constant_feature=constant_feature,
                    constant_bridge=constant_bridge, inject_layers=plan["inject_layers"],
                    dose=r1.TRAIN_DOSE,
                )
                emit("vm3_cell", host=host_key, seed=seed, packet=key[0], text=key[1],
                     present_mix=r1.mix(cache[key], "present"), absent_mix=r1.mix(cache[key], "absent"))
            cells[mode][arm] = cache[key]
    if not r1.samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during rung 3 evaluation")
    metrics = {mode: r1.mode_metrics(cells[mode], mode) for mode in r1.MODES}
    rung1_reading = r1.reading_of(metrics)
    reading = part2_reading(metrics, coco)
    emit("vm3_seed_reading", host=host_key, seed=seed, **reading)
    stem = (CANONICAL_R3B_STEM + args.artifact_suffix
            if args.figure == "mesh" else CANONICAL_STEM)
    payload = {"track": r1.TRACK, "experiment": f"{stem}-{host_key}-s{seed}",
            "registration": REGISTRATION, "reuses": r1.REGISTRATION,
            "git_head": r1.git_head(), "host_machine": platform.node(), "device": str(device),
            "dtype": str(dtype).removeprefix("torch."), "host": {"key": host_key, **r1.HOSTS[host_key],
            "depth": depth, "hidden_size": hidden}, "seed": seed, "smoke": bool(args.smoke),
            "figure": args.figure, "level": selected_level, "manifest": manifest, "layer_plan": plan,
            "doors": {"mode": mode_record, "constant": constant_record, "dose": r1.TRAIN_DOSE,
                      "trained_here": False}, "frames": {"eval": len(frames)},
            "arms": cells, "metrics": metrics, "rung1_reading": rung1_reading,
            "reading": reading,
            "wall_seconds": round(time.perf_counter() - started, 3)}
    if args.order_seed is not None or args.constant_variant != "same":
        payload.update({"order_seed": args.order_seed,
                        "constant_variant": args.constant_variant})
    return payload


def pooled_part2(host_key: str, per_seed: Sequence[Mapping[str, Any]], coco: Mapping[str, float],
                 artifact_suffix: str) -> dict[str, Any]:
    metrics = {mode: r1.mode_metrics({arm: [row for payload in per_seed for row in payload["arms"][mode][arm]]
                                     for arm in r1.ARMS}, mode) for mode in r1.MODES}
    rung1_reading = r1.reading_of(metrics)
    reading = part2_reading(metrics, coco)
    seed_readings = {str(payload["seed"]): payload["reading"]["reading"] for payload in per_seed}
    votes: dict[str, int] = {}
    for value in seed_readings.values():
        votes[value] = votes.get(value, 0) + 1
    majority = max(votes.items(), key=lambda item: (item[1], item[0]))
    stem = (CANONICAL_R3B_STEM + artifact_suffix
            if per_seed[0]["figure"] == "mesh" else CANONICAL_STEM)
    payload = {"track": r1.TRACK, "experiment": f"{stem}-{host_key}-pooled",
            "registration": REGISTRATION, "git_head": r1.git_head(), "host_machine": platform.node(),
            "host": per_seed[0]["host"], "seeds": [p["seed"] for p in per_seed],
            "smoke": any(p["smoke"] for p in per_seed), "figure": per_seed[0]["figure"],
            "level": per_seed[0]["level"],
            "manifest": per_seed[0]["manifest"], "per_seed_readings": seed_readings,
            "majority_reading": {"reading": majority[0], "votes": majority[1], "of": len(per_seed),
                                 "decisive": majority[1] * 2 > len(per_seed)},
            "pooled_metrics": metrics, "pooled_rung1_reading": rung1_reading,
            "pooled_reading": reading,
            "per_seed_leans": {str(p["seed"]): p["reading"]["leans"] for p in per_seed},
            "per_seed_grounding": {str(p["seed"]): p["reading"]["grounding"] for p in per_seed},
            "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3)}
    if "constant_variant" in per_seed[0]:
        payload.update({"order_seed": per_seed[0]["order_seed"],
                        "constant_variant": per_seed[0]["constant_variant"]})
    return payload


def write_part2_markdown(payload: Mapping[str, Any], path: Path, *, pooled: bool) -> None:
    title = (f"# V-line rung 3 — {payload['host']['label']}, pooled over door seeds {payload['seeds']}"
             if pooled else f"# V-line rung 3 — {payload['host']['label']}, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    reading = payload["pooled_reading"] if pooled else payload["reading"]
    lines += [f"Registration: {payload['registration']}.", "",
              (f"Rendered {'mesh level' if payload.get('figure') == 'mesh' else 'level L' + str(payload['level'])}; "
               f"frame manifest `{payload['manifest']['sha256']}`."), "",
              f"Reading: **{reading['reading']}**.", ""]
    lines += r1.metrics_table(payload["pooled_metrics"] if pooled else payload["metrics"])
    if pooled:
        lines += ["", "Per-seed readings: " + json.dumps(payload["per_seed_readings"]),
                  "Per-seed leans: " + json.dumps(payload["per_seed_leans"]),
                  "Per-seed grounding: " + json.dumps(payload["per_seed_grounding"])]
    lines += ["", "COCO pooled leans used by the HOST-BLIND comparison: " + json.dumps(reading["coco_leans"])]
    path.write_text("\n".join(lines) + "\n")


def detector_artifact_for_part2(base: Path, manifest_sha256: str, *, figure: str) -> Path:
    detector_stem = CANONICAL_R3B_DETECTOR_STEM if figure == "mesh" else CANONICAL_DETECTOR_STEM
    candidates = [base.with_suffix(".json"),
                  base.with_name(f"{base.name}-detector").with_suffix(".json"),
                  r1.ROOT / "findings" / f"{detector_stem}.json"]
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if (payload.get("experiment") == detector_stem and
                payload.get("manifest", {}).get("sha256") == manifest_sha256):
            return path
    raise SystemExit("part 2 is blocked: no detector artifact for this frame manifest; run part 1 first")


def run_part2(args, base: Path) -> None:
    if args.figure == "capsule" and args.level is None:
        raise SystemExit("--part 2 requires --level {0,1}")
    if args.figure == "mesh" and args.level is not None:
        raise SystemExit("--figure mesh has one mesh level; drop --level")
    selected_level: int | str = "mesh" if args.figure == "mesh" else args.level
    manifest_path = figure_frames_dir(args) / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"rendered frame manifest missing: {display_path(manifest_path)}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("figure", "capsule") != args.figure:
        raise SystemExit(f"rendered frame manifest is for {manifest.get('figure', 'capsule')}, "
                         f"not --figure {args.figure}")
    if manifest_sha(manifest.get("records", [])) != manifest.get("sha256"):
        raise SystemExit("rendered frame manifest SHA does not match its records")
    detector_path = detector_artifact_for_part2(base, manifest["sha256"], figure=args.figure)
    detector_payload = json.loads(detector_path.read_text())
    accuracies = detector_payload["reading"]["accuracy_by_level"]
    if max(accuracies.values()) < PART2_GATE:
        suffix = "at the mesh level" if args.figure == "mesh" else "at both levels"
        raise SystemExit(f"part 2 is blocked: yolo26s accuracy is below 0.75 {suffix}")
    selected_key = level_key(selected_level)
    if accuracies[selected_key] < PART2_GATE:
        raise SystemExit(f"part 2 is blocked: chosen {selected_key} did not clear yolo26s accuracy 0.75")
    frames = level_frames(manifest, selected_level)
    expected = SMOKE_PER_CLASS if args.smoke else EVAL_PER_CLASS
    if len(frames) != expected * 2 or any(not Path(frame["file"]).is_file() for frame in frames):
        raise SystemExit("rendered frame manifest does not match this run")
    seeds = list(r1.SEEDS) if args.all_seeds else [args.seed if args.seed is not None else r1.SEEDS[0]]
    host_arg = args.host or "450m"
    hosts = list(r1.HOST_ORDER) if host_arg == "all" else [host_arg]
    for host_key in hosts:
        device = torch.device(args.device if args.device != "auto"
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        processor, model, tower, depth, hidden, dtype = r1.load_host(
            host_key, device, local_files_only=args.local_files_only)
        plan, coco = r1.layer_plan(depth), coco_leans(host_key)
        bank, base_samples = r1.FrameBank(processor, device, dtype), r1.parameter_samples(model)
        emit("vm3_host", host=host_key, level=selected_level, depth=depth, plan=plan)
        per_seed = []
        for seed in seeds:
            payload = run_part2_seed(args, host_key=host_key, seed=seed, model=model, tower=tower,
                                     processor=processor, device=device, dtype=dtype, depth=depth,
                                     hidden=hidden, frames=frames, manifest=manifest, bank=bank,
                                     base_samples=base_samples, plan=plan, coco=coco,
                                     selected_level=selected_level)
            prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
            atomic_json_dump(payload, prefix.with_suffix(".json"))
            write_part2_markdown(payload, prefix.with_suffix(".md"), pooled=False)
            per_seed.append(payload)
        pooled = pooled_part2(host_key, per_seed, coco, args.artifact_suffix)
        prefix = base.with_name(f"{base.name}-{host_key}")
        atomic_json_dump(pooled, prefix.with_suffix(".json"))
        write_part2_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
        emit("vm3_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
             majority=pooled["majority_reading"])
        del model, tower, bank
        r1.release_device_memory(device)


# --------------------------------------------------------------------------
# rung 3b capsule host-probe
# --------------------------------------------------------------------------

def probe_frames(frames_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = frames_dir / "manifest.json"
    guidance = (
        "the capsule host-probe needs rung 3's existing full L1 capsule frames; "
        f"run rung 3 part 1 without --smoke using --frames-dir {display_path(frames_dir)} first"
    )
    if not manifest_path.is_file():
        raise SystemExit(f"{guidance} (manifest missing)")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("figure", "capsule") != "capsule":
        raise SystemExit(f"{guidance} (the manifest is not a capsule draw)")
    l1 = level_frames(manifest, 1)
    present = sorted((row for row in l1 if row["label"] == "present"),
                     key=lambda row: row["image_id"])[:PROBE_PER_CLASS]
    absent = sorted((row for row in l1 if row["label"] == "absent"),
                    key=lambda row: row["image_id"])[:PROBE_PER_CLASS]
    frames = present + absent
    if len(present) != PROBE_PER_CLASS or len(absent) != PROBE_PER_CLASS:
        raise SystemExit(f"{guidance} (need 30 present and 30 absent L1 frames)")
    missing = [row["file"] for row in frames if not Path(row["file"]).is_file()]
    if missing:
        raise SystemExit(f"{guidance} (selected frame missing: {missing[0]})")
    return manifest, frames


@torch.no_grad()
def run_probe_arm(*, arm: str, model, tower, tokenizer, device, bank, frames,
                  constant_bridge, constant_feature, inject_layers) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    rows = []
    for index, frame in enumerate(frames):
        position = r1.held_out_position(index)
        if arm == "text":
            mode_text = r1.PHRASES["caution"][position]
            inputs = bank.inputs(frame, mode_text=mode_text)
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=r1.MAX_NEW,
                use_cache=True, pad_token_id=eos,
            )
            new_ids = generated[0, inputs["input_ids"].shape[1]:]
        elif arm == "habit":
            mode_text = None
            _packet, payloads = r1.payloads_of(constant_bridge, constant_feature, device)
            new_ids = r1.generate_with_packet(
                model, tower, bank.inputs(frame, mode_text=None), payloads,
                inject_layers, r1.TRAIN_DOSE, eos=eos,
            )
        else:
            raise ValueError(arm)
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        action, how = r1.parse_action(decoded)
        rows.append({"image_id": frame["image_id"], "file": frame["file"],
                     "label": frame["label"], "position": position,
                     "mode_text": mode_text, "decoded": decoded,
                     "action": action, "parse": how})
    return rows


def probe_reading(discrimination: float) -> dict[str, Any]:
    if discrimination >= 0.30:
        reading = "HOST SEES"
    elif discrimination < 0.10:
        reading = "HOST BLIND"
    else:
        reading = "MIXED"
    return {"reading": reading, "discrimination": discrimination,
            "bands": {"host_sees_at_least": 0.30, "host_blind_below": 0.10}}


def write_probe_markdown(payload: Mapping[str, Any], path: Path) -> None:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    reading = payload["reading"]
    shares = payload["stop_shares"]
    lines = ["# V-line rung 3b — capsule host-probe at LFM2.5-VL-3B", "",
             f"Registration: {payload['registration']}.", "",
             (f"Reading: **{reading['reading']}**; caution-text discrimination "
              f"{reading['discrimination']:.3f}."), "",
             (f"Caution TEXT stop-share: present {shares['text']['present']:.3f}, "
              f"absent {shares['text']['absent']:.3f}. Habit stop-share: present "
              f"{shares['habit']['present']:.3f}, absent {shares['habit']['absent']:.3f}."), "",
             f"Frame manifest SHA-256: `{payload['manifest']['sha256']}`.", "",
             "| image id | label | parity | caution TEXT decode | HABIT decode |",
             "|---:|---|---:|---|---|"]
    habit = {row["image_id"]: row for row in payload["arms"]["habit"]}
    for row in payload["arms"]["text"]:
        lines.append(f"| {row['image_id']} | {row['label']} | {row['position']} | "
                     f"{clean(row['decoded'])} | {clean(habit[row['image_id']]['decoded'])} |")
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, base: Path) -> None:
    if args.figure != "capsule":
        raise SystemExit("--part probe uses rung 3's L1 capsule frames; use --figure capsule")
    if args.host not in (None, PROBE_HOST):
        raise SystemExit("--part probe is registered only for --host 3b")
    if args.all_seeds:
        raise SystemExit("--part probe runs one constant-door seed; use --seed (default 42)")
    if args.smoke or args.regenerate:
        raise SystemExit("--part probe reuses existing full rung 3 frames; drop --smoke/--regenerate")
    seed = args.seed if args.seed is not None else r1.SEEDS[0]
    manifest, frames = probe_frames(Path(args.frames_dir).expanduser())
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = r1.load_host(
        PROBE_HOST, device, local_files_only=args.local_files_only)
    plan = r1.layer_plan(depth)
    base_samples = r1.parameter_samples(model)
    bank = r1.FrameBank(processor, device, dtype)
    r1.seed_everything(seed, device)
    constant_bridge, constant_record = r1.load_door(
        PROBE_HOST, "constant", seed, hidden=hidden,
        inject_layers=plan["inject_layers"], device=device)
    if constant_record["source_layer"] != plan["source_layer"]:
        raise SystemExit("the constant door's source layer is not this host plan's")
    features = r1.bank_features(model, processor.tokenizer, device, plan["source_layer"])
    constant_feature = r1.train_tap_stack(features).mean(0)
    arms = {arm: run_probe_arm(
        arm=arm, model=model, tower=tower, tokenizer=processor.tokenizer,
        device=device, bank=bank, frames=frames, constant_bridge=constant_bridge,
        constant_feature=constant_feature, inject_layers=plan["inject_layers"])
        for arm in ("text", "habit")}
    if not r1.samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the rung 3b capsule probe")
    stop_shares = {arm: {label: r1.share(rows, label, "stop")
                         for label in ("present", "absent")}
                   for arm, rows in arms.items()}
    discrimination = round(stop_shares["text"]["present"] -
                           stop_shares["text"]["absent"], 6)
    reading = probe_reading(discrimination)
    payload = {"track": r1.TRACK, "experiment": CANONICAL_PROBE_STEM,
               "registration": REGISTRATION, "reuses": r1.REGISTRATION,
               "git_head": r1.git_head(), "host_machine": platform.node(),
               "device": str(device), "dtype": str(dtype).removeprefix("torch."),
               "host": {"key": PROBE_HOST, **r1.HOSTS[PROBE_HOST],
                        "depth": depth, "hidden_size": hidden},
               "seed": seed, "figure": "capsule", "level": 1,
               "manifest": manifest, "frames": {"present": PROBE_PER_CLASS,
                                                   "absent": PROBE_PER_CLASS},
               "layer_plan": plan, "door": {**constant_record, "dose": r1.TRAIN_DOSE,
                                              "trained_here": False},
               "arms": arms, "stop_shares": stop_shares, "reading": reading}
    atomic_json_dump(payload, base.with_suffix(".json"))
    write_probe_markdown(payload, base.with_suffix(".md"))
    emit("vm3_probe_reading", host=PROBE_HOST, seed=seed, **reading)
    del model, tower, bank
    r1.release_device_memory(device)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_part(value: str) -> int | str:
    return int(value) if value in ("1", "2") else value


def level_key(level: int | str) -> str:
    return "mesh" if level == "mesh" else f"l{level}"


def figure_frames_dir(args) -> Path:
    root = Path(args.frames_dir).expanduser()
    return root / "mesh" if args.figure == "mesh" else root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=parse_part, choices=(1, 2, "probe"), required=True)
    parser.add_argument("--figure", choices=("capsule", "mesh"), default="capsule")
    parser.add_argument("--host", choices=(*r1.HOSTS, "all"), default=None,
                        help="all = the two LFM hosts; 27b must be named explicitly (rung 3f)")
    parser.add_argument("--seed", type=int, choices=r1.SEEDS, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--door-variant", choices=("registered", "x2", "mx", "mx2"), default="registered",
                        help="x2 loads the vm-r1x2 door pair (rung 3c); mx loads vm-r1mx "
                             "(rung 3d); each writes its matching artifact family")
    parser.add_argument("--order-seed", type=int, choices=r1.SEEDS, default=None)
    parser.add_argument("--constant-variant", choices=("same", "registered"), default="same")
    parser.add_argument("--level", type=int, choices=LEVELS, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--frames-dir", default="/tmp/vm-r3-frames")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--all-seeds runs every registered seed; drop --seed")
    if args.order_seed is not None and args.door_variant != "mx2":
        raise SystemExit("--order-seed requires --door-variant mx2")
    if args.order_seed is not None and args.part != 2:
        raise SystemExit("--order-seed applies to part 2 only")
    if args.constant_variant == "registered" and args.door_variant != "mx2":
        raise SystemExit("--constant-variant registered requires --door-variant mx2")
    if args.constant_variant == "registered" and args.order_seed is not None:
        raise SystemExit("--constant-variant registered cannot be combined with --order-seed")
    args.artifact_suffix = ""
    if args.door_variant != "registered":
        if args.part != 2:
            raise SystemExit(f"--door-variant {args.door_variant} applies to part 2 only")
        r1.DOOR_STEM = {"x2": "vm-r1x2", "mx": "vm-r1mx", "mx2": "vm-r1mx2"}[args.door_variant]
        args.artifact_suffix = {"x2": "-x2door", "mx": "-mxdoor",
                                "mx2": "-mx2door"}[args.door_variant]
    if args.order_seed is not None:
        r1.DOOR_STEM = f"vm-r1mx2o{args.order_seed}"
        args.artifact_suffix = f"-mx2o{args.order_seed}door"
    elif args.constant_variant == "registered":
        r1.CONSTANT_DOOR_STEM = "vm-r1"
        args.artifact_suffix = "-mx2ndoor"
    if args.part == "probe":
        stem = CANONICAL_PROBE_STEM
    elif args.figure == "mesh":
        stem = (CANONICAL_R3B_DETECTOR_STEM if args.part == 1 else
                CANONICAL_R3B_STEM + args.artifact_suffix)
    else:
        stem = CANONICAL_DETECTOR_STEM if args.part == 1 else CANONICAL_STEM
    if args.output_prefix is None:
        if args.smoke:
            raise SystemExit("--smoke requires a non-canonical --output-prefix")
        base = r1.ROOT / "findings" / stem
    else:
        base = Path(args.output_prefix).expanduser()
        canonical_parent = (r1.ROOT / "findings").resolve()
        canonical_name = base.name.startswith((CANONICAL_STEM, CANONICAL_DETECTOR_STEM,
                                               CANONICAL_R3B_STEM, CANONICAL_R3B_DETECTOR_STEM,
                                               CANONICAL_PROBE_STEM))
        if args.smoke and base.parent.resolve() == canonical_parent and canonical_name:
            raise SystemExit("a smoke may not write the canonical family")
    base.parent.mkdir(parents=True, exist_ok=True)
    if args.part == 1:
        run_part1(args, base)
    elif args.part == 2:
        run_part2(args, base)
    else:
        run_probe(args, base)


if __name__ == "__main__":
    main()
