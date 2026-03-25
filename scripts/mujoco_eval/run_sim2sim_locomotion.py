#!/usr/bin/env python3
"""Sim2Sim locomotion evaluation in MuJoCo.

Loads an ONNX policy exported from IsaacLab and replays it in MuJoCo with
environment alignment (observation semantics, PD control, terrain, timing).

Usage examples:
    # Interactive viewer with keyboard teleop (flat ground)
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task flat_stand \
        --render --teleop keyboard

    # Headless batch evaluation with video recording
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task slope_comprehensive \
        --num-episodes 10 --save-video --output-dir eval_results

    # Deploy mode with optional deploy.yaml for PD gains override
    python scripts/mujoco_eval/run_sim2sim_locomotion.py \
        --robot g1 --onnx path/to/policy.onnx --task flat_stand \
        --render --deploy --follow --deploy-yaml path/to/deploy.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure unitree_lab.mujoco_utils is importable even when the full Isaac Lab
# environment is not available (unitree_lab.__init__ imports isaaclab_tasks).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_source_pkg = PROJECT_ROOT / "source" / "unitree_lab"
for _p in [str(PROJECT_ROOT), str(_source_pkg)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Prevent the top-level unitree_lab package from triggering Isaac Lab imports.
# We only need the mujoco_utils subpackage here.
import types

_ul = types.ModuleType("unitree_lab")
_ul.__path__ = [str(_source_pkg / "unitree_lab")]
_ul.__package__ = "unitree_lab"
sys.modules.setdefault("unitree_lab", _ul)

import numpy as np

ROBOT_CONFIGS: dict[str, dict[str, Any]] = {
    "g1": {
        "xml_flat": "g1/scene_29dof.xml",
        "xml_terrain": "g1/scene_29dof_terrain.xml",
        "dof": 29,
    },
}


def _mid_range(r: Any) -> float:
    if r is None:
        return 0.1
    if isinstance(r, (tuple, list)) and len(r) >= 2:
        return (float(r[0]) + float(r[1])) / 2.0
    return float(r)


def _terrain_dict_to_gen_cfg(terrain: dict) -> dict | None:
    """Map bfm-style :class:`~unitree_lab.mujoco_utils.evaluation.eval_task.EvalTask` terrain dict to generator kwargs."""
    from unitree_lab.mujoco_utils.evaluation.eval_task import TERRAIN_FLAT

    if terrain is None or terrain == TERRAIN_FLAT:
        return {"terrain_type": "flat"}
    if len(terrain) != 1:
        return None
    (name, spec), = terrain.items()
    if float(spec.get("proportion", 1.0)) < 0.999:
        return None
    sub = {k: v for k, v in spec.items() if k != "proportion"}

    if name == "plane":
        return {"terrain_type": "flat"}
    if name == "RandomUniform":
        return {
            "terrain_type": "random_uniform",
            "noise_range": tuple(sub.get("noise_range", (-0.05, 0.05))),
            "noise_step": float(sub.get("noise_step", 0.01)),
        }
    if name == "PyramidStairs":
        sh = sub.get("step_height_range", (0.1, 0.15))
        return {
            "terrain_type": "pyramid_stairs",
            "step_height": _mid_range(sh),
            "step_width": float(sub.get("step_width", 0.3)),
        }
    if name == "InvertedPyramidStairs":
        sh = sub.get("step_height_range", (0.1, 0.15))
        return {
            "terrain_type": "pyramid_stairs_inv",
            "step_height": _mid_range(sh),
            "step_width": float(sub.get("step_width", 0.3)),
        }
    if name == "PyramidSloped":
        sr = sub.get("slope_range", (0.25, 0.25))
        return {
            "terrain_type": "pyramid_sloped",
            "slope_angle": _mid_range(sr),
        }
    if name == "InvertedPyramidSloped":
        sr = sub.get("slope_range", (0.25, 0.25))
        return {
            "terrain_type": "pyramid_sloped_inv",
            "slope_angle": _mid_range(sr),
        }
    if name == "Rails":
        return {
            "terrain_type": "rails",
            "rail_height_range": tuple(sub.get("rail_height_range", (0.02, 0.06))),
            "rail_spacing_range": tuple(sub.get("rail_spacing_range", (0.4, 0.2))),
        }
    return None


def _task_ref_velocity_tuple(task: Any) -> tuple[float, float, float]:
    if hasattr(task, "get_velocity_command"):
        w = float(getattr(task, "warmup", 0.0))
        v = task.get_velocity_command(w)
        return (float(v[0]), float(v[1]), float(v[2]))
    return (0.5, 0.0, 0.0)


def _find_asset_dir() -> Path:
    """Locate the robots_xml asset directory."""
    candidates = [
        Path(__file__).resolve().parents[2] / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
        Path(__file__).resolve().parents[1] / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
        PROJECT_ROOT / "source" / "unitree_lab" / "unitree_lab" / "assets" / "robots_xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find robots_xml directory. Searched: {[str(c) for c in candidates]}"
    )


def _load_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load deploy.yaml and return as a flat dict."""
    import yaml

    path = Path(path)
    if not path.exists():
        print(f"[Warning] deploy.yaml not found: {path}")
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _deploy_yaml_to_config_override(deploy: dict) -> dict[str, Any]:
    """Convert deploy.yaml contents to OnnxConfig-compatible override dict.

    Handles both direct metadata keys (from ONNX metadata_json) and the
    mjlab-style deploy.yaml schema produced by train.py's
    ``_build_deploy_yaml_from_metadata``.
    """
    override: dict[str, Any] = {}

    # Direct keys (ONNX metadata style)
    _DIRECT_KEYS = [
        "joint_stiffness", "joint_damping", "joint_names", "default_joint_pos",
        "action_scale", "action_offset", "observation_names", "observation_dims",
        "observation_scales", "sim_dt", "decimation", "joint_armature",
        "history_length", "single_frame_dims",
    ]
    for k in _DIRECT_KEYS:
        if k in deploy:
            override[k] = deploy[k]

    # mjlab-style aliases produced by train.py
    if "stiffness" in deploy and "joint_stiffness" not in override:
        override["joint_stiffness"] = deploy["stiffness"]
    if "damping" in deploy and "joint_damping" not in override:
        override["joint_damping"] = deploy["damping"]
    if "armature" in deploy and "joint_armature" not in override:
        override["joint_armature"] = deploy["armature"]

    # actions.JointPositionAction.{scale, offset}
    actions = deploy.get("actions", {})
    jpa = actions.get("JointPositionAction", {})
    if "scale" in jpa and "action_scale" not in override:
        override["action_scale"] = jpa["scale"]
    if "offset" in jpa and "action_offset" not in override:
        override["action_offset"] = jpa["offset"]

    # observations.<term>.scale -> observation_scales dict
    obs_dict = deploy.get("observations", {})
    if isinstance(obs_dict, dict) and obs_dict and "observation_scales" not in override:
        scales: dict[str, float] = {}
        for term, cfg in obs_dict.items():
            if isinstance(cfg, dict) and "scale" in cfg:
                scales[term] = cfg["scale"]
        if scales:
            override["observation_scales"] = scales

    return override


def _generate_course_xml(xml_path: Path, task: Any) -> tuple[Path, float]:
    """Generate a temporary XML with box-geom course terrain inserted.

    Returns (temp_xml_path, spawn_z_offset).  The caller should load the
    returned XML instead of the original terrain XML.
    """
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    if not hasattr(task, "get_terrain_config"):
        raise TypeError(
            "Course XML generation requires get_terrain_config(); EvalTask uses terrain dicts only."
        )
    terrain_cfg = task.get_terrain_config()
    segments = terrain_cfg.get("course_segments", ())
    step_h = float(terrain_cfg.get("step_height", 0.05))
    step_w = float(terrain_cfg.get("step_width", 0.50))
    slope_angle = float(terrain_cfg.get("slope_angle", 0.15))
    platform_h = float(terrain_cfg.get("course_platform_height", 0.50))
    noise_amp = 0.0
    nr = terrain_cfg.get("noise_range", (-0.01, 0.01))
    if nr:
        noise_amp = max(abs(nr[0]), abs(nr[1]))
    half_y = 3.0  # half-width in Y direction

    total_len = sum(s[1] for s in segments)

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    xml_dir = xml_path.parent

    # Remove heightfield asset + geom (we'll use boxes instead).
    asset_el = root.find("asset")
    if asset_el is not None:
        for hf in list(asset_el.findall("hfield")):
            asset_el.remove(hf)

    wb = root.find("worldbody")
    if wb is not None:
        for g in list(wb.findall("geom")):
            if g.get("type") == "hfield":
                wb.remove(g)
    else:
        wb = ET.SubElement(root, "worldbody")

    # Ensure groundplane material exists
    if asset_el is None:
        asset_el = ET.SubElement(root, "asset")

    # Add a flat ground plane at z=0 as fallback (prevents falling through gaps).
    gp = ET.SubElement(wb, "geom")
    gp.set("name", "ground_plane")
    gp.set("type", "plane")
    gp.set("size", f"{total_len} {half_y + 1} 0.01")
    gp.set("material", "groundplane")
    gp.set("friction", "1 0.005 0.0001")
    gp.set("contype", "1")
    gp.set("conaffinity", "1")

    # Build course along +X, origin at center.
    # Each geom is a solid box from z=0 up to its top surface, preventing
    # foot entrapment. A small x-overlap between adjacent segments ensures
    # continuous contact surfaces at transitions.
    x_start = -total_len / 2.0
    cur_x = x_start
    cur_h = 0.0
    geom_idx = 0
    spawn_z = 0.5  # default
    x_overlap = 0.02  # 2cm overlap between segments to seal gaps

    def _add_solid_box(wb_, name, cx, cz_top, half_sx, half_sy, rgba="0.4 0.45 0.5 1"):
        """Add a box geom that extends from z=0 to cz_top."""
        nonlocal geom_idx
        if cz_top < 0.001:
            return
        bx = ET.SubElement(wb_, "geom")
        bx.set("name", name)
        bx.set("type", "box")
        bx.set("size", f"{half_sx} {half_sy} {cz_top / 2}")
        bx.set("pos", f"{cx} 0 {cz_top / 2}")
        bx.set("friction", "1 0.005 0.0001")
        bx.set("contype", "1")
        bx.set("conaffinity", "1")
        bx.set("rgba", rgba)
        geom_idx += 1

    for seg_idx, (seg_type, seg_len) in enumerate(segments):
        seg_type = seg_type.strip().lower()

        if seg_type in ("flat", "platform"):
            if cur_h > 0.001:
                _add_solid_box(wb, f"course_{geom_idx}",
                               cur_x + seg_len / 2, cur_h,
                               seg_len / 2 + x_overlap, half_y, "0.35 0.45 0.55 1")
            cur_x += seg_len

        elif seg_type == "slope_up":
            rise = min(seg_len * np.tan(slope_angle), platform_h - cur_h)
            rise = max(rise, 0.0)
            n_slices = max(1, int(seg_len / 0.10))
            slice_w = seg_len / n_slices
            for sl in range(n_slices):
                frac = (sl + 1) / n_slices
                top_h = cur_h + frac * rise
                sx_pos = cur_x + sl * slice_w + slice_w / 2
                _add_solid_box(wb, f"course_{geom_idx}", sx_pos, top_h,
                               slice_w / 2 + x_overlap / 2, half_y, "0.3 0.5 0.4 1")
            cur_h += rise
            cur_x += seg_len

        elif seg_type == "slope_down":
            drop = min(seg_len * np.tan(slope_angle), cur_h)
            drop = max(drop, 0.0)
            n_slices = max(1, int(seg_len / 0.10))
            slice_w = seg_len / n_slices
            for sl in range(n_slices):
                frac = (sl + 1) / n_slices
                top_h = cur_h - frac * drop
                top_h = max(top_h, 0.001)
                sx_pos = cur_x + sl * slice_w + slice_w / 2
                _add_solid_box(wb, f"course_{geom_idx}", sx_pos, top_h,
                               slice_w / 2 + x_overlap / 2, half_y, "0.3 0.5 0.4 1")
            cur_h -= drop
            cur_h = max(cur_h, 0.0)
            cur_x += seg_len

        elif seg_type == "stairs_up":
            target_rise = min(platform_h - cur_h, platform_h)
            target_rise = max(target_rise, 0.0)
            n_steps = max(1, round(target_rise / step_h)) if step_h > 1e-6 else 1
            actual_step_w = seg_len / n_steps
            for s in range(n_steps):
                sh = cur_h + (s + 1) * step_h
                sx_pos = cur_x + s * actual_step_w + actual_step_w / 2
                _add_solid_box(wb, f"course_{geom_idx}", sx_pos, sh,
                               actual_step_w / 2 + x_overlap, half_y, "0.4 0.45 0.5 1")
            cur_h += n_steps * step_h
            cur_x += seg_len

        elif seg_type == "stairs_down":
            target_drop = min(cur_h, platform_h)
            target_drop = max(target_drop, 0.0)
            n_steps = max(1, round(target_drop / step_h)) if step_h > 1e-6 else 1
            actual_step_w = seg_len / n_steps
            for s in range(n_steps):
                sh = cur_h - (s + 1) * step_h
                sh = max(sh, 0.001)
                sx_pos = cur_x + s * actual_step_w + actual_step_w / 2
                _add_solid_box(wb, f"course_{geom_idx}", sx_pos, sh,
                               actual_step_w / 2 + x_overlap, half_y, "0.4 0.45 0.5 1")
            cur_h -= n_steps * step_h
            cur_h = max(cur_h, 0.0)
            cur_x += seg_len

        elif seg_type == "rough":
            # 碎石地: discrete stone-like bumps (larger, sparser).
            rng = np.random.default_rng(42 + geom_idx)
            n_bumps = int(seg_len * half_y * 2 * 2)  # ~2 bumps per m²
            for _ in range(n_bumps):
                bw = rng.uniform(0.08, 0.20)
                bh_val = rng.uniform(0.002, noise_amp * 2) if noise_amp > 0 else 0.005
                bx_pos = cur_x + rng.uniform(0, seg_len)
                by_pos = rng.uniform(-half_y, half_y)
                bz_pos = cur_h + bh_val / 2
                bump = ET.SubElement(wb, "geom")
                bump.set("name", f"course_bump_{geom_idx}")
                bump.set("type", "box")
                bump.set("size", f"{bw/2} {bw/2} {bh_val/2}")
                bump.set("pos", f"{bx_pos} {by_pos} {bz_pos}")
                bump.set("friction", "1 0.005 0.0001")
                bump.set("contype", "1")
                bump.set("conaffinity", "1")
                bump.set("rgba", "0.3 0.4 0.5 1")
                geom_idx += 1
            cur_x += seg_len

        elif seg_type == "rough_ground":
            # Continuous heightfield with gentle undulations.
            # Use coarser grid + extra smoothing to avoid foot-snagging artifacts.
            rng = np.random.default_rng(43 + geom_idx)
            res = 0.10  # 10cm grid (coarser = fewer sharp edges for contacts)
            ncol = max(2, int(seg_len / res))
            nrow = max(2, int((2 * half_y) / res))
            h_raw = rng.uniform(-1.0, 1.0, (nrow, ncol))
            # Two passes of 3x3 box filter for a smoother surface
            h_smooth = h_raw.astype(np.float64)
            for _ in range(2):
                h_pad = np.pad(h_smooth, 1, mode="edge")
                h_smooth = (
                    h_pad[0:-2, 0:-2] + h_pad[0:-2, 1:-1] + h_pad[0:-2, 2:]
                    + h_pad[1:-1, 0:-2] + h_pad[1:-1, 1:-1] + h_pad[1:-1, 2:]
                    + h_pad[2:, 0:-2] + h_pad[2:, 1:-1] + h_pad[2:, 2:]
                ) / 9.0
            amp = max(0.002, min(0.005, noise_amp * 0.4)) if noise_amp > 0 else 0.003
            h_min, h_max = h_smooth.min(), h_smooth.max()
            heights_phys = amp * (h_smooth - h_min) / (h_max - h_min + 1e-8)
            elev_range = float(np.ptp(heights_phys))
            elev_range = max(elev_range, 0.001)
            hf_base_thickness = 0.05
            heights_norm = (heights_phys / elev_range).astype(np.float32)
            hf_name = f"rough_ground_{geom_idx}"
            hf_file = xml_dir / f"{hf_name}.bin"
            with open(hf_file, "wb") as f:
                f.write(np.array([nrow, ncol], dtype=np.int32).tobytes())
                f.write(heights_norm.flatten(order="C").tobytes())
            half_x, half_y_hf = seg_len / 2.0, half_y
            hf_el = ET.SubElement(asset_el, "hfield")
            hf_el.set("name", hf_name)
            hf_el.set("nrow", str(nrow))
            hf_el.set("ncol", str(ncol))
            hf_el.set("size", f"{half_x} {half_y_hf} {elev_range} {hf_base_thickness}")
            hf_el.set("file", str(hf_file.resolve()))
            geom_hf = ET.SubElement(wb, "geom")
            geom_hf.set("name", f"course_{geom_idx}")
            geom_hf.set("type", "hfield")
            geom_hf.set("hfield", hf_name)
            hf_z = cur_h - hf_base_thickness
            geom_hf.set("pos", f"{cur_x + seg_len / 2} 0 {hf_z}")
            geom_hf.set("friction", "0.8 0.005 0.0001")
            geom_hf.set("contype", "1")
            geom_hf.set("conaffinity", "1")
            geom_hf.set("rgba", "0.45 0.45 0.45 1")
            geom_idx += 1
            cur_x += seg_len

        else:
            cur_x += seg_len

    # Spawn at x=0 (centre).  The flat segment is designed to cover x=0.
    # Ground plane is at z=0, so robot spawns on it.
    spawn_z = 0.5

    # Write temporary XML in same directory as source so relative includes/meshdir work.
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="course_terrain_", dir=str(xml_dir))
    os.close(fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    print(f"[Terrain] Generated course XML with {geom_idx} geoms -> {tmp_path}")
    return Path(tmp_path), spawn_z


def _setup_terrain(
    simulator: Any,
    task: Any,
) -> float:
    """Generate and inject terrain heightfield data into the simulator model.

    Returns the recommended spawn z-offset for the floating base.
    """
    from unitree_lab.mujoco_utils.terrain.generator import MujocoTerrainGenerator
    from unitree_lab.mujoco_utils.terrain.xml_generation import setup_terrain_data_in_model

    terrain = getattr(task, "terrain", None)
    base_cfg = _terrain_dict_to_gen_cfg(terrain if isinstance(terrain, dict) else None)
    if base_cfg is None:
        print(
            "[Warning] Terrain layout not supported for runtime heightfield injection "
            "(e.g. RandomGrid / mixed); using XML heightfield as-is."
        )
        return 0.0
    terrain_type = base_cfg.get("terrain_type", "flat")

    if terrain_type == "flat":
        return 0.0

    # Build TerrainConfig from the task's terrain params, matching the XML heightfield dimensions.
    hfield_name = "terrain_hfield"
    import mujoco as mj

    hfield_id = mj.mj_name2id(simulator.model, mj.mjtObj.mjOBJ_HFIELD, hfield_name)
    if hfield_id < 0:
        print("[Warning] No heightfield found in model; skipping terrain injection.")
        return 0.0

    nrow = int(simulator.model.hfield_nrow[hfield_id])
    ncol = int(simulator.model.hfield_ncol[hfield_id])
    sx = float(simulator.model.hfield_size[hfield_id, 0])
    sy = float(simulator.model.hfield_size[hfield_id, 1])
    world_x = sx * 2
    world_y = sy * 2

    gen_cfg = {
        "terrain_type": terrain_type,
        "size": (world_x, world_y),
        "horizontal_scale": world_x / max(1, ncol - 1),
    }
    for k, v in base_cfg.items():
        if k != "terrain_type":
            gen_cfg[k] = v

    generator = MujocoTerrainGenerator(gen_cfg)
    generator.nx = ncol
    generator.ny = nrow
    generator.generate()

    # Record min height before injection (injection shifts min to 0).
    hf = generator.heightfield
    min_h = float(np.min(hf)) if hf is not None and hf.size else 0.0
    shift = -min_h if min_h < 0.0 else 0.0

    setup_terrain_data_in_model(simulator.model, generator, hfield_name=hfield_name)

    # spawn_z uses the pre-shift heightfield; correct for the shift applied during injection.
    raw_spawn_z = generator.get_spawn_height(0.0, 0.0)
    spawn_z = raw_spawn_z + shift
    return max(0.0, spawn_z)


class KeyboardTeleop:
    """Keyboard velocity command controller for the MuJoCo viewer.

    Uses arrow keys / Page keys (GLFW key codes) to avoid conflicts with
    MuJoCo viewer's built-in shortcuts (W=wireframe, S=shadow, A=autoconnect,
    D=static body, Q=camera, E=equality — all toggle rendering flags).

    Key bindings:
      UP/DOWN      -> vx (forward/backward)
      LEFT/RIGHT   -> wz (yaw left/right)
      PgUp/PgDn    -> vy (lateral left/right)
      Keypad +/−   -> spawn z offset +0.5 / −0.5 (per press)
      Backspace    -> zero velocity (vx, vy, wz)

    IMPORTANT: ``handle_key`` is invoked from the GLFW render thread.
    All I/O (print, logging) is deferred to the main simulation thread via
    ``_pending_msg`` to avoid stdout contention / GIL deadlocks that freeze
    the terminal.
    """

    VX_STEP = 0.1
    VY_STEP = 0.1
    WZ_STEP = 0.5
    VX_RANGE = (-2.0, 2.0)
    VY_RANGE = (-0.0, 0.0)
    WZ_RANGE = (-2.0, 2.0)

    # GLFW key codes — arrow keys + page keys don't conflict with MuJoCo viewer
    _KEY_UP = 265
    _KEY_DOWN = 264
    _KEY_LEFT = 263
    _KEY_RIGHT = 262
    _KEY_PAGE_UP = 266
    _KEY_PAGE_DOWN = 267
    _KEY_BACKSPACE = 259
    _KEY_KP_ADD = 334
    _KEY_KP_SUBTRACT = 333

    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._dirty = False
        self._pending_msg: str | None = None

    def handle_key(self, key: int) -> None:
        """Handle a key press (GLFW key code).

        Called from the GLFW render thread – must NOT do any I/O here.
        """
        if key == self._KEY_UP:
            self.vx = min(self.vx + self.VX_STEP, self.VX_RANGE[1])
        elif key == self._KEY_DOWN:
            self.vx = max(self.vx - self.VX_STEP, self.VX_RANGE[0])
        elif key == self._KEY_PAGE_UP:
            self.vy = min(self.vy + self.VY_STEP, self.VY_RANGE[1])
        elif key == self._KEY_PAGE_DOWN:
            self.vy = max(self.vy - self.VY_STEP, self.VY_RANGE[0])
        elif key == self._KEY_LEFT:
            self.wz = min(self.wz + self.WZ_STEP, self.WZ_RANGE[1])
        elif key == self._KEY_RIGHT:
            self.wz = max(self.wz - self.WZ_STEP, self.WZ_RANGE[0])
        elif key == self._KEY_BACKSPACE:
            self.vx = self.vy = self.wz = 0.0
        else:
            return

        self._dirty = True
        self._pending_msg = (
            f"[Teleop] vx={self.vx:+.1f}  vy={self.vy:+.1f}  wz={self.wz:+.1f} "
        )

    def flush_msg(self) -> None:
        """Print any pending status message (call from main thread only)."""
        msg = self._pending_msg
        if msg is not None:
            self._pending_msg = None
            print(msg)

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


def _build_overlay_text(
    teleop: KeyboardTeleop | None,
    simulator: Any,
    step: int,
    fps: float = 0.0,
) -> list[tuple]:
    """Build text overlay tuples for viewer.set_texts."""
    import mujoco

    texts = []

    # Velocity command overlay (bottom-left)
    if teleop is not None:
        vx, vy, wz = teleop.command
        cmd_left = "Velocity Cmd\nvx\nvy\nwz"
        cmd_right = f"(Arrow/PgUp/PgDn)\n{vx:+.2f} m/s\n{vy:+.2f} m/s\n{wz:+.2f} rad/s"
    else:
        vx, vy, wz = simulator.velocity_command
        cmd_left = "Velocity Cmd\nvx\nvy\nwz"
        cmd_right = f"(fixed)\n{vx:+.2f} m/s\n{vy:+.2f} m/s\n{wz:+.2f} rad/s"
    texts.append((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        cmd_left,
        cmd_right,
    ))

    # Step counter + base height + FPS (bottom-right)
    z = simulator.base_pos[2]
    texts.append((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        f"Step\nHeight\nFPS",
        f"{step}\n{z:.3f} m\n{int(fps)}",
    ))

    return texts


def run_interactive(
    simulator: Any,
    task: Any,
    teleop: str = "off",
    follow: bool = False,
    velocity: tuple[float, float, float] | None = None,
    max_steps_per_episode: int | None = None,
    initial_spawn_z: float = 0.0,
    save_video: bool = False,
    output_dir: str = "eval_results",
    video_steps: int = 500,
) -> None:
    """Run interactive sim2sim with MuJoCo viewer.

    Real-time pacing: each policy step targets ``simulator.policy_dt`` of
    wall-clock time so the viewer runs at the correct real-time speed
    (typically 50 Hz for policy_dt=0.02 s).  If the physics computation
    takes longer than policy_dt the viewer will still remain responsive.

    By default (max_steps_per_episode is None) the simulation runs until the robot
    terminates (e.g. falls) or you close the viewer; no time-based reset. Pass a
    positive int to force a reset after that many steps.

    If save_video is True, the session is recorded to output_dir from start until
    the viewer is closed, using the same tracking camera as headless mode.
    """
    import mujoco
    import mujoco.viewer

    teleop_ctrl: KeyboardTeleop | None = None
    key_callback = None
    if teleop == "keyboard":
        teleop_ctrl = KeyboardTeleop()
        if velocity:
            teleop_ctrl.vx, teleop_ctrl.vy, teleop_ctrl.wz = velocity
        key_callback = teleop_ctrl.handle_key
        print("[Teleop] Keyboard: UP/DOWN=vx, PgUp/PgDn=vy, LEFT/RIGHT=wz, Keypad+/-=z+0.5, Backspace=zero vel")

    if velocity and teleop != "keyboard":
        simulator.set_velocity_command(*velocity)
    elif teleop != "keyboard" and hasattr(task, "get_velocity_command"):
        v0 = _task_ref_velocity_tuple(task)
        simulator.set_velocity_command(*v0)

    viewer = mujoco.viewer.launch_passive(
        simulator.model, simulator.data, key_callback=key_callback,
    )

    policy_dt = getattr(simulator, "policy_dt", 0.02)
    target_fps = int(round(1.0 / policy_dt)) if policy_dt > 0 else 50
    print(f"[Sim2Sim] Interactive mode: target FPS = {target_fps} (policy_dt={policy_dt:.4f}s)")

    _fps_window: list[float] = []
    _FPS_WINDOW_SIZE = 50
    _fps_display = 0
    _fps_print_interval = 100

    # Optional video recording from start until viewer is closed (same tracking camera as headless)
    _video_frames: list[np.ndarray] = []

    try:
        while viewer.is_running():
            simulator.spawn_root_z_offset = initial_spawn_z
            simulator.reset()
            if teleop_ctrl is not None:
                simulator.set_velocity_command(*teleop_ctrl.command)
            elif velocity:
                simulator.set_velocity_command(*velocity)
            elif hasattr(task, "get_velocity_command"):
                v0 = task.get_velocity_command(0.0)
                simulator.set_velocity_command(float(v0[0]), float(v0[1]), float(v0[2]))

            _prev_time = time.monotonic()
            _fps_window.clear()
            _step = 0

            while True:
                if not viewer.is_running():
                    break

                if teleop_ctrl is not None and teleop_ctrl._dirty:
                    simulator.set_velocity_command(*teleop_ctrl.command)
                    teleop_ctrl._dirty = False
                    teleop_ctrl.flush_msg()
                elif teleop_ctrl is None and velocity is None and hasattr(task, "get_velocity_command"):
                    t_cmd = _step * policy_dt
                    vt = task.get_velocity_command(t_cmd)
                    simulator.set_velocity_command(float(vt[0]), float(vt[1]), float(vt[2]))

                simulator.step()

                # Capture frame for video (no duration limit; write when viewer closes)
                if save_video:
                    _video_frames.append(_render_one_frame_tracking(simulator))

                if follow:
                    _update_camera_follow(viewer, simulator)

                viewer.set_texts(
                    _build_overlay_text(teleop_ctrl, simulator, _step + 1, fps=_fps_display)
                )
                viewer.sync()

                if simulator._check_termination():
                    print(f"[Sim2Sim] Terminated at step {_step + 1}, FPS: {_fps_display}, resetting...")
                    time.sleep(0.3)
                    break

                # Optional: reset after a fixed number of steps (when max_steps_per_episode is set)
                if max_steps_per_episode is not None and max_steps_per_episode > 0 and _step + 1 >= max_steps_per_episode:
                    print(f"[Sim2Sim] Reached {max_steps_per_episode} steps, resetting...")
                    break

                # Real-time pacing: sleep remainder of policy_dt budget
                now = time.monotonic()
                compute_elapsed = now - _prev_time
                remaining = policy_dt - compute_elapsed
                if remaining > 0:
                    time.sleep(remaining)

                # Measure true frame-to-frame interval (includes sleep)
                now = time.monotonic()
                frame_dt = now - _prev_time
                _prev_time = now

                _fps_window.append(frame_dt)
                if len(_fps_window) > _FPS_WINDOW_SIZE:
                    _fps_window.pop(0)
                _fps_display = int(round(len(_fps_window) / sum(_fps_window)))

                if _step > 0 and _step % _fps_print_interval == 0:
                    print(f"[Sim2Sim] Step {_step:5d}  FPS: {_fps_display} (target: {target_fps})")

                _step += 1
    except KeyboardInterrupt:
        print("\n[Sim2Sim] Interrupted by user.")
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        if save_video and _video_frames:
            print(f"[Sim2Sim] Writing {len(_video_frames)} frames to video...")
            _write_frames_to_video(
                _video_frames, task, output_dir, simulator.policy_dt,
            )


def _update_camera_follow(viewer: Any, simulator: Any) -> None:
    """Update viewer camera to follow the robot base."""
    try:
        pos = simulator.base_pos
        cam = viewer.cam
        cam.lookat[:] = pos
    except Exception:
        pass


def run_headless(
    simulator: Any,
    task: Any,
    num_episodes: int = 10,
    save_video: bool = False,
    output_dir: str = "eval_results",
    video_steps: int = 500,
    velocity: tuple[float, float, float] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run headless evaluation episodes."""
    from unitree_lab.mujoco_utils.evaluation.metrics import (
        compute_locomotion_metrics,
        print_metrics,
    )

    policy_dt = float(getattr(simulator, "policy_dt", 0.02))
    n_steps = max_steps
    if n_steps is None and hasattr(task, "duration"):
        n_steps = max(1, int(np.ceil(float(task.duration) / policy_dt)))
    elif n_steps is None:
        n_steps = 1000

    vel_cmd = velocity if velocity is not None else _task_ref_velocity_tuple(task)
    episode_results = []

    for ep_idx in range(num_episodes):
        print(f"  Episode {ep_idx + 1}/{num_episodes}", end="\r", flush=True)
        result = simulator.run_episode(
            max_steps=n_steps,
            render=False,
            velocity_command=vel_cmd,
        )
        episode_results.append(result)

    print()

    metrics = compute_locomotion_metrics(
        episode_results, np.array(vel_cmd, dtype=np.float64), simulator.policy_dt
    )
    print_metrics(metrics, task.name)

    summary = {
        "task": task.name,
        "num_episodes": num_episodes,
        "survival_rate": metrics.survival_rate,
        "mean_velocity_error": metrics.mean_velocity_error,
        "mean_forward_distance": metrics.mean_forward_distance,
    }

    if save_video:
        video_path = _record_video_headless(simulator, task, output_dir, video_steps, velocity=velocity)
        if video_path:
            summary["video"] = video_path

    return summary


def _render_one_frame_tracking(simulator: Any) -> np.ndarray:
    """Render one frame with tracking camera and visualization overlays."""
    import mujoco
    width, height = 1280, 720
    if getattr(simulator, "_video_renderer", None) is None:
        if simulator.model.vis.global_.offwidth < width:
            simulator.model.vis.global_.offwidth = width
        if simulator.model.vis.global_.offheight < height:
            simulator.model.vis.global_.offheight = height
        r = mujoco.Renderer(simulator.model, height, width)
        simulator._video_renderer = r
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        base_body_id = getattr(simulator, "_base_body_id", None)
        cam.trackbodyid = base_body_id if (base_body_id is not None and base_body_id > 0) else 1
        cam.distance = 3.0
        cam.azimuth = -150
        cam.elevation = -20
        cam.lookat[:] = [0, 0, 0.8]
        simulator._video_cam = cam
    simulator._video_renderer.update_scene(simulator.data, camera=simulator._video_cam)
    frame = simulator._video_renderer.render()

    try:
        from unitree_lab.mujoco_utils.visualization.panels import create_combined_visualization
        from scipy.spatial.transform import Rotation as Rot

        quat = simulator.base_quat
        r = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
        base_lin_vel = r.inv().apply(simulator.base_lin_vel)
        torque_util = np.zeros(simulator.num_actions, dtype=np.float32)
        cmd = np.array(simulator.velocity_command, dtype=np.float32)
        frame = create_combined_visualization(
            sim_frame=frame,
            torque_util=torque_util,
            num_actions=simulator.num_actions,
            base_lin_vel=base_lin_vel,
            exteroception=None,
            exteroception_type=None,
            robot_z=simulator.base_pos[2],
            sim_time=simulator.data.time,
            command=cmd,
            command_idx=0,
        )
    except ImportError:
        pass

    return frame


def _write_frames_to_video(
    frames: list[np.ndarray],
    task: Any,
    output_dir: str,
    policy_dt: float,
) -> None:
    """Write captured frames to a playable H.264 MP4.

    Pipe raw RGB frames to ffmpeg via stdin — no intermediate file, no codec
    compatibility issues, and works everywhere ffmpeg is installed.
    Falls back to imageio, then cv2+ffmpeg transcode.
    """
    if not frames:
        return
    import subprocess
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_file = out_path / f"{task.name}_sim2sim.mp4"
    fps = int(1.0 / policy_dt) if policy_dt > 0 else 50
    height, width = frames[0].shape[:2]

    # Method 1: pipe raw frames to ffmpeg (best: no temp file, direct H.264)
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                str(video_file),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            print(f"[Sim2Sim] Video saved ({len(frames)} frames): {video_file}")
            return
        print(f"[Warning] ffmpeg pipe exited with code {proc.returncode}")
    except FileNotFoundError:
        print("[Warning] ffmpeg not found, trying imageio fallback")
    except Exception as e:
        print(f"[Warning] ffmpeg pipe failed: {e}")

    # Method 2: imageio
    try:
        import imageio
        writer = imageio.get_writer(str(video_file), fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Sim2Sim] Video saved ({len(frames)} frames): {video_file}")
        return
    except Exception as e:
        print(f"[Warning] imageio failed: {e}")

    # Method 3: cv2 mp4v (may not open in all players)
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[Sim2Sim] Video saved (mp4v fallback, {len(frames)} frames): {video_file}")
    except Exception as e:
        print(f"[Warning] All video writers failed: {e}")


def _record_video_headless(
    simulator: Any,
    task: Any,
    output_dir: str,
    duration_steps: int = 500,
    velocity: tuple[float, float, float] | None = None,
) -> str | None:
    """Record an evaluation video without a viewer window.

    Uses a tracking camera that follows the robot, matching what you see
    in the interactive ``run_interactive`` viewer with ``--follow``.
    """
    try:
        import cv2
        import mujoco
    except ImportError:
        print("[Warning] cv2 or mujoco not available; skipping video recording.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    if simulator.model.vis.global_.offwidth < width:
        simulator.model.vis.global_.offwidth = width
    if simulator.model.vis.global_.offheight < height:
        simulator.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(simulator.model, height, width)

    # Setup a tracking camera that follows the robot base body.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # Track the robot's torso / floating-base body.
    base_body_id = getattr(simulator, "_base_body_id", None)
    if base_body_id is not None and base_body_id > 0:
        cam.trackbodyid = base_body_id
    else:
        cam.trackbodyid = 1
    cam.distance = 3.0
    cam.azimuth = -150
    cam.elevation = -20
    cam.lookat[:] = [0, 0, 0.8]

    simulator.reset()
    policy_dt = float(getattr(simulator, "policy_dt", 0.02))

    try:
        from unitree_lab.mujoco_utils.visualization.panels import create_combined_visualization
        from scipy.spatial.transform import Rotation as Rot
        _has_viz = True
    except ImportError:
        _has_viz = False

    frames: list[np.ndarray] = []
    for step_i in range(duration_steps):
        cmd = None
        if velocity is not None:
            simulator.set_velocity_command(*velocity)
            cmd = np.array(velocity, dtype=np.float32)
        elif hasattr(task, "get_velocity_command"):
            vt = task.get_velocity_command(step_i * policy_dt)
            simulator.set_velocity_command(float(vt[0]), float(vt[1]), float(vt[2]))
            cmd = np.asarray(vt, dtype=np.float32)
        else:
            simulator.set_velocity_command(0.5, 0.0, 0.0)
            cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        renderer.update_scene(simulator.data, camera=cam)
        frame = renderer.render()

        if _has_viz:
            quat = simulator.base_quat
            r = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
            base_lin_vel = r.inv().apply(simulator.base_lin_vel)
            torque_util = np.zeros(simulator.num_actions, dtype=np.float32)
            frame = create_combined_visualization(
                sim_frame=frame,
                torque_util=torque_util,
                num_actions=simulator.num_actions,
                base_lin_vel=base_lin_vel,
                exteroception=None,
                exteroception_type=None,
                robot_z=simulator.base_pos[2],
                sim_time=simulator.data.time,
                command=cmd,
                command_idx=step_i,
            )

        frames.append(frame)
        simulator.step()

    video_file = out_path / f"{task.name}_sim2sim.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(1.0 / simulator.policy_dt) if simulator.policy_dt > 0 else 50
    writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    try:
        import subprocess

        tmp = video_file.with_suffix(".h264_tmp.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_file),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(video_file)
    except Exception:
        pass

    print(f"  Video saved: {video_file}")
    return str(video_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sim2Sim locomotion evaluation in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--robot", type=str, default="g1", choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot name (default: g1)")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX policy file")
    parser.add_argument("--xml", type=str, default=None,
                        help="Path to MuJoCo XML (overrides auto-detection)")
    parser.add_argument("--task", type=str, default="flat_stand",
                        help="Evaluation task name (see --list-tasks)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available evaluation tasks and exit")

    # Rendering & interaction
    parser.add_argument("--render", action="store_true",
                        help="Open interactive MuJoCo viewer")
    parser.add_argument("--follow", action="store_true",
                        help="Camera follows the robot")
    parser.add_argument("--teleop", type=str, default="off", choices=["keyboard", "off"],
                        help="Teleop mode: keyboard or off")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy mode (interactive with auto-reset)")

    # Velocity command override
    parser.add_argument("--velocity", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="Fixed velocity command [vx, vy, wz]")
    parser.add_argument("--skip-terrain-inject", action="store_true",
                        help="Do NOT overwrite heightfield data at runtime; use XML terrain as-is")

    # Deploy config
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml for PD gains/joint config override")
    parser.add_argument("--config-override", type=str, default=None,
                        help="JSON string of config overrides")

    # Headless evaluation
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes (headless)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max episode steps from task")
    parser.add_argument("--save-video", action="store_true",
                        help="Record evaluation video")
    parser.add_argument("--video-steps", type=int, default=500,
                        help="Steps per video recording")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results/videos")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # List tasks
    if args.list_tasks:
        from unitree_lab.mujoco_utils.evaluation.eval_task import list_eval_tasks
        print("Available evaluation tasks:")
        for name in list_eval_tasks():
            print(f"  - {name}")
        return

    # Validate ONNX path
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[Error] ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Get task
    from unitree_lab.mujoco_utils.evaluation.eval_task import TERRAIN_FLAT, get_eval_task

    task = get_eval_task(args.task)
    print(f"\n{'='*60}")
    print(f"  Sim2Sim Locomotion: {task.name}")
    print(f"  {task.description}")
    print(f"{'='*60}\n")

    is_flat_terrain = task.terrain == TERRAIN_FLAT

    # Robot config
    robot_cfg = ROBOT_CONFIGS[args.robot]
    asset_dir = _find_asset_dir()

    if args.xml:
        xml_path = Path(args.xml)
        uses_terrain = (not is_flat_terrain) and ("terrain" in xml_path.name.lower())
    else:
        uses_terrain = not is_flat_terrain
        xml_key = "xml_terrain" if uses_terrain else "xml_flat"
        xml_path = asset_dir / robot_cfg[xml_key]
        if not xml_path.exists():
            print(f"[Warning] Terrain XML not found ({xml_path}), falling back to flat.")
            xml_path = asset_dir / robot_cfg["xml_flat"]

    print(f"[Sim2Sim] Robot: {args.robot}")
    print(f"[Sim2Sim] XML: {xml_path}")
    print(f"[Sim2Sim] ONNX: {onnx_path}")

    # Build config override
    config_override: dict[str, Any] = {}

    if args.deploy_yaml:
        deploy = _load_deploy_yaml(args.deploy_yaml)
        config_override.update(_deploy_yaml_to_config_override(deploy))
        print(f"[Sim2Sim] Loaded deploy.yaml: {args.deploy_yaml}")

    if args.config_override:
        import json
        try:
            extra = json.loads(args.config_override)
            config_override.update(extra)
        except Exception as e:
            print(f"[Warning] Failed to parse --config-override: {e}")

    # For legacy course terrain, generate a temporary XML with box geoms before loading.
    course_spawn_z = 0.0
    _tmp_xml: Path | None = None
    if uses_terrain and getattr(task, "terrain_type", None) == "course" and (not bool(args.skip_terrain_inject)):
        xml_path, course_spawn_z = _generate_course_xml(xml_path, task)
        _tmp_xml = xml_path
        uses_terrain = False  # terrain is already baked into the XML

    # Create simulator
    from unitree_lab.mujoco_utils.simulation.base_simulator import BaseMujocoSimulator

    simulator = BaseMujocoSimulator(
        xml_path=str(xml_path),
        onnx_path=str(onnx_path),
        config_override=config_override if config_override else None,
    )

    if course_spawn_z > 0:
        simulator.spawn_root_z_offset = course_spawn_z
        print(f"[Sim2Sim] Course terrain spawn z-offset: {course_spawn_z:.2f}m")

    # Setup terrain if needed (unless user wants to use XML terrain as-is)
    if uses_terrain and (not bool(args.skip_terrain_inject)):
        spawn_z = _setup_terrain(simulator, task)
        if spawn_z > 0:
            simulator.spawn_root_z_offset = spawn_z
            print(f"[Sim2Sim] Terrain spawn z-offset: {spawn_z:.2f}m")

    # Velocity override
    velocity = tuple(args.velocity) if args.velocity else None

    policy_dt = float(getattr(simulator, "policy_dt", 0.02))
    max_episode_steps = max(1, int(np.ceil(float(task.duration) / policy_dt)))
    if args.max_steps is not None:
        max_episode_steps = int(args.max_steps)

    # Run
    if args.render or args.deploy:
        # Interactive: no step-based reset; only reset on termination or viewer close.
        # Pass max_steps_per_episode=None. Use --max-steps to force a step limit if needed.
        max_steps = getattr(args, "max_steps", None)
        if max_steps is not None:
            max_steps = int(max_steps)

        run_interactive(
            simulator=simulator,
            task=task,
            teleop=args.teleop,
            follow=args.follow,
            velocity=velocity,
            max_steps_per_episode=max_steps,
            initial_spawn_z=getattr(simulator, "spawn_root_z_offset", 0.0),
            save_video=args.save_video,
            output_dir=args.output_dir,
            video_steps=args.video_steps,
        )
    else:
        summary = run_headless(
            simulator=simulator,
            task=task,
            num_episodes=args.num_episodes,
            save_video=args.save_video,
            output_dir=args.output_dir,
            video_steps=args.video_steps,
            velocity=velocity,
            max_steps=max_episode_steps,
        )
        print(f"\nTask: {summary['task']}")
        print(f"  Episodes:         {summary['num_episodes']}")
        print(f"  Survival rate:    {summary['survival_rate']:.1f}%")
        print(f"  Mean vel error:   {summary['mean_velocity_error']:.3f} m/s")
        print(f"  Mean fwd dist:    {summary['mean_forward_distance']:.2f} m")
        if "video" in summary:
            print(f"  Video:            {summary['video']}")

    # Clean up temporary XML and heightfield bin files.
    if _tmp_xml is not None:
        try:
            xml_dir = _tmp_xml.parent
            _tmp_xml.unlink(missing_ok=True)
            for f in xml_dir.glob("rough_ground_*.bin"):
                f.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
