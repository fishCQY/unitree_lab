# Copyright (c) 2024-2026, unitree_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Batch evaluator for running multiple eval tasks in parallel."""

from __future__ import annotations

import importlib
import importlib.util
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging import logger
from .eval_task import get_eval_task, list_eval_tasks
from .metrics import EvalResult


def _resolve_gl_backend(onnx_path: str) -> str:
    """Pick MuJoCo GL backend for eval workers.

    Priority:
    1) UNITREE_LAB_EVAL_MUJOCO_GL
    2) ONNX metadata auto-detect (depth -> egl)
    3) fallback: osmesa
    """
    forced = os.environ.get("UNITREE_LAB_EVAL_MUJOCO_GL")
    if forced:
        return forced.strip().lower()
    try:
        from ..core.onnx_utils import get_onnx_config
        cfg = get_onnx_config(onnx_path)
        if str(cfg.get("exteroception_type", "")).lower() == "depth":
            return "egl"
    except Exception as exc:
        logger.warning(f"[eval] Failed to inspect ONNX metadata for GL backend: {exc}")
    return "osmesa"


def _resolve_gpu_ids() -> list[int]:
    raw = os.environ.get("UNITREE_LAB_EVAL_GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        ids: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                ids.append(int(token))
            except ValueError:
                logger.warning(f"[eval] Ignoring invalid GPU id token: '{token}'")
        if ids:
            return ids
    return [0]


def _select_mp_context(gl_backend: str) -> mp.context.BaseContext:
    methods = set(mp.get_all_start_methods())
    forced = os.environ.get("UNITREE_LAB_EVAL_MP_START_METHOD")
    if forced:
        method = forced.strip().lower()
        if method in methods:
            return mp.get_context(method)
        logger.warning(f"[eval] Invalid start method '{forced}', falling back to spawn.")
    if sys.platform.startswith("linux") and gl_backend != "egl" and "fork" in methods:
        return mp.get_context("fork")
    if sys.platform.startswith("linux") and "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


def _set_subprocess_env(gl_backend: str, gpu_id: int | None) -> None:
    os.environ["UNITREE_LAB_LIGHTWEIGHT"] = "1"
    os.environ["MUJOCO_GL"] = gl_backend
    os.environ["PYOPENGL_PLATFORM"] = gl_backend
    if gl_backend == "egl" and gpu_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)


@dataclass
class BatchEvalConfig:
    num_workers: int = 16
    task_names: list[str] | None = None
    save_torque_data: bool = False
    timeout_per_task: float = 100.0
    save_mixed_terrain_video: bool = True
    num_worst_videos: int = 2


@dataclass
class BatchEvalResult:
    results: list  # list[EvalResult]
    video_paths: dict[str, str] | None = None

    def get_video_path(self, task_name: str) -> str | None:
        return self.video_paths.get(task_name) if self.video_paths else None

    def to_wandb_dict(self, prefix: str = "sim2sim_eval") -> dict:
        log_dict = {}
        for r in self.results:
            task_prefix = f"{prefix}/{r.task_name}"
            if r.survival_rate is not None:
                log_dict[f"{task_prefix}/survival_rate"] = r.survival_rate
            if r.linear_velocity_error is not None:
                log_dict[f"{task_prefix}/linear_velocity_error"] = r.linear_velocity_error
            if r.angular_velocity_error is not None:
                log_dict[f"{task_prefix}/angular_velocity_error"] = r.angular_velocity_error
        return log_dict

    def summary(self) -> str:
        lines = ["=" * 70, "BATCH EVALUATION RESULTS", "=" * 70]
        for r in sorted(self.results, key=lambda x: x.task_name):
            if r.error:
                lines.append(f"[ERR] {r.task_name}: {r.error[:50]}...")
            else:
                sr = f"{r.survival_rate:.0%}" if r.survival_rate else "N/A"
                lve = f"{r.linear_velocity_error:.3f}" if r.linear_velocity_error else "N/A"
                ave = f"{r.angular_velocity_error:.3f}" if r.angular_velocity_error else "N/A"
                lines.append(
                    f"{r.task_name:30s} | surv: {sr:>4s} | lin_vel_err: {lve:>6s} m/s | ang_vel_err: {ave:>6s} rad/s"
                )
        lines.append("=" * 70)
        return "\n".join(lines)


def _run_single_task(args: tuple) -> EvalResult:
    (
        task_name, task_idx, onnx_path, robot_model_path,
        video_file, torque_data_dir, simulation_fn_path,
        gl_backend, gpu_ids,
    ) = args

    gpu_id = None
    if gl_backend == "egl":
        gpu_id = gpu_ids[task_idx % len(gpu_ids)]
    _set_subprocess_env(gl_backend, gpu_id)

    random.seed()
    np.random.seed()

    render = bool(video_file)

    try:
        eval_task = get_eval_task(task_name)

        if simulation_fn_path:
            module_path, fn_name = simulation_fn_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            mujoco_simulation = getattr(module, fn_name)
        else:
            sim_file = (
                Path(__file__).resolve().parent.parent.parent
                / "tasks" / "locomotion" / "mujoco_eval" / "simulator.py"
            )
            spec = importlib.util.spec_from_file_location("_mj_sim", sim_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mujoco_simulation = mod.run_locomotion_simulation

        result = mujoco_simulation(
            onnx_path=onnx_path,
            mujoco_model_path=robot_model_path,
            eval_task=eval_task,
            render=render,
            headless=True,
            video_file=video_file,
            torque_data_dir=torque_data_dir,
        )
        return result if result else EvalResult.from_error(task_name, "Simulation returned None")
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return EvalResult.from_error(task_name, error_msg)


def _determine_videos_to_keep(results: list, config: BatchEvalConfig) -> list[str]:
    keep_tasks = set()
    if config.save_mixed_terrain_video:
        for r in results:
            if r.task_name == "mixed_terrain" and not r.error:
                keep_tasks.add("mixed_terrain")
                break
    if config.num_worst_videos > 0:
        valid_results = [r for r in results if not r.error and r.survival_rate is not None]
        sorted_results = sorted(
            valid_results,
            key=lambda r: (r.survival_rate, -(r.angular_velocity_error or 0)),
        )
        for r in sorted_results[: config.num_worst_videos]:
            keep_tasks.add(r.task_name)
    return list(keep_tasks)


def _run_batch_headless(
    task_names: list[str],
    onnx_path: str,
    robot_model_path: str,
    torque_data_dir: str,
    config: BatchEvalConfig,
    simulation_fn_path: str | None = None,
) -> list:
    gl_backend = _resolve_gl_backend(onnx_path)
    gpu_ids = _resolve_gpu_ids()
    workers_per_gpu = max(1, int(os.environ.get("UNITREE_LAB_EVAL_WORKERS_PER_GPU", "1")))

    num_workers = min(config.num_workers, len(task_names))
    if gl_backend == "egl":
        max_egl_workers = max(1, len(gpu_ids) * workers_per_gpu)
        num_workers = min(num_workers, max_egl_workers)

    logger.info(f"[eval] backend={gl_backend}, gpu_ids={gpu_ids}, workers={num_workers}")

    start_time = time.time()
    args_list = [
        (name, idx, onnx_path, robot_model_path, "", torque_data_dir,
         simulation_fn_path, gl_backend, gpu_ids)
        for idx, name in enumerate(task_names)
    ]

    prev_lightweight = os.environ.get("UNITREE_LAB_LIGHTWEIGHT")
    prev_mujoco_gl = os.environ.get("MUJOCO_GL")
    prev_pyopengl = os.environ.get("PYOPENGL_PLATFORM")
    os.environ["UNITREE_LAB_LIGHTWEIGHT"] = "1"
    os.environ["MUJOCO_GL"] = gl_backend
    os.environ["PYOPENGL_PLATFORM"] = gl_backend

    results = []
    completed = 0
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=_select_mp_context(gl_backend),
        ) as executor:
            futures = {executor.submit(_run_single_task, args): args[0] for args in args_list}
            completed_futures = set()
            try:
                for future in as_completed(futures, timeout=config.timeout_per_task * len(task_names)):
                    task_name = futures[future]
                    completed_futures.add(future)
                    try:
                        result = future.result(timeout=config.timeout_per_task)
                        results.append(result)
                        sr = f"{result.survival_rate:.0%}" if result.survival_rate else "ERR"
                    except Exception as e:
                        results.append(EvalResult.from_error(task_name, str(e)))
                        sr = "ERR"
                    completed += 1
                    elapsed = time.time() - start_time
                    logger.info(f"[eval] [{completed}/{len(task_names)}] {task_name} surv={sr} ({elapsed:.1f}s)")
            except TimeoutError:
                for future, task_name in futures.items():
                    if future not in completed_futures:
                        results.append(EvalResult.from_error(task_name, "Task timed out"))
                        future.cancel()
    finally:
        if prev_lightweight is None:
            os.environ.pop("UNITREE_LAB_LIGHTWEIGHT", None)
        else:
            os.environ["UNITREE_LAB_LIGHTWEIGHT"] = prev_lightweight
        if prev_mujoco_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = prev_mujoco_gl
        if prev_pyopengl is None:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ["PYOPENGL_PLATFORM"] = prev_pyopengl

    return results


def _run_video_tasks_serial(
    task_names: list[str],
    onnx_path: str,
    robot_model_path: str,
    video_dir: str,
    timeout: float,
    simulation_fn_path: str | None = None,
) -> dict[str, str]:
    video_paths = {}
    gl_backend = _resolve_gl_backend(onnx_path)
    gpu_ids = _resolve_gpu_ids()

    for i, task_name in enumerate(task_names):
        logger.info(f"[video] [{i + 1}/{len(task_names)}] Recording {task_name}...")
        video_file = f"{video_dir}/{task_name}.mp4"
        args = (task_name, i, onnx_path, robot_model_path, video_file, "",
                simulation_fn_path, gl_backend, gpu_ids)
        try:
            result = _run_single_task(args)
            if not result.error and os.path.exists(video_file):
                video_paths[task_name] = video_file
            else:
                logger.warning(f"[video] {task_name} failed: {result.error[:50] if result.error else 'no video'}")
        except Exception as e:
            logger.error(f"[video] {task_name} exception: {e}")

    return video_paths


def run_batch_eval(
    onnx_path: str,
    robot_model_path: str,
    config: BatchEvalConfig | None = None,
    video_dir: str | None = None,
    torque_data_dir: str | None = None,
    simulation_fn_path: str | None = None,
) -> BatchEvalResult:
    """Run batch evaluation with two-phase strategy.

    Phase 1: Run ALL tasks WITHOUT rendering (parallel, fast).
    Phase 2: Re-run SELECTED tasks WITH rendering (serial, stable).
    """
    config = config or BatchEvalConfig()
    task_names = config.task_names or list_eval_tasks()

    effective_torque_dir = (torque_data_dir or "") if config.save_torque_data else ""
    need_videos = video_dir and (config.save_mixed_terrain_video or config.num_worst_videos > 0)

    logger.info(f"Starting batch eval: {len(task_names)} tasks, {config.num_workers} workers")
    start_time = time.time()

    results = _run_batch_headless(
        task_names=task_names,
        onnx_path=onnx_path,
        robot_model_path=robot_model_path,
        torque_data_dir=effective_torque_dir,
        config=config,
        simulation_fn_path=simulation_fn_path,
    )

    video_paths: dict[str, str] = {}
    if need_videos and video_dir:
        video_tasks = _determine_videos_to_keep(results, config)
        if video_tasks:
            os.makedirs(video_dir, exist_ok=True)
            logger.info(f"Recording videos for {len(video_tasks)} tasks: {video_tasks}")
            video_paths = _run_video_tasks_serial(
                task_names=video_tasks,
                onnx_path=onnx_path,
                robot_model_path=robot_model_path,
                video_dir=video_dir,
                timeout=config.timeout_per_task + 60,
                simulation_fn_path=simulation_fn_path,
            )

    elapsed = time.time() - start_time
    logger.info(f"Batch eval completed in {elapsed:.1f}s ({len(results)} tasks)")
    return BatchEvalResult(results=results, video_paths=video_paths or None)
