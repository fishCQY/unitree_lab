#!/usr/bin/env python3
"""Split existing AMP pkl clips into individual files and extract time segments.

Two jobs:
  1. Split lafan_all_clips.pkl (already 50fps with derived features) into one
     pkl per clip: e.g. walk1_subject1.pkl, dance1_subject1.pkl, ...
  2. For each specified time segment, slice the corresponding clip and save
     as a separate pkl: e.g. walk1_subject1_3.36_39.1.pkl

Also converts any CSV in the LAFAN g1 directory that is NOT already present
in lafan_all_clips.pkl (safety net).

Usage:
    cd unitree_lab/scripts && python extract_lafan_segments.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "source" / "unitree_lab" / "unitree_lab" / "data"
AMP_DIR = DATA_ROOT / "AMP"
ALL_CLIPS_PKL = AMP_DIR / "lafan_all_clips.pkl"
CSV_DIR = DATA_ROOT / "LAFAN1_Retargeting_Dataset" / "g1"
OUTPUT_DIR = AMP_DIR / "individual"

TARGET_FPS = 50

# Segment specs: (clip_name, start_sec, end_sec)
SEGMENTS = [
    ("walk1_subject1",   3.36,   39.1),
    ("walk1_subject1",  81.86,  120.4),
    ("walk1_subject2",  78.13,  132.23),
    ("walk1_subject2", 173.03,  218.2),
    ("run1_subject2",  116.0,   159.0),
    ("run1_subject5",    5.6,    56.6),
]


def load_pickle_compat(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("numpy._core"):
                import numpy.core as nc
                sys.modules.setdefault("numpy._core", nc)
                sys.modules.setdefault("numpy._core.multiarray", nc.multiarray)
                sys.modules.setdefault("numpy._core.numeric", nc.numeric)
                sys.modules.setdefault("numpy._core.umath", nc.umath)
                f.seek(0)
                return pickle.load(f)
            raise


def save_pkl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def slice_clip(clip: dict, start_sec: float, end_sec: float) -> dict:
    """Slice a clip dict by time range (seconds at the clip's fps)."""
    fps = int(clip["fps"])
    s = max(0, int(round(start_sec * fps)))
    e = min(int(round(end_sec * fps)), clip["dof_pos"].shape[0])
    out = {}
    for k, v in clip.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == clip["dof_pos"].shape[0]:
            out[k] = v[s:e].copy()
        else:
            out[k] = v
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ALL_CLIPS_PKL.name} ...")
    all_clips: dict[str, dict] = load_pickle_compat(ALL_CLIPS_PKL)
    print(f"  {len(all_clips)} clips loaded\n")

    # ------------------------------------------------------------------
    # Part 1: split each clip into individual pkl
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Part 1: Split all clips into individual pkl files")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    for clip_name, clip_data in sorted(all_clips.items()):
        out_path = OUTPUT_DIR / f"{clip_name}.pkl"
        save_pkl({clip_name: clip_data}, out_path)
        frames = clip_data["dof_pos"].shape[0]
        dur = frames / float(clip_data["fps"])
        print(f"  {out_path.name:<55s}  {frames:6d} frames  {dur:7.1f}s")

    # ------------------------------------------------------------------
    # Part 2: extract specified time segments
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Part 2: Extract specified time segments")
    print("=" * 70)

    for clip_name, start_sec, end_sec in SEGMENTS:
        if clip_name not in all_clips:
            print(f"  [SKIP] clip '{clip_name}' not found in {ALL_CLIPS_PKL.name}")
            continue

        sliced = slice_clip(all_clips[clip_name], start_sec, end_sec)
        out_name = f"{clip_name}_{start_sec}_{end_sec}.pkl"
        out_path = OUTPUT_DIR / out_name
        save_pkl({clip_name: sliced}, out_path)

        frames = sliced["dof_pos"].shape[0]
        dur = frames / TARGET_FPS
        print(f"  {out_name:<55s}  {frames:6d} frames  {dur:7.1f}s  "
              f"[{start_sec}s - {end_sec}s]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_pkls = sorted(OUTPUT_DIR.glob("*.pkl"))
    print(f"\nDone. {len(all_pkls)} pkl files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
