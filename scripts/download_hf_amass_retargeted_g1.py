#!/usr/bin/env python3
"""Download the full AMASS_Retargeted_for_G1 dataset from Hugging Face.

By default, files are downloaded into:
  source/unitree_lab/unitree_lab/data/AMASS_Retargeted_for_G1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download full Hugging Face dataset: ember-lab-berkeley/AMASS_Retargeted_for_G1"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ember-lab-berkeley/AMASS_Retargeted_for_G1",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="source/unitree_lab/unitree_lab/data",
        help="Parent directory for downloaded dataset.",
    )
    parser.add_argument(
        "--target-subdir",
        type=str,
        default="AMASS_Retargeted_for_G1",
        help="Subdirectory name created under --output-dir.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint or HF dataset page URL (default uses hf-mirror).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token for private/gated datasets (optional).",
    )
    return parser.parse_args()


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    raise ValueError(f"Invalid endpoint/url: {endpoint}")


def main() -> None:
    args = parse_args()
    endpoint = normalize_endpoint(args.endpoint)

    os.environ["HF_ENDPOINT"] = endpoint

    output_root = Path(args.output_dir).resolve()
    target_dir = output_root / args.target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HF] Repo: {args.repo_id}")
    print(f"[HF] Endpoint: {endpoint}")
    print(f"[HF] Downloading to: {target_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        endpoint=endpoint,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=args.token,
    )

    print("[HF] Download completed.")
    print(f"[HF] Files saved under: {target_dir}")


if __name__ == "__main__":
    main()

