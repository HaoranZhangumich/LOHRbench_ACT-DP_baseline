#!/usr/bin/env python3
"""
Rechunk LoHRbench HDF5 files for fast random-access image reads.
================================================================

The original HDF5 files use large chunk sizes (likely entire trajectories)
with gzip compression. Reading a single 48 KB image decompresses megabytes
of data, causing ~18 ms/image reads (97.7% of DataLoader time).

This script rewrites the files with:
  - Per-image chunk shape: (1, H, W, 3)
  - Fast compression: lzf (~10-20x faster decompression than gzip)
  - Same internal structure (drop-in replacement)

Expected: single-image reads drop from ~18 ms to ~0.5-2 ms.

Usage:
  python preprocess_lohrbench.py \
    --data-root /path/to/LoHRbench \
    --output-root /path/to/LoHRbench_rechunked \
    --compression lzf
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from typing import List, Optional, Tuple

import h5py
import numpy as np
from tqdm.auto import tqdm

from train_lohrbench import (
    TASK_TYPES,
    REQUIRED_H5_SUFFIX,
    discover_lohrbench_h5_files,
)

# Datasets that contain image data and should be rechunked
IMAGE_DATASET_PATHS = [
    "obs/sensor_data/base_camera/rgb",
    "obs/sensor_data/hand_camera/rgb",
]


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def inspect_dataset(ds: h5py.Dataset) -> dict:
    """Return chunk/compression info for an HDF5 dataset."""
    info = {
        "shape": ds.shape,
        "dtype": ds.dtype,
        "chunks": ds.chunks,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "nbytes": ds.nbytes,
        "id_storage_size": ds.id.get_storage_size(),
    }
    return info


def rechunk_dataset(
    src_ds: h5py.Dataset,
    dst_group: h5py.Group,
    ds_name: str,
    is_image: bool,
    compression: Optional[str],
    batch_size: int = 64,
):
    """Copy a dataset with new chunking and compression.

    For image datasets: chunks=(1, H, W, C), fast compression.
    For other datasets: copy as-is (small, not worth rechunking).
    """
    shape = src_ds.shape
    dtype = src_ds.dtype

    if is_image and len(shape) == 4:
        # Image dataset: (T, H, W, C) -> chunks per single image
        chunks = (1,) + shape[1:]
        dst_ds = dst_group.create_dataset(
            ds_name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )
        # Copy in batches to avoid loading entire trajectory at once
        T = shape[0]
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            dst_ds[start:end] = src_ds[start:end]
    else:
        # Non-image dataset: copy in batches if large, else direct copy
        if src_ds.nbytes > 50 * 1024 * 1024 and len(shape) >= 1:
            # Large dataset (>50 MB): copy in batches along first axis
            dst_ds = dst_group.create_dataset(
                ds_name, shape=shape, dtype=dtype,
            )
            T = shape[0]
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                dst_ds[start:end] = src_ds[start:end]
        else:
            data = src_ds[:]
            dst_group.create_dataset(ds_name, data=data)


def _resolve_relative_path(h5_path: str, data_root: str) -> str:
    """Get the path of h5_path relative to data_root."""
    return os.path.relpath(h5_path, data_root)


def rechunk_file(
    src_path: str,
    dst_path: str,
    compression: Optional[str],
    batch_size: int,
) -> Tuple[int, int]:
    """Rechunk a single HDF5 file. Returns (src_size, dst_size)."""
    src_size = os.path.getsize(src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    inspected_first = False

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy top-level attributes
        for attr_name, attr_val in src.attrs.items():
            dst.attrs[attr_name] = attr_val

        traj_keys = sorted([k for k in src.keys() if k.startswith("traj_")])

        pbar = tqdm(traj_keys, desc=f"  trajs", leave=False, dynamic_ncols=True)
        for traj_key in pbar:
            src_traj = src[traj_key]
            dst_traj = dst.create_group(traj_key)

            # Copy trajectory-level attributes
            for attr_name, attr_val in src_traj.attrs.items():
                dst_traj.attrs[attr_name] = attr_val

            # Recursively process all datasets in this trajectory
            _copy_group_recursive(
                src_traj, dst_traj, traj_key,
                compression=compression,
                batch_size=batch_size,
                inspected_first=inspected_first,
            )
            inspected_first = True

        # Copy any non-traj top-level items
        for key in src.keys():
            if not key.startswith("traj_") and key not in dst:
                src.copy(src[key], dst, name=key)

    dst_size = os.path.getsize(dst_path)
    return src_size, dst_size


def _copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    traj_key: str,
    compression: Optional[str],
    batch_size: int,
    inspected_first: bool,
    path_prefix: str = "",
):
    """Recursively copy all items in an HDF5 group."""
    for name in src_group.keys():
        item = src_group[name]
        current_path = f"{path_prefix}/{name}" if path_prefix else name

        if isinstance(item, h5py.Group):
            sub_group = dst_group.create_group(name)
            # Copy group attributes
            for attr_name, attr_val in item.attrs.items():
                sub_group.attrs[attr_name] = attr_val
            _copy_group_recursive(
                item, sub_group, traj_key,
                compression=compression,
                batch_size=batch_size,
                inspected_first=inspected_first,
                path_prefix=current_path,
            )
        elif isinstance(item, h5py.Dataset):
            # Check if this is an image dataset by matching the full path
            is_image = any(current_path == ip or current_path.endswith(ip)
                          for ip in IMAGE_DATASET_PATHS)

            # Print diagnostic for first trajectory
            if not inspected_first and is_image:
                info = inspect_dataset(item)
                print(f"\n  [DIAG] {traj_key}/{current_path}:")
                print(f"    shape={info['shape']}, dtype={info['dtype']}")
                print(f"    chunks={info['chunks']}, compression={info['compression']}"
                      f" (opts={info['compression_opts']})")
                print(f"    raw={fmt_bytes(info['nbytes'])}, "
                      f"stored={fmt_bytes(info['id_storage_size'])}, "
                      f"ratio={info['nbytes'] / max(info['id_storage_size'], 1):.1f}x")

            rechunk_dataset(
                item, dst_group, name,
                is_image=is_image,
                compression=compression,
                batch_size=batch_size,
            )

            # Copy dataset attributes
            for attr_name, attr_val in item.attrs.items():
                dst_group[name].attrs[attr_name] = attr_val


def verify_file(src_path: str, dst_path: str, num_checks: int = 3) -> bool:
    """Spot-check that images in dst match src exactly. Returns True if all match."""
    ok = True
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "r") as dst:
        traj_keys = sorted([k for k in src.keys() if k.startswith("traj_")])
        if not traj_keys:
            return True

        # Pick random trajectories to check
        check_keys = random.sample(traj_keys, min(num_checks, len(traj_keys)))

        for traj_key in check_keys:
            for img_path in IMAGE_DATASET_PATHS:
                try:
                    src_ds = src[traj_key][img_path]
                    dst_ds = dst[traj_key][img_path]
                except KeyError:
                    continue

                if src_ds.shape != dst_ds.shape:
                    print(f"  [VERIFY FAIL] {traj_key}/{img_path}: shape mismatch "
                          f"{src_ds.shape} vs {dst_ds.shape}")
                    ok = False
                    continue

                if src_ds.dtype != dst_ds.dtype:
                    print(f"  [VERIFY FAIL] {traj_key}/{img_path}: dtype mismatch "
                          f"{src_ds.dtype} vs {dst_ds.dtype}")
                    ok = False
                    continue

                T = src_ds.shape[0]
                # Check first, last, and a random frame
                check_indices = [0, T - 1]
                if T > 2:
                    check_indices.append(random.randint(1, T - 2))

                for idx in check_indices:
                    src_img = src_ds[idx]
                    dst_img = dst_ds[idx]
                    if not np.array_equal(src_img, dst_img):
                        print(f"  [VERIFY FAIL] {traj_key}/{img_path}[{idx}]: "
                              f"pixel mismatch! max_diff={np.max(np.abs(src_img.astype(int) - dst_img.astype(int)))}")
                        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk LoHRbench HDF5 files for fast random-access reads"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Original dataset root")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Output directory for rechunked files")
    parser.add_argument("--compression", type=str, default="lzf",
                        choices=["lzf", "gzip", "none"],
                        help="Compression for image datasets (default: lzf)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of frames to read/write at a time (memory control)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    compression = args.compression if args.compression != "none" else None

    print("=" * 70)
    print("  LoHRbench HDF5 Rechunker")
    print("=" * 70)
    print(f"  Source:      {args.data_root}")
    print(f"  Output:      {args.output_root}")
    print(f"  Compression: {args.compression}")
    print(f"  Batch size:  {args.batch_size}")
    print()

    # Discover files
    file_list = discover_lohrbench_h5_files(args.data_root)
    if not file_list:
        print(f"ERROR: No HDF5 files found at {args.data_root} ending with '{REQUIRED_H5_SUFFIX}'")
        sys.exit(1)
    print(f"  Found {len(file_list)} H5 files")

    # Calculate total source size
    total_src_size = sum(os.path.getsize(fp) for _, _, fp in file_list)
    print(f"  Total source size: {fmt_bytes(total_src_size)}")

    # Check output disk space
    try:
        os.makedirs(args.output_root, exist_ok=True)
        usage = shutil.disk_usage(args.output_root)
        # Estimate output size: lzf is ~1.5-2x larger than gzip for image data
        est_factor = {"lzf": 2.0, "gzip": 1.0, "none": 4.0}.get(args.compression, 2.0)
        est_output = total_src_size * est_factor
        print(f"  Output disk free:  {fmt_bytes(usage.free)}")
        print(f"  Estimated output:  {fmt_bytes(est_output)} "
              f"(~{est_factor:.1f}x source with {args.compression})")
        if usage.free < est_output * 1.1:
            print(f"\n  WARNING: May not have enough disk space!")
            print(f"  Free: {fmt_bytes(usage.free)}, Need: ~{fmt_bytes(est_output)}")
            print(f"  Proceeding anyway (will fail if disk fills up)...")
    except Exception as e:
        print(f"  Could not check disk space: {e}")

    print()

    # Process each file
    total_src = 0
    total_dst = 0
    t_start = time.time()

    for i, (ttype, task_name, src_path) in enumerate(file_list):
        rel_path = _resolve_relative_path(src_path, args.data_root)
        dst_path = os.path.join(args.output_root, rel_path)

        print(f"\n[{i + 1}/{len(file_list)}] {ttype}/{task_name}")
        print(f"  src: {src_path}")
        print(f"  dst: {dst_path}")

        if os.path.exists(dst_path) and not args.force:
            dst_size = os.path.getsize(dst_path)
            src_size = os.path.getsize(src_path)
            print(f"  SKIP (already exists, {fmt_bytes(dst_size)}). Use --force to overwrite.")
            total_src += src_size
            total_dst += dst_size
            continue

        t0 = time.time()
        src_size, dst_size = rechunk_file(
            src_path, dst_path,
            compression=compression,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        total_src += src_size
        total_dst += dst_size
        ratio = dst_size / max(src_size, 1)
        print(f"  Done in {elapsed:.1f}s: {fmt_bytes(src_size)} -> {fmt_bytes(dst_size)} "
              f"({ratio:.2f}x)")

        # Verify integrity: spot-check random images
        if verify_file(src_path, dst_path):
            print(f"  [VERIFY OK] Spot-checked images match exactly.")
        else:
            print(f"  [VERIFY FAILED] Some images do not match! Check output.")

    total_elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total source:  {fmt_bytes(total_src)}")
    print(f"  Total output:  {fmt_bytes(total_dst)}")
    print(f"  Size ratio:    {total_dst / max(total_src, 1):.2f}x")
    print(f"  Time elapsed:  {total_elapsed:.1f}s")
    print()
    print(f"  Output root: {args.output_root}")
    print(f"  Use with training:")
    print(f"    python train_lohrbench.py --preprocessed-root {args.output_root} ...")
    print()
    print(f"  Verify with benchmark:")
    print(f"    python benchmark_dataloader.py --data-root {args.output_root} --quick")
    print("=" * 70)


if __name__ == "__main__":
    main()
