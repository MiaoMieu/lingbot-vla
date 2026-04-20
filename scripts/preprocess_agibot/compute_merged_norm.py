#!/usr/bin/env python3
"""
Compute merged norm stats (sliced) across all Agibot sub-datasets.

Reads info.json from the first sub-dataset to determine slice indices,
applies the same slicing as AgibotDataset training code, and outputs
norm stats with only the effective dimensions (e.g. 16-D state, 16-D action).

Usage:
    python scripts/preprocess_agibot/compute_merged_norm.py \
        --data_root /mnt/zhaonanshu/train_data/instruction \
        --output assets/norm_stats/instruction_merged.json

    # With waist joint:
    python scripts/preprocess_agibot/compute_merged_norm.py \
        --data_root /mnt/zhaonanshu/train_data/instruction \
        --output assets/norm_stats/instruction_merged_waist.json \
        --use_waist
"""
import argparse
import json
import os
import pathlib
import sys
from glob import glob

import numpy as np
import pyarrow.parquet as pq

NUM_BINS = 10000


def parse_agibot_slices(info: dict, feature_key: str, use_waist: bool = True) -> list[tuple[int, int]]:
    """Identical logic to _parse_agibot_slices in base_dataset.py."""
    descs = info["features"][feature_key].get("field_descriptions", {})

    def _indices(suffix: str) -> list[int]:
        for name, desc in descs.items():
            if name.endswith(suffix):
                return desc.get("indices", [])
        return []

    slices: list[tuple[int, int]] = []

    joint_idx = _indices("/joint/position")
    if len(joint_idx) >= 14:
        slices.append((joint_idx[0], joint_idx[6] + 1))
        slices.append((joint_idx[7], joint_idx[13] + 1))

    for side in ("left", "right"):
        eff_idx = _indices(f"/{side}_effector/position")
        if eff_idx:
            slices.append((eff_idx[0], eff_idx[-1] + 1))

    if use_waist:
        waist_idx = _indices("/waist/position")
        if len(waist_idx) >= 5:
            slices.append((waist_idx[4], waist_idx[4] + 1))

    return slices


def slices_to_indices(slices: list[tuple[int, int]]) -> list[int]:
    indices = []
    for s, e in slices:
        indices.extend(range(s, e))
    return indices


def apply_slices(data: np.ndarray, indices: list[int]) -> np.ndarray:
    """Select columns from (N, D) array by index list -> (N, len(indices))."""
    return data[:, indices]


def main():
    parser = argparse.ArgumentParser(
        description="Compute merged & sliced norm stats for Agibot datasets",
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing sub-dataset folders")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--use_waist", action="store_true", default=False,
                        help="Include waist joint in slicing (adds 1 extra dim)")
    args = parser.parse_args()

    sub_dirs = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    if not sub_dirs:
        print(f"ERROR: no sub-directories found under {args.data_root}")
        sys.exit(1)

    info_path = None
    for sd in sub_dirs:
        candidate = os.path.join(args.data_root, sd, "meta", "info.json")
        if os.path.isfile(candidate):
            info_path = candidate
            break
    if info_path is None:
        print(f"ERROR: no meta/info.json found in any sub-directory under {args.data_root}")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)
    state_slices = parse_agibot_slices(info, "observation.state", use_waist=args.use_waist)
    action_slices = parse_agibot_slices(info, "action", use_waist=args.use_waist)
    state_indices = slices_to_indices(state_slices)
    action_indices = slices_to_indices(action_slices)

    print(f"Loaded slice config from: {info_path}")
    print(f"  state slices: {state_slices} -> {len(state_indices)}-D")
    print(f"  action slices: {action_slices} -> {len(action_indices)}-D")
    print(f"  use_waist: {args.use_waist}")

    all_parquet_files = sorted(
        glob(os.path.join(args.data_root, "**", "data", "**", "*.parquet"), recursive=True)
    )
    if not all_parquet_files:
        all_parquet_files = sorted(
            glob(os.path.join(args.data_root, "**", "*.parquet"), recursive=True)
        )
    if not all_parquet_files:
        print(f"ERROR: no parquet files found under {args.data_root}")
        sys.exit(1)

    dir_counts = {}
    for pf in all_parquet_files:
        rel = os.path.relpath(pf, args.data_root)
        top = rel.split(os.sep)[0]
        dir_counts[top] = dir_counts.get(top, 0) + 1
    print(f"\nFound {len(all_parquet_files)} parquet files across {len(dir_counts)} sub-datasets:")
    for sd_name in sorted(dir_counts):
        print(f"  {sd_name}: {dir_counts[sd_name]} files")

    count = 0
    s_mean = s_mos = s_min = s_max = None
    a_mean = a_mos = a_min = a_max = None

    print("\nPass 1/2: computing global statistics (sliced) ...")
    for i, pf in enumerate(all_parquet_files):
        if i % 50 == 0:
            print(f"  [{i+1}/{len(all_parquet_files)}] {os.path.relpath(pf, args.data_root)}")
        table = pq.read_table(pf)
        cols = table.column_names
        if "observation.state" not in cols or "action" not in cols:
            print(f"  SKIP (missing columns): {pf}")
            continue
        state_raw = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        action_raw = np.stack(table.column("action").to_pylist()).astype(np.float64)
        state = apply_slices(state_raw, state_indices)
        action = apply_slices(action_raw, action_indices)
        n = len(state)
        if count == 0:
            s_mean, s_mos = np.mean(state, 0), np.mean(state ** 2, 0)
            s_min, s_max = np.min(state, 0), np.max(state, 0)
            a_mean, a_mos = np.mean(action, 0), np.mean(action ** 2, 0)
            a_min, a_max = np.min(action, 0), np.max(action, 0)
        else:
            total = count + n
            s_mean = (count * s_mean + n * np.mean(state, 0)) / total
            s_mos = (count * s_mos + n * np.mean(state ** 2, 0)) / total
            s_min = np.minimum(s_min, np.min(state, 0))
            s_max = np.maximum(s_max, np.max(state, 0))
            a_mean = (count * a_mean + n * np.mean(action, 0)) / total
            a_mos = (count * a_mos + n * np.mean(action ** 2, 0)) / total
            a_min = np.minimum(a_min, np.min(action, 0))
            a_max = np.maximum(a_max, np.max(action, 0))
        count += n

    if count == 0:
        print("ERROR: no valid samples found")
        sys.exit(1)

    sd_dim, ad_dim = len(s_mean), len(a_mean)
    print(f"\nTotal samples: {count}, sliced state_dim: {sd_dim}, sliced action_dim: {ad_dim}")

    s_edges = [np.linspace(s_min[i] - 1e-10, s_max[i] + 1e-10, NUM_BINS + 1) for i in range(sd_dim)]
    a_edges = [np.linspace(a_min[i] - 1e-10, a_max[i] + 1e-10, NUM_BINS + 1) for i in range(ad_dim)]
    s_hist = [np.zeros(NUM_BINS) for _ in range(sd_dim)]
    a_hist = [np.zeros(NUM_BINS) for _ in range(ad_dim)]

    print("Pass 2/2: building histograms (sliced) ...")
    for i, pf in enumerate(all_parquet_files):
        if i % 50 == 0:
            print(f"  [{i+1}/{len(all_parquet_files)}]")
        table = pq.read_table(pf)
        cols = table.column_names
        if "observation.state" not in cols or "action" not in cols:
            continue
        state_raw = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        action_raw = np.stack(table.column("action").to_pylist()).astype(np.float64)
        state = apply_slices(state_raw, state_indices)
        action = apply_slices(action_raw, action_indices)
        for j in range(sd_dim):
            h, _ = np.histogram(state[:, j], bins=s_edges[j])
            s_hist[j] += h
        for j in range(ad_dim):
            h, _ = np.histogram(action[:, j], bins=a_edges[j])
            a_hist[j] += h

    def _quantile(histograms, edges, q, total_count):
        target = q * total_count
        vals = []
        for hist, edge in zip(histograms, edges):
            idx = np.searchsorted(np.cumsum(hist), target)
            vals.append(edge[min(idx, len(edge) - 1)])
        return np.array(vals)

    std_s = np.sqrt(np.maximum(s_mos - s_mean ** 2, 0))
    std_a = np.sqrt(np.maximum(a_mos - a_mean ** 2, 0))

    norm_stats = {
        "norm_stats": {
            "observation.state": {
                "mean": s_mean.tolist(),
                "std": std_s.tolist(),
                "q01": _quantile(s_hist, s_edges, 0.01, count).tolist(),
                "q99": _quantile(s_hist, s_edges, 0.99, count).tolist(),
                "q02": _quantile(s_hist, s_edges, 0.02, count).tolist(),
                "q98": _quantile(s_hist, s_edges, 0.98, count).tolist(),
            },
            "action": {
                "mean": a_mean.tolist(),
                "std": std_a.tolist(),
                "q01": _quantile(a_hist, a_edges, 0.01, count).tolist(),
                "q99": _quantile(a_hist, a_edges, 0.99, count).tolist(),
                "q02": _quantile(a_hist, a_edges, 0.02, count).tolist(),
                "q98": _quantile(a_hist, a_edges, 0.98, count).tolist(),
            },
        },
        "sliced": True,
        "use_waist": args.use_waist,
        "state_dim": sd_dim,
        "action_dim": ad_dim,
        "count": count,
    }

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\nSaved: {args.output} ({count} samples, state_dim={sd_dim}, action_dim={ad_dim})")


if __name__ == "__main__":
    main()
