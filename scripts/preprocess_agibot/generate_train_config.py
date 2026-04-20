#!/usr/bin/env python3
"""
生成 LingBot-VLA 训练任务：计算 merged & sliced norm stats + 生成训练配置 yaml。

用法:
  python3 scripts/preprocess_agibot/generate_train_config.py /mnt/instruction \
      --use-waist --remove-depth

  python3 scripts/preprocess_agibot/generate_train_config.py /mnt/instruction /mnt/other_tasks \
      --no-waist --keep-depth

生成内容:
  1. assets/norm_stats/<config_name>_merged.json  (合并所有子数据集、已 slice 到有效维度)
  2. configs/vla/<config_name>.yaml               (训练配置，可直接用于启动训练)
"""
import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

sys.path.insert(0, SCRIPT_DIR)
from compute_merged_norm import (
    parse_agibot_slices,
    slices_to_indices,
    apply_slices,
)


YAML_TEMPLATE = """\
model:
  model_path: models/Robbyant/lingbot-vla-4b
  tokenizer_path: models/Qwen/Qwen2.5-VL-3B-Instruct
  post_training: true
  adanorm_time: true
  old_adanorm: true

data:
  datasets_type: vla
  data_name: {data_name}
  train_path: {train_path}
  num_workers: 8
  norm_type: bounds_99_woclip
  norm_stats_file: {norm_stats_file}
  use_waist: {use_waist}

train:
  output_dir: output/
  loss_type: L1_fm
  data_parallel_mode: fsdp2
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true
  use_wandb: false
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 48
  action_dim: {action_dim}
  max_action_dim: 75
  max_state_dim: 75
  lr: 1.0e-4
  lr_decay_style: constant
  num_train_epochs: 100
  micro_batch_size: 16
  global_batch_size: 128
  max_steps: 100000
  ckpt_manager: dcp
  save_steps: 100000
  save_epochs: 100
  enable_fp32: true
  enable_resume: true
"""


def discover_datasets(root_dirs):
    datasets = []
    for root in root_dirs:
        root = os.path.abspath(root)
        if os.path.isfile(os.path.join(root, "meta", "info.json")):
            datasets.append(root)
        else:
            for f in sorted(glob.glob(os.path.join(root, "*", "meta", "info.json"))):
                datasets.append(os.path.dirname(os.path.dirname(f)))
    return datasets


def compute_merged_norm(dataset_paths, output_path, use_waist):
    info_path = None
    for ds_path in dataset_paths:
        candidate = os.path.join(ds_path, "meta", "info.json")
        if os.path.isfile(candidate):
            info_path = candidate
            break
    if info_path is None:
        print("    ERROR: no meta/info.json found in any dataset")
        return False

    with open(info_path) as f:
        info = json.load(f)
    state_slices = parse_agibot_slices(info, "observation.state", use_waist=use_waist)
    action_slices = parse_agibot_slices(info, "action", use_waist=use_waist)
    state_indices = slices_to_indices(state_slices)
    action_indices = slices_to_indices(action_slices)
    print(f"  slice config from: {info_path}")
    print(f"  state slices: {state_slices} -> {len(state_indices)}-D")
    print(f"  action slices: {action_slices} -> {len(action_indices)}-D")

    all_parquet_files = []
    for ds_path in dataset_paths:
        all_parquet_files.extend(
            sorted(glob.glob(os.path.join(ds_path, "data", "**", "*.parquet"), recursive=True))
        )
    if not all_parquet_files:
        print("    WARNING: no parquet files found, skipping norm")
        return False

    NUM_BINS = 10000
    count = 0
    s_mean = s_mos = s_min = s_max = None
    a_mean = a_mos = a_min = a_max = None

    print(f"  Pass 1/2: computing global statistics ({len(all_parquet_files)} files) ...")
    for i, pf in enumerate(all_parquet_files):
        if i % 50 == 0:
            print(f"    [{i+1}/{len(all_parquet_files)}]")
        table = pq.read_table(pf)
        cols = table.column_names
        if "observation.state" not in cols or "action" not in cols:
            continue
        state_raw = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
        action_raw = np.stack(table.column("action").to_pylist()).astype(np.float64)
        state = apply_slices(state_raw, state_indices)
        action = apply_slices(action_raw, action_indices)
        n = len(state)
        if count == 0:
            s_mean, s_mos = np.mean(state, 0), np.mean(state**2, 0)
            s_min, s_max = np.min(state, 0), np.max(state, 0)
            a_mean, a_mos = np.mean(action, 0), np.mean(action**2, 0)
            a_min, a_max = np.min(action, 0), np.max(action, 0)
        else:
            total = count + n
            s_mean = (count * s_mean + n * np.mean(state, 0)) / total
            s_mos = (count * s_mos + n * np.mean(state**2, 0)) / total
            s_min = np.minimum(s_min, np.min(state, 0))
            s_max = np.maximum(s_max, np.max(state, 0))
            a_mean = (count * a_mean + n * np.mean(action, 0)) / total
            a_mos = (count * a_mos + n * np.mean(action**2, 0)) / total
            a_min = np.minimum(a_min, np.min(action, 0))
            a_max = np.maximum(a_max, np.max(action, 0))
        count += n

    if count == 0:
        print("    ERROR: no valid samples found")
        return False

    sd_dim, ad_dim = len(s_mean), len(a_mean)
    print(f"  Total samples: {count}, sliced state_dim: {sd_dim}, sliced action_dim: {ad_dim}")

    s_edges = [np.linspace(s_min[i] - 1e-10, s_max[i] + 1e-10, NUM_BINS + 1) for i in range(sd_dim)]
    a_edges = [np.linspace(a_min[i] - 1e-10, a_max[i] + 1e-10, NUM_BINS + 1) for i in range(ad_dim)]
    s_hist = [np.zeros(NUM_BINS) for _ in range(sd_dim)]
    a_hist = [np.zeros(NUM_BINS) for _ in range(ad_dim)]

    print(f"  Pass 2/2: building histograms ...")
    for i, pf in enumerate(all_parquet_files):
        if i % 50 == 0:
            print(f"    [{i+1}/{len(all_parquet_files)}]")
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

    std_s = np.sqrt(np.maximum(s_mos - s_mean**2, 0))
    std_a = np.sqrt(np.maximum(a_mos - a_mean**2, 0))

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
        "use_waist": use_waist,
        "state_dim": sd_dim,
        "action_dim": ad_dim,
        "count": count,
    }
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  norm saved: {output_path} ({count} samples, state_dim={sd_dim}, action_dim={ad_dim})")
    return True, ad_dim


def manage_depth(info_path, remove_depth):
    with open(info_path, "r") as f:
        info = json.load(f)
    features = info["features"]
    depth_keys = [k for k in features if k.startswith("observation.images.") and "depth" in k]
    changed = False

    if remove_depth and depth_keys:
        for k in depth_keys:
            del features[k]
        changed = True
        print(f"    removed depth features: {', '.join(depth_keys)}")
    elif not remove_depth and not depth_keys:
        rgb_keys = [k for k in features if k.startswith("observation.images.") and "depth" not in k]
        fps = info.get("fps", 30.0)
        added = []
        for rgb_key in rgb_keys:
            depth_key = rgb_key + "_depth"
            depth_entry = {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": True,
                    "video.fps": fps,
                    "video.codec": "png",
                    "video.pix_fmt": "gray16be",
                    "has_audio": False,
                },
                "shape": features[rgb_key].get("shape", [256, 256, 3])[:],
                "names": ["height", "width", "channel"],
            }
            features[depth_key] = depth_entry
            added.append(depth_key)
        if added:
            changed = True
            print(f"    added depth features: {', '.join(added)}")

    if changed:
        try:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
        except OSError as e:
            print(f"    WARNING: cannot write {info_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multiple agibot datasets: compute merged sliced norms, manage depth, generate training config"
    )
    parser.add_argument("root_dirs", nargs="+", help="Root directories containing sub-datasets")
    waist_group = parser.add_mutually_exclusive_group()
    waist_group.add_argument("--use-waist", action="store_true", default=True, help="Include waist joint (default)")
    waist_group.add_argument("--no-waist", action="store_true", help="Exclude waist joint")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--remove-depth", action="store_true", default=True, help="Remove depth features from info.json (default)")
    depth_group.add_argument("--keep-depth", action="store_true", help="Keep/add depth features in info.json")
    parser.add_argument("--config-name", default=None, help="Config yaml name (default: first root dir basename)")
    parser.add_argument("--skip-norm", action="store_true", help="Skip norm computation (use existing norm files)")
    args = parser.parse_args()

    use_waist = not args.no_waist
    remove_depth = not args.keep_depth

    datasets = discover_datasets(args.root_dirs)
    if not datasets:
        print("ERROR: No datasets found", file=sys.stderr)
        sys.exit(1)

    config_name = args.config_name or os.path.basename(os.path.abspath(args.root_dirs[0]).rstrip("/"))
    print(f"Config name: {config_name}")
    print(f"Use waist: {use_waist}")
    print(f"Remove depth: {remove_depth}")
    print(f"Found {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  {ds}")
    print()

    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        info_path = os.path.join(ds_path, "meta", "info.json")
        print(f"[{ds_name}] managing depth ...")
        manage_depth(info_path, remove_depth)

    norm_output = os.path.join(PROJECT_ROOT, "assets", "norm_stats", f"{config_name}_merged.json")
    action_dim = None
    if args.skip_norm and os.path.isfile(norm_output):
        print(f"Norm exists, skipping: {norm_output}")
        with open(norm_output) as f:
            existing = json.load(f)
        action_dim = existing.get("action_dim")
    else:
        print(f"\nComputing merged sliced norm across {len(datasets)} dataset(s) ...")
        result = compute_merged_norm(datasets, norm_output, use_waist)
        if not result:
            print("ERROR: norm computation failed", file=sys.stderr)
            sys.exit(1)
        _, action_dim = result

    train_paths = [ds for ds in datasets]
    data_names = ["agibot"] * len(train_paths)

    config_path = os.path.join(PROJECT_ROOT, "configs", "vla", f"{config_name}.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    yaml_content = YAML_TEMPLATE.format(
        data_name=", ".join(data_names),
        train_path=", ".join(train_paths),
        norm_stats_file=norm_output,
        use_waist=str(use_waist).lower(),
        action_dim=action_dim or 17,
    )

    with open(config_path, "w") as f:
        f.write(yaml_content)
    print(f"\nConfig saved: {config_path}")
    print(f"Datasets: {len(train_paths)}")
    print(f"Action dim: {action_dim}")
    print(f"Norm file: {norm_output}")
    print()
    print("To start training:")
    print(f"  bash scripts/train_agibot.sh {config_path}")


if __name__ == "__main__":
    main()
