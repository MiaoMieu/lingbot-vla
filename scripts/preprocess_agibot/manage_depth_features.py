#!/usr/bin/env python3
"""
管理 lerobot 数据集 info.json 中的 depth image features。

支持传入多个路径，每个路径可以是：
  - 单个数据集目录（含 meta/info.json）
  - 根目录（自动扫描所有含 meta/info.json 的子目录）

用法:
  删除所有 depth features:
    python scripts/manage_depth_features.py /path/to/root1 /path/to/root2 --remove

  添加 depth features:
    python scripts/manage_depth_features.py /path/to/root --add

  查看当前 depth features:
    python scripts/manage_depth_features.py /path/to/root --list
"""
import argparse
import copy
import glob
import json
import os
import sys


DEPTH_TEMPLATE = {
    "dtype": "video",
    "video_info": {
        "video.is_depth_map": True,
        "video.fps": 30.0,
        "video.codec": "png",
        "video.pix_fmt": "gray16be",
        "has_audio": False,
    },
    "shape": [256, 256, 3],
    "names": ["height", "width", "channel"],
}


def discover_datasets(paths):
    datasets = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(os.path.join(p, "meta", "info.json")):
            datasets.append(p)
        else:
            found = sorted(glob.glob(os.path.join(p, "*", "meta", "info.json")))
            for f in found:
                datasets.append(os.path.dirname(os.path.dirname(f)))
    if not datasets:
        print("ERROR: No datasets found in given paths", file=sys.stderr)
        sys.exit(1)
    return datasets


def load_info(dataset_path):
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        print(f"ERROR: {info_path} not found", file=sys.stderr)
        return None, None
    with open(info_path, "r") as f:
        return json.load(f), info_path


def save_info(info, info_path):
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"Saved: {info_path}")


def get_image_keys(features):
    return [k for k in features if k.startswith("observation.images.")]


def get_depth_keys(features):
    return [k for k in get_image_keys(features) if "depth" in k]


def get_rgb_keys(features):
    return [k for k in get_image_keys(features) if "depth" not in k]


def list_depth(dataset_path):
    info, _ = load_info(dataset_path)
    if info is None:
        return
    features = info["features"]
    depth_keys = get_depth_keys(features)
    rgb_keys = get_rgb_keys(features)

    print(f"Dataset: {dataset_path}")
    print(f"  RGB images ({len(rgb_keys)}): {', '.join(rgb_keys)}")
    if depth_keys:
        print(f"  Depth images ({len(depth_keys)}): {', '.join(depth_keys)}")
    else:
        print(f"  Depth images (0): (none)")


def remove_depth(dataset_path):
    info, info_path = load_info(dataset_path)
    if info is None:
        return
    features = info["features"]
    depth_keys = get_depth_keys(features)

    if not depth_keys:
        print(f"[{dataset_path}] No depth features, skip.")
        return

    for k in depth_keys:
        del features[k]

    save_info(info, info_path)
    print(f"[{dataset_path}] Removed {len(depth_keys)}: {', '.join(depth_keys)}")


def add_depth(dataset_path):
    info, info_path = load_info(dataset_path)
    if info is None:
        return
    features = info["features"]
    rgb_keys = get_rgb_keys(features)
    existing_depth = get_depth_keys(features)

    if not rgb_keys:
        print(f"[{dataset_path}] No RGB image features, skip.")
        return

    fps = info.get("fps", 30.0)
    added = []
    for rgb_key in rgb_keys:
        depth_key = rgb_key + "_depth"
        if depth_key in existing_depth:
            continue

        depth_entry = copy.deepcopy(DEPTH_TEMPLATE)
        depth_entry["video_info"]["video.fps"] = fps
        rgb_shape = features[rgb_key].get("shape", [256, 256, 3])
        depth_entry["shape"] = [rgb_shape[0], rgb_shape[1], rgb_shape[2]]

        features[depth_key] = depth_entry
        added.append(depth_key)

    if added:
        save_info(info, info_path)
        print(f"[{dataset_path}] Added {len(added)}: {', '.join(added)}")
    else:
        print(f"[{dataset_path}] All depth features already exist, skip.")


def main():
    parser = argparse.ArgumentParser(description="Manage depth features in lerobot dataset info.json")
    parser.add_argument("paths", nargs="+", help="Dataset dirs or root dirs containing sub-datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--remove", action="store_true", help="Remove all depth image features")
    group.add_argument("--add", action="store_true", help="Add depth features for each RGB image")
    group.add_argument("--list", action="store_true", help="List current image features")
    args = parser.parse_args()

    datasets = discover_datasets(args.paths)
    print(f"Found {len(datasets)} dataset(s):\n")

    action = list_depth if args.list else (remove_depth if args.remove else add_depth)
    for ds in datasets:
        action(ds)
        print()


if __name__ == "__main__":
    main()
