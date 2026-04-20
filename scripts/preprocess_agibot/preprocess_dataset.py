#!/usr/bin/env python3
"""
Dataset preprocessing: copy the raw dataset to a target path, then:
  0. Add modality.json (if missing)
  1. Normalize video resolution: scale with aspect ratio + pad to 256×256
  2. Language: run update_highlevel_instruction.py to refresh high_level_instruction,
     then update tasks.jsonl and parquet to use high_level text

Usage:
  python scripts/preprocess_dataset.py /path/to/source_dataset /path/to/output_dataset
  python scripts/preprocess_dataset.py /path/to/source_dataset /path/to/output_dataset --skip_video
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TARGET_VIDEO_SIZE = (256, 256)


def _build_modality_from_info(info: dict, *, include_waist: bool = True) -> dict:
    """Build modality.json from info.json ``field_descriptions``.

    Rules:
      - joint/position: first 7 dims → left_arm_joint_position, next 7 → right_arm_joint_position
      - left_effector/position, right_effector/position: map as-is
      - waist/position: 5th index (last of five); omit when ``include_waist`` is False
      - video / annotation: discover video keys from ``features``
    """
    features = info.get("features", {})

    def _extract_indices(feature_key: str, field_suffix: str) -> list[int] | None:
        feat = features.get(feature_key, {})
        descs = feat.get("field_descriptions", {})
        for name, desc in descs.items():
            if name.endswith(field_suffix):
                return desc.get("indices", [])
        return None

    def _build_arm_and_effector(feature_key: str) -> dict:
        result = {}
        joint_indices = _extract_indices(feature_key, "/joint/position")
        if joint_indices and len(joint_indices) >= 14:
            left_start = joint_indices[0]
            left_end = joint_indices[6] + 1
            right_start = joint_indices[7]
            right_end = joint_indices[13] + 1
            result["left_arm_joint_position"] = {"start": left_start, "end": left_end}
            result["right_arm_joint_position"] = {"start": right_start, "end": right_end}

        for side in ("left", "right"):
            eff_indices = _extract_indices(feature_key, f"/{side}_effector/position")
            if eff_indices:
                result[f"{side}_effector_position"] = {
                    "start": eff_indices[0],
                    "end": eff_indices[-1] + 1,
                }

        if include_waist:
            waist_indices = _extract_indices(feature_key, "/waist/position")
            if waist_indices and len(waist_indices) >= 5:
                fifth = waist_indices[4]
                result["waist_position"] = {"start": fifth, "end": fifth + 1}

        return result

    state_section = _build_arm_and_effector("observation.state")
    action_section = _build_arm_and_effector("action")

    video_section = {}
    video_key_map = {
        "observation.images.top_head": "top_head",
        "observation.images.hand_left": "hand_left",
        "observation.images.hand_right": "hand_right",
    }
    for orig_key, short_name in video_key_map.items():
        if orig_key in features and features[orig_key].get("dtype") == "video":
            video_section[short_name] = {"original_key": orig_key}

    annotation_section = {"language.action_text": {"original_key": "task_index"}}

    return {
        "state": state_section,
        "action": action_section,
        "video": video_section,
        "annotation": annotation_section,
    }


def _ignore_tar_on_copy(_dir: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ignore: skip ``.tar`` / split-archive ``.tar.*`` files and depth directories."""
    ignored: set[str] = set()
    for name in names:
        if "tar.gz" in name:
            ignored.add(name)
        if "depth" in name.lower():
            ignored.add(name)
    return ignored


def resize_keep_aspect(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Scale with aspect ratio preserved + pad to target size (e.g. 256×256)."""
    h, w, _ = frame.shape
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (new_w, new_h))
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded


def process_video(
    src_path: Path,
    dst_path: Path,
    target_size: tuple[int, int] = TARGET_VIDEO_SIZE,
    fps: float = 30,
) -> None:
    """Resize each frame with aspect ratio + pad to target size, then write the video."""
    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("decord required for video preprocessing") from exc

    vr = VideoReader(str(src_path))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    target_h, target_w = target_size

    processed = np.array([resize_keep_aspect(f, target_h, target_w) for f in frames])
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_path), fourcc, fps, (target_w, target_h))
    for f in processed:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


def step0_add_modality(dataset_path: Path, modality_template: Path | None, *, include_waist: bool = True) -> None:
    """Always build modality.json from info.json ``field_descriptions`` (unless a template is given)."""
    meta = dataset_path / "meta"
    modality_path = meta / "modality.json"

    if modality_template and modality_template.exists():
        shutil.copy(modality_template, modality_path)
        print(f"  [modality] copied from template: {modality_template} -> {modality_path}")
        return

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    modality_obj = _build_modality_from_info(info, include_waist=include_waist)
    with open(modality_path, "w", encoding="utf-8") as f:
        json.dump(modality_obj, f, indent=4, ensure_ascii=False)
    print(f"  [modality] generated from info.json: {modality_path}")


def _process_video_task(args: tuple) -> Path:
    """Worker for ProcessPoolExecutor; returns the processed path."""
    mp4_path, target_h, target_w, fps = args
    process_video(mp4_path, mp4_path, target_size=(target_h, target_w), fps=fps)
    return mp4_path


def step1_video_resolution(
    dataset_path: Path,
    target_size: tuple[int, int],
    workers: int = 1,
) -> None:
    """Normalize all videos to target size (aspect-ratio scale + pad); optional parallel workers."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not video_keys:
        print("  [video] no video features, skipping")
        return

    target_h, target_w = target_size
    fps = info.get("fps", 30)

    # Collect all videos to process
    videos_dir = dataset_path / "videos"
    tasks = []
    for chunk_dir in sorted(videos_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for video_key in video_keys:
            video_subdir = chunk_dir / video_key
            if "depth" in str(video_subdir):
                continue
            if not video_subdir.exists():
                continue
            print(video_subdir)
            for mp4 in sorted(video_subdir.glob("*.mp4")):
                if "depth" not in str(mp4):
                    tasks.append((mp4, target_h, target_w, fps))
    # Parallel or sequential processing
    if workers <= 1:
        for args in tasks:
            _process_video_task(args)
            print(f"  [video] done: {args[0]}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_video_task, t): t for t in tasks}
            for future in as_completed(futures):
                mp4_path = future.result()
                print(f"  [video] done: {mp4_path}")

    # Update video shapes in info.json
    for k in video_keys:
        if k in features:
            features[k]["shape"] = [target_h, target_w, 3]
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"  [video] updated video shapes in info.json to {target_h}x{target_w}")


def step2_language(dataset_path: Path, script_dir: Path) -> None:
    """1) Run update_highlevel_instruction.py to refresh high_level_instruction.
    2) Update tasks.jsonl and parquet to use high_level (one instruction per episode).
    """
    # 2a. Run update_highlevel_instruction.py
    update_script = script_dir / "update_highlevel_instruction.py"
    if not update_script.exists():
        raise FileNotFoundError(f"update_highlevel_instruction.py not found: {update_script}")

    ret = subprocess.run(
        [sys.executable, str(update_script), str(dataset_path)],
        cwd=str(script_dir.parent),
        check=False,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"update_highlevel_instruction.py failed with exit code {ret.returncode}")

    # 2b. Read high_level_instruction
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    hl_map = info.get("high_level_instruction", {})
    if not hl_map:
        print("  [language] no high_level_instruction, skipping tasks.jsonl update")
        return

    # 2c. Sort by episode_index
    ep_indices = sorted(hl_map.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    tasks_lines = []
    for ep_str in ep_indices:
        entry = hl_map.get(ep_str, {})
        if isinstance(entry, dict):
            text = entry.get("high_level_instruction", "")
        else:
            text = str(entry)
        tasks_lines.append({"task_index": int(ep_str), "task": text})

    # Write tasks.jsonl (one line per episode)
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for line in tasks_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"  [language] updated tasks.jsonl: {len(tasks_lines)} episodes")

    # 2d. Read episodes.jsonl for length
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    length_map: dict[int, int] = {}
    if episodes_path.exists():
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if "length" in ep:
                    length_map[ep["episode_index"]] = ep["length"]

    # 2e. Write episodes.jsonl: tasks from high_level_instruction + preserved length
    episodes_lines = []
    for ep_str in ep_indices:
        entry = hl_map.get(ep_str, {})
        if isinstance(entry, dict):
            text = entry.get("high_level_instruction", "")
        else:
            text = str(entry)
        ep_idx = int(ep_str)
        line_obj: dict = {"episode_index": ep_idx, "tasks": text}
        if ep_idx in length_map:
            line_obj["length"] = length_map[ep_idx]
        episodes_lines.append(line_obj)

    with open(episodes_path, "w", encoding="utf-8") as f:
        for line_obj in episodes_lines:
            f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
    print(f"  [language] updated episodes.jsonl: {len(episodes_lines)} lines (tasks + length)")


def step3_add_episode_index_from_filename(dataset_path: Path) -> None:
    """Walk ``data/**/*.parquet``, parse episode id from filename, set episode_index and task_index columns."""
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        print("  [episode_index] data directory missing, skipping")
        return

    parquet_files = list(data_dir.rglob("*.parquet"))
    pattern = re.compile(r"episode_(\d+)\.parquet", re.IGNORECASE)
    updated = 0

    for parquet_path in parquet_files:
        match = pattern.search(parquet_path.name)
        if not match:
            continue
        ep_idx = int(match.group(1))
        table = pq.read_table(parquet_path)
        n = len(table)
        ep_col = pa.array([ep_idx] * n, type=pa.int64())

        for col_name in ("episode_index", "task_index"):
            if col_name in table.column_names:
                idx = table.schema.get_field_index(col_name)
                table = table.set_column(idx, col_name, ep_col)
            else:
                table = table.append_column(col_name, ep_col)

        pq.write_table(table, parquet_path)
        updated += 1

    print(f"  [episode_index] updated {updated} parquet files")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing: copy, modality.json, video resolution, language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess_dataset.py /path/to/source /path/to/output
  python scripts/preprocess_dataset.py /path/to/source /path/to/output --skip_video --modality-template test_train_data/meta/modality.json
        """,
    )
    parser.add_argument("--source", type=str, default="./source_dataset", help="Source dataset directory")
    parser.add_argument("--output", type=str, default="./output_dataset", help="Output dataset directory (copy + preprocess)")
    parser.add_argument(
        "--skip_copy",
        action="store_true",
        help="When output exists, skip re-copying from source (process existing output)",
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip video resolution preprocessing",
    )
    parser.add_argument(
        "--modality-template",
        type=str,
        default=None,
        help="Path to modality.json template (optional)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Target video resolution H W (default: 256 256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker processes for video preprocessing (default: CPU count)",
    )
    parser.add_argument(
        "--no-waist",
        action="store_true",
        help="Omit waist_position from modality.json (included by default)",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    script_dir = Path(__file__).resolve().parent
    modality_template = Path(args.modality_template).resolve() if args.modality_template else None
    target_size = tuple(args.target_size)

    if not source.exists():
        print(f"Error: source directory does not exist: {source}")
        return 1
    if not (source / "meta" / "info.json").exists():
        print(f"Error: info.json not found: {source}/meta/info.json")
        return 1

    print(f"Source: {source}")
    print(f"Output: {output}")

    # 0. Copy
    if output.exists() and not args.skip_copy:
        shutil.rmtree(output)
        shutil.copytree(source, output, ignore=_ignore_tar_on_copy)
        print(f"Copied to: {output} (excluded .tar / .tar.00, etc.)")
    elif output.exists() and args.skip_copy:
        print(f"--skip_copy set; skipping copy, using output directory: {output}")
    else:
        shutil.copytree(source, output, ignore=_ignore_tar_on_copy)
        print(f"Copied to: {output} (excluded .tar / .tar.00, etc.)")

    # 1. modality.json
    print("\n[Step 0] Add modality.json")
    step0_add_modality(output, modality_template, include_waist=not args.no_waist)

    # 2. Video resolution
    if not args.skip_video:
        print(f"\n[Step 1] Normalize video resolution to target size (workers={args.workers})")
        step1_video_resolution(output, target_size, workers=args.workers)
    else:
        print("\n[Step 1] Skip video preprocessing (--skip_video)")

    # 3. Language
    print("\n[Step 2] Language processing")
    step2_language(output, script_dir)

    # 4. episode_index / task_index from parquet filenames
    print("\n[Step 3] Write episode_index / task_index from parquet filenames")
    step3_add_episode_index_from_filename(output)

    print("\n✓ Preprocessing complete")


if __name__ == "__main__":
    exit(main() or 0)
