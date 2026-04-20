#!/usr/bin/env python3
"""
Remove static frames from episodes: read each episode's active frame range from
``instruction_segments`` in info.json, keep only frames within
[min(start_frame_index), max(end_frame_index)], and reindex ``frame_index``.
Also trim the matching videos.

Usage:
  python scripts/trim_static_frames.py /path/to/dataset
  python scripts/trim_static_frames.py /path/to/dataset --dry-run
  python scripts/trim_static_frames.py /path/to/dataset --workers 8
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _get_frame_range(segments: list[dict]) -> tuple[int, int] | None:
    """Extract min(start_frame_index) and max(end_frame_index) from instruction_segments."""
    if not segments:
        return None
    starts = [s["start_frame_index"] for s in segments if "start_frame_index" in s]
    ends = [s["end_frame_index"] for s in segments if "end_frame_index" in s]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def trim_parquet(
    parquet_path: Path,
    min_frame: int,
    max_frame: int,
    frame_index_col: str = "frame_index",
) -> int:
    """
    Keep parquet rows whose frame_index lies in [min_frame, max_frame],
    and reindex ``frame_index`` and ``index`` from 0.
    Returns the number of rows kept.
    """
    table = pq.read_table(parquet_path)
    if frame_index_col not in table.column_names:
        raise ValueError(f"parquet missing column {frame_index_col}: {parquet_path}")

    col = table.column(frame_index_col)
    indices = [col[i].as_py() for i in range(len(col))]
    mask = [min_frame <= idx <= max_frame for idx in indices]
    if not any(mask):
        return 0

    filtered = table.filter(pa.array(mask))
    n = len(filtered)

    new_indices = pa.array(list(range(n)), type=pa.int64())
    idx_pos = filtered.schema.get_field_index(frame_index_col)
    new_table = filtered.set_column(idx_pos, frame_index_col, new_indices)

    if "index" in new_table.column_names:
        idx_type = new_table.schema.field("index").type
        new_index = pa.array(list(range(n)), type=idx_type)
        idx_pos = new_table.schema.get_field_index("index")
        new_table = new_table.set_column(idx_pos, "index", new_index)

    pq.write_table(new_table, parquet_path)
    return n


def trim_video(
    video_path: Path,
    min_frame: int,
    max_frame: int,
    fps: float = 30,
) -> None:
    """Trim video to frames in [min_frame, max_frame] (inclusive) and overwrite the file."""
    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("decord required for video trimming") from exc

    vr = VideoReader(str(video_path))
    total = len(vr)
    if max_frame >= total:
        max_frame = total - 1
    if min_frame < 0:
        min_frame = 0

    indices = list(range(min_frame, max_frame + 1))
    frames = vr.get_batch(indices).asnumpy()
    # decord returns RGB; cv2 VideoWriter expects BGR
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

    h, w = frames_bgr[0].shape[:2]
    tmp_path = video_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
    for f in frames_bgr:
        out.write(f)
    out.release()
    tmp_path.replace(video_path)


def update_instruction_segments(
    segments: list[dict],
    min_frame: int,
) -> list[dict]:
    """Shift instruction_segments frame indices to the trimmed timeline (subtract min_frame)."""
    result = []
    for seg in segments:
        new_seg = dict(seg)
        if "start_frame_index" in new_seg:
            new_seg["start_frame_index"] = new_seg["start_frame_index"] - min_frame
        if "end_frame_index" in new_seg:
            new_seg["end_frame_index"] = new_seg["end_frame_index"] - min_frame
        if "success_frame_index" in new_seg:
            new_seg["success_frame_index"] = new_seg["success_frame_index"] - min_frame
        result.append(new_seg)
    return result


def _process_episode(
    parquet_path: Path,
    ep_idx: int,
    segs: list[dict],
    min_frame: int,
    max_frame: int,
    frame_index_col: str,
    videos_dir: Path,
    video_keys: list[str],
    fps: float,
) -> tuple[int, int, list[dict], list[str]]:
    """Trim one episode (parquet + videos); return (ep_idx, n_frames, new_segs, messages)."""
    messages: list[str] = []

    total_before = len(pq.read_table(parquet_path, columns=[frame_index_col]))

    if total_before == 0 or (min_frame == 0 and max_frame >= total_before - 1):
        messages.append(f"  [skip] episode {ep_idx} no trim (range already covers all frames)")
        return ep_idx, total_before, segs, messages

    head_removed = min_frame
    tail_removed = max(total_before - 1 - max_frame, 0)
    removed_parts: list[str] = []
    if head_removed > 0:
        removed_parts.append(f"head [0, {min_frame - 1}] {head_removed} frames")
    if tail_removed > 0:
        removed_parts.append(
            f"tail [{max_frame + 1}, {total_before - 1}] {tail_removed} frames"
        )
    total_removed = head_removed + tail_removed

    n_frames = trim_parquet(
        parquet_path, min_frame, max_frame, frame_index_col=frame_index_col
    )
    new_segs = update_instruction_segments(segs, min_frame)

    messages.append(
        f"  [trim] episode {ep_idx}: {total_before} -> {n_frames} frames, "
        f"removed {total_removed} frames ({', '.join(removed_parts)})"
    )

    chunk_match = re.search(r"chunk-(\d+)", str(parquet_path))
    chunk = int(chunk_match.group(1)) if chunk_match else (ep_idx // 10)
    chunk_str = f"chunk-{chunk:03d}"
    for video_key in video_keys:
        video_subdir = videos_dir / chunk_str / video_key
        mp4_path = video_subdir / f"episode_{ep_idx:06d}.mp4"
        if mp4_path.exists():
            trim_video(mp4_path, min_frame, max_frame, fps=fps)
            messages.append(f"         video={video_key} trimmed")

    return ep_idx, n_frames, new_segs, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove static frames using instruction_segments; trim parquet and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/trim_static_frames.py /path/to/dataset
  python scripts/trim_static_frames.py /path/to/dataset --dry-run
  python scripts/trim_static_frames.py /path/to/dataset --workers 8
        """,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions only; do not modify files",
    )
    parser.add_argument(
        "--frame-index-col",
        type=str,
        default="frame_index",
        help="Frame index column in parquet (default: frame_index)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"Error: info.json not found: {info_path}")
        return 1

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    instruction_segments = info.get("instruction_segments", {})
    if not instruction_segments:
        print("No instruction_segments in info.json, skipping")
        return 0

    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"
    fps = info.get("fps", 30)
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    video_keys = [k for k in video_keys if "depth" not in str(k).lower()]

    parquet_files = list(data_dir.rglob("*.parquet"))
    pattern = re.compile(r"episode_(\d+)\.parquet", re.IGNORECASE)

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_list: list[dict] = []
    if episodes_path.exists():
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes_list.append(json.loads(line))

    tasks: list[tuple[Path, int, list[dict], int, int]] = []
    for parquet_path in sorted(parquet_files):
        match = pattern.search(parquet_path.name)
        if not match:
            continue
        ep_idx = int(match.group(1))
        segs = instruction_segments.get(str(ep_idx))
        if not segs:
            print(f"  [skip] episode {ep_idx} has no instruction_segments")
            continue

        frame_range = _get_frame_range(segs)
        if not frame_range:
            print(f"  [skip] episode {ep_idx} could not parse frame range")
            continue

        min_frame, max_frame = frame_range
        if args.dry_run:
            print(f"  [dry-run] episode {ep_idx}: keep frames [{min_frame}, {max_frame}]")
            continue

        tasks.append((parquet_path, ep_idx, segs, min_frame, max_frame))

    if args.dry_run:
        print("Dry run complete; no files modified")
        return 0

    if not tasks:
        print("No episodes need trimming, skipping")
        return 0

    new_lengths: dict[int, int] = {}
    new_instruction_segments: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_episode,
                parquet_path,
                ep_idx,
                segs,
                min_frame,
                max_frame,
                args.frame_index_col,
                videos_dir,
                video_keys,
                fps,
            ): ep_idx
            for parquet_path, ep_idx, segs, min_frame, max_frame in tasks
        }
        for future in as_completed(futures):
            ep_idx_key = futures[future]
            try:
                ep_idx, n_frames, new_segs, messages = future.result()
                for msg in messages:
                    print(msg)
                new_lengths[ep_idx] = n_frames
                new_instruction_segments[str(ep_idx)] = new_segs
            except Exception as e:
                print(f"  [error] episode {ep_idx_key}: {e}")

    if episodes_path.exists() and new_lengths:
        new_episodes = []
        for ep in episodes_list:
            ep_idx = ep["episode_index"]
            if ep_idx in new_lengths:
                ep = dict(ep)
                ep["length"] = new_lengths[ep_idx]
            new_episodes.append(ep)
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in new_episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        print(f"  [episodes.jsonl] updated length for {len(new_lengths)} episodes")

    if new_instruction_segments:
        info["instruction_segments"] = new_instruction_segments
    if new_lengths:
        total = sum(
            new_lengths.get(ep["episode_index"], ep.get("length", 0)) or 0
            for ep in episodes_list
        )
        info["total_frames"] = total
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print("  [info.json] updated instruction_segments and total_frames")

    print("\n✓ Static-frame removal complete")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
