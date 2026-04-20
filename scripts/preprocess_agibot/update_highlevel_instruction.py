#!/usr/bin/env python3
"""
刷写数据集的 meta/info.json 中的 high_level_instruction 字段
用法:
    # 从 instruction_segments 中自动拼接
    python update_high_level_instruction.py /path/to/dataset
    # 为所有 episode 设置统一的高级指令
    python update_high_level_instruction.py /path/to/dataset --high-level-instruction "抓取物体"
"""
import json
import argparse
from pathlib import Path
from typing import Optional


def _get_hl_text(entry) -> str:
    """兼容两种 high_level_instruction entry 格式：
    1) 新格式：str
    2) 旧格式：{"high_level_instruction": str}
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        v = entry.get("high_level_instruction", "")
        return v if isinstance(v, str) else str(v)
    return str(entry)
def update_high_level_instruction(
    info_json_path: str,
    high_level_instruction: str | None = None,
) -> None:
    """
    更新info.json中的high_level_instruction
    Args:
        info_json_path: info.json文件路径
        high_level_instruction: 如果提供，则所有episode使用该值；
                               如果为None，则从instruction_segments中拼接
    """
    with open(info_json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    # 确保 high_level_instruction 字段存在
    if "high_level_instruction" not in info:
        info["high_level_instruction"] = {}
    # 兼容旧格式：{"0": {"high_level_instruction": "..."}}
    # 统一内部处理为 dict[str, str]
    if isinstance(info["high_level_instruction"], dict):
        info["high_level_instruction"] = {
            str(k): _get_hl_text(v) for k, v in info["high_level_instruction"].items()
        }
    else:
        info["high_level_instruction"] = {}
    # 确保 instruction_segments 字段存在
    instruction_segments = info.get("instruction_segments", {})
    # 获取所有 episode 索引
    if high_level_instruction is not None:
        # 如果提供了值，使用该值填充所有 episode
        # 优先从 total_episodes 获取，否则从 instruction_segments 或已有的 high_level_instruction 获取
        total_episodes = info.get("total_episodes", 0)
        if total_episodes > 0:
            episode_indices = [str(i) for i in range(total_episodes)]
        else:
            # 从 instruction_segments 或已有的 high_level_instruction 获取所有 key
            episode_indices = set(instruction_segments.keys()) | set(info["high_level_instruction"].keys())
            episode_indices = sorted(episode_indices, key=lambda x: int(x) if x.isdigit() else 0)
        for idx_str in episode_indices:
            info["high_level_instruction"][str(idx_str)] = high_level_instruction
        print(f"已更新所有 {len(episode_indices)} 个 episode 的 high_level_instruction 为: {high_level_instruction}")
    else:
        # 从 instruction_segments 中拼接
        updated_count = 0
        for idx_str, segments in instruction_segments.items():
            if not segments:
                continue
            # 提取所有 instruction
            instructions = [seg.get("instruction", "") for seg in segments if seg.get("instruction")]
            if len(instructions) == 1:
                # 只有一个 instruction，直接使用
                result = instructions[0]
            else:
                # 多个 instruction，先去掉末尾标点符号，再用逗号拼接
                punctuation = ".,!?;:，。！？；：、"
                cleaned = [inst.rstrip(punctuation) for inst in instructions]
                result = ", ".join(cleaned)
            # 更新 high_level_instruction
            info["high_level_instruction"][str(idx_str)] = result
            updated_count += 1
        print(f"已从 instruction_segments 更新 {updated_count} 个 episode 的 high_level_instruction")
    with open(info_json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
def main():
    parser = argparse.ArgumentParser(
        description="刷写数据集的 meta/info.json 中的 high_level_instruction 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 instruction_segments 中自动拼接
  python update_high_level_instruction.py /path/to/dataset
  # 为所有 episode 设置统一的高级指令
  python update_high_level_instruction.py /path/to/dataset --high-level-instruction "抓取物体"
        """,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="数据集目录路径（应包含 meta/info.json）",
    )
    parser.add_argument(
        "--high-level-instruction",
        type=str,
        default=None,
        help="高级指令文本。如果提供，则所有 episode 使用该值；如果不提供，则从 instruction_segments 中自动拼接",
    )
    args = parser.parse_args()
    # 检查数据集目录
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"错误: 数据集目录不存在: {dataset_path}")
        return 1
    if not dataset_path.is_dir():
        print(f"错误: 不是目录: {dataset_path}")
        return 1
    # 检查 info.json
    info_json_path = dataset_path / "meta" / "info.json"
    if not info_json_path.exists():
        print(f"错误: info.json 不存在: {info_json_path}")
        return 1
    print(f"数据集目录: {dataset_path}")
    print(f"info.json 路径: {info_json_path}")
    if args.high_level_instruction:
        print(f"高级指令: {args.high_level_instruction}")
    else:
        print("模式: 从 instruction_segments 自动拼接")
    print()
    # 更新 high_level_instruction
    try:
        update_high_level_instruction(
            str(info_json_path),
            high_level_instruction=args.high_level_instruction,
        )
        print(f"\n✓ 更新完成！")
        return 0
    except Exception as e:
        print(f"\n✗ 更新失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
if __name__ == "__main__":
    exit(main())