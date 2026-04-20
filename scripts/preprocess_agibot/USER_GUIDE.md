# Agibot 数据处理与 LingBot-VLA 训练流水线

本文档描述从原始 Agibot 数据到 LingBot-VLA 模型训练的完整流程。

所有脚本均位于 `scripts/preprocess_agibot/` 目录下。

---

## 目录

- [总览](#总览)
- [Step 1: 数据预处理](#step-1-数据预处理)
- [Step 2: 生成训练配置](#step-2-生成训练配置)
- [Step 3: 启动训练](#step-3-启动训练)
- [Step 4: 部署推理](#step-4-部署推理)
- [附录: 工具脚本](#附录-工具脚本)

---

## 总览

```
原始数据集                          预处理后数据集                     训练配置 + 模型
    │                                   │                               │
    │  Step 1                           │  Step 2                       │  Step 3
    │  preprocess_agibot_data.sh        │  generate_train_config.py     │  train_agibot.sh
    ▼                                   ▼                               ▼
┌──────────────┐   ──────────>   ┌──────────────┐   ──────────>   ┌──────────────┐
│ 原始采集数据  │                 │ 标准化数据集   │                 │ configs/     │
│ (任意分辨率)  │                 │ (256×256 视频) │                 │   vla/*.yaml │
│              │                 │ (去除静止帧)   │                 │ assets/      │
│              │                 │ (语义已刷新)   │                 │   norm_stats │
└──────────────┘                 └──────────────┘                 └──────────────┘
```

**完整命令示例（从头到尾）：**

```bash
# 1. 预处理每个子数据集
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw/task_a /data/processed/task_a
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw/task_b /data/processed/task_b

# 2. 生成训练配置（扫描所有子数据集，计算合并 norm，生成 yaml）
python3 scripts/preprocess_agibot/generate_train_config.py /data/processed \
    --use-waist --remove-depth

# 3. 启动训练
bash scripts/train_agibot.sh configs/vla/processed.yaml
```

---

## Step 1: 数据预处理

将原始 Agibot 数据集转换为 LingBot-VLA 可用的标准格式。通过 `preprocess_agibot_data.sh` 一键执行，内部依次运行两个阶段。

### 1.1 一键运行

```bash
bash scripts/preprocess_agibot/preprocess_agibot_data.sh <SOURCE> <OUTPUT>
```

**执行内容：**

| 阶段 | 脚本 | 功能 |
|------|------|------|
| 阶段 1 | `preprocess_dataset.py` | 拷贝数据集、生成 modality.json、视频裁剪到 256×256（保持比例 + 居中 pad）、语义指令刷新 |
| 阶段 2 | `trim_static_frames.py` | 根据 `instruction_segments` 去除每个 episode 首尾的静止帧 |

**常用示例：**

```bash
# 基本用法
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw_dataset /data/preprocessed_dataset

# 跳过视频处理（视频已是 256×256 时使用）
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw /data/out --skip_video

# 自定义两个阶段的 worker 数（用 -- 分隔）
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw /data/out --workers 8 -- --workers 16

# 预览裁剪效果（不修改文件）
bash scripts/preprocess_agibot/preprocess_agibot_data.sh /data/raw /data/out -- --dry-run
```

### 1.2 阶段 1 详解: `preprocess_dataset.py`

四个处理步骤：

| 步骤 | 说明 |
|------|------|
| **Step 0** | 从 `info.json` 的 `field_descriptions` 生成 `meta/modality.json`（或从模板拷贝） |
| **Step 1** | 将所有视频帧统一到 256×256（保持宽高比缩放 + 居中黑边填充），使用 `cv2.resize` |
| **Step 2** | 刷新语义标注：运行 `update_highlevel_instruction.py` 更新 `high_level_instruction`，重写 `tasks.jsonl` 和 `episodes.jsonl` |
| **Step 3** | 根据 parquet 文件名写入 `episode_index` / `task_index` 列 |

```bash
python3 scripts/preprocess_agibot/preprocess_dataset.py \
    --source /path/to/raw_dataset \
    --output /path/to/output_dataset
```

| 参数 | 说明 |
|------|------|
| `--source` | 源数据集目录（须含 `meta/info.json`） |
| `--output` | 输出目录（拷贝源目录后就地处理） |
| `--skip_copy` | 跳过拷贝，直接处理已有的输出目录 |
| `--skip_video` | 跳过视频分辨率处理 |
| `--modality-template PATH` | 使用自定义 `modality.json` 模板 |
| `--target-size H W` | 目标分辨率（默认 `256 256`） |
| `--workers N` | 视频处理并行 worker 数（默认 CPU 核数） |
| `--no-waist` | 生成的 modality.json 中不包含 waist_position |

### 1.3 阶段 2 详解: `trim_static_frames.py`

根据 `info.json` 中的 `instruction_segments` 字段确定每个 episode 的有效帧范围，裁剪 parquet 数据和视频文件，并重新索引。

```bash
python3 scripts/preprocess_agibot/trim_static_frames.py /path/to/output_dataset
```

| 参数 | 说明 |
|------|------|
| `--dry-run` | 仅预览裁剪计划，不修改文件 |
| `--frame-index-col NAME` | parquet 中的帧索引列名（默认 `frame_index`） |
| `--workers N` | 并发线程数（默认 `4`） |

### 1.4 预处理后数据集结构

```
output_dataset/
├── meta/
│   ├── info.json              # 数据集元信息（features, fps, instruction_segments 等）
│   ├── modality.json          # 模态配置（自动生成或来自模板）
│   ├── tasks.jsonl            # 每个 episode 一条任务描述
│   └── episodes.jsonl         # Episode 元信息（index, tasks, length）
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.top_head/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.hand_left/
        │   └── ...
        └── observation.images.hand_right/
            └── ...
```

---

## Step 2: 生成训练配置

所有子数据集预处理完成后，使用 `generate_train_config.py` 一键完成：
1. 管理 depth features（默认移除）
2. 跨所有子数据集计算合并的 **sliced norm stats**（仅保留有效关节维度）
3. 生成训练配置 yaml

### 2.1 用法

```bash
python3 scripts/preprocess_agibot/generate_train_config.py <ROOT_DIRS...> [OPTIONS]
```

`<ROOT_DIRS>` 可以是：
- 包含多个子数据集的根目录（脚本自动扫描含 `meta/info.json` 的子目录）
- 单个数据集目录

**示例：**

```bash
# 扫描 /data/processed 下所有子数据集，带腰，移除深度
python3 scripts/preprocess_agibot/generate_train_config.py /data/processed \
    --use-waist --remove-depth

# 多个根目录，不带腰
python3 scripts/preprocess_agibot/generate_train_config.py /data/task_a /data/task_b \
    --no-waist

# 自定义配置文件名
python3 scripts/preprocess_agibot/generate_train_config.py /data/processed \
    --config-name my_experiment

# 跳过 norm 计算（使用已有 norm 文件）
python3 scripts/preprocess_agibot/generate_train_config.py /data/processed \
    --skip-norm
```

| 参数 | 说明 |
|------|------|
| `root_dirs` | 数据集根目录（支持多个） |
| `--use-waist` | 包含腰关节（默认） |
| `--no-waist` | 不包含腰关节 |
| `--remove-depth` | 从 info.json 中移除 depth features（默认） |
| `--keep-depth` | 保留 depth features |
| `--config-name NAME` | 配置文件名（默认取第一个根目录的 basename） |
| `--skip-norm` | 跳过 norm 计算，使用已有 norm 文件 |

### 2.2 输出

| 输出文件 | 说明 |
|---------|------|
| `assets/norm_stats/<name>_merged.json` | 合并的 sliced norm stats（含 `"sliced": true` 标记） |
| `configs/vla/<name>.yaml` | 训练配置文件 |

**Norm stats 格式：**
```json
{
  "norm_stats": {
    "observation.state": {"mean": [...], "std": [...], "q01": [...], "q99": [...], ...},
    "action": {"mean": [...], "std": [...], "q01": [...], "q99": [...], ...}
  },
  "sliced": true,
  "use_waist": true,
  "state_dim": 17,
  "action_dim": 17,
  "count": 123456
}
```

> `"sliced": true` 表示 norm stats 已经是 slice 后的有效维度（16-D 或 17-D），训练代码加载时会跳过 re-slicing。

### 2.3 Norm 计算原理

`compute_merged_norm.py` 的核心逻辑：

1. 从第一个子数据集的 `info.json` 的 `field_descriptions` 解析出 state/action 的有效维度索引
2. 两遍扫描所有 parquet 文件：
   - Pass 1: 增量计算 mean、mean-of-squares、min、max（sliced 维度）
   - Pass 2: 构建直方图，计算分位数（q01/q99/q02/q98）
3. 输出合并的 norm stats JSON

**Slice 规则（186-D → 16/17-D）：**

| 维度 | 来源 |
|------|------|
| 0–6 | left arm joint position (7) |
| 7–13 | right arm joint position (7) |
| 14 | left effector position (1) |
| 15 | right effector position (1) |
| 16 | waist position 第 5 个分量（仅 `--use-waist` 时） |

---

## Step 3: 启动训练

```bash
bash scripts/train_agibot.sh configs/vla/<name>.yaml
```

训练脚本自动检测可用 GPU 数量，使用 `torchrun` 分布式训练。

**环境变量：**

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU | 所有可用 GPU |
| `NNODES` | 节点数 | `1` |
| `NODE_RANK` | 当前节点编号 | `0` |
| `MASTER_ADDR` | 主节点地址 | `0.0.0.0` |
| `MASTER_PORT` | 主节点端口 | `62500` |

**示例：**

```bash
# 单机 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_agibot.sh configs/vla/instruction.yaml

# 额外参数追加在 yaml 路径后面
bash scripts/train_agibot.sh configs/vla/instruction.yaml --train.max_steps 50000
```

---

## Step 4: 部署推理

训练完成后，使用 WebSocket 部署推理服务：

```bash
python3 deploy/lingbot_agibot_policy.py \
    --model_path /path/to/checkpoint \
    --use_waist \
    --port 8006
```

| 参数 | 说明 |
|------|------|
| `--model_path` | checkpoint 目录 |
| `--use_waist` | 启用腰关节 |
| `--use_depth` | 启用深度图 |
| `--port` | WebSocket 端口（默认 `8006`） |
| `--norm_stats_file` | 覆盖 norm stats 文件路径 |
| `--data_repo` | 数据集路径（用于读取 state/action 的 slice 配置） |
| `--action_horizon` | 截断返回的 action 序列长度 |

**客户端 payload（msgpack 编码）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `images.top_head` | ndarray (H,W,3) uint8 | 顶部相机图像 |
| `images.hand_left` | ndarray (H,W,3) uint8 | 左手相机图像 |
| `images.hand_right` | ndarray (H,W,3) uint8 | 右手相机图像 |
| `state` | list / ndarray (32-D) | 机器人状态向量 |
| `prompt` / `task_name` | str | 语言指令 |

**客户端 state 布局（32-D）：**

| 索引 | 内容 |
|------|------|
| 0–6 | 左臂关节位置 (7) |
| 7–13 | 右臂关节位置 (7) |
| 14 | 左夹爪位置 (1) |
| 15 | 右夹爪位置 (1) |
| 16–20 | 腰部位置 (5)，仅使用第 5 个分量 |

> 当 `task_name` 不包含 `"sorting_packages"` 时，即使模型带腰输出，返回的 action 也会去掉腰位维度。

---

## 附录: 工具脚本

### manage_depth_features.py

独立管理 info.json 中的 depth features，支持批量操作。

```bash
# 查看
python3 scripts/preprocess_agibot/manage_depth_features.py /data/processed --list

# 移除
python3 scripts/preprocess_agibot/manage_depth_features.py /data/processed --remove

# 添加
python3 scripts/preprocess_agibot/manage_depth_features.py /data/processed --add
```

### compute_merged_norm.py

独立计算合并的 sliced norm stats（通常由 `generate_train_config.py` 自动调用）。

```bash
python3 scripts/preprocess_agibot/compute_merged_norm.py \
    --data_root /data/processed \
    --output assets/norm_stats/merged.json \
    --use_waist
```

### update_highlevel_instruction.py

独立刷新 info.json 中的 `high_level_instruction` 字段。

```bash
# 从 instruction_segments 自动拼接
python3 scripts/preprocess_agibot/update_highlevel_instruction.py /path/to/dataset

# 统一设置
python3 scripts/preprocess_agibot/update_highlevel_instruction.py /path/to/dataset \
    --high-level-instruction "左臂抓取桌上的蓝色台球"
```

---

## 文件清单

```
scripts/preprocess_agibot/
├── USER_GUIDE.md                    # 本文档
├── preprocess_agibot_data.sh        # Step 1 一键脚本（数据预处理）
├── preprocess_dataset.py            # Step 1 阶段 1：拷贝、modality、视频、语义
├── trim_static_frames.py            # Step 1 阶段 2：去除静止帧
├── update_highlevel_instruction.py  # 语义指令刷新工具
├── generate_train_config.py         # Step 2：生成 norm stats + 训练 yaml
├── compute_merged_norm.py           # 合并 sliced norm stats 计算
└── manage_depth_features.py         # Depth features 管理工具
```
