# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json
import yaml
from PIL import Image
from .transform import Normalizer, prepare_action, prepare_images, prepare_language, prepare_state

from ...utils import logging

class VlaDataset(Dataset):
    def __init__(
        self,
        repo_id="path2dataset",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
        action_name="action",
    ):
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            action_name: [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
        )
        self.action_name = action_name

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )

class liberoDataset(Dataset):
    def __init__(
        self,
        repo_id="libero",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "actions": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='libero',
            norm_type={
                "image": "identity",
                "wrist_image": "identity",
                "state": data_config.norm_type,
                "actions": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["image"] * 255).to(torch.uint8)
        wrist_image = (normalized_item["wrist_image"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image},
            "state": normalized_item["state"].to(torch.float32),
            "action": normalized_item["actions"].to(torch.float32),
            "action_is_pad": normalized_item["actions_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict,  use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }

        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

class RobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='robotwin',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_item["observation.images.cam_left_wrist"] * 255).to(
            torch.uint8
        )
        right_wrist_image = (normalized_item["observation.images.cam_right_wrist"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )


def _parse_agibot_slices(info: dict, feature_key: str, use_waist: bool = True) -> list[tuple[int, int]]:
    """Parse slice ranges for the effective sub-fields of an Agibot state or action vector.

    Field selection logic (same as preprocess_dataset.py):
      - joint/position        : first 7 → left arm, next 7 → right arm
      - left/right_effector   : all indices
      - waist/position        : 5th index only (index 4 within the field), only when use_waist=True

    Returns a list of (start, end) slices in order:
      [left_arm_joints, right_arm_joints, left_effector, right_effector, (waist)]
    """
    descs = info["features"][feature_key].get("field_descriptions", {})

    def _indices(suffix: str) -> list[int]:
        for name, desc in descs.items():
            if name.endswith(suffix):
                return desc.get("indices", [])
        return []

    slices: list[tuple[int, int]] = []

    joint_idx = _indices("/joint/position")
    if len(joint_idx) >= 14:
        slices.append((joint_idx[0], joint_idx[6] + 1))   # left arm: first 7
        slices.append((joint_idx[7], joint_idx[13] + 1))  # right arm: next 7

    for side in ("left", "right"):
        eff_idx = _indices(f"/{side}_effector/position")
        if eff_idx:
            slices.append((eff_idx[0], eff_idx[-1] + 1))

    if use_waist:
        waist_idx = _indices("/waist/position")
        if len(waist_idx) >= 5:
            slices.append((waist_idx[4], waist_idx[4] + 1))   # 5th element only

    return slices


def _apply_slices(tensor: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([tensor[..., s:e] for s, e in slices], dim=-1)


class AgibotDataset(Dataset):
    def __init__(
        self,
        repo_id="agibot_dataset",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        local_root = repo_id if os.path.isdir(repo_id) else None
        hub_repo_id = os.path.basename(repo_id.rstrip("/")) if local_root else repo_id
        self.dataset_meta = LeRobotDatasetMetadata(hub_repo_id, root=local_root)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=hub_repo_id,
            root=local_root,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        info_path = os.path.join(repo_id, "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)
        use_waist = getattr(data_config, "use_waist", True)
        self.state_slices = _parse_agibot_slices(info, "observation.state", use_waist=use_waist)
        self.action_slices = _parse_agibot_slices(info, "action", use_waist=use_waist)

        with open(self.norm_stats_file) as f:
            norm_json = json.load(f)
        raw_norm = norm_json['norm_stats']
        already_sliced = norm_json.get("sliced", False)
        if not already_sliced:
            slice_indices_state = []
            for s, e in self.state_slices:
                slice_indices_state.extend(range(s, e))
            slice_indices_action = []
            for s, e in self.action_slices:
                slice_indices_action.extend(range(s, e))
            for stat_key in raw_norm.get("observation.state", {}):
                arr = raw_norm["observation.state"][stat_key]
                raw_norm["observation.state"][stat_key] = [arr[i] for i in slice_indices_state]
            for stat_key in raw_norm.get("action", {}):
                arr = raw_norm["action"][stat_key]
                raw_norm["action"][stat_key] = [arr[i] for i in slice_indices_action]
        self.normalizer = Normalizer(
            norm_stats=raw_norm,
            from_file=True,
            data_type='customized',
            norm_type={
                "observation.images.top_head": "identity",
                "observation.images.hand_left": "identity",
                "observation.images.hand_right": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]

        state_vec = _apply_slices(item["observation.state"].to(torch.float32), self.state_slices)
        action_vec = _apply_slices(item["action"].to(torch.float32), self.action_slices)

        to_normalize = {
            "observation.state": state_vec,
            "action": action_vec,
        }
        normalized = self.normalizer.normalize(to_normalize)

        base_image = (item["observation.images.top_head"] * 255).to(torch.uint8)
        left_wrist_image = (item["observation.images.hand_left"] * 255).to(torch.uint8)
        right_wrist_image = (item["observation.images.hand_right"] * 255).to(torch.uint8)

        batch_dict = {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized["observation.state"].to(torch.float32),
            "action": normalized["action"].to(torch.float32),
            "action_is_pad": item["action_is_pad"],
            "prompt": [task],
        }
        state = prepare_state(self.config, batch_dict)
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict)
        actions = prepare_action(self.config, batch_dict)
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )


class CustomizedRobotwinDataset(Dataset):
    def __init__(
        self,
        repo_id="robotwin",
        config=PI0Config,
        tokenizer=AutoTokenizer,
        data_config=None,
        image_processor=None,
        use_depth_align=False,
    ):
        image_transforms = Resize((data_config.img_size, data_config.img_size))
        self.image_processor = image_processor
        # [i / 30 for i in range(50)] represents action chunks in 50 steps at 30 FPS.
        # The timestamps are set to 0 for the images and state, as we only use current obs.
        self.config = config
        self.tokenizer = tokenizer
        self.norm_stats_file = data_config.norm_stats_file
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        delta_timestamps = {
            "action": [t / self.dataset_meta.fps for t in range(50)],
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)
        self.normalizer = Normalizer(
            # norm_stats=self.dataset.meta.stats,
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='customized',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": data_config.norm_type,
                "action": data_config.norm_type,
            },
        )
        self.use_depth_align = use_depth_align

    def __len__(self):
        return len(self.dataset)

    def getdata(self, idx):
        item = self.dataset[idx]
        task = self.dataset_meta.tasks[int(item['task_index'])]
        assert task == item['task']

        normalized_item = self.normalizer.normalize(item)
        base_image = (normalized_item["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_item["observation.images.cam_left_wrist"] * 255).to(
            torch.uint8
        )
        right_wrist_image = (normalized_item["observation.images.cam_right_wrist"] * 255).to(
            torch.uint8
        )
        batch_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_item["observation.state"].to(torch.float32),
            "action": normalized_item["action"].to(torch.float32),
            "action_is_pad": normalized_item["action_is_pad"],
            "prompt": [item["task"]],
        }
        state = prepare_state(self.config, batch_dict) # bs,8 -> bs,32
        lang_tokens, lang_masks = prepare_language(self.config, self.tokenizer, batch_dict) # bs, seq_len
        actions = prepare_action(self.config, batch_dict) # bs,50,7 -> bs,50,32 , 7
        images, img_masks, pil_images = prepare_images(self.config, self.image_processor, batch_dict, use_depth_align=self.use_depth_align)

        batch_dict = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'actions': actions,
            'action_is_pad': batch_dict['action_is_pad'],
        }
        if self.use_depth_align: batch_dict['pil_images'] = pil_images

        return batch_dict

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )