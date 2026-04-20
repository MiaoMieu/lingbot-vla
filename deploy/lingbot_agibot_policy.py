"""
LingBot-VLA Agibot deployment server (WebSocket + msgpack).

Wire-compatible with GR00T N1.6 serve_gr00t_websocket.py.

Usage:
  python deploy/lingbot_agibot_policy.py \
    --model_path /path/to/checkpoint \
    --use_waist \
    --port 8006

Payload format (Agibot three cameras + flattened state):
  images: top_head / hand_left / hand_right — HWC uint8.
  state: flattened list / ndarray, layout depends on embodiment:
    - Without waist (16-D): left arm 7 | right arm 7 | left gripper 1 | right gripper 1
    - With waist (17-D):    left arm 7 | right arm 7 | left gripper 1 | right gripper 1 | waist 1
  prompt / task_name: language instruction string.

Response (same as GR00T):
  actions: list of action vectors (each is a flat list).
  actions_by_key: actions split by joint key names.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import asyncio
import argparse
import base64
import io
import json
import logging
import random
import socket
import time
import traceback
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import yaml
from PIL import Image
from safetensors import safe_open
from tqdm import tqdm

import torch
from torch import Tensor
from transformers import AutoConfig

from lerobot.configs.policies import PreTrainedConfig
from lingbotvla.models.vla.pi0.modeling_pi0 import PI0Policy
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.data.vla_data.transform import (
    Normalizer, prepare_images, prepare_language, prepare_state,
)
from lingbotvla.data.vla_data.base_dataset import _parse_agibot_slices, _apply_slices
from lingbotvla.models import build_processor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed_everywhere(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL_PATH = {
    'pi0': os.environ.get('PALIGEMMA_PATH', './paligemma-3b-pt-224/'),
    'lingbotvla': os.environ.get('QWEN25_PATH', os.path.join(CURRENT_DIR, '../models/Qwen/Qwen2.5-VL-3B-Instruct/')),
}

IMAGE_KEYS = ("top_head", "hand_left", "hand_right")

WAIST_TASK_NAMES = {"sorting_packages"}

# Client sends 32-D state:
# [0:7] left arm | [7:14] right arm | [14:15] left gripper | [15:16] right gripper | [16:21] waist (5)
# Training uses only waist index 4 (the 5th element) → client index 20

CLIENT_STATE_SLICES_NO_WAIST = [
    ("left_arm_joint_position", slice(0, 7)),
    ("right_arm_joint_position", slice(7, 14)),
    ("left_effector_position", slice(14, 15)),
    ("right_effector_position", slice(15, 16)),
]

CLIENT_STATE_SLICES_WITH_WAIST = [
    ("left_arm_joint_position", slice(0, 7)),
    ("right_arm_joint_position", slice(7, 14)),
    ("left_effector_position", slice(14, 15)),
    ("right_effector_position", slice(15, 16)),
    ("waist_position", slice(20, 21)),
]

# ---------------------------------------------------------------------------
# msgpack helpers — wire-compatible with GR00T
# ---------------------------------------------------------------------------

def _make_pack_unpack():
    from deploy.msgpack_numpy import pack_array, unpack_array

    def _bytes_keys_to_str(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if isinstance(k, (bytes, bytearray)):
                    try:
                        k = k.decode("utf-8")
                    except Exception:
                        k = str(k)
                out[k] = _bytes_keys_to_str(v)
            return out
        if isinstance(x, list):
            return [_bytes_keys_to_str(v) for v in x]
        return x

    def pack(obj: Any) -> bytes:
        return msgpack.packb(obj, default=pack_array)

    def unpack(data: bytes) -> Any:
        raw = msgpack.unpackb(
            data, raw=True, strict_map_key=False,
            object_hook=unpack_array,
        )
        return _bytes_keys_to_str(raw)

    return pack, unpack


# ---------------------------------------------------------------------------
# Image / state helpers — wire-compatible with GR00T
# ---------------------------------------------------------------------------

def _ensure_ndarray(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.array(x)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _decode_maybe_b64_ndarray(img: Any) -> np.ndarray:
    if isinstance(img, str):
        buf = base64.b64decode(img)
        return np.load(io.BytesIO(buf), allow_pickle=False)
    return _ensure_ndarray(img)


def _prepare_frame_hwc_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Action formatting — wire-compatible with GR00T
# ---------------------------------------------------------------------------

def action_to_wire(
    action_array: np.ndarray,
    use_waist: bool = False,
    horizon: int | None = None,
    task_name: str = "",
) -> dict[str, Any]:
    """Convert (T, action_dim) ndarray to GR00T-compatible wire format.

    action_array layout (no waist, 16-D):
      0-7 left arm | 7-14 right arm | 14 left gripper | 15 right gripper
    action_array layout (waist, 17-D):
      0-7 left arm | 7-14 right arm | 14 left gripper | 15 right gripper | 16 waist

    Waist is only included when task_name contains a WAIST_TASK_NAMES entry.
    """
    if action_array.ndim == 1:
        action_array = action_array[np.newaxis, :]

    T = action_array.shape[0]
    if horizon is not None:
        T = min(T, max(0, int(horizon)))
        action_array = action_array[:T]

    left = action_array[:, 0:7]
    right = action_array[:, 7:14]
    left_grip = action_array[:, 14:15]
    right_grip = action_array[:, 15:16]

    if use_waist and action_array.shape[-1] >= 17 and any(t in task_name for t in WAIST_TASK_NAMES):
        waist = action_array[:, 16:17]
        pad_zeros = np.zeros((T, 4), dtype=np.float32)
        cmds = np.concatenate(
            [left, right, left_grip, right_grip, pad_zeros, waist], axis=-1,
        )
    else:
        waist = None
        cmds = np.concatenate([left, right, left_grip, right_grip], axis=-1)

    by_key: dict[str, list] = {
        "left_arm_joint_position": left.tolist(),
        "right_arm_joint_position": right.tolist(),
        "left_effector_position": left_grip.tolist(),
        "right_effector_position": right_grip.tolist(),
    }
    if waist is not None:
        by_key["waist_position"] = waist.tolist()

    return {
        "actions": [cmd.tolist() for cmd in cmds],
        "actions_by_key": by_key,
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model_weights(policy, path_to_pi_model, strict=True):
    all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
    merged_weights = {}
    for file_path in tqdm(all_safetensors):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)
    policy.load_state_dict(merged_weights, strict=strict)


def merge_qwen_config(policy_config, qwen_config):
    if hasattr(qwen_config, 'to_dict'):
        config_dict = qwen_config.to_dict()
    else:
        config_dict = qwen_config

    text_keys = {
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
        "rope_theta", "vocab_size", "max_position_embeddings",
        "hidden_act", "tie_word_embeddings", "tokenizer_path",
    }
    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
            print(f"✅ Merged LLM: {key} = {config_dict[key]}")

    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    else:
        print("⚠️ Warning: 'vision_config' not found in qwen_config!")
    return policy_config


# ---------------------------------------------------------------------------
# PolicyPreprocessMixin + inference wrappers
# ---------------------------------------------------------------------------

class PolicyPreprocessMixin:

    @torch.no_grad()
    def select_action(
        self,
        observation: dict[str, Tensor],
        use_bf16: bool = False,
        vlm_causal: bool = False,
        noise: Tensor | None = None,
    ):
        self.eval()
        device = 'cuda'
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        s1 = time.time()

        if len(observation['images'].shape) == 4:
            observation['images'] = observation['images'].unsqueeze(0)
            observation['img_masks'] = observation['img_masks'].unsqueeze(0)

        if 'expert_imgs' in observation:
            actions = self.model.sample_actions(
                observation['images'].to(dtype=dtype, device=device),
                observation['img_masks'].to(device=device),
                observation['lang_tokens'].unsqueeze(0).to(device=device),
                observation['lang_masks'].unsqueeze(0).to(device=device),
                observation['state'].unsqueeze(0).to(dtype=dtype, device=device),
                observation['expert_imgs'].to(dtype=dtype, device=device),
                vlm_causal=vlm_causal,
            )
        else:
            actions = self.model.sample_actions(
                observation['images'].to(dtype=dtype, device=device),
                observation['img_masks'].to(device=device),
                observation['lang_tokens'].unsqueeze(0).to(device=device),
                observation['lang_masks'].unsqueeze(0).to(device=device),
                observation['state'].unsqueeze(0).to(dtype=dtype, device=device),
                vlm_causal=vlm_causal,
            )
        delta_time = time.time() - s1
        print(f'sample_actions cost {delta_time} s')
        observation['action'] = actions.squeeze(0)[:, :self.action_dim].to(
            dtype=torch.float32, device='cpu',
        )
        if use_bf16:
            observation['state'] = observation['state'].to(dtype=torch.float32)
        data = self.normalizer.unnormalize(observation)
        return data


class LingBotVlaInferencePolicy(PolicyPreprocessMixin, LingbotVlaPolicy):
    pass


class PI0InfernecePolicy(PolicyPreprocessMixin, PI0Policy):
    pass


# ---------------------------------------------------------------------------
# AgibotVlaServer — model loading + inference
# ---------------------------------------------------------------------------

class AgibotVlaServer:
    """Policy wrapper for Agibot deployment.

    Accepts GR00T-compatible payloads and returns GR00T-compatible responses.
    """

    def __init__(
        self,
        path_to_pi_model="",
        use_waist=False,
        use_depth=False,
        use_length=1,
        chunk_ret=False,
        use_bf16=True,
        use_fp32=False,
        norm_stats_file=None,
        action_horizon=None,
        data_repo=None,
    ) -> None:
        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'
        self.use_length = use_length
        self.chunk_ret = chunk_ret
        self.use_waist = use_waist
        self.use_depth = use_depth
        self.norm_stats_file_override = norm_stats_file
        self.action_horizon = action_horizon
        self.data_repo = data_repo
        self.task_description = None

        self.vla = self.load_vla(path_to_pi_model)
        self.vla = self.vla.cuda().eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        elif use_fp32:
            self.vla.model.float()
        self.global_step = 0
        self.last_action_chunk = None
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32

    # ---- checkpoint path helpers ----

    def _find_weights_dir(self, path_to_pi_model: str) -> Path:
        ckpt = Path(path_to_pi_model)
        hf_ckpt = ckpt / "hf_ckpt"
        if hf_ckpt.is_dir() and list(hf_ckpt.glob("*.safetensors")):
            return hf_ckpt
        if list(ckpt.glob("*.safetensors")):
            return ckpt
        raise FileNotFoundError(
            f"Cannot find *.safetensors in {ckpt} or {hf_ckpt}. "
            f"Please run mereg_dcp_to_hf.py first if this is a DCP checkpoint."
        )

    def _find_config_dir(self, path_to_pi_model: str) -> Path:
        ckpt = Path(path_to_pi_model)
        candidates = [ckpt / "hf_ckpt", ckpt, ckpt / "model_assets"]
        for i in range(1, 4):
            parent = ckpt.parents[i - 1] if i <= len(ckpt.parents) else None
            if parent is not None:
                candidates.append(parent / "model_assets")
                candidates.append(parent)
        for d in candidates:
            if (d / "config.json").is_file():
                return d
        raise FileNotFoundError(
            f"Cannot find config.json near {path_to_pi_model}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    def _find_training_config(self, path_to_pi_model: str) -> Path:
        ckpt = Path(path_to_pi_model)
        candidates = [ckpt]
        for i in range(1, 4):
            if i <= len(ckpt.parents):
                candidates.append(ckpt.parents[i - 1])
        for d in candidates:
            p = d / "lingbotvla_cli.yaml"
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"Cannot find lingbotvla_cli.yaml near {path_to_pi_model}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    # ---- model loading ----

    def load_vla(self, path_to_pi_model) -> LingbotVlaPolicy:
        print(f"loading model from: {path_to_pi_model}")

        weights_dir = self._find_weights_dir(path_to_pi_model)
        print(f"loading weights from: {weights_dir}")

        config_dir = self._find_config_dir(path_to_pi_model)
        print(f"loading config from: {config_dir}")
        config = PreTrainedConfig.from_pretrained(str(config_dir))

        training_config_path = self._find_training_config(path_to_pi_model)
        print(f"loading training config from: {training_config_path}")
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)

        training_model_config = training_config['model']
        training_model_config.update(training_config['train'])
        for k, v in training_model_config.items():
            v = getattr(config, k, training_model_config[k])
            setattr(config, k, v)
        config.attention_implementation = 'eager'

        training_base_model = training_config['model']['tokenizer_path']
        if 'paligemma' in training_base_model:
            model_name = 'pi0'
            config.vocab_size = 257152
        elif 'qwen2' in training_base_model.lower():
            model_name = 'lingbotvla'
        else:
            raise ValueError(f"Unsupported base model of {path_to_pi_model}")
        base_model_path = BASE_MODEL_PATH[model_name]
        config.tokenizer_path = base_model_path
        self.model_name = model_name

        qwen_config = AutoConfig.from_pretrained(base_model_path)
        config = merge_qwen_config(config, qwen_config)

        if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
            config.vocab_size = training_config['model']['vocab_size']

        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        data_config = SimpleNamespace(**training_config['data'])

        print('Initializing model ... ')
        if 'paligemma' in training_base_model:
            policy = PI0InfernecePolicy(config, tokenizer_path=base_model_path)
        else:
            policy = LingBotVlaInferencePolicy(config, tokenizer_path=base_model_path)

        load_model_weights(policy, str(weights_dir), strict=True)

        policy.feature_transform = None
        self.data_config = data_config
        self.config = config
        self.joint_max_dim = training_config['train']['max_action_dim']
        self.action_dim = training_config['train']['action_dim']
        self.chunk_size = training_config['train']['chunk_size']
        policy.action_dim = self.action_dim
        policy.chunk_size = self.chunk_size

        if self.norm_stats_file_override:
            self.norm_stats_file = self.norm_stats_file_override
        else:
            self.norm_stats_file = data_config.norm_stats_file
            if ',' in self.norm_stats_file:
                self.norm_stats_file = self.norm_stats_file.split(',')[0].strip()
                print(f"Multiple norm_stats_files detected, using first: {self.norm_stats_file}")

        self.use_depth_align = 'align_params' in training_config['train']

        with open(self.norm_stats_file) as f:
            norm_json = json.load(f)
        raw_norm = norm_json['norm_stats']
        already_sliced = norm_json.get("sliced", False)

        info_path = None
        if already_sliced:
            print(f"Norm stats already sliced: state {len(raw_norm['observation.state']['mean'])}-D, "
                  f"action {len(raw_norm['action']['mean'])}-D")
        else:
            if self.data_repo:
                data_repo_path = self.data_repo
                if ',' in data_repo_path:
                    data_repo_path = data_repo_path.split(',')[0].strip()
                info_path = os.path.join(data_repo_path, "meta", "info.json")
            else:
                data_dir = getattr(data_config, 'data_dir', '')
                if ',' in data_dir:
                    data_dir = data_dir.split(',')[0].strip()
                info_path = os.path.join(data_dir, "meta", "info.json")

            if os.path.isfile(info_path):
                with open(info_path) as f:
                    dataset_info = json.load(f)
                self.state_slices = _parse_agibot_slices(
                    dataset_info, "observation.state", use_waist=self.use_waist,
                )
                self.action_slices = _parse_agibot_slices(
                    dataset_info, "action", use_waist=self.use_waist,
                )
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
                print(f"Sliced norm_stats: state {len(slice_indices_state)}-D, "
                      f"action {len(slice_indices_action)}-D (use_waist={self.use_waist})")
            else:
                print(f"Warning: info.json not found at {info_path}, "
                      f"using raw norm_stats without slicing. "
                      f"Set --data_repo to the dataset root directory.")
                self.state_slices = None
                self.action_slices = None

        policy.normalizer = Normalizer(
            norm_stats=raw_norm,
            from_file=True,
            data_type='customized',
            norm_type={
                "observation.images.top_head": "identity",
                "observation.images.hand_left": "identity",
                "observation.images.hand_right": "identity",
                "observation.state": self.data_config.norm_type,
                "action": self.data_config.norm_type,
            },
        )
        print('Model initialized ... ')
        return policy

    # ---- runtime helpers ----

    def reset(self, robo_name=None, path_to_pi_model=None) -> None:
        if path_to_pi_model is not None:
            self.vla = self.load_vla(path_to_pi_model)
            self.vla = self.vla.cuda().eval()
            if self.use_bf16:
                self.vla = self.vla.to(torch.bfloat16)
            elif self.use_fp32:
                self.vla.model.float()

        self.global_step = 0
        self.last_action_chunk = None

        if getattr(self.data_config, 'norm_type', None) is None:
            self.data_config.norm_type = 'meanstd'
        if getattr(self.config, 'vlm_causal', None) is None:
            self.config.vlm_causal = False
        if getattr(self.config, 'qwenvl_bos', None) is None:
            self.config.qwenvl_bos = False

        if path_to_pi_model is not None:
            all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
            merged_weights = {}
            for file_path in tqdm(all_safetensors):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        merged_weights[key] = f.get_tensor(key)
            self.vla.load_state_dict(merged_weights, strict=True)

    # GR00T 同款：训练视频是 256×256 居中 pad 格式
    VIDEO_SIZE = 256

    def _resize_image_for_model(self, img_hwc_u8: np.ndarray) -> np.ndarray:
        """Mimic training pipeline: keep-aspect + center-pad to VIDEO_SIZE (using
        cv2.resize, same as preprocess_dataset.py), then torchvision Resize to
        img_size (same as LeRobotDataset image_transforms).
        Returns (img_size, img_size, 3) HWC uint8.
        """
        import cv2
        from torchvision.transforms.v2 import Resize as TvResize
        h, w = img_hwc_u8.shape[:2]
        target = self.VIDEO_SIZE
        scale = min(target / h, target / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(img_hwc_u8, (new_w, new_h))
        pad_h = target - resized.shape[0]
        pad_w = target - resized.shape[1]
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            resized = np.pad(
                resized,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        image_size = getattr(self.data_config, 'img_size', 224)
        if resized.shape[0] != image_size or resized.shape[1] != image_size:
            chw = torch.from_numpy(np.transpose(resized, (2, 0, 1)))
            chw = TvResize((image_size, image_size))(chw)
        else:
            chw = torch.from_numpy(np.transpose(resized, (2, 0, 1)))
        return (chw.float() / 255.0).numpy()

    # ---- payload parsing (GR00T-compatible) ----

    def _parse_payload(self, payload: dict[str, Any]) -> dict:
        """Convert GR00T-format payload to internal observation dict."""
        if "images" not in payload:
            raise ValueError("Payload must contain 'images' key")

        images = payload["images"]
        observation = {}
        for key in IMAGE_KEYS:
            if key not in images:
                raise ValueError(f"Payload missing images.{key}")
            img = _decode_maybe_b64_ndarray(images[key])
            if isinstance(img, list):
                img = np.array(img, dtype=np.uint8)
            img = _prepare_frame_hwc_u8(img)
            observation[f"observation.images.{key}"] = self._resize_image_for_model(img)

        st = payload.get("state")
        if st is None:
            raise ValueError("Payload requires 'state'")
        state = _ensure_ndarray(st, dtype=np.float32).reshape(-1)
        slices = CLIENT_STATE_SLICES_WITH_WAIST if self.use_waist else CLIENT_STATE_SLICES_NO_WAIST
        expected_dim = sum(sl.stop - sl.start for _, sl in slices)
        if state.size == expected_dim:
            selected = state
        else:
            max_idx = max(sl.stop for _, sl in slices)
            if state.size < max_idx:
                raise ValueError(
                    f"State dim {state.size} is neither the sliced dim ({expected_dim}) "
                    f"nor large enough for the expected raw state (need >= {max_idx})"
                )
            selected = np.concatenate([state[sl] for _, sl in slices])
        observation["observation.state"] = selected

        prompt = payload.get("prompt") or payload.get("task_name") or ""
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode("utf-8", errors="replace")
        observation["task"] = str(prompt)

        return observation

    # ---- inference ----

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accepts GR00T-compatible payload, returns GR00T-compatible response."""
        if payload.get("reset"):
            self.reset(
                robo_name=payload.get("robo_name", "agibot"),
                path_to_pi_model=payload.get("path_to_pi_model"),
            )
            return {"actions": [], "actions_by_key": {}}

        observation = self._parse_payload(payload)
        task_name = observation.get("task", "")

        for k, v in observation.items():
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)

        if self.use_length == -1 or self.global_step % self.use_length == 0:
            normalized_observation = self.vla.normalizer.normalize(observation)
            base_image = (normalized_observation[f"observation.images.{IMAGE_KEYS[0]}"] * 255).to(torch.uint8)
            left_image = (normalized_observation[f"observation.images.{IMAGE_KEYS[1]}"] * 255).to(torch.uint8)
            right_image = (normalized_observation[f"observation.images.{IMAGE_KEYS[2]}"] * 255).to(torch.uint8)
            obs_dict = {
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": left_image,
                    "right_wrist_0_rgb": right_image,
                },
                "state": normalized_observation["observation.state"].to(torch.float32),
                "prompt": [observation["task"]],
            }
            state = prepare_state(self.config, obs_dict)
            lang_tokens, lang_masks = prepare_language(
                self.config, self.language_tokenizer, obs_dict,
            )
            images, img_masks, _ = prepare_images(
                self.config, self.image_processor, obs_dict,
                use_depth_align=self.use_depth,
            )

            if not hasattr(self, '_dumped_pre_qwen'):
                dump_path = os.path.join(os.path.dirname(__file__), "debug_infer_pre_qwen.pt")
                torch.save({
                    "base_image": base_image.cpu(),
                    "left_image": left_image.cpu(),
                    "right_image": right_image.cpu(),
                    "state": obs_dict["state"].cpu(),
                }, dump_path)
                print(f"[DUMP] Saved pre-Qwen inference data to {dump_path}")
                print(f"  base_image: shape={base_image.shape}, dtype={base_image.dtype}, "
                      f"min={base_image.float().min():.1f}, max={base_image.float().max():.1f}")
                print(f"  left_image: shape={left_image.shape}, dtype={left_image.dtype}")
                print(f"  right_image: shape={right_image.shape}, dtype={right_image.dtype}")
                self._dumped_pre_qwen = True
            observation = {
                'images': images,
                'img_masks': img_masks,
                'state': state,
                'lang_tokens': lang_tokens,
                'lang_masks': lang_masks,
            }
            if self.use_bf16:
                observation['state'] = observation['state'].to(torch.bfloat16)

            if not hasattr(self, '_dumped_infer_obs'):
                dump_path = os.path.join(os.path.dirname(__file__), "debug_infer_observation.pt")
                torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in observation.items()}, dump_path)
                print(f"[DUMP] Saved inference observation to {dump_path}")
                for k, v in observation.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                              f"min={v.float().min().item():.4f}, max={v.float().max().item():.4f}")
                self._dumped_infer_obs = True

        if self.chunk_ret:
            action = self.vla.select_action(
                observation, self.use_bf16, self.config.vlm_causal,
            )['action'].float().cpu().numpy()
            action = action[:self.use_length, :self.action_dim]
        else:
            if self.use_length == -1 or self.global_step % self.use_length == 0:
                action = self.vla.select_action(
                    observation, self.use_bf16, self.config.vlm_causal,
                )['action']
                self.last_action_chunk = action.float().cpu().numpy()

            if self.use_length > 0:
                action = self.last_action_chunk[self.global_step % self.use_length]
            action = action[:, :self.action_dim]
            print(f"on server step: {self.global_step}")
            self.global_step += 1

        return action_to_wire(
            action, use_waist=self.use_waist, horizon=self.action_horizon,
            task_name=task_name,
        )


# ---------------------------------------------------------------------------
# WebSocket server — wire-compatible with GR00T serve_gr00t_websocket.py
# ---------------------------------------------------------------------------

async def _handler(
    ws: Any,
    policy: AgibotVlaServer,
    pack,
    unpack,
) -> None:
    logger.info("Connection from %s opened", ws.remote_address)

    meta = {
        "embodiment": "agibot_genie1_waist" if policy.use_waist else "agibot_genie1",
        "video_keys": list(IMAGE_KEYS),
        "state_keys": [name for name, _ in (
            CLIENT_STATE_SLICES_WITH_WAIST if policy.use_waist else CLIENT_STATE_SLICES_NO_WAIST
        )],
        "action_keys": [name for name, _ in (
            CLIENT_STATE_SLICES_WITH_WAIST if policy.use_waist else CLIENT_STATE_SLICES_NO_WAIST
        )],
    }
    await ws.send(pack(meta))

    while True:
        try:
            raw = await ws.recv()
            payload = unpack(raw)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload, got {type(payload)}")
            resp = policy.infer(payload)
            await ws.send(pack(resp))
        except Exception:
            logger.exception("Inference error")
            await ws.send(pack({
                "error": "inference_failed",
                "traceback": traceback.format_exc(),
            }))
            try:
                await ws.close(1011, "Internal error")
            except Exception:
                pass
            break


def serve_forever(policy: AgibotVlaServer, host: str, port: int) -> None:
    import websockets.asyncio.server as ws_server

    pack, unpack = _make_pack_unpack()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"

    embodiment = "agibot_genie1_waist" if policy.use_waist else "agibot_genie1"
    logger.info(
        "LingBot-VLA WebSocket server: host=%s ip=%s port=%s embodiment=%s",
        hostname, local_ip, port, embodiment,
    )

    async def run() -> None:
        async with ws_server.serve(
            lambda ws: _handler(ws, policy, pack, unpack),
            host,
            port,
            compression=None,
            max_size=2**28,
        ) as server:
            await server.serve_forever()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LingBot-VLA Agibot WebSocket policy server "
                    "(wire-compatible with GR00T serve_gr00t_websocket.py)",
    )
    parser.add_argument("--model_path", type=str, default="/home/zy/workspace/checkpoints/instruction_lingbotvla_develop/checkpoints/global_step_18000/",
                        help="Path to the model checkpoint directory")
    parser.add_argument("--use_waist", action="store_true", default=False,
                        help="Whether the model was trained with waist joint")
    parser.add_argument("--use_depth", action="store_true", default=False,
                        help="Whether the model uses depth alignment features")
    parser.add_argument("--use_length", type=int, default=50,
                        help="Used length of action chunk")
    parser.add_argument("--chunk_ret", action="store_true", default=True,
                        help="Return full action chunk (True) or single-step (False)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8888,
                        help="WebSocket server port")
    parser.add_argument("--norm_stats_file", type=str, default=None,
                        help="Override the norm_stats file path")
    parser.add_argument("--action_horizon", type=int, default=None,
                        help="If set, return only the first N timesteps of actions")
    parser.add_argument("--data_repo", type=str, default=None,
                        help="Path to the Agibot dataset root (contains meta/info.json). "
                             "Required for correct state/action slicing from 186-D raw state.")
    args = parser.parse_args()

    model = AgibotVlaServer(
        path_to_pi_model=args.model_path,
        use_waist=args.use_waist,
        use_depth=args.use_depth,
        use_length=args.use_length,
        chunk_ret=args.chunk_ret,
        norm_stats_file=args.norm_stats_file,
        action_horizon=args.action_horizon,
        data_repo=args.data_repo,
    )
    serve_forever(model, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
