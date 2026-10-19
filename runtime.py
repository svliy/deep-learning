"""随机种子与单设备选择。"""

import random

import numpy as np
import torch

from config import DEVICE_PATTERN


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子；manual_seed 包含 CUDA。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    """解析设备名，校验 CUDA 可用性及编号，保留完整的可见设备索引。"""
    if not isinstance(name, str) or not DEVICE_PATTERN.fullmatch(name):
        raise ValueError("device 必须是 auto、cpu、cuda 或 cuda:N（例如 cuda:1）")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请使用 --device cpu")
    index = torch.cuda.current_device() if name == "cuda" else int(name.split(":")[1])
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise RuntimeError(f"CUDA 设备编号 {index} 越界，当前仅有 {count} 张可见 GPU")
    return torch.device("cuda", index)
