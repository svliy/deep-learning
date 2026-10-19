"""共用图像预处理、离线样例与 DataLoader 设置。"""

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import TrainConfig

NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def make_transform(size: int, *, training: bool = False) -> transforms.Compose:
    """训练增强仅用于花卉方案；其他场景固定缩放，再执行同一归一化。"""
    spatial = (
        [transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip()]
        if training else [transforms.Resize((size, size))]
    )
    return transforms.Compose(
        [
            *spatial,
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def create_fake_datasets(
    size: int,
    classes: Sequence[str],
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> tuple[datasets.FakeData, datasets.FakeData]:
    """构造 64/32 个离线样本，用不同偏移区分训练与验证图像。"""
    train_set = datasets.FakeData(
        size=64,
        image_size=(3, size, size),
        num_classes=len(classes),
        transform=train_transform,
        random_offset=0,
    )
    val_set = datasets.FakeData(
        size=32,
        image_size=(3, size, size),
        num_classes=len(classes),
        transform=val_transform,
        random_offset=64,
    )
    train_set.classes = list(classes)
    val_set.classes = list(classes)
    return train_set, val_set


def create_loaders(
    train_set: Dataset,
    val_set: Dataset,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """固定训练抽样种子，验证顺序不打乱；设备由训练入口解析。"""
    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
    }
    generator = torch.Generator().manual_seed(config.seed)
    
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    return train_loader, val_loader
