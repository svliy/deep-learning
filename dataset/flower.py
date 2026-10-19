"""读取 flower_data/train、val，并提供 AlexNet / VGG 共用的图像预处理。"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import TrainConfig, project_path

from .common import create_fake_datasets, create_loaders, make_transform

CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")


def image_transform(training: bool = False) -> transforms.Compose:
    """训练时增强图像，验证和预测时使用相同的固定预处理。"""
    return make_transform(224, training=training)


def build_loaders(config: TrainConfig, smoke_test: bool = False) -> tuple[DataLoader, DataLoader]:
    """仅读取本地 train/val 划分，或构造互不重叠的离线样例。"""
    train_transform = image_transform(training=True)
    val_transform = image_transform()
    if smoke_test:
        train_set, val_set = create_fake_datasets(224, CLASSES, train_transform, val_transform)
    else:
        root = project_path(config.data_dir)
        for split in ("train", "val"):
            if not (root / split).is_dir():
                raise FileNotFoundError(f"缺少本地花卉数据目录：{root / split}，请检查 data_dir")
        # 使用 ImageFolder 读取清理后的花卉训练集和验证集，要求类别编号一致。
        train_set = datasets.ImageFolder(root / "train", transform=train_transform)
        val_set = datasets.ImageFolder(root / "val", transform=val_transform)
        if train_set.class_to_idx != val_set.class_to_idx:
            raise ValueError("花卉训练集和验证集的类别及编号必须一致")
    return create_loaders(train_set, val_set, config)
