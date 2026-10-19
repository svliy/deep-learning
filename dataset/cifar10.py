"""本地 CIFAR-10 数据加载及与预测共用的预处理。"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import TrainConfig, project_path

from .common import create_fake_datasets, create_loaders, make_transform

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def image_transform() -> transforms.Compose:
    """验证和预测使用相同的固定缩放与归一化。"""
    return make_transform(32)


def build_loaders(config: TrainConfig, smoke_test: bool = False) -> tuple[DataLoader, DataLoader]:
    """读取本地 CIFAR-10；缺失时报告目录，禁止自动下载。"""
    transform = image_transform()
    if smoke_test:
        train_set, val_set = create_fake_datasets(32, CLASSES, transform, transform)
    else:
        root = project_path(config.data_dir)
        try:
            train_set = datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
            val_set = datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
        except RuntimeError as error:
            raise RuntimeError(
                f"无法加载本地 CIFAR-10（{root}）。"
                "请将完整的 cifar-10-batches-py 目录放入该目录，"
                "或使用 --data-dir 指定其父目录。"
                "离线验证可使用 --smoke-test。"
            ) from error

    return create_loaders(train_set, val_set, config)
