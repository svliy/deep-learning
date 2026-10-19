"""根据模型配置选择对应的数据加载和图像预处理。"""

from types import ModuleType

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config import TrainConfig

from . import cifar10, flower

__all__ = ["build_loaders", "image_transform"]


def _dataset_module(model_name: str) -> ModuleType:
    modules = {
        "alexnet": flower,
        "lenet": cifar10,
        "vgg11": flower,
        "vgg13": flower,
        "vgg16": flower,
        "vgg19": flower,
    }
    try:
        return modules[model_name]
    except KeyError:
        raise ValueError("model_name 必须是 lenet、alexnet、vgg11、vgg13、vgg16 或 vgg19") from None


def build_loaders(config: TrainConfig, smoke_test: bool = False) -> tuple[DataLoader, DataLoader]:
    """按既定的模型与数据集配对创建训练、验证加载器。"""
    return _dataset_module(config.model_name).build_loaders(config, smoke_test)


def image_transform(model_name: str) -> Compose:
    """取得与模型匹配的验证、预测预处理。"""
    return _dataset_module(model_name).image_transform()
