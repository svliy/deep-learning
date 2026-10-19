"""图像分类网络与模型工厂。"""

from torch import nn

from .alexnet import AlexNet
from .lenet import LeNet
from .vgg import VGG, VGG_BLOCKS, vgg

__all__ = ["AlexNet", "LeNet", "VGG", "vgg", "build_model"]


def build_model(model_name: str, num_classes: int, *, init_weights: bool = True) -> nn.Module:
    """按实际类别数构建模型；加载已有权重时跳过额外的显式初始化。"""
    if model_name == "alexnet":
        return AlexNet(num_classes=num_classes, init_weights=init_weights)
    if model_name == "lenet":
        if num_classes != 10:
            raise ValueError("当前 LeNet 用于 CIFAR-10，类别数必须是 10")
        return LeNet()
    if model_name in VGG_BLOCKS:
        return vgg(model_name, num_classes=num_classes, init_weights=init_weights)
    raise ValueError("model_name 必须是 lenet、alexnet、vgg11、vgg13、vgg16 或 vgg19")
