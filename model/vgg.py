"""VGG11/13/16/19，无 BatchNorm；参考 WZMIAOMIAO 的 Test3_vggnet 教程。

参考：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test3_vggnet
"""

import torch
from torch import nn

# 五个阶段的卷积层数；每个阶段后接一次池化。
VGG_BLOCKS = {
    "vgg11": (1, 1, 2, 2, 2),
    "vgg13": (2, 2, 2, 2, 2),
    "vgg16": (2, 2, 3, 3, 3),
    "vgg19": (2, 2, 4, 4, 4),
}


class VGG(nn.Module):
    """输入 [N, 3, 224, 224]，输出 [N, num_classes] 的 logits。"""

    def __init__(
        self,
        model_name: str = "vgg16",
        num_classes: int = 1000,
        init_weights: bool = False,
    ) -> None:
        super().__init__()
        if model_name not in VGG_BLOCKS:
            raise ValueError(f"未知 VGG 型号：{model_name}，可选：{', '.join(VGG_BLOCKS)}")

        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels, count in zip(
            (64, 128, 256, 512, 512), VGG_BLOCKS[model_name], strict=True
        ):
            for _ in range(count):
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # 五次池化：224 -> 112 -> 56 -> 28 -> 14 -> 7。
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224):
            raise ValueError("VGG 输入必须是 [N, 3, 224, 224]")
        x = self.features(x)  # [N, 512, 7, 7]
        return self.classifier(x.flatten(start_dim=1))

    def _initialize_weights(self) -> None:
        """沿用教程：卷积和全连接权重使用 Xavier uniform，偏置置零。"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


def vgg(
    model_name: str = "vgg16",
    num_classes: int = 1000,
    init_weights: bool = False,
) -> VGG:
    """保留教程的 vgg(model_name=..., num_classes=..., init_weights=...) 调用方式。"""
    return VGG(model_name=model_name, num_classes=num_classes, init_weights=init_weights)
