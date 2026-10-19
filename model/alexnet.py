"""参考 WZMIAOMIAO 仓库 Test2_alexnet/model.py 的 AlexNet 实现。"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """输入为 [N, 3, 224, 224]，输出为 [N, num_classes] 的 logits。"""

    def __init__(self, num_classes: int = 1000, init_weights: bool = False) -> None:
        super().__init__()
        # 1. 特征提取：5 个卷积层，空间尺寸依次为 224 -> 55 -> 27 -> 13 -> 6。
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # [N, 48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # [N, 128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # [N, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # [N, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # [N, 128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 128, 6, 6]
        )

        # 2. 分类器：展平后输入 3 个全连接层，Dropout 用于训练时随机丢弃部分特征。
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),  # [N, 4608] -> [N, 2048]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # [N, 2048] -> [N, num_classes]
        )

        # 3. 可选的显式初始化；默认使用 PyTorch 各层自带的初始化。
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [N, 3, 224, 224] -> [N, 128, 6, 6]
        x = torch.flatten(x, start_dim=1)  # [N, 128, 6, 6] -> [N, 4608]
        x = self.classifier(x)  # [N, 4608] -> [N, num_classes]
        return x

    def _initialize_weights(self) -> None:
        """卷积层使用 Kaiming 初始化，全连接层使用正态分布初始化，偏置置零。"""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    print(model)
