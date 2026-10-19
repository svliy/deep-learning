import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LeNet(nn.Module):
    """用于 CIFAR-10 的 LeNet，输入形状为 [N, 3, 32, 32]。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise ValueError("LeNet 输入必须是 [N, 3, 32, 32]")
        # 空间尺寸：32 -> 28 -> 14 -> 10 -> 5。
        x = F.relu(self.conv1(x))  # [B, 3, 32, 32] -> [B, 6, 28, 28]
        x = self.pool1(x)  # [B, 6, 28, 28] -> [B, 6, 14, 14]
        x = F.relu(self.conv2(x))  # [B, 6, 14, 14] -> [B, 16, 10, 10]
        x = self.pool2(x)  # [B, 16, 10, 10] -> [B, 16, 5, 5]
        x = x.flatten(start_dim=1)  # [B, 16, 5, 5] -> [B, 400]
        x = F.relu(self.fc1(x))  # [B, 400] -> [B, 120]
        x = F.relu(self.fc2(x))  # [B, 120] -> [B, 84]
        x = self.fc3(x)  # [B, 84] -> [B, 10]
        return x


if __name__ == "__main__":
    model = LeNet()
    print(model)
