"""独立的单轮训练与验证循环，指标按实际样本数加权。"""

import logging
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class EpochMetrics(TypedDict):
    loss: float
    accuracy: float


class _MetricAccumulator:
    """在设备端累积脱离计算图的统计量，仅在读取指标时同步。"""

    def __init__(self, device: torch.device) -> None:
        self.loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.correct = torch.zeros((), device=device, dtype=torch.int64)
        self.samples = 0

    def update(self, loss: Tensor, logits: Tensor, labels: Tensor) -> None:
        count = labels.size(0)
        self.samples += count
        self.loss_sum.add_(loss.detach().to(torch.float64) * count)
        self.correct.add_((logits.detach().argmax(dim=1) == labels).sum())

    def compute(self) -> EpochMetrics:
        if self.samples == 0:
            raise ValueError("数据集不能为空")
        loss_sum, correct = torch.stack((self.loss_sum, self.correct)).cpu().tolist()
        return {"loss": loss_sum / self.samples, "accuracy": correct / self.samples}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer,
    logger: logging.Logger,
    log_interval: int,
) -> EpochMetrics:
    """训练一轮，按间隔及最后一个 batch 输出进度；损失函数须按 batch 取均值。"""
    model.train()
    metrics = _MetricAccumulator(device)
    total_steps = len(loader)
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        metrics.update(loss, logits, labels)

        if step % log_interval == 0 or step == total_steps:
            current = metrics.compute()
            logger.info(
                "Step %d/%d loss=%.4f avg_loss=%.4f accuracy=%.2f%% lr=%.6g",
                step,
                total_steps,
                loss.item(),
                current["loss"],
                current["accuracy"] * 100,
                optimizer.param_groups[0]["lr"],
            )
    return metrics.compute()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """验证一轮；使用评估模式且不构建计算图，损失函数须按 batch 取均值。"""
    model.eval()
    metrics = _MetricAccumulator(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        metrics.update(loss, logits, labels)
    return metrics.compute()
