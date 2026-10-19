"""统计数据集样本数量，并保存样例图和类别分布图。"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from torch.utils.data import Dataset

from config import load_config, project_path
from config.cli import parse_args
from dataset import build_loaders
from dataset.common import NORMALIZE_MEAN, NORMALIZE_STD
from runtime import seed_everything


def class_sample_indices(dataset: Dataset) -> list[list[int]]:
    """按类别收集样本下标，用于统计数量和选择预览图片。"""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        # FakeData 不保存 targets，需要逐个读取标签。
        targets = [dataset[index][1] for index in range(len(dataset))]
    indices = [[] for _ in dataset.classes]
    for index, label in enumerate(targets):
        indices[int(label)].append(index)
    return indices


def plot_class_distribution(
    classes: Sequence[str],
    train_counts: Sequence[int],
    val_counts: Sequence[int],
    path: str | Path,
) -> None:
    """绘制各类别的训练、验证样本数，并保存图片。"""
    figure = Figure(figsize=(max(8, len(classes) * 1.5), 4.5))
    axis = figure.subplots()
    positions = np.arange(len(classes))
    train_bars = axis.bar(positions - 0.2, train_counts, width=0.4, label="Train", color="#4c78a8")
    val_bars = axis.bar(positions + 0.2, val_counts, width=0.4, label="Validation", color="#f58518")
    axis.bar_label(train_bars, padding=3)
    axis.bar_label(val_bars, padding=3)
    axis.set_xticks(positions, classes, rotation=25 if len(classes) > 5 else 0)
    axis.set_ylabel("Number of images")
    axis.set_title("Class distribution")
    axis.set_ylim(0, max(1, *train_counts, *val_counts) * 1.2)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(project_path(path), dpi=150, bbox_inches="tight")


def plot_dataset_samples(
    train_set: Dataset,
    val_set: Dataset,
    train_indices: Sequence[Sequence[int]],
    val_indices: Sequence[Sequence[int]],
    path: str | Path,
) -> None:
    """每类展示一张训练、一张验证图片，标题标出集合与类别。"""
    classes = train_set.classes
    figure = Figure(figsize=(max(8, len(classes) * 2.6), 6))
    axes = figure.subplots(2, len(classes), squeeze=False)
    for row, (name, dataset, indices) in enumerate(
        (
            ("Train", train_set, train_indices),
            ("Validation", val_set, val_indices),
        )
    ):
        for label, category in enumerate(classes):
            axis = axes[row, label]
            axis.set_title(f"{name}: {category}", fontsize=11)
            axis.axis("off")
            if not indices[label]:
                axis.text(0.5, 0.5, "No samples", ha="center", va="center")
                continue
            image, _ = dataset[random.choice(indices[label])]
            # 当前两套预处理均以 0.5 归一化，将 [-1, 1] 还原为 [0, 1] 后显示。
            image = image.detach().cpu()
            mean = image.new_tensor(NORMALIZE_MEAN).view(3, 1, 1)
            std = image.new_tensor(NORMALIZE_STD).view(3, 1, 1)
            image = (image * std + mean).clamp(0, 1)
            axis.imshow(image.permute(1, 2, 0).numpy())
    figure.suptitle("Dataset samples after preprocessing", fontsize=16)
    figure.tight_layout()
    figure.savefig(project_path(path), dpi=150, bbox_inches="tight")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv, preview=True)
    config = load_config(args.config, args)
    seed_everything(config.seed)
    train_loader, val_loader = build_loaders(config, smoke_test=args.smoke_test)
    train_set, val_set = train_loader.dataset, val_loader.dataset
    classes = train_set.classes
    if args.smoke_test:
        print("SMOKE TEST: 当前预览为随机图像，不是真实数据集样例。")

    train_indices = class_sample_indices(train_set)
    val_indices = class_sample_indices(val_set)
    train_counts = [len(indices) for indices in train_indices]
    val_counts = [len(indices) for indices in val_indices]
    print(f"训练集样本数: {len(train_set)}")
    print(f"验证集样本数: {len(val_set)}")
    print(f"{'编号':<6} {'类别':<16} {'训练集':>8} {'验证集':>8}")
    for index, category in enumerate(classes):
        print(f"{index:<6} {category:<16} {train_counts[index]:>8} {val_counts[index]:>8}")

    output_dir = project_path(config.output_dir) / config.experiment_name / "dataset_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.png"
    counts_path = output_dir / "class_distribution.png"
    plot_dataset_samples(train_set, val_set, train_indices, val_indices, samples_path)
    plot_class_distribution(classes, train_counts, val_counts, counts_path)
    print(f"样例图片: {samples_path}")
    print(f"类别分布: {counts_path}")
