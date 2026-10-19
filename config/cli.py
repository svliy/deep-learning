"""训练、预测和预览的命令行参数；未指定的覆盖项保留为 None。"""

import argparse
from collections.abc import Sequence

from config import DEFAULT_CONFIG


def parse_args(argv: Sequence[str] | None = None, *, preview: bool = False) -> argparse.Namespace:
    """预览复用配置参数，并兼容原有训练参数集合。"""
    description = (
        "数据集统计与样例预览" if preview else "AlexNet / VGG 花卉分类与 LeNet / CIFAR-10 训练 demo"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件")
    parser.add_argument("--experiment-name", help="实验名称，对应输出根目录下的实验文件夹")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--log-interval", type=int, help="每训练多少个 step 输出一次进度，须为正整数")
    parser.add_argument("--device", help="auto、cpu、cuda 或 cuda:N（例如 cuda:1）")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", help="实验输出根目录")
    parser.add_argument("--smoke-test", action="store_true", help="使用随机图像离线验证流程")
    return parser.parse_args(argv)


def parse_predict_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析单图预测参数。"""
    parser = argparse.ArgumentParser(description="加载实验权重进行单图预测")
    parser.add_argument("--image", required=True, help="待预测图片")
    parser.add_argument("--checkpoint", required=True, help="某次实验运行中的 best.pt 或 last.pt")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:N（例如 cuda:1）")
    return parser.parse_args(argv)
