#!/usr/bin/env bash
set -euo pipefail

# 运行前激活项目的 Python 环境；从任意目录执行均可。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# AlexNet 花卉分类：最佳权重（best.pt）。
python predict.py \
  --checkpoint outputs/alexnet_flowers/20261019_200938_844466/best.pt \
  --image dataset/data/flower_data/val/sunflowers/6053739964_a1d9ab3ed1_n.jpg \
  --device cpu

# AlexNet 花卉分类：最后一轮权重（last.pt）。
python predict.py \
  --checkpoint outputs/alexnet_flowers/20261019_200938_844466/last.pt \
  --image dataset/data/flower_data/val/sunflowers/6053739964_a1d9ab3ed1_n.jpg \
  --device cpu
