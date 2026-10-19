"""根据 YAML 配置训练 AlexNet / VGG 花卉分类或 LeNet / CIFAR-10。"""

from collections.abc import Sequence
from dataclasses import asdict
from time import perf_counter

import torch.nn as nn
import torch.optim as optim

from config import load_config, project_path, save_config
from config.cli import parse_args
from dataset import build_loaders
from engine import evaluate, train_one_epoch
from experiment import (
    log_tensorboard_metrics,
    save_class_indices,
    save_model_source,
    save_training_results,
    tensorboard_writer,
    training_session,
)
from model import build_model
from runtime import get_device, seed_everything


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, args)

    # 配置读取成功后，直接在对应的实验目录中建立日志。
    with training_session(config, argv=argv) as (logger, run_dir):
        logger.info("读取配置文件：%s；命令行参数：%s", args.config, vars(args))

        if args.smoke_test:
            config.epochs = args.epochs if args.epochs is not None else 1
            logger.warning("SMOKE TEST: 使用随机图像，准确率不代表真实数据集效果。")
        logger.info("训练配置：%s", asdict(config))

        seed_everything(config.seed)
        device = get_device(config.device)
        config.device = str(device)
        save_config(config, run_dir / "config.yaml")
        logger.info("已保存配置：config.yaml")
        logger.info("计算设备：%s；随机种子：%d", device, config.seed)

        # TensorBoard 与文本日志共用本次运行目录，退出时自动刷新并关闭。
        with tensorboard_writer(run_dir / "tensorboard", config) as writer:
            logger.info("TensorBoard 目录：%s", run_dir / "tensorboard")

            logger.info("开始加载数据；本地数据目录：%s", project_path(config.data_dir))
            train_loader, val_loader = build_loaders(config, smoke_test=args.smoke_test)
            logger.info("数据加载完成：train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))
            
            classes = list(train_loader.dataset.classes)
            save_class_indices(classes, run_dir / "class_indices.json")
            logger.info("类别编号：%s", dict(enumerate(classes)))
            logger.info("已保存类别映射：class_indices.json")

            model = build_model(config.model_name, len(classes)).to(device)
            save_model_source(type(model), run_dir / "model.py")
            logger.info("已保存模型源码：model.py")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            logger.info("模型：%s", model)

            history, best_accuracy = [], -1.0
            for epoch in range(1, config.epochs + 1):
                # 训练
                epoch_start = perf_counter()
                logger.info("Epoch %d/%d 开始训练", epoch, config.epochs)
                train_start = perf_counter()
                train_metrics = train_one_epoch(model, train_loader, criterion, device, optimizer, logger, config.log_interval)
                train_seconds = perf_counter() - train_start
                # 验证
                logger.info("Epoch %d/%d 开始验证", epoch, config.epochs)
                val_start = perf_counter()
                val_metrics = evaluate(model, val_loader, criterion, device)
                val_seconds = perf_counter() - val_start
                timings = {"train_seconds": train_seconds, "test_seconds": val_seconds, "epoch_seconds": perf_counter() - epoch_start}

                history.append(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "time": timings,
                    }
                )
                log_tensorboard_metrics(
                    writer,
                    epoch,
                    train_metrics,
                    val_metrics,
                    optimizer.param_groups[0]["lr"],
                    timings,
                )
                logger.info(
                    "Epoch %d/%d train_loss=%.4f train_accuracy=%.2f%% "
                    "val_loss=%.4f val_accuracy=%.2f%% "
                    "训练耗时=%.2f秒 测试耗时=%.2f秒 本轮总耗时=%.2f秒",
                    epoch,
                    config.epochs,
                    train_metrics["loss"],
                    train_metrics["accuracy"] * 100,
                    val_metrics["loss"],
                    val_metrics["accuracy"] * 100,
                    train_seconds,
                    val_seconds,
                    timings["epoch_seconds"],
                )

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "classes": classes,
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "smoke_test": args.smoke_test,
                }
                is_best = val_metrics["accuracy"] > best_accuracy
                save_training_results(checkpoint, history, run_dir, is_best=is_best)
                if is_best:
                    best_accuracy = val_metrics["accuracy"]
                logger.info("已保存权重和 history.json；更新最佳权重：%s", is_best)

            logger.info("训练完成，最佳验证准确率：%.2f%%；输出：%s", best_accuracy * 100, run_dir)


if __name__ == "__main__":
    main()
