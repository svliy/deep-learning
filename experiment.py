"""实验运行日志、结果归档与 TensorBoard；产物集中在同一运行目录。"""

from __future__ import annotations

import inspect
import json
import logging
import shlex
import shutil
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import yaml

from config import TrainConfig, project_path
from engine import EpochMetrics

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class _LogStream(TextIOBase):
    """将第三方代码的标准输出、标准错误按行转交给日志记录器。"""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer)
        self.buffer = ""

    def isatty(self) -> bool:
        return False


@contextmanager
def training_session(
    config: TrainConfig,
    *,
    argv: Sequence[str] | None = None,
) -> Generator[tuple[logging.Logger, Path], None, None]:
    """配置读取成功后，在实验目录记录运行日志、异常及结束状态。"""
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    # 根据配置直接创建实验运行目录，日志与权重、指标等结果保存在一起。
    run_dir = project_path(config.output_dir) / config.experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_file = run_dir / "train.log"
    logger = logging.getLogger(f"train.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 禁止向上级记录器传播，避免重复输出。
    # 每条日志包含时间、级别和消息。
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers = [logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # print 等标准输出按 INFO 记录，标准错误按 ERROR 记录。
    stdout = _LogStream(logger, logging.INFO)
    stderr = _LogStream(logger, logging.ERROR)
    start = perf_counter()
    status = "正常结束"
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            logger.info("程序开始")
            logger.info("实验名称：%s", config.experiment_name)
            logger.info(
                "启动命令：%s",
                shlex.join(
                    [sys.executable, sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
                ),
            )
            logger.info("工作目录：%s", Path.cwd())
            logger.info("Python：%s；PyTorch：%s", sys.version.split()[0], torch.__version__)
            logger.info("运行目录：%s", run_dir)
            logger.info("日志文件：%s", log_file)
            try:
                yield logger, run_dir
            except KeyboardInterrupt:
                status = "用户中断"
                logger.warning("训练被用户中断", exc_info=True)
                raise
            except SystemExit as error:
                if error.code not in (None, 0):
                    status = f"退出码 {error.code}"
                    logger.exception("程序退出失败")
                raise
            except Exception:
                status = "运行失败"
                logger.exception("程序运行失败")
                raise
            finally:
                stdout.flush()
                stderr.flush()
                logger.info("程序结束：%s；总耗时：%.2f 秒", status, perf_counter() - start)
    finally:
        # 关闭文件和终端处理器，释放资源。
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def save_class_indices(classes: Sequence[str], path: str | Path) -> None:
    """按训练时的类别顺序保存“类别编号 -> 类别名称”映射。"""
    with project_path(path).open("w", encoding="utf-8") as file:
        json.dump(dict(enumerate(classes)), file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_model_source(model_class: type[torch.nn.Module], path: str | Path) -> None:
    """保存模型类所在的完整 Python 源文件，记录本次训练使用的网络定义。"""
    shutil.copy2(inspect.getfile(model_class), project_path(path))


@contextmanager
def tensorboard_writer(
    log_dir: str | Path, config: TrainConfig
) -> Generator[SummaryWriter, None, None]:
    """创建独立的 TensorBoard 记录，写入配置并确保结束时关闭。"""
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        config_text = yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False)
        writer.add_text("config", f"```yaml\n{config_text}\n```", 0)
        yield writer


def log_tensorboard_metrics(
    writer: SummaryWriter,
    epoch: int,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    learning_rate: float,
    timings: Mapping[str, float],
) -> None:
    """按轮次记录指标、学习率及训练、测试和整轮耗时，并及时写入磁盘。"""
    for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
        for name, value in metrics.items():
            writer.add_scalar(f"{phase}/{name}", value, epoch)
    writer.add_scalar("train/learning_rate", learning_rate, epoch)
    for name, seconds in timings.items():
        writer.add_scalar(f"time/{name}", seconds, epoch)
    writer.flush()


def save_training_results(
    checkpoint: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    is_best: bool = False,
) -> None:
    """将权重和完整训练指标保存到已创建的运行目录。"""
    output_dir = project_path(output_dir)
    torch.save(checkpoint, output_dir / "last.pt")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)
