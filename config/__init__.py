"""训练配置的数据结构、默认路径及字段校验。"""

import re
from argparse import Namespace
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "alexnet.yaml"
DEVICE_PATTERN = re.compile(r"(?:auto|cpu|cuda(?::(?:0|[1-9][0-9]*))?)")


@dataclass
class TrainConfig:
    model_name: str = "alexnet"
    experiment_name: str = "alexnet_flowers"
    data_dir: str = "dataset/data/flower_data"
    output_dir: str = "outputs"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0002
    log_interval: int = 20
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        """校验字段类型和取值范围，确保训练配置合法。"""
        if self.model_name not in ("lenet", "alexnet", "vgg11", "vgg13", "vgg16", "vgg19"):
            raise ValueError("model_name 必须是 lenet、alexnet、vgg11、vgg13、vgg16 或 vgg19")
        for name in ("epochs", "batch_size", "log_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if (
            type(self.learning_rate) not in (int, float)
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate 必须是有限正数")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers 必须是非负整数")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        if not isinstance(self.device, str) or not DEVICE_PATTERN.fullmatch(self.device):
            raise ValueError("device 必须是 auto、cpu、cuda 或 cuda:N（例如 cuda:1）")
        if (
            not isinstance(self.experiment_name, str)
            or not self.experiment_name.strip()
            or self.experiment_name != self.experiment_name.strip()
            or self.experiment_name in (".", "..")
            or any(char in self.experiment_name for char in ("/", "\\", "\0"))
        ):
            raise ValueError("experiment_name 必须是非空目录名，不能包含路径分隔符或为 .、..")
        for name in ("data_dir", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空路径字符串")


def project_path(path: str | Path) -> Path:
    """数据、输出和预测文件的相对路径以项目根目录为基准。"""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def override_config(config: TrainConfig, args: Namespace) -> None:
    """仅用显式指定的命令行参数覆盖配置字段。"""
    for field in fields(config):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, value)


def load_config(path: str | Path = DEFAULT_CONFIG, args: Namespace | None = None) -> TrainConfig:
    """按默认值、YAML、命令行的顺序合并配置，并校验最终结果。"""
    with Path(path).expanduser().open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("YAML 配置必须是参数名到参数值的映射")
    unknown = values.keys() - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"未知配置字段：{', '.join(sorted(map(str, unknown)))}")
    config = TrainConfig(**values)
    if args is not None:
        override_config(config, args) # 使用命令行参数覆盖 YAML 配置
    config.validate()
    return config


def save_config(config: TrainConfig, path: str | Path) -> None:
    """保存实际使用的配置，供后续复现实验。"""
    project_path(path).write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
