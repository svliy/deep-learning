import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig, load_config
from dataset import build_loaders
from engine import evaluate, train_one_epoch
from model import LeNet


class DemoTests(unittest.TestCase):
    def test_cifar10_shapes_preserve_batch(self):
        model = LeNet()
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    tuple(model(torch.randn(batch_size, 3, 32, 32)).shape),
                    (batch_size, 10),
                )
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 28, 28))

    def test_training_updates_weights_and_testing_does_not(self):
        model = LeNet()
        model.eval()
        loader = DataLoader(
            TensorDataset(
                torch.randn(3, 3, 32, 32),
                torch.tensor([0, 1, 2]),
            ),
            batch_size=2,
        )
        original = model.fc3.weight.detach().clone()
        train_one_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            torch.optim.Adam(model.parameters(), lr=0.001),
            Mock(),
            100,
        )
        self.assertTrue(model.training)
        self.assertFalse(torch.equal(original, model.fc3.weight))
        trained = {key: value.clone() for key, value in model.state_dict().items()}
        forward_states = []
        with model.register_forward_hook(
            lambda module, inputs, output: forward_states.append(
                (
                    module.training,
                    torch.is_grad_enabled(),
                    output.requires_grad,
                    torch.is_inference_mode_enabled(),
                )
            )
        ):
            evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertEqual(forward_states, [(False, False, False, True)] * len(loader))
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(trained[key], value))

    def test_testing_weights_uneven_batches_by_sample_count(self):
        # 三个样本、两个 batch；只有第一个样本分类正确。
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(nn.Identity(), loader, criterion, torch.device("cpu"))
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["loss"], criterion(logits, labels).item(), places=6)

    def test_training_logs_interval_and_final_step_with_weighted_metrics(self):
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
        labels = torch.tensor([0, 1, 0, 0, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        for interval, expected_steps in ((1, [1, 2, 3]), (2, [2, 3]), (10, [3])):
            with self.subTest(interval=interval):
                model = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    model.weight.copy_(torch.eye(2))
                # lr=0 保持预测固定，便于核对最后一个不足 batch 的加权统计。
                optimizer = torch.optim.SGD(model.parameters(), lr=0)
                logger = Mock()
                metrics = train_one_epoch(
                    model,
                    loader,
                    criterion,
                    torch.device("cpu"),
                    optimizer,
                    logger,
                    interval,
                )
                messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
                self.assertEqual(len(messages), len(expected_steps))
                for message, step in zip(messages, expected_steps, strict=True):
                    self.assertTrue(message.startswith(f"Step {step}/3 "))
                expected_loss = criterion(logits, labels).item()
                self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
                # 日志保留四位小数，允许批次归约的浮点误差落在舍入边界两侧。
                logged_loss = float(messages[-1].split("avg_loss=")[1].split()[0])
                self.assertAlmostEqual(logged_loss, expected_loss, delta=1e-4)
                self.assertIn("accuracy=60.00%", messages[-1])

    def test_invalid_training_config(self):
        for values in (
            {"epochs": 0},
            {"batch_size": -1},
            {"learning_rate": float("nan")},
            {"learning_rate": "0.001"},
            {"log_interval": 0},
            {"log_interval": -1},
            {"log_interval": 1.5},
            {"log_interval": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig(**values).validate()

    def test_yaml_overrides_do_not_reparse_command_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            path.write_text(
                "# 测试 YAML 注释及部分字段配置\n"
                "epochs: 3\nbatch_size: 36\nnum_workers: 2\nlog_interval: 3\n",
                encoding="utf-8",
            )
            args = Namespace(
                epochs=None, batch_size=16, num_workers=0, smoke_test=True, log_interval=2
            )
            # 配置函数必须使用传入的参数，不能自行再次解析进程参数。
            with patch("sys.argv", ["test", "--unrelated-option"]):
                original = load_config(path)
                overridden = load_config(path, args)
            self.assertEqual(original.batch_size, 36)
            self.assertEqual(original.num_workers, 2)
            self.assertEqual(original.log_interval, 3)
            self.assertEqual(overridden.epochs, 3)
            self.assertEqual(overridden.batch_size, 16)
            self.assertEqual(overridden.num_workers, 0)
            self.assertEqual(overridden.log_interval, 2)
            self.assertFalse(hasattr(overridden, "smoke_test"))

    def test_cli_overrides_are_validated(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            load_config(args=Namespace(epochs=0))

    def test_empty_loaders_report_an_error_for_training_and_validation(self):
        model = nn.Linear(2, 2)
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "数据集不能为空"):
            evaluate(model, loader, criterion, device)
        with self.assertRaisesRegex(ValueError, "数据集不能为空"):
            train_one_epoch(
                model,
                loader,
                criterion,
                device,
                torch.optim.SGD(model.parameters(), lr=0.1),
                Mock(),
                1,
            )

    def test_unknown_yaml_fields_report_their_names(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.yaml"
            path.write_text("batch_szie: 8\nunknown: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知配置字段：batch_szie, unknown"):
                load_config(path)

    def test_missing_local_dataset_does_not_start_download(self):
        with TemporaryDirectory() as directory:
            config = TrainConfig(model_name="lenet", data_dir=directory)
            with patch("torchvision.datasets.CIFAR10.download") as download:
                with self.assertRaisesRegex(RuntimeError, "无法加载本地 CIFAR-10"):
                    build_loaders(config)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
