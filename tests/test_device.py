import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import torch

import train
from config import DEFAULT_CONFIG, TrainConfig, load_config
from config.cli import parse_args, parse_predict_args
from dataset import build_loaders
from runtime import get_device


class DeviceTests(unittest.TestCase):
    def test_explicit_arguments_do_not_read_process_arguments(self):
        with patch("sys.argv", ["host", "--unrelated-option"]):
            args = parse_args(["--device", "cpu", "--batch-size", "7"])
            prediction = parse_predict_args(
                [
                    "--image",
                    "image.png",
                    "--checkpoint",
                    "model.pt",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(prediction.image, "image.png")
        self.assertEqual(prediction.checkpoint, "model.pt")

    def test_config_and_entrypoints_accept_indexed_cuda(self):
        for name in ("cuda:0", "cuda:1", "cuda:2"):
            with self.subTest(device=name):
                TrainConfig(device=name).validate()
                with patch("sys.argv", ["train.py", "--device", name]):
                    self.assertEqual(parse_args().device, name)
                with patch(
                    "sys.argv",
                    [
                        "predict.py",
                        "--image",
                        "image.png",
                        "--checkpoint",
                        "model.pt",
                        "--device",
                        name,
                    ],
                ):
                    self.assertEqual(parse_predict_args().device, name)

    def test_device_selection_with_multiple_visible_gpus(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=3),
            patch("torch.cuda.current_device", return_value=1),
        ):
            self.assertEqual(get_device("cuda:2"), torch.device("cuda:2"))
            self.assertEqual(get_device("cuda:0"), torch.device("cuda:0"))
            self.assertEqual(get_device("cuda"), torch.device("cuda:1"))
            self.assertEqual(get_device("auto"), torch.device("cuda:1"))
            self.assertEqual(get_device("cpu"), torch.device("cpu"))

    def test_gpu_index_is_checked_before_creating_torch_device(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=3),
        ):
            # 大编号也必须报错，不能在 torch.device 中发生整数截断。
            for name in ("cuda:3", "cuda:256"):
                with self.subTest(device=name), self.assertRaisesRegex(RuntimeError, "越界"):
                    get_device(name)

    def test_cpu_fallback_and_explicit_cuda_failure(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(get_device("auto"), torch.device("cpu"))
            for name in ("cuda", "cuda:0", "cuda:2"):
                with (
                    self.subTest(device=name),
                    self.assertRaisesRegex(RuntimeError, "无法使用 CUDA"),
                ):
                    get_device(name)

    def test_invalid_device_names(self):
        for name in ("cuda:-1", "cuda:abc", "cuda:01", "cpu:1", "mps", "", None):
            with self.subTest(device=name):
                with self.assertRaisesRegex(ValueError, "device"):
                    TrainConfig(device=name).validate()
                with self.assertRaisesRegex(ValueError, "device"):
                    get_device(name)

    def test_indexed_cuda_enables_pinned_memory(self):
        loaders = build_loaders(TrainConfig(device="cuda:2"), smoke_test=True)
        for loader in loaders:
            self.assertTrue(loader.pin_memory)

    def test_training_preserves_gpu_index_in_loaders_and_config_snapshot(self):
        device = torch.device("cuda:2")
        model = Mock()
        model.to.return_value = model
        model.state_dict.return_value = {}
        loader = MagicMock()
        loader.dataset.__len__.return_value = 1
        loader.dataset.classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
        with TemporaryDirectory() as directory:
            args = Namespace(
                config=DEFAULT_CONFIG,
                epochs=1,
                output_dir=directory,
                smoke_test=True,
                device="cuda:2",
            )
            # 用替身跳过 CUDA 运算，检查训练入口是否在传递配置时丢失编号。
            with (
                patch.multiple(
                    train,
                    parse_args=Mock(return_value=args),
                    seed_everything=Mock(),
                    get_device=Mock(return_value=device),
                    build_loaders=Mock(return_value=(loader, loader)),
                    build_model=Mock(return_value=model),
                    train_one_epoch=Mock(return_value={"loss": 1.0, "accuracy": 0.5}),
                    evaluate=Mock(return_value={"loss": 1.0, "accuracy": 0.5}),
                    save_training_results=Mock(),
                    save_model_source=Mock(),
                    training_session=Mock(return_value=nullcontext((Mock(), Path(directory)))),
                    tensorboard_writer=Mock(return_value=nullcontext(Mock())),
                ),
                patch("train.optim.Adam"),
            ):
                train.main()
                self.assertEqual(train.build_loaders.call_args.args[0].device, "cuda:2")
                self.assertEqual(load_config(Path(directory) / "config.yaml").device, "cuda:2")
                train.build_model.assert_called_once_with("alexnet", 5)
                model.to.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
