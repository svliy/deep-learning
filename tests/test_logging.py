import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
from config import PROJECT_ROOT, TrainConfig
from experiment import log_tensorboard_metrics, tensorboard_writer, training_session


class LoggingTests(unittest.TestCase):
    def test_full_log_captures_output_and_runs_are_separate(self):
        with TemporaryDirectory() as directory:
            config = TrainConfig(output_dir=directory)
            with redirect_stderr(io.StringIO()):
                with training_session(config) as (logger, first_run):
                    logger.info("第一轮运行")
                    print("标准输出", end="")
                    sys.stdout.flush()
                    sys.stderr.write("标准错误\n")
                self.assertEqual(logger.handlers, [])
                with training_session(config) as (_, second_run):
                    print("第二轮运行")
            self.assertNotEqual(first_run, second_run)
            first = (first_run / "train.log").read_text(encoding="utf-8")
            second = (second_run / "train.log").read_text(encoding="utf-8")
            for message in (
                "程序开始",
                "启动命令",
                "第一轮运行",
                "标准输出",
                "标准错误",
                "程序结束：正常结束",
                "总耗时",
            ):
                self.assertIn(message, first)
            self.assertIn("第二轮运行", second)
            self.assertNotIn("第一轮运行", second)

    def test_failure_before_tensorboard_starts_is_logged(self):
        with TemporaryDirectory() as directory:
            with (
                patch(
                    "sys.argv",
                    ["train.py", "--output-dir", directory, "--experiment-name", "failure_test"],
                ),
                patch("train.get_device", side_effect=RuntimeError("设备不可用")),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "设备不可用"):
                    train.main()
            (run_dir,) = (Path(directory) / "failure_test").iterdir()
            content = (run_dir / "train.log").read_text(encoding="utf-8")
            for message in (
                "程序开始",
                "Traceback",
                "RuntimeError: 设备不可用",
                "程序结束：运行失败",
            ):
                self.assertIn(message, content)
            self.assertFalse((run_dir / "tensorboard").exists())

    def test_invalid_config_does_not_create_output_directory(self):
        with TemporaryDirectory() as directory:
            config_file = Path(directory) / "invalid.yaml"
            output_dir = Path(directory) / "results"
            config_file.write_text("epochs: 0\n", encoding="utf-8")
            with patch(
                "sys.argv",
                ["train.py", "--config", str(config_file), "--output-dir", str(output_dir)],
            ):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    train.main()
            self.assertFalse(output_dir.exists())

    def test_tensorboard_metrics_can_be_read_after_writer_closes(self):
        with TemporaryDirectory() as directory:
            with tensorboard_writer(directory, TrainConfig(device="cpu")) as writer:
                for epoch in (1, 2):
                    log_tensorboard_metrics(
                        writer,
                        epoch,
                        {"loss": 1.25, "accuracy": 0.5},
                        {"loss": 1.5, "accuracy": 0.4},
                        0.001,
                        {"train_seconds": 1.25, "test_seconds": 0.75, "epoch_seconds": 2.0},
                    )
            events = EventAccumulator(directory).Reload()
            expected = {
                "train/loss": 1.25,
                "train/accuracy": 0.5,
                "val/loss": 1.5,
                "val/accuracy": 0.4,
                "train/learning_rate": 0.001,
                "time/epoch_seconds": 2.0,
                "time/train_seconds": 1.25,
                "time/test_seconds": 0.75,
            }
            self.assertEqual(set(events.Tags()["scalars"]), set(expected))
            self.assertIn("config/text_summary", events.Tags()["tensors"])
            for tag, value in expected.items():
                values = events.Scalars(tag)
                self.assertEqual([item.step for item in values], [1, 2])
                for item in values:
                    self.assertAlmostEqual(item.value, value, places=6)

    def test_keyboard_interrupt_is_logged(self):
        with TemporaryDirectory() as directory:
            config = TrainConfig(output_dir=directory)
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    with training_session(config) as (_, run_dir):
                        raise KeyboardInterrupt()
            content = (run_dir / "train.log").read_text(encoding="utf-8")
            self.assertIn("KeyboardInterrupt", content)
            self.assertIn("程序结束：用户中断", content)

    def test_training_process_exits_cleanly_and_keeps_log_and_events(self):
        # 独立进程覆盖 TensorBoard 退出清理，以及两种模型的完整训练/预测流程。
        script = """
import sys
from pathlib import Path
import train
import predict
from config import DEFAULT_CONFIG
from PIL import Image
root = Path(sys.argv[1])
model_name = sys.argv[2]
train.main(['--config', str(DEFAULT_CONFIG.with_name(f'{model_name}.yaml')),
            '--smoke-test', '--device', 'cpu', '--experiment-name', 'logging_test',
            '--output-dir', str(root / 'experiments'), '--log-interval', '3', '--batch-size', '16'])
run_dir = next((root / 'experiments/logging_test').iterdir())
image_path = root / 'sample.png'
Image.new('L', (80, 60), color=128).save(image_path)
predict.main(['--image', str(image_path), '--checkpoint', str(run_dir / 'best.pt'),
              '--device', 'cpu'])
"""
        for model_name, num_classes in (("alexnet", 5), ("lenet", 10)):
            with self.subTest(model=model_name), TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", script, directory, model_name],
                    cwd=PROJECT_ROOT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    },
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("Exception ignored", result.stderr)
                self.assertIn("预测类别：", result.stdout)
                self.assertIn(f"模型：{model_name}；设备：cpu", result.stdout)
                self.assertIn("SMOKE TEST 权重", result.stdout)
                self.assertFalse((Path(directory) / "outputs/_startup").exists())
                runs = list((Path(directory) / "experiments/logging_test").iterdir())
                self.assertEqual(len(runs), 1)
                content = (runs[0] / "train.log").read_text(encoding="utf-8")
                self.assertIn("Epoch 1/1 train_loss=", content)
                self.assertIn("Step 3/4 loss=", content)
                self.assertIn("Step 4/4 loss=", content)
                self.assertNotIn("Step 1/4 loss=", content)
                for label in ("训练耗时=", "测试耗时=", "本轮总耗时="):
                    self.assertIn(label, content)
                self.assertIn("程序结束：正常结束", content)
                for name in (
                    "best.pt",
                    "last.pt",
                    "history.json",
                    "config.yaml",
                    "class_indices.json",
                    "model.py",
                ):
                    self.assertTrue((runs[0] / name).is_file())
                class_indices = json.loads(
                    (runs[0] / "class_indices.json").read_text(encoding="utf-8")
                )
                checkpoint = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["model_name"], model_name)
                self.assertEqual(len(checkpoint["classes"]), num_classes)
                self.assertEqual(
                    class_indices,
                    {str(index): name for index, name in enumerate(checkpoint["classes"])},
                )
                self.assertEqual(
                    (runs[0] / "model.py").read_bytes(),
                    (PROJECT_ROOT / f"model/{model_name}.py").read_bytes(),
                )
                events = EventAccumulator(str(runs[0] / "tensorboard")).Reload()
                self.assertEqual(len(events.Scalars("train/loss")), 1)
                history = json.loads((runs[0] / "history.json").read_text(encoding="utf-8"))
                timings = history[0]["time"]
                self.assertGreaterEqual(
                    timings["epoch_seconds"],
                    timings["train_seconds"] + timings["test_seconds"],
                )
                for name in ("train_seconds", "test_seconds", "epoch_seconds"):
                    self.assertGreater(timings[name], 0)
                    self.assertAlmostEqual(
                        events.Scalars(f"time/{name}")[0].value,
                        timings[name],
                        places=5,
                    )


if __name__ == "__main__":
    unittest.main()
