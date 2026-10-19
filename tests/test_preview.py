import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from config import DEFAULT_CONFIG, PROJECT_ROOT


class PreviewTests(unittest.TestCase):
    def test_preview_saves_images_and_reports_counts_for_both_datasets(self):
        for model_name, classes in (("alexnet", 5), ("lenet", 10)):
            with self.subTest(model=model_name), TemporaryDirectory() as directory:
                # 同时覆盖既有模块入口和新增的显式 argv 调用。
                entrypoint = (
                    ["-m", "dataset"]
                    if model_name == "alexnet"
                    else ["-c", "import sys; from dataset.preview import main; main(sys.argv[1:])"]
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        *entrypoint,
                        "--config",
                        str(DEFAULT_CONFIG.with_name(f"{model_name}.yaml")),
                        "--smoke-test",
                        "--device",
                        "cpu",
                        "--output-dir",
                        directory,
                        "--experiment-name",
                        "preview_test",
                    ],
                    cwd=PROJECT_ROOT,
                    env={
                        **os.environ,
                        # 即便环境指定了桌面后端，保存图片也不应依赖显示服务。
                        "MPLBACKEND": "TkAgg",
                        "MPLCONFIGDIR": directory,
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    },
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("SMOKE TEST", result.stdout)
                self.assertIn("训练集样本数: 64", result.stdout)
                self.assertIn("验证集样本数: 32", result.stdout)
                counts = [
                    line.split()
                    for line in result.stdout.splitlines()
                    if line.split() and line.split()[0].isdigit()
                ]
                self.assertEqual(len(counts), classes)
                self.assertEqual(sum(int(row[-2]) for row in counts), 64)
                self.assertEqual(sum(int(row[-1]) for row in counts), 32)
                root = Path(directory) / "preview_test/dataset_preview"
                for name in ("samples.png", "class_distribution.png"):
                    with Image.open(root / name) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertGreater(min(image.size), 100)
