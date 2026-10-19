import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

import predict
from config import TrainConfig
from dataset import build_loaders
from model import AlexNet, LeNet, build_model


class FlowerTests(unittest.TestCase):
    def make_dataset(self, root, val_classes=("daisy", "tulips")):
        for split, classes in (("train", ("daisy", "tulips")), ("val", val_classes)):
            for category in classes:
                folder = root / split / category
                folder.mkdir(parents=True)
                Image.new("L", (96, 80), color=128).save(folder / "sample.png")

    def test_loaders_use_clean_splits_and_model_matches_classes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            # 原始备份中的额外类别不能混入训练、验证数据。
            backup = root / "flower_photos/other"
            backup.mkdir(parents=True)
            Image.new("RGB", (40, 30)).save(backup / "unused.jpg")
            config = TrainConfig(data_dir=directory, batch_size=2, device="cpu")
            train_loader, val_loader = build_loaders(config)
            self.assertEqual(len(train_loader.dataset), 2)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(train_loader.dataset.classes, ["daisy", "tulips"])
            self.assertEqual(train_loader.dataset.class_to_idx, val_loader.dataset.class_to_idx)
            train_images, _ = next(iter(train_loader))
            images, labels = next(iter(val_loader))
            self.assertEqual(tuple(train_images.shape), (2, 3, 224, 224))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(labels.tolist(), [0, 1])
            torch.testing.assert_close(images, next(iter(val_loader))[0])
            torch.testing.assert_close(images[0, 0], images[0, 1])
            model = build_model(config.model_name, len(train_loader.dataset.classes)).eval()
            with torch.inference_mode():
                self.assertEqual(tuple(model(images).shape), (2, 2))

    def test_train_and_val_must_share_class_mapping(self):
        with TemporaryDirectory() as directory:
            self.make_dataset(Path(directory), val_classes=("daisy",))
            with self.assertRaisesRegex(ValueError, "类别及编号必须一致"):
                build_loaders(TrainConfig(data_dir=directory))

    def test_missing_flower_split_reports_local_path(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "缺少本地花卉数据目录"):
                build_loaders(TrainConfig(data_dir=directory))

    def test_legacy_lenet_checkpoint_still_predicts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "legacy.pt"
            image_path = root / "input.png"
            Image.new("RGBA", (50, 40)).save(image_path)
            torch.save(
                {
                    "model_state_dict": LeNet().state_dict(),
                    "classes": [f"class_{index}" for index in range(10)],
                },
                checkpoint,
            )
            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "predict.py",
                        "--image",
                        str(image_path),
                        "--checkpoint",
                        str(checkpoint),
                        "--device",
                        "cpu",
                    ],
                ),
                redirect_stdout(output),
            ):
                predict.main()
            self.assertIn("预测类别：class_", output.getvalue())

    def test_alexnet_predicts_grayscale_image_using_checkpoint_class_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "alexnet.pt"
            image_path = root / "input.png"
            Image.new("L", (50, 40), color=128).save(image_path)
            # 调整类别顺序，并固定最后一层输出，以验证加载、预处理和标签映射。
            classes = ["tulips", "daisy", "sunflowers", "roses", "dandelion"]
            model = AlexNet(num_classes=len(classes))
            with torch.no_grad():
                model.classifier[-1].weight.zero_()
                model.classifier[-1].bias.zero_()
                model.classifier[-1].bias[2] = 10
            torch.save(
                {
                    "model_name": "alexnet",
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                },
                checkpoint,
            )
            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "predict.py",
                        "--image",
                        str(image_path),
                        "--checkpoint",
                        str(checkpoint),
                        "--device",
                        "cpu",
                    ],
                ),
                redirect_stdout(output),
            ):
                predict.main()
            self.assertIn("模型：alexnet；设备：cpu", output.getvalue())
            self.assertIn("预测类别：sunflowers；置信度：99.98%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
