import unittest
from unittest.mock import patch

import torch
from torch import nn

from config import DEFAULT_CONFIG, TrainConfig, load_config
from dataset import build_loaders, image_transform
from model import VGG, build_model, vgg


class VGGTests(unittest.TestCase):
    def test_all_variants_preserve_standard_architecture_and_batch_shape(self):
        # 标准分类器有上亿参数，使用 meta 检查全部变体的完整图，避免重复分配大权重。
        for model_name, convolutions in (("vgg11", 8), ("vgg13", 10), ("vgg16", 13), ("vgg19", 16)):
            with self.subTest(model=model_name), torch.device("meta"):
                model = build_model(model_name, 5)
                inputs = torch.randn(2, 3, 224, 224)
                features = model.features(inputs)
                self.assertIsInstance(model, VGG)
                self.assertEqual(tuple(features.shape), (2, 512, 7, 7))
                self.assertEqual(
                    sum(isinstance(m, nn.Conv2d) for m in model.modules()), convolutions
                )
                self.assertEqual(sum(isinstance(m, nn.MaxPool2d) for m in model.modules()), 5)
                self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
                linear_shapes = [
                    (m.in_features, m.out_features)
                    for m in model.classifier
                    if isinstance(m, nn.Linear)
                ]
                self.assertEqual(linear_shapes, [(512 * 7 * 7, 4096), (4096, 4096), (4096, 5)])
                self.assertEqual(tuple(model(inputs).shape), (2, 5))

    def test_full_vgg16_graph_supports_cross_entropy_backward(self):
        with torch.device("meta"):
            model = build_model("vgg16", 5)
            logits = model(torch.randn(1, 3, 224, 224))
            loss = nn.CrossEntropyLoss()(logits, torch.tensor([2]))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(parameter.grad.shape, parameter.shape, name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 134281029)

    def test_tutorial_factory_and_initialization_switch(self):
        with torch.device("meta"), patch("model.vgg.VGG._initialize_weights") as initialize:
            model = vgg()
            self.assertEqual(model.classifier[-1].out_features, 1000)
            initialize.assert_not_called()
            build_model("vgg16", 5, init_weights=False)
            initialize.assert_not_called()
            build_model("vgg16", 5)
            initialize.assert_called_once()

    def test_vgg_config_and_all_variants_reuse_flower_data_and_preprocessing(self):
        config = load_config(DEFAULT_CONFIG.with_name("vgg16.yaml"))
        self.assertEqual(config.model_name, "vgg16")
        config.device = "cpu"
        reference = image_transform("alexnet")
        for model_name in ("vgg11", "vgg13", "vgg16", "vgg19"):
            with self.subTest(model=model_name):
                config.model_name = model_name
                config.validate()
                train_loader, val_loader = build_loaders(config, smoke_test=True)
                self.assertEqual(len(train_loader.dataset), 64)
                self.assertEqual(len(val_loader.dataset), 32)
                self.assertEqual(len(train_loader.dataset.classes), 5)
                image, _ = val_loader.dataset[0]
                self.assertEqual(tuple(image.shape), (3, 224, 224))
                self.assertEqual(str(image_transform(model_name)), str(reference))

    def test_unknown_variants_and_wrong_input_shape_report_errors(self):
        for name in ("vgg", "vgg15", "vgg16_bn"):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "model_name"):
                TrainConfig(model_name=name).validate()
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "未知 VGG 型号"):
                vgg(name)
        with torch.device("meta"):
            model = vgg("vgg16", num_classes=5)
            with self.assertRaisesRegex(ValueError, "VGG 输入必须"):
                model(torch.randn(1, 3, 32, 32))
