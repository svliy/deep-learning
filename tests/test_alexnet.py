import unittest
from unittest.mock import patch

import torch
from torch import nn

from model import AlexNet, build_model


class AlexNetTests(unittest.TestCase):
    def test_factory_can_skip_initialization_when_loading_weights(self):
        with patch("model.alexnet.AlexNet._initialize_weights") as initialize:
            build_model("alexnet", 5, init_weights=False)
            initialize.assert_not_called()
            build_model("alexnet", 5)
            initialize.assert_called_once()

    def test_forward_preserves_batch_and_supports_custom_classes(self):
        for batch_size, num_classes in ((1, 1000), (2, 10)):
            with self.subTest(batch_size=batch_size, num_classes=num_classes):
                model = AlexNet(num_classes=num_classes).eval()
                with torch.inference_mode():
                    logits = model(torch.randn(batch_size, 3, 224, 224))
                self.assertEqual(tuple(logits.shape), (batch_size, num_classes))
                self.assertTrue(torch.isfinite(logits).all())

    def test_initialized_model_supports_cross_entropy_backward(self):
        model = AlexNet(num_classes=5, init_weights=True)
        logits = model(torch.randn(2, 3, 224, 224))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for weight in (model.features[0].weight, model.classifier[-1].weight):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(torch.count_nonzero(weight.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
