"""加载实验权重，对一张图片进行花卉或 CIFAR-10 分类。"""

from collections.abc import Sequence

from matplotlib import pyplot as plt
import torch
from PIL import Image

from config import project_path
from config.cli import parse_predict_args
from dataset import image_transform
from model import build_model
from runtime import get_device


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_predict_args(argv)
    device = get_device(args.device)

    # 加载模型权重, 仅加载模型参数，避免在 CPU 上分配大模型。
    checkpoint = torch.load(
        project_path(args.checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    # 读取模型名称和类别列表
    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]

    # 在 meta 设备上构建结构，直接接管 CPU 权重，避免 VGG 先分配另一份大参数。
    with torch.device("meta"):
        model = build_model(model_name, len(classes), init_weights=False)
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    model.to(device).eval()

    transform = image_transform(model_name)
    image = Image.open(project_path(args.image))
    inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)
    plt.imshow(image)
    plt.show()
    
    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)

    # 类别顺序以训练权重中的记录为准。
    print(f"模型：{model_name}；设备：{device}")
    if checkpoint.get("smoke_test", False):
        print("SMOKE TEST 权重：预测仅用于流程验证。")
    print(f"预测类别：{classes[index.item()]}；置信度：{confidence.item():.2%}")


if __name__ == "__main__":
    main()
