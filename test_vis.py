import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# SSH 会话可能没有继承 WSLg 的显示地址。
if not os.environ.get("DISPLAY") and Path("/mnt/wslg/.X11-unix/X0").exists():
    os.environ["DISPLAY"] = ":0"
plt.switch_backend("TkAgg")

with Image.open(Path(__file__).resolve().parent / "asserts/images/cat.jpg") as image:
    plt.imshow(image)
plt.axis("off")
plt.show(block=True)
