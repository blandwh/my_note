# Datasets & DataLoaders

## 数据集的可视化

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

`figure` 代表整张画布，`figsize` 设置画布的尺寸，`cols rows` 代表在其中放三行三列。`torch.randint(len(training_data), size=(1,))` 会产生一个整数张量，范围在 `[0, len(training_data)-1]`。`add_subplot(3, 3, i)` 在 `3×3` 网格的第 i 个位置创建小图；位置按从左到右、从上到下编号。`img.squeeze()` 这张灰度图片的张量形状通常是 `[1, 28, 28]`；去掉长度为 1 的通道维后变成 `[28, 28]`，方便 imshow 当作二维灰度图显示。

## 创建自己的数据集

CSV 文件记录“图片文件名 → 标签”，图片本身放在文件夹里。每次用索引取样本时，程序才去读取对应的图片。

```python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

`__init__` 函数：`pd.read_csv(...)` 把 CSV 读成一张表，存入 `self.img_labels`。`img_dir` 图片所在的文件夹。`transform` 可选的图片处理函数，例如缩放、转换数据类型。`target_transform` 可选的标签处理函数。

`__len__` 函数：CSV 有多少行数据，数据集就有多少个样本。

`__getitem__` 函数：`iloc[idx, 0]` 取 CSV 第 idx 行、第 0 列，也就是图片文件名。`os.path.join(...)` 把文件夹和文件名拼成图片路径。`decode_image(img_path)` 读取图片，得到形状通常为 `[C, H, W]` 的张量



