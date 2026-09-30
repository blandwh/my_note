# PyTorch 相关

以下知识点基本都基于[官方教程](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## 数据集相关的准备

涉及到两个库，`Dataset` 以及 `DataLoader` 前者包含了各种数据集（也就是样本与标签），后者则将前者进一步封装成可迭代的形式，方便进行批处理等操作。

以 MNIST 数据集为例，下面是代码：

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
```

其中 train 表示是否选用训练集（否代表选用测试集），download 先检测 root 下是否有数据集，如果没有则进行下载。Compose 表示依次进行列表当中的两个操作，ToImage 表示把图片转为可处理的张量，ToDtype 表示转变数据类型，并对像素值做缩放，按范围映射到 $[0,1]$。

```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

batch_size 表示批大小，X y 分别对应每一批数据（也就是 sample 以及对应的 label）

## Model 的创建

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

首先是设备的选择，这里的第一条代码表示，如果有可以选择的加速器（CUDA、MPS、XPU 等等），那么就使用加速器，否则使用 CPU。之后模型的创建往往都继承 `nn.Module`。`__init__` 函数用来搭建层，构建框架，需要先 `super().__init__()` 父类，之后在进行自己的操作。`Flatten` 表示把图片张量占平，但对于某一批数据，其 batch 维度 $N$ 会被保留。之后 `Sequential` 表示依次搭建这些层，`Linear(512, 10)` 表示经过线性层，单个数据的维度从 512 变成 10。Linear 当中有需要训练的权重和偏置。

然后 `forward` 表示数据如何经过我们搭建的层，这里就是先展平之后经过 `linear_relu_stack`。

最后 `NeuralNetwork()` 创建模型，并初始化其中的可学习参数；`.to(device)` 把模型放到选定设备上；`print(model)` 展示模型的层结构。之后把图片送进模型时，图片也要放到同一个设备，例如 `images = images.to(device)`。

## 优化 Model 参数

需要 `loss function` 以及 `optimizer`。

