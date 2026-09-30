# PyTorch 相关流程速查

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

需要 `loss function` 以及 `optimizer`，比如以下例子，使用了随机梯度下降以及交叉熵损失。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

接下来看一个训练过程当中的反向传播以及参数更新。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

其中 `size` 代表数据集总共的大小。`model.train()` 是一个开关，代表把模型切换带训练模式，这个语句本身不会引发模型的参数更新，需要这个语句，是因为有一些层，比如 dropout，在训练以及测试的时候状态不同，需要进行模式切换。

`enumerate` 表示额外给出 batch 编号，之后逐 batch 进行参数更新，也就是循环当中的内容。

`X.to(device)` 和 `y.to(device)` 把数据放到与模型相同的计算设备上，假设 batch 大小为 64，X 通常是 [64, 1, 28, 28]，y 是 \[64]（注意这里时批处理的方式）。

`model(X)` 调用 forward 进行前向计算，之后计算结果和 y 一起送进损失函数算出损失。

`loss.backward()`：计算梯度，结果存在参数的 `.grad` 中。这一步还没有修改权重。`optimizer.step()`：优化器根据刚算出的梯度和学习率，更新权重、偏置。`optimizer.zero_grad()`：清理梯度。 PyTorch 的梯度默认会累积；这里在更新之后清理，保证下一批重新计算自己的梯度。也常见把这句放在每批计算的开头，本质上都是确保更新前没有混入上一批的梯度。

最后是打印部分，`loss.item` 把只有一个值的损失张量转化为普通的数字，便于格式化打印。

接下来是测试的流程：

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

`test_loss`、`correct`：分别用来累计各批损失和预测正确的图片数。`with torch.no_grad()` 在下面的对应缩进当中开启临时环境，表示在缩进块当中，不记录用于反向传播的梯度（因为在 test 当中不需要进行反向传播，这里加上 `with` 可以减少内存占用）。`pred.argmax(1)` 代表沿着 `pred` 的第一个维度（在这里就是类别维度，`pred` 的第零维是 batch 维度，在这里也就是 64）查找最大值，如果最大值等于 y，说明预测对了。

## 保存 Model

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

简单理解就是 `model.state_dict()` 取得模型的状态字典，之后序列化到该目录下的 `model.pth` 文件。

## 加载 Model

```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

可以理解成几个部分：`NeuralNetwork().to(device)` 创建一个固定架构的空白模型，`torch.load("model.pth", weight_only=True)` 表示取出之前保存的模型状态（对应之前训练出来的模型权重），之后 `load_state_dict` 装进空白模型。`weights_only=True` 代表使用受限的方式读取保存文件，适合读取由张量组成的 `state_dict`。它限制反序列化时可以创建或调用的对象，减少加载文件时执行任意代码的风险。


