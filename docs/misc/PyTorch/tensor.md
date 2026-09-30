# Tensors 相关

PyTorch 当中的一种特殊的数据结构，用来编码模型的输入输出以及模型当中的参数。官方文档声称其和 `NumPy` 当中的 `ndarray` 很相似。先简单看下 `ndarray` 的部分。

## ndarray 相关

`ndarray` 可以当作高维数组。

```python
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
], dtype=np.float32)

print(a.shape)  # (2, 3)
print(a.ndim)   # 2
print(a.size)   # 6
print(a.dtype)  # float32
print(a[1, 2])  # 6.0
```

其中 `a.ndim` 表示查看这个数组的维度。

## Tensor 的初始化

- 直接从 Python 数据创建

    ```python
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    ```

    `data` 是普通的嵌套列表，之后转化为 int 类型的张量。

- 从 ndarray 创建

    ```python
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    ```

    第一行把 Python 列表变成 NumPy 的 ndarray；第二行把这个数组变成 PyTorch tensor。

    `torch.from_numpy` 有一个值得记住的细节：`np_array` 和 `x_np` 共享底层数据。例如在 CPU 上修改 `np_array[0, 0]`，`x_np[0, 0]` 也会随之改变。相比之下，上面的 `torch.tensor(data)` 会根据输入创建自己的数据副本。

- 参照已有 tensor 创建新的 tensor

    ```python
    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")
    ```

    `ones_like` 表示创建一个和原来类似的全 1 张量，注意这里“类似”指的是数值类型以及形状一样。`rand_like` 填入的则是 `[0,1)` 当中的自然数。

- 随机初始化或者使用常数初始化

    ```python
    shape = (2,3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    ```

## 查看张量的属性

`tensor.shape` 查看形状，`tensor.dtype` 查看数值类型，`tensor.device` 查看所在设备。

## 张量操作

[官方的超详细 API 列表](https://docs.pytorch.org/docs/2.14/torch.html)

```python
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```

tensor 默认在 CPU 上创建，我们需要显式地把 tensor 移到可使用的加速设备上。

- 索引和切片操作

    ```python
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:,1] = 0
    print(tensor)
    ```

    `tensor[0]` 表示选出第 0 行，`tensor[:, 0]` 表示取出所有行的第 0 列的数，也就是第 0 列，`tensor[..., -1]` 表示前面所有维度都保留，取最后一维的最后一个位置，这里就是最后一列。

    其中 `:` 表示“这个维度上的所有位置”。`...` 表示“中间省略的维度全部选取”；因为这里只有行、列两个维度，所以 `tensor[..., -1]` 等价于 `tensor[:, -1]`。-1 表示倒数第一个索引，也就是列号 3。

- 张量的结合

    ```python
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    ```

    其中 `dim=1` 表示结合的方向是第 1 维的方向，也就是沿列的方向拼接，依次划过每一列。

- 算术操作

    ```python
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)


    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    ```

    `tensor.T` 代表矩阵转置。`@ matmul` 都是矩阵乘法，也就是行列点积。`* mul` 代表另一种乘法，也就是逐元素相乘。

- 单元素的张量

    ```python
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
    ```

    `tensor.sum()` 代表把张量当中的所有元素求和，得到一个张量，这个张量只含有一个数。`agg.item()` 可以把这个张量当中的数值取出，得到普通的 Python 数值类型。

- 张量的原地操作

    ```python
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)
    ```

    计算结果直接写回原来的张量，原张量本身会变。PyTorch 的这类方法通常以尾部下划线 `_` 标记，例如 `add_()`、`copy_()`、`t_()`。

    原地操作有时能少创建一个新张量，节省一些内存；但训练模型时，自动求导可能需要使用修改前的值来计算梯度。若提前覆盖了它，反向传播可能报错。因此初学写训练代码时，先优先用普通运算，需要原地操作时再确认它是否影响求导。
