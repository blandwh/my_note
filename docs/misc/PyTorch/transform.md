# Transforms 相关

使用 `transform` 的目的是，把原始类型的数据，变换成我们可以用来训练的类型。所有 TorchVision 数据集都包含两个参数，`transform` 修改特征，`target_transform` 修改标签。

## Lambda Transforms

使用 `Lambda transform` 可以实现自定义的函数。

```python
target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```

`lambda y:` 定义一个接受标签 y 的函数，之后 `tensor` 转化成整数张量，之后 `one_hot` 转化成 `one_hot` 向量，最后 `v2.Lambda` 把这个函数封装成一个 `transform`。

