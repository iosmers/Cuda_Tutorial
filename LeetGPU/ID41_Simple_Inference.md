# LeetGPU Simple Inference 解题思路

> **难度**：Easy  
> **题型**：PyTorch Linear forward / 小模型推理  
> **接口**：
>
> ```python
> def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
>     pass
> ```

---

## 1. 题目要求

给定 GPU 上的输入 tensor 和一个训练好的 `torch.nn.Linear` 模型，计算：

```text
output = input @ weight.T + bias
```

题目允许使用 PyTorch 内置函数。

---

## 2. PyTorch 可提交代码

```python
import torch
import torch.nn as nn

# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.no_grad():
        output.copy_(model(input))
```

---

## 3. 如果手写公式

`nn.Linear` 的权重形状是：

```text
weight: [output_size, input_size]
bias:   [output_size]
```

所以公式是：

```python
output[:] = input @ model.weight.T + model.bias
```

---

## 4. 讲课重点

- 这题不是 CUDA C++ kernel 题，而是 PyTorch GPU tensor 题。
- `output.copy_(...)` 可以保持题目要求的输出 tensor 不变。
- `torch.no_grad()` 避免构建计算图，推理更轻量。
