# LoRALinear
## 定义lora的rank decomposition matrices
```python
# weight.size(1)是in_channel，weight.size(1)是out_channel
self.lora_right_weight = torch.nn.Parameter(weight.new_zeros((lora_dim, weight.size(0))))
self.lora_left_weight = torch.nn.Parameter(weight.new_zeros((weight.size(1), lora_dim)))
```
## Initialization
```python
# 降维矩阵采用kaiming初始化，升维矩阵初始化为零
# 有一些工作专门研究初始化方法，更好更快的微调
# 例如：PiSSA
torch.nn.init.kaiming_uniform_(self.lora_left_weight, a=math.sqrt(5))
torch.nn.init.zeros_(self.lora_right_weight)
```
## forward()
```python
result = F.linear(input, self.weight, bias=self.bias)    
result += (input @ self.lora_left_weight @ self.lora_right_weight) * self.lora_scaling
```
```python
# input (batch_size, in_features)
# weight (out_features, in_features)
# bias (out_features,)
torch.nn.functional.linear(input, weight, bias=None)
```

# LoRA learning curve

| ![Image 1](./results/lora_1/learning_curve.png) | ![Image 2](./results/lora_2/learning_curve.png) | ![Image 3](./results/lora_4/learning_curve.png) |
|---------------------------|---------------------------|---------------------------|
| ![Image 4](./results/lora_8/learning_curve.png) | ![Image 5](./results/lora_16/learning_curve.png) | ![Image 6](./results/lora_32/learning_curve.png) |

# Generation
