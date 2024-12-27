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


# 如果不小心commit了大文件，无法同步到远程仓库怎么办？

1. 查找大文件的pack id，从小到大排列
   ```
   git verify-pack -v .git/objects/pack/pack-*.idx | sort -k 3 -g | tail -5
   ```
2. 查找涉及到的文件地址
    ```
    git rev-list --objects --all | grep <id>
    ```
3. 将该文件从历史记录中删除
   ```
   git log --pretty=oneline --branches -- <file_path>
   ```
4. 重写所有commit
   ```
   git filter-branch --index-filter 'git rm --cached --ignore-unmatch <file_path>' -- --all
   ```
5. 完全删除引用
   ```
   rm -Rf .git/refs/original //删除git的备份
   rm -Rf .git/logs/ // 删除logs
   git gc //收集所有松散对象并将它们存入 packfile（垃圾回收）
   git prune //删除所有过期的、不可达的且未被打包的松散对象
   ```
6. 提交
   ```
   git push origin xxx --force
   ```

# 设置.gitignore忽略不想提交的文件

1. 通过在项目的某个文件夹下定义`.gitignore`文件，在该文件中定义相应的忽略规则，来管理当前文件夹下的文件的Git提交行为，在每一行指定一个忽略规则
2. 设置全局的git .gitignore文件来管理所有Git项目的行为，创建相应的`.gitignore`文件
   ```
   git config --global core.excludesfile ~/.gitignore
   ```

此处`.gitignore`文件中添加一行`*.safetensors`忽略以safetensors结尾的模型