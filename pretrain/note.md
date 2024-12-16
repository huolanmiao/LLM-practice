# First commit
## CausalSelfAttention: 将QKV和多头的运算，利用分块矩阵乘法的性质，只做一次矩阵乘法
1. 定义layer，3 * config.n_embd分别是QKV的weights
```python
# n_embed = n_head * head_size
# key, query, value projections for all heads, but in a batch
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
# output projection
self.c_proj = nn.Linear(config.n_embd, config.n_embd)
```
2. QKV一次算出来，然后分成Q、K、V，再分成多头
```python
# 计算QKV然后分块
qkv = self.c_attn(x)
q, k, v = qkv.split(self.n_embd, dim=2)

# 分成多个头，每个头分别做self-attention
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
```

3. 对每个头分别做attention matrix的计算
```python
# Batched attention mechanisms
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```
4. Attention mask是通过将希望mask的位置赋为-inf，使得softmax之后概率为0。
```python
# 实现注意力掩码，然后对Value做加权
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```
5. 合并每个head的结果，得到hidden_size与最初的embedding_dim相同 
```python
# .contiguous()确保返回一个连续的张量。
y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
```
6. 最后做一次proj (n_embed * n_embed)，得到hidden_states
```python
# output projection
y = self.c_proj(y)
```

## MLP层：采用GELU激活函数，采用一种近似估计
https://pytorch.org/docs/stable/generated/torch.nn.GELU.html 

<img src="./figures/gelu.bmp" alt="Python Logo" width="500"/>

## Small Summerize
- **Attention**: map, get the relationship across tokens.
- **MLP**: reduce, think individually.

## Block forward: layernorm前置，residual不做layernorm
1. We want clean residual pass way
2. 同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。
https://kexue.fm/archives/9009

<img src="./figures/layernorm.bmp" alt="Python Logo" width="500"/>

## GPTConfig() and GPT class
1. vocab_size的来源
```python
block_size: int = 1024 # max sequence length
vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
```
2. wte和wpe的维度
```python
# Token Embedding [50257, 768]
wte = nn.Embedding(config.vocab_size, config.n_embd),
# Positional Embedding [1024, 768]
wpe = nn.Embedding(config.block_size, config.n_embd),
```
3. from_pretrained 负责从预训练模型加载权重

# Add forward() function of GPT2 nn.Module
1. 输入的形状是(B, T)，即(batch, token_length)，这里的B是batched calculation计算一次的大小，并不等于用于更新梯度的batchsize，batchsize = B * T * num_accum_steps(需要串行的部分) * num_processes(可以放多张卡上并行)。token_length取决于设置的context_length，最多能根据多少个上文的token来预测下一个token。
2. 中间运算过程：先将token embedding和positional embedding加起来，其中pos embedding对于每一行相同，需要利用广播机制。然后，循环经过每个block，每个block中做attention和mlp，其中有前面提到的，clean residual和前置layernorm。最后做一次layernorm。相当于对下图，下面的部分迭代多次，再做最上面的layernorm。
   
<img src="./figures/prenorm.bmp" alt="Python Logo" width="400"/>

3. 输出将hidden_states的维度 (B, T, n_embd) 经过 lm_head 映射到logits的维度(B, T, vocab_size)。每一行（总共B个句子）T个token，每个token都 tend to 前面的token，得到自己的hidden_state，以此预测自己的下一个token的概率。每一个位置预测的都是对应的下一个token的概率，(B, T)的输入得到的是(B, T)的预测输出，也就是将进行B*T次loss的计算。

# Generate from the model
1. 预测下一个token。因为是在做inference，所以只需要拿到最后一个token预测出的下一个token的logits即可。
```python
logits = model(x) # (B, T, vocab_size)
# take the logits at the last position
logits = logits[:, -1, :] # (B, vocab_size)
```
2. TopK选取概率最高的k个token，重新归一化，然后采样。这样可以避免sample到显然不合理的低概率值。topk_indices记录了选取的50个token到原词表中token的位置的映射。
```python
topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
```
3. torch.gather根据给定索引从一个张量中提取元素。
```python
# torch.gather(input, dim, index, out=None)
ix = torch.multinomial(topk_probs, 1) # (B, 1)
# gather the corresponding indices
xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
```