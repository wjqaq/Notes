# 代码
#### 对比解码代码
$$
P_{vcd}(y|v,v',x) = softmax[(1+\alpha)logit_{\theta}(y|v,x) - \alpha logit_{\theta}(y|v',x)]
$$
$$
\mathcal{V}_{head}(y<_t) = \{ y_t \in \mathcal{V} : p_{\theta}(y_t|v,x,y<_t) \geq \beta  \max_{w}p_{\theta}(w|v,x,y<_t) \},
$$
$$
p_{vcd}(y_t|v,v',x) = 0 \; if \; y_t \notin \mathcal{V}_{head}(y<_t)
$$
源文件：
版本1：
```python
	probs = nn.functional.softmax(next_token_logits, dim=-1)
	cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values
```
版本2：
```python
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
```
