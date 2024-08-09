## Dropout
```
torch.nn.Dropout(p=0.5, inplace=False)
```

功能：Dropout是一种**正则化**手段，能够减少神经元对上层神经元的依赖，一般放在fc层后防止过拟合、提高模型的泛化能力。
只会在model.train()时激活，在model.eval()时失效。

输入：
- p：元素归零的概率
- inplace：是否改变输入数据，如果设置为True，则会直接修改输入数据；如果设置为False，则不对输入数据做修改


## Linear
```
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```
功能：为输入的特征向量做线性变换
$y = xW^T + b$
- w的shape: $(\text{out\_features}, \text{in\_features})$
  
输入：
- in_features: size of each input sample, shape: $(*, H_{in})$,$H_{in} = \text{in\_features}$
- out_features: size of each output sample,shape:$(*, H_{out})$,$H_{out} = \text{out\_features}$
- bias: If set to ``False``, the layer will not learn an additive bias.
Default: ``True``
## 余弦相似度
```
torch.nn.CosineSimilarity(dim=1, eps=1e-08)
```
$\text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.$

即：$\text{cosine\_similarity} = cos(\theta) = \dfrac{A \cdot B}{\Vert A \Vert  \Vert B\Vert}$

参数：
- dim：相似度计算的维度
- eps：防止除以零所设的下线。

输入：$x_1:(\ast_1, D, \ast_2) ; x_2:(\ast_1, D, \ast_2)$

输出：$(\ast_1, \ast_2)$ 取值为$(-1,1)$

