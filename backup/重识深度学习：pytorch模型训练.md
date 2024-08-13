
# 数据准备

pytorch能够通过`DataLoader`去迭代输出`Dataset`中的每条batch数据，解耦代码提高可读性与更好的模块化。
## Dataset
自定义`Dataset`类需要实现至少如下三个函数：`__init__，__len__,__getitem__`，基础的实现方式如下所示：
```python
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data['input']
        label = self.data['label']
        return input, label
```
## DataLoader
`DataLoader`能够方便的读取与迭代数据，  其创建方式如下：
```
torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='')
```
1. dataset (必需): 用于加载数据的数据集，通常是torch.utils.data.Dataset的子类实例。
2. batch_size (可选): 每个批次的数据样本数。默认值为1。
3. shuffle (可选): 是否在每个周期开始时打乱数据。默认为False。
4. sampler (可选): 定义从数据集中抽取样本的策略。如果指定，则忽略shuffle参数。
5. batch_sampler (可选): 与sampler类似，但一次返回一个批次的索引。不能与batch_size、shuffle和sampler同时使用。
6. num_workers (可选): 用于数据加载的子进程数量。默认为0，意味着数据将在主进程中加载。
7. collate_fn (可选): 如何将多个数据样本整合成一个批次。通常不需要指定。
8. drop_last (可选): 如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次。默认为False。

在训练时，直接迭代DataLoader，每次迭代的输出即为__getitem__的输出（本例即input与label）
```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for inputs, labels in train_loader:
    do_train()
```
# 损失与优化器

损失函数与优化器是编译一个神经网络模型必备的两个参数。
## 损失函数
损失函数是指用于计算标签值和预测值之间差异的函数，在机器学习过程中，有多种损失函数可供选择，典型的有距离向量，绝对值向量等。
随着迭代次数的增加，代表预测值与真实值之间误差的损失函数体现了模型的拟合效果。

### 常见的损失函数

在pytorch中，所有损失函数都属于nn.modules.loss，在初始化时，若`reduction`设置为`none`，则损失等于$L$（此时输出的形状与输入的形状相同，否则输出会是标量）

否则：
$\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}$

#### 1. **L1Loss**

$L$ 代表着所有x与y相差绝对值的集合。

$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|$, 其中 $N$ 为 batch size
```
torch.nn.L1Loss(reduction='mean')
```

#### 2. **MSELoss**
平方损失函数，又称L2Loss，与L1的区别是对差值取平方

$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2$

```
torch.nn.MSELoss(reduction='mean')
```

#### 3. **BCELoss**
测量目标与输入概率的二进制交叉熵
$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, $

$\quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]$
```
torch.nn.BCELoss(weight=None, reduction='mean')
```

- 参数weight（Tensor，可选） – 手动重新缩放权重，用于每个批次元素的损失。如果给定，则必须是大小为 *nbatch* 的 Tensor。
- BCEWithLogitsLoss，添加了sigmoid:
$
       \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],
$

#### 4. **CrossEntropyLoss**
重量级，待更新

## 优化器

基类`torch.optim.Optimizer(params,defaults)`

- 参数
  - params (iterable) —— Variable 或者 dict的iterable。指定了什么参数应当被优化。
  - defaults —— (dict)：包含了优化选项默认值的字典（一个参数组没有指定的参数选项将会使用默认值）。
- 方法
  - load_state_dict(state_dict)：加载optimizer状态。
  - state_dict()：以dict返回optimizer的状态。包含两项：state - 一个保存了当前优化状态的dict，param_groups - 一个包含了全部参数组的dict。
  - add_param_group(param_group)：给 optimizer 管理的参数组中增加一组参数，可为该组参数定制 lr,momentum, weight_decay 等，在 finetune 中常用。
  - step(closure) ：进行单次优化 (参数更新)。
  - zero_grad() ：清空所有被优化过的Variable的梯度。

使用方式:
```python
定义:
optimizer=Optimizer(model.parameters, lr = lr)
optimizer=Optimizer([var1,var2], lr = lr)
```

# 模型定义

## nn.Module
个人理解，pytorch不像tensorflow那么底层，也不像keras那么高层，这里先比较keras和pytorch的一些小区别。

（1）keras更常见的操作是通过继承Layer类来实现自定义层，不推荐去继承Model类定义模型，详细原因可以参见官方文档

（2）pytorch中其实一般没有特别明显的Layer和Module的区别，不管是自定义层、自定义块、自定义模型，都是通过继承Module类完成的，这一点很重要。其实Sequential类也是继承自Module类的。

**注意**：我们当然也可以直接通过继承torch.autograd.Function类来自定义一个层，但是这很不推荐，不提倡，至于为什么后面会介绍。

**总结**：pytorch里面一切自定义操作基本上都是继承nn.Module类来实现的
<p>这是一个普通段落：</p>

```python
class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
        
'''
有一部分没有完全列出来
'''
```

我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：

（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
    
（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。