
## 数据准备

pytorch能够通过`DataLoader`去迭代输出`Dataset`中的每条batch数据，解耦代码提高可读性与更好的模块化。
### Dataset
自定义`Dataset`类需要实现至少如下三个函数：`__init__，__len__,__getitem__`，基础的实现方式如下所示：
```
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
### DataLoader
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
```
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for inputs, labels in train_loader:
    do_train()
```
## 模型、损失与优化器加载
模型直接加载定义好的模型即可。

损失函数与优化器是编译一个神经网络模型必备的两个参数。
损失函数是指用于计算标签值和预测值之间差异的函数，在机器学习过程中，有多种损失函数可供选择，典型的有距离向量，绝对值向量等。
