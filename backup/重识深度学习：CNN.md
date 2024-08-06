# 一、pytorch实现
CNN的基础架构：
![CNN基本结构](https://i-blog.csdnimg.cn/blog_migrate/dd24ffc1b67ac2aa6553ded74168bc47.png)
其中主要的是卷积层与池化层的实现。

```
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 第一个卷积层:输入通道数为1,输出通道数为16,卷积核大小为3x3,步长为1,填充为1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层,池化核大小为2x2,步长为2
        
        # 第二个卷积层:输入通道数为16,输出通道数为32,卷积核大小为3x3,步长为1,填充为1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层,池化核大小为2x2,步长为2
        
        # 全连接层:输入特征向量大小为32*7*7(经过卷积和池化后的特征图大小),输出大小为num_classes
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
 
    def forward(self, x):
        # 前向传播过程:依次经过卷积层、ReLU激活函数、最大池化层,最后通过全连接层得到输出
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 将特征图展平成一维向量
        x = self.fc(x)
        return x

```

## 输入输出
```
model_input: [batch_size,depth,height,width]
model_output: [batch_size,output]
```

# 二、模型拆解

## 1. 卷积核
```
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```
### 输入输出
- Input: $(N, C_{in}, H_{in}, W_{in})$ or $(C_{in}, H_{in}, W_{in})$
- Output: $(N, C_{out}, H_{out}, W_{out})$ or $(C_{out}, H_{out}, W_{out})$

维度计算：

$$
H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
\\

W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
$$

$L =(M-D*(K-1)-1+2P)/S+1$

$D=1时，有L =(M-K+2P)/S+1$

### 计算公式：
$\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)$

 其中：

$\star$ is the valid 2D `cross-correlation`_ operator

$N$ 为 batch size

$C$ 是 number_of_channels,

$H$ is a height of input planes in pixels, and $W$ is width in pixels.



### 参数详解：
- `kernel_size` 卷积核尺寸 `Tuple`&`Int` 例：`（2，3）`代表卷积核为2\*3，输入`3`代表卷积核为3\*3
- `stride` 步长 `Tuple`&`Int` 
- `padding` 填充 `Tuple`&`Int` or `String {‘valid’, ‘same’}` 
- `dilation` `Tuple`&`Int` kernel points的间距,[更多可视化](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) ![dilation](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/dilation.gif) 

- `padding_mode` padding的模式 `String{'zeros'（default）, 'reflect', 'replicate' or 'circular'} `
- `groups` 并排运行几个cov层，每有一个group则将输入与输出拆成一份，`in_channels`与`out_channels`必须被其整除。

## 2.池化层

```
#最大池化
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) 
#均值池化
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```
### 输入输出
- Input: $(N, C, H_{in}, W_{in})$ or $(C, H_{in}, W_{in})$
- Output: $(N, C, H_{out}, W_{out})$ or $(C, H_{out}, W_{out})$

维度计算：

$H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}\times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor$

$W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}\times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor$

### 计算公式（最大池化为例）

$\begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
\end{aligned}$

其中$K$表示$Kernel$



# 三、相关概念：
## padding 方法（padding to where）:
- valid padding（有效填充）：完全不使用填充。
- half/same padding（半填充/相同填充）：保证输入和输出的feature map尺寸相同。
- full padding（全填充）：在卷积操作过程中，每个像素在每个方向上被访问的次数相同。
- arbitrary padding（任意填充）：人为设定填充。
## padding mode (padding with what)：
- zeros： 零填充，即用0进行填充
- reflect： 镜像填充，以矩阵边缘为对称轴，将矩阵中的元素对称的填充到最外围
- replicate：重复填充，直接用边缘的像素值进行填充
- circular：循环填充，



# 四、维度变化
（不考虑batch）

## 输入维度：$(C,H_{in},W_{in})$

- 经过卷积核 
$(N_{Kernel},C,H_{ks},W_{ks})$

`nn.Conv2d(in_channels=C, out_channels=N, kernel_size=K, stride=S, padding=P, dilation=D)` 

## 卷积后维度 $(N_{Kernel},H_{Kout},W_{Kout})$
其中$H_{Kout} =(H_{in}-D*(K-1)-1+2P)/S+1$

- 经过激活层，不改变维度
- 经过池化层

## 池化后维度 $(N_{Kernel}, H_{Pout}, W_{Pout})$
其中的计算方法同上

$H_{Kout} =(H_{in}-D*(K-1)-1+2P)/S+1$
![alt text](image-4.png)



特殊情况
- $P=0, S=K时维度变为H/S--(S-K+2P=0时均成立)$


# 五、外部连接
- [可交互的CNN网络可视化](https://poloclub.github.io/cnn-explainer/)