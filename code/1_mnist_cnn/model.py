from torch import nn

class CNN(nn.Module):
    """
    卷积网络

    两层设计：
    卷积(Conv2d) -> 激励函数(ReLU) -> 池化(MaxPooling) -> 
    卷积(Conv2d) -> 激励函数(ReLU) -> 池化(MaxPooling) -> 
    展平多维的卷积成的特征图->接入全连接层(Linear)->输出 
    """
    def __init__(self):
        super().__init__()

        
        self.conv1 = nn.Sequential(
            # 输入(1,28,28)
            # 第一个卷积con2d
            nn.Conv2d(
                in_channels=1,  # 输入通道数
                out_channels=16, # 输出通道数，使用16个卷积核，生成16张特征图
                kernel_size=5,  # 卷积和大小
                stride=1,       # 卷积步长
                padding=2       # 四周各填充2个像素，stride=1时输出仍为28×28；不填充则为24×24
            ),
            # 输出(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 最大池化，2x2下采样
            nn.MaxPool2d(kernel_size=2),
            # 输出(16,14,14)
        )

        self.conv2 = nn.Sequential(
            # 输入(16,14,14)
            # 第二个卷积con2d
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出(32,7,7)
        )

        # 全连接层：输出 y, yi代表数字i的原始预测分数
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """前向传播"""
        # 输入 x = (B, 1,28,28) 待识别数字
        x = self.conv1(x)   # 第一层卷积 x = (B, 16,14,14)
        x = self.conv2(x)   # 第二层卷积 x = (B, 32,7,7)
        x = x.view(x.size(0), -1)   # 展平特征图 X = (B, 32 * 7 * 7)
        return self.out(x)  # 全连接线性层
