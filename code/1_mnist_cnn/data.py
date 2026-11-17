from torchvision import datasets, transforms

def load_data(data_dir, train, download=True):
    """加载 MNIST 数据集"""
    return datasets.MNIST(
        root=data_dir,  # 保存位置
        train=train,    # true训练集，false测试集
        transform=transforms.ToTensor(),    # 将图片转成张量
        download=download,  # 首次加载需要先下载
    )
