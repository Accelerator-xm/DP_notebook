# DP Notebook · 深度学习实验与笔记

从可运行的小实验出发，记录模型背后的知识、张量如何流动，以及怎样读懂训练结果。

**在线笔记：[深度学习 · 实验与笔记](https://accelerator-xm.github.io/DP_notebook/)**（首次 Pages 部署成功后可访问）

## 实验与笔记

| 实验 | Demo | 笔记 |
| --- | --- | --- |
| 01 · MNIST 手写数字识别 | [code/1_mnist_cnn](code/1_mnist_cnn/) | [实验入口](docs/1_mnist_cnn/index.md)、[数据与张量](docs/1_mnist_cnn/data.md)、[卷积网络](docs/1_mnist_cnn/cnn.md)、[训练与梯度](docs/1_mnist_cnn/training.md)、[评估与误差分析](docs/1_mnist_cnn/evaluation.md) |

`code/` 与 `docs/` 按相同实验名称对应。基础知识在第一次涉及的实验中解释，后续实验通过链接引用，只补充新的模型、方法与观察，避免反复重写相同内容。

当前保留已有六个 Python 文件及其注释。笔记包含原理示意图，以及依据原实验记录绘制的学习曲线、混淆矩阵、错误样例和特征图。图注区分示意计算与真实结果；98.92% 是一次已保存实验的结果，重新训练不保证得到同一数值。

## 运行 MNIST Demo

使用 Conda 管理 Python 环境，以下命令从仓库根目录执行。已有可用的 `deeplearn` 环境时，可直接激活使用。

```bash
conda create -n dp-notebook python=3.11 pip -y
conda activate dp-notebook
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python code/1_mnist_cnn/train.py
python code/1_mnist_cnn/evaluate.py
```

上述安装适用于 Linux / Windows 的 CPU 环境。使用 NVIDIA GPU 时，首次安装可按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/) 选择与驱动适配的 CUDA 安装包；原实验记录使用 PyTorch 2.14.1、torchvision 0.29.1 的 CUDA 12.6 版本。代码通过 `torch.cuda.is_available()` 自动选择 CUDA 或 CPU。

首次训练会下载 MNIST 到 `code/1_mnist_cnn/data/`，随后将权重、训练曲线数据和最终指标写入 `code/1_mnist_cnn/results/`。评估入口读取刚保存的 `cnn.pth`，因此需要先训练。再次训练会更新结果目录；笔记中的历史实验记录保存在 `docs/1_mnist_cnn/records/`，与新运行结果分开。

## 本地阅读笔记

阅读笔记只需要文档环境，无需安装 PyTorch、下载数据或训练模型：

```bash
conda create -n dp-notebook-docs python=3.12 pip -y
conda activate dp-notebook-docs
python -m pip install -r requirements-docs.txt
python -m mkdocs serve --dev-addr 127.0.0.1:8002
```

打开 <http://127.0.0.1:8002/>。环境只需创建一次，之后激活对应环境再启动即可。图片均保存在仓库内，不依赖外部图床；图中的数字也可结合保留的 CSV / JSON 核对。

## GitHub Pages

将仓库内容推送到 `main` 后，在 [Settings → Pages](https://github.com/Accelerator-xm/DP_notebook/settings/pages) 的 **Build and deployment → Source** 中选择 **GitHub Actions**。仓库已包含工作流，跳过创建模板，再到 **Actions → Build and deploy notes → Run workflow** 运行一次。

工作流只安装文档依赖并构建 `docs/`，不运行训练，也不需要 GPU 或额外 Secrets。后续修改笔记、配图或文档配置并推送后会自动更新；PR 只检查构建。站点地址根据仓库所有者和仓库名自动生成，无需上传 `site/`。

## 目录

```text
DP_notebook/
├── code/
│   └── 1_mnist_cnn/             # 实验一的原始 Python 代码
├── docs/
│   ├── index.md                # 实验入口与知识导航
│   ├── 1_mnist_cnn/
│   │   ├── index.md            # 实验与知识点的对应关系
│   │   ├── data.md             # 数据、张量与批次
│   │   ├── cnn.md              # 卷积、激活、池化与分类
│   │   ├── training.md         # 损失、梯度与参数更新
│   │   ├── evaluation.md       # 指标、误差与泛化
│   │   ├── assets/             # 原理图和实验分析图
│   │   └── records/            # 原实验的 CSV / JSON 记录
│   └── assets/                 # 网站图标与样式
├── .github/workflows/docs.yml  # 文档检查与 Pages 发布
├── mkdocs.yml                  # 主题、导航与搜索
└── requirements-docs.txt       # 文档构建依赖
```

新增实验时，创建同名的 `code/2_实验名/`、`docs/2_实验名/`，并在 `mkdocs.yml` 加入导航。已有知识点可直接链接到第一篇对应章节；实验产生的数据和权重保留在本地，计划展示的分析图与记录放入相应笔记目录。
