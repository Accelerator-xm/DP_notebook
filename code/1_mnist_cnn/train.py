import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from data import load_data
from engine import evaluate, train_epoch
from model import CNN

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

# 指导书中需要补齐的三个超参数。
EPOCH = 5
BATCH_SIZE = 50
LR = 0.001


def main():
    RESULTS.mkdir(exist_ok=True)

    # 加载训练集
    train_data = load_data(DATA_DIR, train=True)
    # 加载测试集
    test_data = load_data(DATA_DIR, train=False)
    
    # 数据加载器，包装数据的工具，有效迭代数据进行批训练
    # 批处理 batch_size：50个sample，输入(50, 1, 28, 28)
    # shuffle：是否打乱数据
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 兼容 cpu和gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建预测模型
    cnn = CNN().to(device)
    # Adam 优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 交叉熵损失，目标标签直接使用 MNIST 的整数类别
    loss_func = nn.CrossEntropyLoss()

    # 环境与训练配置打印与存储====
    environment = {
        "conda_env": Path(sys.prefix).name,
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    (RESULTS / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "config.json").write_text(
        json.dumps({"epochs": EPOCH, "batch_size": BATCH_SIZE, "lr": LR}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Device: {device}; {environment['device_name']}; "
          f"train={len(train_data)}, test={len(test_data)}", flush=True)
    # =========

    history = []
    start = time.perf_counter()

    # 开始训练
    # 训练轮次 epoch
    for epoch in range(1, EPOCH + 1):
        print(f"Epoch {epoch}/{EPOCH}", flush=True)
        # 每轮训练循环
        epoch_history = train_epoch(cnn, train_loader, loss_func, optimizer, device, test_loader)
        for record in epoch_history:
            history.append({"epoch": epoch,
                            "step": (epoch - 1) * len(train_loader) + record["step"],
                            "train_loss": record["train_loss"],
                            "test_accuracy": record["test_accuracy"]})

    # 保存模型参数
    torch.save(cnn.state_dict(), RESULTS / "cnn.pth")
    with (RESULTS / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # 评估
    test_metrics = evaluate(cnn, test_loader, device)


    # 训练最终结果保存：
    # 评估结果、参数量、训练集大小、测试集大小、最终损失、训练时间、准确率是否达标
    metrics = {"test": test_metrics, "parameters": sum(p.numel() for p in cnn.parameters()),
               "train_size": len(train_data), "test_size": len(test_data),
               "final_train_loss": history[-1]["train_loss"],
               "training_seconds": time.perf_counter() - start,
               "meets_requirement": test_metrics["accuracy"] >= 0.98}
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {RESULTS / 'cnn.pth'}; final test accuracy={test_metrics['accuracy']:.2%}",
          flush=True)


if __name__ == "__main__":
    main()
