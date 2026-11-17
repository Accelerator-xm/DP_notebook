import torch


def train_epoch(cnn, train_loader, loss_func, optimizer, device, test_loader):
    """训练循环"""
    cnn.train() # 训练模式
    history = []

    # 开始训练
    for step, (b_x, b_y) in enumerate(train_loader, start=1):
        # 获取数据并加载到设备上
        b_x, b_y = b_x.to(device), b_y.to(device)
        # 前向传播 - 预测值
        output = cnn(b_x)
        # 交叉熵损失
        loss = loss_func(output, b_y)
        # 清除梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 应用梯度
        optimizer.step()

        if step % 50 == 0:
            eval_metrics = evaluate(cnn, test_loader, device)
            cnn.train()  # 测评结束后继续训练
            print(f"step: {step} | train loss: {loss.item():.4f} | "
                  f"test accuracy: {eval_metrics['accuracy']:.2%}", flush=True)
            
            if step % 100 == 0:
                history.append({"step": step, "train_loss": loss.item(),
                                "test_accuracy": eval_metrics["accuracy"]})

    return history


def evaluate(cnn, test_loader, device):
    """测评"""

    cnn.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = cnn(b_x)
            pred_y = output.argmax(1)
            correct += (pred_y == b_y).sum().item()
            total += b_y.size(0)

    return {"accuracy": correct / total, "correct": correct, "count": total}
