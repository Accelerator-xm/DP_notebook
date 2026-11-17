"""加载指导书格式的 state_dict，独立复核测试集准确率。"""

import json

import torch
from torch.utils.data import DataLoader

from data import load_data
from engine import evaluate
from model import CNN
from train import BATCH_SIZE, DATA_DIR, RESULTS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(RESULTS / "cnn.pth", map_location=device, weights_only=True))
    test_data = load_data(DATA_DIR, train=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate(cnn, test_loader, device)
    metrics["device"] = str(device)
    (RESULTS / "reevaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device}; accuracy={metrics['accuracy']:.2%} "
          f"({metrics['correct']}/{metrics['count']})")


if __name__ == "__main__":
    main()
