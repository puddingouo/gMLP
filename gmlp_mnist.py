"""
將CNN參數轉換為gMLP可用的訓練配置
基於原CNN程式的訓練策略和數據處理方式
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from g_mlp_pytorch import gMLPVision
from tqdm import tqdm
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


# 原CNN的數據增強策略，適配gMLP
train_transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


# MNIST Dataset 取代 CIFAR10
class MNIST_dataset(Dataset):
    def __init__(self, partition="train", transform=None):
        print("\nLoading MNIST ", partition, " Dataset...")
        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.MNIST(".data/", train=True, download=True)
        else:
            self.data = torchvision.datasets.MNIST(".data/", train=False, download=True)
        print("\tTotal Len.: ", len(self.data), "\n", 50 * "-")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image_tensor = self.transform(image)
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()
        return {"img": image_tensor, "label": label}


def create_gmlp_model():
    """創建對應MNIST複雜度的gMLP模型"""
    # MNIST: 單通道灰階圖像 28x28
    model = gMLPVision(
        image_size=28,  # MNIST圖像大小
        patch_size=4,  # 適合28x28的patch大小
        num_classes=10,  # MNIST類別數
        dim=128,  # MNIST較簡單，維度可調低
        depth=12,  # 深度可調低
        ff_mult=3,  # 特徵維度擴展倍數
        channels=1,  # MNIST為單通道
        prob_survival=1.0,
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_gmlp_model():
    """使用MNIST資料集訓練gMLP，添加mixup功能"""

    use_mixup = False  # 已取消mixup
    print("🎨 Mixup配置: 關閉")

    # MNIST資料集載入
    train_dataset = MNIST_dataset(partition="train", transform=train_transform)
    test_dataset = MNIST_dataset(partition="test", transform=test_transform)

    batch_size = 100
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print("Num workers", num_workers)

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    net = create_gmlp_model()
    net.to(device)
    print("gMLP Model:")
    print(f"Parameters: {count_parameters(net):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, min_lr=0.00001
    )
    epochs = 50  # MNIST較簡單，訓練輪數可減少

    print("\n---- Start Training gMLP ----")
    best_accuracy = -1
    best_epoch = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_correct = 0, 0
        net.train()
        lam = 1.0
        with tqdm(
            iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch"
        ) as tepoch:
            for batch in tepoch:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                labels_idx = torch.argmax(labels, dim=1)
                pred = torch.argmax(outputs, dim=1)
                train_correct += pred.eq(labels_idx).sum().item()
                train_loss += loss.item()
        train_loss /= len(train_dataloader.dataset) / batch_size
        train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)
        test_loss, test_correct = 0, 0
        net.eval()
        with torch.no_grad():
            with tqdm(
                iter(test_dataloader), desc="Test " + str(epoch), unit="batch"
            ) as tepoch:
                for batch in tepoch:
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)
                    outputs = net(images)
                    test_loss += criterion(outputs, labels)
                    labels_idx = torch.argmax(labels, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                    test_correct += pred.eq(labels_idx).sum().item()
        lr_scheduler.step(test_loss)
        test_loss /= len(test_dataloader.dataset) / batch_size
        test_accuracy = 100.0 * test_correct / len(test_dataloader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        test_losses.append(test_loss.item())
        test_accs.append(test_accuracy)
        epoch_time = time.time() - start_time
        print(
            "[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}% - Time: {:.1f}s".format(
                epoch + 1,
                train_loss,
                test_loss,
                train_accuracy,
                test_accuracy,
                epoch_time,
            )
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(net.state_dict(), "best_gmlp_model.pt")
            print(f"   💾 New best model saved: {test_accuracy:.2f}%")
    print(f"\nBEST TEST ACCURACY: {best_accuracy:.2f}% in epoch {best_epoch + 1}")
    return train_losses, train_accs, test_losses, test_accs, best_accuracy


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """繪製訓練歷史"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, "b-", label="Train Loss")
    plt.plot(test_losses, "r-", label="Test Loss")
    plt.title("gMLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, "b-", label="Train Acc")
    plt.plot(test_accs, "r-", label="Test Acc")
    plt.title("gMLP Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("🚀 gMLP訓練 - MNIST資料集 (無Mixup)")
    print("🎯 模型配置: depth=8, dim=128, ff_mult=2")
    print("📚 優化器: AdamW (更適合Transformer)")
    print("📈 調度器: ReduceLROnPlateau")
    print("🔢 批次大小: 100")
    print("🎨 Mixup增強: 關閉")
    print("=" * 70)

    try:
        train_losses, train_accs, test_losses, test_accs, best_acc = train_gmlp_model()
        plot_training_history(train_losses, train_accs, test_losses, test_accs)

        print(f"\n🎉 訓練完成!")
        print(f"   • 最佳準確率: {best_acc:.2f}%")
        print(f"   • 模型已保存: best_gmlp_model.pt")
        print(f"   • Mixup增強: 關閉")
        print(f"   • 訓練策略: AdamW + 梯度裁剪")
    except KeyboardInterrupt:
        print("\n❌ 訓練被中斷")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
