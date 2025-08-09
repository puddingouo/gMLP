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


# Mixup 功能 - 參考 model_16.py
def mixup_data(x, y, alpha=0.1, lam=1.0, count=0, device="cpu"):
    """Mixup 數據增強 - 基於 model_16.py 實現"""
    if count == 0:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 損失函數 - 基於 model_16.py 實現"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


class CIFAR10_dataset(Dataset):
    def __init__(self, partition="train", transform=None):
        print("\nLoading CIFAR10 ", partition, " Dataset...")
        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10(
                ".data/", train=True, download=True
            )
        else:
            self.data = torchvision.datasets.CIFAR10(
                ".data/", train=False, download=True
            )
        print("\tTotal Len.: ", len(self.data), "\n", 50 * "-")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Image
        image = self.data[idx][0]
        image_tensor = self.transform(image)

        # Label - 保持與原程式相同的one-hot編碼格式
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}


def create_gmlp_model():
    """創建對應CNN複雜度的gMLP模型"""
    # 基於CNN的深度和複雜度調整gMLP參數
    model = gMLPVision(
        image_size=32,  # CIFAR-10圖像大小
        patch_size=4,  # 適合32x32的patch大小
        num_classes=10,  # CIFAR-10類別數
        dim=256,  # 較大的維度對應CNN的通道數增長
        depth=12,  # 對應CNN的9層卷積 + 3層全連接的深度
        ff_mult=4,  # 對應CNN中特徵維度的擴展倍數
        channels=3,  # RGB圖像
        prob_survival=0.9,  # 添加隨機深度以對應CNN的dropout
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_gmlp_model():
    """使用CNN的訓練策略訓練gMLP，添加mixup功能"""

    # Mixup 配置 - 使用 model_16.py 的預設值
    use_mixup = True  # 啟用 mixup
    mixup_alpha = 0.1  # model_16.py 的預設值

    print(f"🎨 Mixup配置: {'啟用' if use_mixup else '關閉'}")
    if use_mixup:
        print(f"   🎭 Alpha 參數: {mixup_alpha}")

    # 數據載入（保持原CNN的配置）
    train_dataset = CIFAR10_dataset(partition="train", transform=train_transform)
    test_dataset = CIFAR10_dataset(partition="test", transform=test_transform)

    batch_size = 100  # 保持原CNN的batch size
    num_workers = multiprocessing.cpu_count() - 1
    print("Num workers", num_workers)

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    # 創建gMLP模型
    net = create_gmlp_model()
    net.to(device)
    print("gMLP Model:")
    print(f"Parameters: {count_parameters(net):,}")

    # 訓練超參數（基於原CNN配置調整）
    criterion = nn.CrossEntropyLoss()

    # 將SGD改為AdamW（更適合Transformer架構）
    optimizer = optim.AdamW(
        net.parameters(),
        lr=0.001,  # 較低的學習率適合AdamW
        weight_decay=1e-4,  # 調整權重衰減
        betas=(0.9, 0.95),  # Transformer常用的beta值
    )

    # 保持原有的學習率調度策略
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, min_lr=0.00001
    )

    epochs = 100  # 保持原訓練輪數

    print("\n---- Start Training gMLP ----")
    best_accuracy = -1
    best_epoch = 0

    # 訓練歷史記錄
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        start_time = time.time()

        # TRAIN NETWORK
        train_loss, train_correct = 0, 0
        net.train()
        lam = 1.0  # mixup lambda 初始值

        with tqdm(
            iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch"
        ) as tepoch:
            for batch in tepoch:
                # 數據處理（保持原格式）
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                # Mixup 數據增強
                if use_mixup:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, mixup_alpha, lam, 0, device
                    )
                    outputs = net(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = net(images)
                    loss = criterion(outputs, labels)

                # Backward
                loss.backward()

                # 添加梯度裁剪（適合Transformer）
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                optimizer.step()

                # 統計（適配mixup）
                if use_mixup:
                    # Mixup 模式下的準確率計算
                    labels_idx_a = torch.argmax(labels_a, dim=1)
                    labels_idx_b = torch.argmax(labels_b, dim=1)
                    pred = torch.argmax(outputs, dim=1)

                    # 基於 lambda 加權的準確率計算
                    correct_a = pred.eq(labels_idx_a).sum().item()
                    correct_b = pred.eq(labels_idx_b).sum().item()
                    train_correct += lam * correct_a + (1 - lam) * correct_b
                else:
                    labels_idx = torch.argmax(labels, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                    train_correct += pred.eq(labels_idx).sum().item()

                train_loss += loss.item()

        train_loss /= len(train_dataloader.dataset) / batch_size
        train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)

        # TEST NETWORK
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

        # 學習率調度
        lr_scheduler.step(test_loss)

        test_loss /= len(test_dataloader.dataset) / batch_size
        test_accuracy = 100.0 * test_correct / len(test_dataloader.dataset)

        # 記錄訓練歷史
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

        # 保存最佳模型
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
    print("🚀 gMLP訓練 - 使用CNN參數配置 + Mixup增強")
    print("🎯 模型配置: depth=12, dim=256, ff_mult=4")
    print("📚 優化器: AdamW (替代SGD，更適合Transformer)")
    print("📈 調度器: ReduceLROnPlateau (保持原策略)")
    print("🔢 批次大小: 100 (保持原設定)")
    print("🎨 Mixup增強: 啟用 (alpha=0.1, 基於model_16.py)")
    print("=" * 70)

    try:
        train_losses, train_accs, test_losses, test_accs, best_acc = train_gmlp_model()
        plot_training_history(train_losses, train_accs, test_losses, test_accs)

        print(f"\n🎉 訓練完成!")
        print(f"   • 最佳準確率: {best_acc:.2f}%")
        print(f"   • 模型已保存: best_gmlp_model.pt")
        print(f"   • Mixup增強: 已啟用 (alpha=0.1)")
        print(f"   • 訓練策略: AdamW + Mixup + 梯度裁剪")

    except KeyboardInterrupt:
        print("\n❌ 訓練被中斷")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
