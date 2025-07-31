<<<<<<< HEAD
"""
超精準版 gMLP 圖像分類測試
包含混合精度訓練、EMA、高級數據增強和詳細監控
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from g_mlp_pytorch import gMLPVision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import random
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class EMA:
    """指數移動平均 (Exponential Moving Average)"""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow參數
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class CutMix:
    """CutMix 數據增強"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)

        # 生成隨機索引
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        # 生成 lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # 生成隨機裁剪區域
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 應用 CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]

        # 調整 lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return images, labels, shuffled_labels, lam


def set_seed(seed=42):
    """設定隨機種子確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar10_data_precision():
    """加載超精準的 CIFAR-10 數據集"""
    print("📦 加載超精準的 CIFAR-10 數據集...")

    # 更精細的數據增強策略
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 反射填充
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
        ]
    )

    # 測試時增強 (TTA準備)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # 載入完整數據集
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # 使用更多訓練數據以提高精度
    train_size = 40000  # 使用80%的訓練數據
    val_size = 10000  # 使用20%作為驗證集

    # 分層採樣確保類別平衡
    train_indices = []
    val_indices = []

    class_counts = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_counts[label].append(idx)

    for class_idx, indices in class_counts.items():
        np.random.shuffle(indices)
        train_split = int(0.8 * len(indices))
        train_indices.extend(indices[:train_split])
        val_indices.extend(indices[train_split:])

    trainset = Subset(trainset, train_indices)
    valset = Subset(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=transform_test
        ),
        val_indices,
    )

    # 創建數據加載器
    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valloader = DataLoader(
        valset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    print(f"   ✓ 訓練樣本: {len(trainset):,}")
    print(f"   ✓ 驗證樣本: {len(valset):,}")
    print(f"   ✓ 測試樣本: {len(testset):,}")
    print(f"   ✓ 類別數: {len(classes)}")

    return trainloader, valloader, testloader, classes


def create_precision_gmlp_model():
    """創建超精準的 gMLP 模型"""
    print("\n🏗️ 創建超精準的 gMLP 模型...")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 補丁大小：更小的patch提高細節捕捉
        num_classes=10,  # 分類數量
        dim=512,  # 增加特徵維度以提高表達能力
        depth=12,  # 增加深度以提高模型容量
        # === 網絡結構參數 ===
        ff_mult=4,  # 前饋倍數
        channels=3,  # 輸入通道
        attn_dim=None,  # 注意力維度
        # === 正則化參數 ===
        dropout=0.15,  # 適度增加dropout
        prob_survival=0.85,  # 隨機深度：更aggressive的stochastic depth
        # === 特殊功能參數 ===
        causal=False,  # 因果遮罩
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 權重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✓ 超精準模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 總參數數量: {total_params:,}")
    print(f"   ✓ 可訓練參數: {trainable_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    return model, device


def train_precision_model(model, trainloader, valloader, device, epochs=50):
    """超精準訓練流程"""
    print(f"\n🏋️ 開始超精準訓練 ({epochs} 個 epochs)...")

    # 設定混合精度訓練
    scaler = GradScaler()

    # 損失函數：使用標籤平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 優化器：使用AdamW + 權重衰減
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8
    )

    # 學習率調度器：餘弦退火 + 預熱
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 第一次重啟的週期
        T_mult=2,  # 每次重啟後週期的倍數
        eta_min=1e-6,  # 最小學習率
    )

    # EMA
    ema = EMA(model, decay=0.9999)

    # CutMix
    cutmix = CutMix(alpha=1.0)

    # 訓練記錄
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []
    epoch_times = []

    # 早停和最佳模型保存
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Learning Rate: {current_lr:.8f}")

        # =============== 訓練階段 ===============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 隨機應用 CutMix
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = cutmix((inputs, targets))
                cutmix_flag = True
            else:
                cutmix_flag = False

            optimizer.zero_grad()

            # 混合精度前向傳播
            with autocast():
                outputs = model(inputs)
                if cutmix_flag:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                        outputs, targets_b
                    )
                else:
                    loss = criterion(outputs, targets)

            # 混合精度反向傳播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 更新 EMA
            ema.update()

            # 統計
            train_loss += loss.item()
            if not cutmix_flag:
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"   Batch {batch_idx + 1:3d}: Loss = {loss.item():.4f}")

        # 更新學習率
        scheduler.step()

        # 計算訓練指標
        avg_train_loss = train_loss / len(trainloader)
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # =============== 驗證階段 ===============
        val_loss, val_acc = validate_model(model, valloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 使用 EMA 進行驗證
        ema.apply_shadow()
        ema_val_loss, ema_val_acc = validate_model(model, valloader, criterion, device)
        ema.restore()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f"Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   EMA Val Loss: {ema_val_loss:.4f}, EMA Val Acc: {ema_val_acc:.2f}%")
        print(f"   Time: {epoch_duration:.2f}s")

        # 早停和最佳模型保存 (使用EMA結果)
        if ema_val_acc > best_val_acc:
            best_val_acc = ema_val_acc
            patience_counter = 0

            # 保存最佳模型 (EMA版本)
            ema.apply_shadow()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "ema_state_dict": ema.shadow,
                },
                "best_precision_model.pth",
            )
            ema.restore()

            print(f"   💾 New best model saved! EMA Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping: No improvement for {patience} epochs")
                break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 訓練完成統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    checkpoint = torch.load("best_precision_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("   • 已載入最佳模型權重")

    return (
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        learning_rates,
        epoch_times,
        total_training_time,
    )


def validate_model(model, dataloader, criterion, device):
    """驗證模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def test_time_augmentation(model, testloader, device, num_crops=5):
    """測試時增強 (TTA)"""
    print(f"\n🔬 執行測試時增強 (TTA) with {num_crops} crops...")

    model.eval()
    all_predictions = []
    all_labels = []

    # TTA transforms
    tta_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 收集所有增強預測
            batch_predictions = []

            # 原始圖像
            with autocast():
                outputs = model(inputs)
                batch_predictions.append(torch.softmax(outputs, dim=1))

            # TTA增強
            for _ in range(num_crops):
                # 對每個樣本應用隨機增強
                augmented_batch = []
                for i in range(batch_size):
                    # 反標準化
                    img = inputs[i].cpu()
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)

                    # 應用TTA變換
                    aug_img = tta_transforms(img)
                    augmented_batch.append(aug_img)

                augmented_batch = torch.stack(augmented_batch).to(device)

                with autocast():
                    outputs = model(augmented_batch)
                    batch_predictions.append(torch.softmax(outputs, dim=1))

            # 平均所有預測
            avg_predictions = torch.stack(batch_predictions).mean(dim=0)
            _, predicted = avg_predictions.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = (
        100.0
        * np.sum(np.array(all_predictions) == np.array(all_labels))
        / len(all_labels)
    )
    print(f"   ✓ TTA Accuracy: {accuracy:.2f}%")

    return accuracy, all_predictions, all_labels


def plot_precision_training_history(
    train_losses, train_accs, val_losses, val_accs, learning_rates, epoch_times
):
    """繪製超精準訓練歷史"""
    print("\n📈 繪製超精準訓練歷史...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Precision gMLP Training History", fontsize=16, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    # 損失曲線
    axes[0, 0].plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    axes[0, 0].plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    axes[0, 0].set_title("Loss Curves", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 準確率曲線
    axes[0, 1].plot(epochs, train_accs, "g-", linewidth=2, label="Training Accuracy")
    axes[0, 1].plot(
        epochs, val_accs, "orange", linewidth=2, label="Validation Accuracy"
    )
    axes[0, 1].set_title("Accuracy Curves", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 學習率曲線
    axes[0, 2].plot(epochs, learning_rates, "purple", linewidth=2)
    axes[0, 2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Learning Rate")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(True, alpha=0.3)

    # 過擬合監控
    if len(train_accs) > 0 and len(val_accs) > 0:
        overfitting = np.array(train_accs) - np.array(val_accs)
        axes[1, 0].plot(epochs, overfitting, "red", linewidth=2)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_title("Overfitting Monitor", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Train - Val Accuracy (%)")
        axes[1, 0].grid(True, alpha=0.3)

    # 每epoch時間
    axes[1, 1].plot(epochs, epoch_times, "brown", linewidth=2, marker="o", markersize=4)
    axes[1, 1].set_title("Training Time per Epoch", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].grid(True, alpha=0.3)

    # 訓練穩定性分析
    if len(val_accs) >= 10:
        window_size = min(5, len(val_accs) // 2)
        val_acc_smooth = np.convolve(
            val_accs, np.ones(window_size) / window_size, mode="valid"
        )
        smooth_epochs = range(window_size, len(val_accs) + 1)
        axes[1, 2].plot(epochs, val_accs, "lightblue", alpha=0.7, label="Raw")
        axes[1, 2].plot(
            smooth_epochs, val_acc_smooth, "darkblue", linewidth=2, label="Smoothed"
        )
        axes[1, 2].set_title("Validation Accuracy Stability", fontweight="bold")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Validation Accuracy (%)")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("precision_gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_precision_model(model, testloader, device, classes):
    """超精準模型評估"""
    print("\n📊 執行超精準模型評估...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 每類別統計
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total

    # TTA評估
    tta_acc, tta_predictions, _ = test_time_augmentation(
        model, testloader, device, num_crops=3
    )

    print(f"\n📈 評估結果:")
    print(f"   • 標準測試準確率: {overall_acc:.3f}%")
    print(f"   • TTA測試準確率: {tta_acc:.3f}%")
    print(f"   • TTA提升: {tta_acc - overall_acc:.3f}%")

    # 詳細可視化
    plot_precision_evaluation(
        all_labels,
        all_predictions,
        tta_predictions,
        all_probabilities,
        classes,
        overall_acc,
        tta_acc,
    )

    return overall_acc, tta_acc


def plot_precision_evaluation(
    labels, predictions, tta_predictions, probabilities, classes, standard_acc, tta_acc
):
    """超精準評估可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Precision gMLP Evaluation Results", fontsize=16, fontweight="bold")

    # 1. 類別準確率比較
    class_accs_std = []
    class_accs_tta = []

    for i in range(len(classes)):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            std_acc = (
                100.0
                * np.sum(np.array(predictions)[class_mask] == i)
                / np.sum(class_mask)
            )
            tta_acc_class = (
                100.0
                * np.sum(np.array(tta_predictions)[class_mask] == i)
                / np.sum(class_mask)
            )
            class_accs_std.append(std_acc)
            class_accs_tta.append(tta_acc_class)
        else:
            class_accs_std.append(0)
            class_accs_tta.append(0)

    x = np.arange(len(classes))
    width = 0.35

    axes[0, 0].bar(x - width / 2, class_accs_std, width, label="Standard", alpha=0.8)
    axes[0, 0].bar(x + width / 2, class_accs_tta, width, label="TTA", alpha=0.8)
    axes[0, 0].set_title("Class-wise Accuracy Comparison")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. 混淆矩陣 (標準)
    cm_std = confusion_matrix(labels, predictions)
    sns.heatmap(
        cm_std,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"Standard Confusion Matrix (Acc: {standard_acc:.2f}%)")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")

    # 3. 混淆矩陣 (TTA)
    cm_tta = confusion_matrix(labels, tta_predictions)
    sns.heatmap(
        cm_tta,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0, 2],
    )
    axes[0, 2].set_title(f"TTA Confusion Matrix (Acc: {tta_acc:.2f}%)")
    axes[0, 2].set_xlabel("Predicted")
    axes[0, 2].set_ylabel("True")

    # 4. 預測信心度分佈
    max_probs = np.max(probabilities, axis=1)
    correct_mask = np.array(predictions) == np.array(labels)

    axes[1, 0].hist(
        max_probs[correct_mask], bins=50, alpha=0.7, label="Correct", density=True
    )
    axes[1, 0].hist(
        max_probs[~correct_mask], bins=50, alpha=0.7, label="Incorrect", density=True
    )
    axes[1, 0].set_title("Prediction Confidence Distribution")
    axes[1, 0].set_xlabel("Max Probability")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 5. 準確率 vs 信心度
    confidence_bins = np.linspace(0, 1, 21)
    bin_accs = []
    bin_counts = []

    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct_mask[mask])
            bin_accs.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0)
            bin_counts.append(0)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, bin_accs, "o-", linewidth=2, markersize=6)
    axes[1, 1].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect Calibration")
    axes[1, 1].set_title("Reliability Diagram")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # 6. 改進分析
    improvement = np.array(tta_predictions) == np.array(labels)
    standard_result = np.array(predictions) == np.array(labels)

    tta_better = improvement & (~standard_result)  # TTA對但標準錯
    tta_worse = (~improvement) & standard_result  # TTA錯但標準對
    both_correct = improvement & standard_result  # 都對
    both_wrong = (~improvement) & (~standard_result)  # 都錯

    categories = ["Both Correct", "TTA Better", "TTA Worse", "Both Wrong"]
    counts = [
        np.sum(both_correct),
        np.sum(tta_better),
        np.sum(tta_worse),
        np.sum(both_wrong),
    ]
    colors = ["green", "blue", "orange", "red"]

    axes[1, 2].pie(counts, labels=categories, colors=colors, autopct="%1.1f%%")
    axes[1, 2].set_title("TTA vs Standard Comparison")

    plt.tight_layout()
    plt.savefig("precision_gmlp_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """主函數"""
    print("🎯 超精準 gMLP 圖像分類測試")
    print("=" * 60)

    # 設定隨機種子
    set_seed(42)

    try:
        # 1. 載入超精準數據
        trainloader, valloader, testloader, classes = load_cifar10_data_precision()

        # 2. 創建超精準模型
        model, device = create_precision_gmlp_model()

        # 3. 超精準訓練
        (
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            learning_rates,
            epoch_times,
            total_training_time,
        ) = train_precision_model(model, trainloader, valloader, device, epochs=50)

        # 4. 繪製訓練歷史
        plot_precision_training_history(
            train_losses, train_accs, val_losses, val_accs, learning_rates, epoch_times
        )

        # 5. 超精準評估
        standard_acc, tta_acc = evaluate_precision_model(
            model, testloader, device, classes
        )

        # 6. 最終報告
        print(f"\n🎊 超精準測試完成！")
        print(f"=" * 60)
        print(f"📊 最終結果:")
        print(f"   • 標準測試準確率: {standard_acc:.3f}%")
        print(f"   • TTA測試準確率: {tta_acc:.3f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.3f}%")
        print(f"   • 總訓練時間: {total_training_time/60:.2f} 分鐘")
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f} 秒")

        # 性能評級
        if tta_acc >= 90:
            grade = "🏆 優秀"
            comment = "模型表現優異，可用於生產環境！"
        elif tta_acc >= 85:
            grade = "🥇 優良"
            comment = "模型表現很好，接近SOTA水準！"
        elif tta_acc >= 80:
            grade = "🥈 良好"
            comment = "模型表現良好，達到預期目標！"
        elif tta_acc >= 75:
            grade = "🥉 及格"
            comment = "模型表現尚可，還有改進空間。"
        else:
            grade = "❌ 需改進"
            comment = "模型表現不佳，需要重新調整。"

        print(f"\n🎯 性能評級: {grade}")
        print(f"💬 評語: {comment}")

        # 技術建議
        print(f"\n🔧 技術分析:")
        overfitting = train_accs[-1] - val_accs[-1] if train_accs and val_accs else 0
        if overfitting > 10:
            print(f"   ⚠️  檢測到過擬合 (差異: {overfitting:.2f}%)")
            print(f"      建議: 增加正則化或減少模型複雜度")
        elif overfitting > 5:
            print(f"   🔶 輕微過擬合 (差異: {overfitting:.2f}%)")
            print(f"      建議: 微調正則化參數")
        else:
            print(f"   ✅ 模型泛化良好 (差異: {overfitting:.2f}%)")

        if tta_acc - standard_acc > 1:
            print(f"   📈 TTA效果顯著 (+{tta_acc - standard_acc:.2f}%)")
            print(f"      建議: 在生產環境中使用TTA")
        else:
            print(f"   📊 TTA效果有限 (+{tta_acc - standard_acc:.2f}%)")
            print(f"      建議: 考慮其他增強策略")

    except Exception as e:
        print(f"❌ 超精準測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
=======
"""
超精準版 gMLP 圖像分類測試
包含混合精度訓練、EMA、高級數據增強和詳細監控
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from g_mlp_pytorch import gMLPVision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import random
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class EMA:
    """指數移動平均 (Exponential Moving Average)"""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow參數
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class CutMix:
    """CutMix 數據增強"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)

        # 生成隨機索引
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        # 生成 lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # 生成隨機裁剪區域
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 應用 CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]

        # 調整 lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return images, labels, shuffled_labels, lam


def set_seed(seed=42):
    """設定隨機種子確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar10_data_precision():
    """加載超精準的 CIFAR-10 數據集"""
    print("📦 加載超精準的 CIFAR-10 數據集...")

    # 更精細的數據增強策略
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 反射填充
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
        ]
    )

    # 測試時增強 (TTA準備)
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # 載入完整數據集
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # 使用更多訓練數據以提高精度
    train_size = 40000  # 使用80%的訓練數據
    val_size = 10000  # 使用20%作為驗證集

    # 分層採樣確保類別平衡
    train_indices = []
    val_indices = []

    class_counts = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_counts[label].append(idx)

    for class_idx, indices in class_counts.items():
        np.random.shuffle(indices)
        train_split = int(0.8 * len(indices))
        train_indices.extend(indices[:train_split])
        val_indices.extend(indices[train_split:])

    trainset = Subset(trainset, train_indices)
    valset = Subset(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=transform_test
        ),
        val_indices,
    )

    # 創建數據加載器
    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valloader = DataLoader(
        valset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    print(f"   ✓ 訓練樣本: {len(trainset):,}")
    print(f"   ✓ 驗證樣本: {len(valset):,}")
    print(f"   ✓ 測試樣本: {len(testset):,}")
    print(f"   ✓ 類別數: {len(classes)}")

    return trainloader, valloader, testloader, classes


def create_precision_gmlp_model():
    """創建超精準的 gMLP 模型"""
    print("\n🏗️ 創建超精準的 gMLP 模型...")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 補丁大小：更小的patch提高細節捕捉
        num_classes=10,  # 分類數量
        dim=512,  # 增加特徵維度以提高表達能力
        depth=12,  # 增加深度以提高模型容量
        # === 網絡結構參數 ===
        ff_mult=4,  # 前饋倍數
        channels=3,  # 輸入通道
        attn_dim=None,  # 注意力維度
        # === 正則化參數 ===
        dropout=0.15,  # 適度增加dropout
        prob_survival=0.85,  # 隨機深度：更aggressive的stochastic depth
        # === 特殊功能參數 ===
        causal=False,  # 因果遮罩
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 權重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✓ 超精準模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 總參數數量: {total_params:,}")
    print(f"   ✓ 可訓練參數: {trainable_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    return model, device


def train_precision_model(model, trainloader, valloader, device, epochs=50):
    """超精準訓練流程"""
    print(f"\n🏋️ 開始超精準訓練 ({epochs} 個 epochs)...")

    # 設定混合精度訓練
    scaler = GradScaler()

    # 損失函數：使用標籤平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 優化器：使用AdamW + 權重衰減
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8
    )

    # 學習率調度器：餘弦退火 + 預熱
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 第一次重啟的週期
        T_mult=2,  # 每次重啟後週期的倍數
        eta_min=1e-6,  # 最小學習率
    )

    # EMA
    ema = EMA(model, decay=0.9999)

    # CutMix
    cutmix = CutMix(alpha=1.0)

    # 訓練記錄
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []
    epoch_times = []

    # 早停和最佳模型保存
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Learning Rate: {current_lr:.8f}")

        # =============== 訓練階段 ===============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 隨機應用 CutMix
            if np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = cutmix((inputs, targets))
                cutmix_flag = True
            else:
                cutmix_flag = False

            optimizer.zero_grad()

            # 混合精度前向傳播
            with autocast():
                outputs = model(inputs)
                if cutmix_flag:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                        outputs, targets_b
                    )
                else:
                    loss = criterion(outputs, targets)

            # 混合精度反向傳播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 更新 EMA
            ema.update()

            # 統計
            train_loss += loss.item()
            if not cutmix_flag:
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"   Batch {batch_idx + 1:3d}: Loss = {loss.item():.4f}")

        # 更新學習率
        scheduler.step()

        # 計算訓練指標
        avg_train_loss = train_loss / len(trainloader)
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # =============== 驗證階段 ===============
        val_loss, val_acc = validate_model(model, valloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 使用 EMA 進行驗證
        ema.apply_shadow()
        ema_val_loss, ema_val_acc = validate_model(model, valloader, criterion, device)
        ema.restore()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f"Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   EMA Val Loss: {ema_val_loss:.4f}, EMA Val Acc: {ema_val_acc:.2f}%")
        print(f"   Time: {epoch_duration:.2f}s")

        # 早停和最佳模型保存 (使用EMA結果)
        if ema_val_acc > best_val_acc:
            best_val_acc = ema_val_acc
            patience_counter = 0

            # 保存最佳模型 (EMA版本)
            ema.apply_shadow()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "ema_state_dict": ema.shadow,
                },
                "best_precision_model.pth",
            )
            ema.restore()

            print(f"   💾 New best model saved! EMA Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping: No improvement for {patience} epochs")
                break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 訓練完成統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    checkpoint = torch.load("best_precision_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("   • 已載入最佳模型權重")

    return (
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        learning_rates,
        epoch_times,
        total_training_time,
    )


def validate_model(model, dataloader, criterion, device):
    """驗證模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def test_time_augmentation(model, testloader, device, num_crops=5):
    """測試時增強 (TTA)"""
    print(f"\n🔬 執行測試時增強 (TTA) with {num_crops} crops...")

    model.eval()
    all_predictions = []
    all_labels = []

    # TTA transforms
    tta_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 收集所有增強預測
            batch_predictions = []

            # 原始圖像
            with autocast():
                outputs = model(inputs)
                batch_predictions.append(torch.softmax(outputs, dim=1))

            # TTA增強
            for _ in range(num_crops):
                # 對每個樣本應用隨機增強
                augmented_batch = []
                for i in range(batch_size):
                    # 反標準化
                    img = inputs[i].cpu()
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)

                    # 應用TTA變換
                    aug_img = tta_transforms(img)
                    augmented_batch.append(aug_img)

                augmented_batch = torch.stack(augmented_batch).to(device)

                with autocast():
                    outputs = model(augmented_batch)
                    batch_predictions.append(torch.softmax(outputs, dim=1))

            # 平均所有預測
            avg_predictions = torch.stack(batch_predictions).mean(dim=0)
            _, predicted = avg_predictions.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = (
        100.0
        * np.sum(np.array(all_predictions) == np.array(all_labels))
        / len(all_labels)
    )
    print(f"   ✓ TTA Accuracy: {accuracy:.2f}%")

    return accuracy, all_predictions, all_labels


def plot_precision_training_history(
    train_losses, train_accs, val_losses, val_accs, learning_rates, epoch_times
):
    """繪製超精準訓練歷史"""
    print("\n📈 繪製超精準訓練歷史...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Precision gMLP Training History", fontsize=16, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    # 損失曲線
    axes[0, 0].plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    axes[0, 0].plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    axes[0, 0].set_title("Loss Curves", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 準確率曲線
    axes[0, 1].plot(epochs, train_accs, "g-", linewidth=2, label="Training Accuracy")
    axes[0, 1].plot(
        epochs, val_accs, "orange", linewidth=2, label="Validation Accuracy"
    )
    axes[0, 1].set_title("Accuracy Curves", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 學習率曲線
    axes[0, 2].plot(epochs, learning_rates, "purple", linewidth=2)
    axes[0, 2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Learning Rate")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(True, alpha=0.3)

    # 過擬合監控
    if len(train_accs) > 0 and len(val_accs) > 0:
        overfitting = np.array(train_accs) - np.array(val_accs)
        axes[1, 0].plot(epochs, overfitting, "red", linewidth=2)
        axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_title("Overfitting Monitor", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Train - Val Accuracy (%)")
        axes[1, 0].grid(True, alpha=0.3)

    # 每epoch時間
    axes[1, 1].plot(epochs, epoch_times, "brown", linewidth=2, marker="o", markersize=4)
    axes[1, 1].set_title("Training Time per Epoch", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].grid(True, alpha=0.3)

    # 訓練穩定性分析
    if len(val_accs) >= 10:
        window_size = min(5, len(val_accs) // 2)
        val_acc_smooth = np.convolve(
            val_accs, np.ones(window_size) / window_size, mode="valid"
        )
        smooth_epochs = range(window_size, len(val_accs) + 1)
        axes[1, 2].plot(epochs, val_accs, "lightblue", alpha=0.7, label="Raw")
        axes[1, 2].plot(
            smooth_epochs, val_acc_smooth, "darkblue", linewidth=2, label="Smoothed"
        )
        axes[1, 2].set_title("Validation Accuracy Stability", fontweight="bold")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Validation Accuracy (%)")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("precision_gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_precision_model(model, testloader, device, classes):
    """超精準模型評估"""
    print("\n📊 執行超精準模型評估...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 每類別統計
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total

    # TTA評估
    tta_acc, tta_predictions, _ = test_time_augmentation(
        model, testloader, device, num_crops=3
    )

    print(f"\n📈 評估結果:")
    print(f"   • 標準測試準確率: {overall_acc:.3f}%")
    print(f"   • TTA測試準確率: {tta_acc:.3f}%")
    print(f"   • TTA提升: {tta_acc - overall_acc:.3f}%")

    # 詳細可視化
    plot_precision_evaluation(
        all_labels,
        all_predictions,
        tta_predictions,
        all_probabilities,
        classes,
        overall_acc,
        tta_acc,
    )

    return overall_acc, tta_acc


def plot_precision_evaluation(
    labels, predictions, tta_predictions, probabilities, classes, standard_acc, tta_acc
):
    """超精準評估可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Precision gMLP Evaluation Results", fontsize=16, fontweight="bold")

    # 1. 類別準確率比較
    class_accs_std = []
    class_accs_tta = []

    for i in range(len(classes)):
        class_mask = np.array(labels) == i
        if np.sum(class_mask) > 0:
            std_acc = (
                100.0
                * np.sum(np.array(predictions)[class_mask] == i)
                / np.sum(class_mask)
            )
            tta_acc_class = (
                100.0
                * np.sum(np.array(tta_predictions)[class_mask] == i)
                / np.sum(class_mask)
            )
            class_accs_std.append(std_acc)
            class_accs_tta.append(tta_acc_class)
        else:
            class_accs_std.append(0)
            class_accs_tta.append(0)

    x = np.arange(len(classes))
    width = 0.35

    axes[0, 0].bar(x - width / 2, class_accs_std, width, label="Standard", alpha=0.8)
    axes[0, 0].bar(x + width / 2, class_accs_tta, width, label="TTA", alpha=0.8)
    axes[0, 0].set_title("Class-wise Accuracy Comparison")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. 混淆矩陣 (標準)
    cm_std = confusion_matrix(labels, predictions)
    sns.heatmap(
        cm_std,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(f"Standard Confusion Matrix (Acc: {standard_acc:.2f}%)")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")

    # 3. 混淆矩陣 (TTA)
    cm_tta = confusion_matrix(labels, tta_predictions)
    sns.heatmap(
        cm_tta,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0, 2],
    )
    axes[0, 2].set_title(f"TTA Confusion Matrix (Acc: {tta_acc:.2f}%)")
    axes[0, 2].set_xlabel("Predicted")
    axes[0, 2].set_ylabel("True")

    # 4. 預測信心度分佈
    max_probs = np.max(probabilities, axis=1)
    correct_mask = np.array(predictions) == np.array(labels)

    axes[1, 0].hist(
        max_probs[correct_mask], bins=50, alpha=0.7, label="Correct", density=True
    )
    axes[1, 0].hist(
        max_probs[~correct_mask], bins=50, alpha=0.7, label="Incorrect", density=True
    )
    axes[1, 0].set_title("Prediction Confidence Distribution")
    axes[1, 0].set_xlabel("Max Probability")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 5. 準確率 vs 信心度
    confidence_bins = np.linspace(0, 1, 21)
    bin_accs = []
    bin_counts = []

    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct_mask[mask])
            bin_accs.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0)
            bin_counts.append(0)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, bin_accs, "o-", linewidth=2, markersize=6)
    axes[1, 1].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect Calibration")
    axes[1, 1].set_title("Reliability Diagram")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # 6. 改進分析
    improvement = np.array(tta_predictions) == np.array(labels)
    standard_result = np.array(predictions) == np.array(labels)

    tta_better = improvement & (~standard_result)  # TTA對但標準錯
    tta_worse = (~improvement) & standard_result  # TTA錯但標準對
    both_correct = improvement & standard_result  # 都對
    both_wrong = (~improvement) & (~standard_result)  # 都錯

    categories = ["Both Correct", "TTA Better", "TTA Worse", "Both Wrong"]
    counts = [
        np.sum(both_correct),
        np.sum(tta_better),
        np.sum(tta_worse),
        np.sum(both_wrong),
    ]
    colors = ["green", "blue", "orange", "red"]

    axes[1, 2].pie(counts, labels=categories, colors=colors, autopct="%1.1f%%")
    axes[1, 2].set_title("TTA vs Standard Comparison")

    plt.tight_layout()
    plt.savefig("precision_gmlp_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """主函數"""
    print("🎯 超精準 gMLP 圖像分類測試")
    print("=" * 60)

    # 設定隨機種子
    set_seed(42)

    try:
        # 1. 載入超精準數據
        trainloader, valloader, testloader, classes = load_cifar10_data_precision()

        # 2. 創建超精準模型
        model, device = create_precision_gmlp_model()

        # 3. 超精準訓練
        (
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            learning_rates,
            epoch_times,
            total_training_time,
        ) = train_precision_model(model, trainloader, valloader, device, epochs=50)

        # 4. 繪製訓練歷史
        plot_precision_training_history(
            train_losses, train_accs, val_losses, val_accs, learning_rates, epoch_times
        )

        # 5. 超精準評估
        standard_acc, tta_acc = evaluate_precision_model(
            model, testloader, device, classes
        )

        # 6. 最終報告
        print(f"\n🎊 超精準測試完成！")
        print(f"=" * 60)
        print(f"📊 最終結果:")
        print(f"   • 標準測試準確率: {standard_acc:.3f}%")
        print(f"   • TTA測試準確率: {tta_acc:.3f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.3f}%")
        print(f"   • 總訓練時間: {total_training_time/60:.2f} 分鐘")
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f} 秒")

        # 性能評級
        if tta_acc >= 90:
            grade = "🏆 優秀"
            comment = "模型表現優異，可用於生產環境！"
        elif tta_acc >= 85:
            grade = "🥇 優良"
            comment = "模型表現很好，接近SOTA水準！"
        elif tta_acc >= 80:
            grade = "🥈 良好"
            comment = "模型表現良好，達到預期目標！"
        elif tta_acc >= 75:
            grade = "🥉 及格"
            comment = "模型表現尚可，還有改進空間。"
        else:
            grade = "❌ 需改進"
            comment = "模型表現不佳，需要重新調整。"

        print(f"\n🎯 性能評級: {grade}")
        print(f"💬 評語: {comment}")

        # 技術建議
        print(f"\n🔧 技術分析:")
        overfitting = train_accs[-1] - val_accs[-1] if train_accs and val_accs else 0
        if overfitting > 10:
            print(f"   ⚠️  檢測到過擬合 (差異: {overfitting:.2f}%)")
            print(f"      建議: 增加正則化或減少模型複雜度")
        elif overfitting > 5:
            print(f"   🔶 輕微過擬合 (差異: {overfitting:.2f}%)")
            print(f"      建議: 微調正則化參數")
        else:
            print(f"   ✅ 模型泛化良好 (差異: {overfitting:.2f}%)")

        if tta_acc - standard_acc > 1:
            print(f"   📈 TTA效果顯著 (+{tta_acc - standard_acc:.2f}%)")
            print(f"      建議: 在生產環境中使用TTA")
        else:
            print(f"   📊 TTA效果有限 (+{tta_acc - standard_acc:.2f}%)")
            print(f"      建議: 考慮其他增強策略")

    except Exception as e:
        print(f"❌ 超精準測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
>>>>>>> 420764095488647da1ecd1309c810893dfec8ea4
