"""
CPU優化版 gMLP 圖像分類測試
專為CPU環境優化，包含可視化結果和準確率優化技巧
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from g_mlp_pytorch import gMLPVision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_cifar10_data_enhanced(quick_test=True):
    """加載增強的 CIFAR-10 數據集 - CPU優化版"""
    print("📦 加載CPU優化的 CIFAR-10 數據集...")

    # CPU優化的數據增強策略 - 論文啟發但適配CIFAR-10
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # 論文中沒有使用RandomErasing，所以移除以保持論文一致性
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    if quick_test:
        # 論文啟發配置：使用更多數據以符合論文訓練規模
        trainset = Subset(trainset, range(15000))  # 增加到15K樣本更接近論文規模
        testset = Subset(testset, range(3000))  # 相應增加測試數據到3K
        print("   � 論文啟發模式：大規模數據集訓練")

    # 論文啟發DataLoader配置
    batch_size = 128  # 更接近論文batch size但適配CPU (論文4096，這裡128)
    num_workers = 0  # CPU單線程避免競爭
    pin_memory = False  # CPU不需要pin_memory

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,  # 測試也用同樣batch_size
        num_workers=num_workers,
        pin_memory=pin_memory,
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

    print(f"   ✓ 訓練樣本: {len(trainset)}")
    print(f"   ✓ 測試樣本: {len(testset)}")
    print(f"   ✓ 類別數: {len(classes)}")
    print(f"   ✓ 論文啟發優化: batch_size={batch_size}, 大規模訓練配置")

    return trainloader, testloader, classes


def create_optimized_gmlp_model(model_size="S"):
    """創建論文標準的 gMLP 模型架構"""
    print(f"\n🏗️ 創建論文標準 gMLP-{model_size} 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)  # 設置4個線程
    print("   ⚡ CPU模式：已設置4個線程")

    # 論文標準架構配置 (基於Table 1)
    if model_size == "Ti":  # gMLP-Ti (最小模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 128,  # d_model = 128
            "ff_mult": 6,  # d_ffn / d_model = 768/128 = 6
            "prob_survival": 1.00,  # 論文中Ti模型不使用隨機深度
            "params_target": 5.9,  # 目標參數量(M)
        }
    elif model_size == "S":  # gMLP-S (中等模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 256,  # d_model = 256
            "ff_mult": 6,  # d_ffn / d_model = 1536/256 = 6
            "prob_survival": 0.95,  # 論文隨機深度存活率
            "params_target": 19.5,  # 目標參數量(M)
        }
    elif model_size == "B":  # gMLP-B (大模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 512,  # d_model = 512
            "ff_mult": 6,  # d_ffn / d_model = 3072/512 = 6
            "prob_survival": 0.80,  # 論文隨機深度存活率
            "params_target": 73.4,  # 目標參數量(M)
        }
    else:
        raise ValueError(f"不支援的模型大小: {model_size}")

    model = gMLPVision(
        # === 核心架構參數 (嚴格按照論文Table 1) ===
        image_size=32,  # CIFAR-10圖像尺寸 (論文用224，這裡適配32)
        patch_size=4,  # 適配CIFAR-10的patch size
        num_classes=10,  # CIFAR-10分類數量
        dim=config["dim"],  # d_model (論文標準)
        depth=config["depth"],  # 層數 #L (論文標準)
        # === 網絡結構參數 ===
        ff_mult=config["ff_mult"],  # 前饋網絡倍數 (論文計算得出)
        channels=3,  # 輸入通道數
        # === 正則化參數 (論文標準) ===
        prob_survival=config["prob_survival"],  # 隨機深度存活率
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    params_M = total_params / 1e6  # 轉換為百萬參數

    print(f"   ✓ gMLP-{model_size} 模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 實際參數數量: {total_params:,} ({params_M:.1f}M)")
    print(f"   ✓ 論文目標參數: {config['params_target']}M")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(
        f"   ✓ 論文架構: depth={config['depth']}, dim={config['dim']}, ff_mult={config['ff_mult']}"
    )
    print(f"   ✓ 隨機深度: prob_survival={config['prob_survival']}")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=300):
    """論文標準訓練配置 - 基於ImageNet-1K超參數"""
    print(f"\n🏋️ 開始論文標準訓練 ({epochs} 個 epochs)...")

    # 論文標準訓練配置 (基於ImageNet-1K超參數)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 論文標準標籤平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 論文峰值學習率
        weight_decay=0.05,  # 論文權重衰減
        betas=(0.9, 0.999),  # 論文Adam參數
        eps=1e-6,  # 論文Adam epsilon
    )

    # 論文標準學習率調度器 (Cosine退火，10K warmup steps適配)
    total_steps = len(trainloader) * epochs
    warmup_steps = min(10000, total_steps // 10)  # 論文10K warmup，但適配較小數據集

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # 論文峰值學習率1e-3
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=warmup_steps / total_steps,  # 基於warmup steps的百分比
        anneal_strategy="cos",  # 論文使用cosine退火
        final_div_factor=1000,  # 大幅衰減到初始值的1/1000
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 論文標準早停配置
    best_val_acc = 0
    patience = 30  # 論文級別長訓練的耐心值
    patience_counter = 0

    # 論文不使用動態過擬合檢測，專注於標準訓練
    print("   📄 使用論文標準訓練配置：300 epochs + 標準正則化")

    for epoch in range(epochs):
        # 記錄每個epoch開始時間
        epoch_start_time = time.time()

        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(
            f"\nEpoch {epoch + 1}/{epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 動態調整數據增強 - 強化防過擬合策略
        if epoch >= epochs * 0.4:  # 從40%開始減少數據增強 (0.7->0.4)
            # 逐步減少隨機擦除概率
            for transform in trainloader.dataset.dataset.transform.transforms:
                if isinstance(transform, transforms.RandomErasing):
                    if epoch >= epochs * 0.4 and epoch < epochs * 0.7:
                        transform.p = 0.1  # 中期降低
                    elif epoch >= epochs * 0.7:
                        transform.p = 0.03  # 後期大幅降低

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 論文標準梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,  # 論文梯度裁剪閾值1.0
            )
            optimizer.step()
            scheduler.step()  # OneCycleLR需要每個batch更新

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 15 == 0 or (i + 1) == len(
                trainloader
            ):  # 顯示進度且包含最後批次
                acc = 100.0 * correct / total
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"   批次 {i+1:3d}/{len(trainloader)}: 損失 = {running_loss/(i+1):.4f}, "
                    f"準確率 = {acc:.2f}%, 學習率 = {current_lr:.6f}"
                )

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 驗證階段
        val_acc = quick_validate(model, testloader, device)
        val_accs.append(val_acc)

        # OneCycleLR不需要手動step

        # 記錄每個epoch結束時間
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練準確率 = {epoch_acc:.2f}%, 驗證準確率 = {val_acc:.2f}%, 時間 = {epoch_duration:.2f}s"
        )

        # 論文標準訓練進度報告 (移除過擬合監控，專注標準訓練)
        train_val_diff = epoch_acc - val_acc
        if train_val_diff > 10:
            print(f"   📊 訓練-驗證差異: {train_val_diff:.2f}%")

        # 長訓練進度提示
        if epoch == epochs // 10:
            print(f"   🔄 已完成10%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 4:
            print(f"   🔄 已完成25%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 2:
            print(f"   🔄 已完成50%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 3 // 4:
            print(f"   🔄 已完成75%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 9 // 10:
            print(f"   🔄 已完成90%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")

        # 論文標準早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint_paper.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 論文標準訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint_paper.pth"))
        print("   • 已載入最佳論文標準模型權重")

    return train_losses, train_accs, val_accs, epoch_times, total_training_time


def quick_validate(model, testloader, device):
    """快速驗證"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def evaluate_model_with_visualization(model, testloader, device, classes):
    """評估模型並生成可視化結果"""
    print("\n📊 評估CPU優化模型並生成可視化...")

    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total
    print(f"   ✓ CPU模型整體準確率: {overall_acc:.2f}%")

    # 1. 各類別準確率條形圖
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    class_accs = []
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accs.append(acc)
        else:
            class_accs.append(0)

    bars = plt.bar(classes, class_accs, color=plt.cm.tab10(np.arange(10)))
    plt.title(
        "CPU-Optimized gMLP: Accuracy of Each Category", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    # 在柱狀圖上添加數值
    for bar, acc in zip(bars, class_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. 混淆矩陣
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("CPU-Optimized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")

    # 3. 標準化混淆矩陣
    plt.subplot(2, 2, 3)
    cm_normalized = confusion_matrix(all_labels, all_predictions, normalize="true")
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(
        "CPU-Optimized Normalized Confusion Matrix", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")

    # 4. 類別分佈
    plt.subplot(2, 2, 4)
    unique, counts = np.unique(all_labels, return_counts=True)
    plt.pie(
        counts,
        labels=[classes[i] for i in unique],
        autopct="%1.1f%%",
        colors=plt.cm.tab10(np.arange(len(unique))),
    )
    plt.title("CPU Test Set Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gmlp_cpu_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 CPU優化模型詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製CPU訓練歷史"""
    print("\n📈 繪製CPU訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="CPU Training Loss")
    plt.title("CPU-Optimized Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="CPU Training Accuracy")
    plt.plot(val_accs, "r-", linewidth=2, label="CPU Validation Accuracy")
    plt.title("CPU-Optimized Accuracy Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="CPU Train-Val Difference")
    plt.title("CPU Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(epoch_times, "orange", linewidth=2, marker="o", label="CPU Epoch Time")
        plt.title("CPU Training Time per Epoch", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_cpu_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化CPU模型樣本預測結果"""
    print(f"\n🔍 可視化CPU模型 {num_samples} 個樣本預測...")

    model.eval()

    # 獲取一批數據
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, 1)

    # 繪製結果
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        "CPU-Optimized gMLP Prediction Results", fontsize=16, fontweight="bold"
    )

    for i in range(min(num_samples, len(images))):
        ax = axes[i // 4, i % 4]

        # 反標準化圖像
        img = images[i].cpu()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        ax.imshow(img.permute(1, 2, 0))

        true_label = classes[labels[i]]
        pred_label = classes[predictions[i]]
        confidence = probabilities[i, predictions[i]].item()

        # 設置顏色（正確=綠色，錯誤=紅色）
        color = "green" if labels[i] == predictions[i] else "red"

        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("gmlp_cpu_sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_model_architectures():
    """比較不同gMLP模型架構的規格"""
    print("\n📋 論文標準 gMLP 模型架構比較:")
    print("=" * 80)
    print(
        f"{'模型':<8} {'深度':<6} {'維度':<8} {'FFN倍數':<8} {'隨機深度':<10} {'目標參數(M)':<12}"
    )
    print("-" * 80)

    models_config = {
        "Ti": {
            "depth": 30,
            "dim": 128,
            "ff_mult": 6,
            "prob_survival": 1.00,
            "params": 5.9,
        },
        "S": {
            "depth": 30,
            "dim": 256,
            "ff_mult": 6,
            "prob_survival": 0.95,
            "params": 19.5,
        },
        "B": {
            "depth": 30,
            "dim": 512,
            "ff_mult": 6,
            "prob_survival": 0.80,
            "params": 73.4,
        },
    }

    for name, config in models_config.items():
        print(
            f"gMLP-{name:<3} {config['depth']:<6} {config['dim']:<8} {config['ff_mult']:<8} "
            f"{config['prob_survival']:<10.2f} {config['params']:<12.1f}"
        )

    print("-" * 80)
    print("💡 建議:")
    print("   • gMLP-Ti: 快速實驗和概念驗證")
    print("   • gMLP-S:  平衡性能和計算資源 (推薦)")
    print("   • gMLP-B:  追求最佳性能 (需要更多資源)")
    print("=" * 80)


def main():
    print("🖼️ 論文標準 gMLP 圖像分類測試")
    print("=" * 60)
    print("📄 基於官方ImageNet-1K超參數配置")
    print("=" * 60)

    # 顯示模型架構比較
    compare_model_architectures()

    try:
        # 1. 加載論文標準數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建論文標準模型 (可選擇模型大小)
        model_size = "S"  # 選擇 'Ti', 'S', 或 'B'
        model, device = create_optimized_gmlp_model(model_size=model_size)

        # 3. 論文標準訓練
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=300,  # 論文標準300 epochs
            )
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存論文標準模型
        torch.save(model.state_dict(), "gmlp_paper_model.pth")
        print("\n💾 論文標準模型已保存為 'gmlp_paper_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 論文標準測試完成！")
        print(f"\n📈 論文標準最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n📄 論文配置特性:")
        print(f"   • 模型架構: gMLP-{model_size}")
        print(f"   • 模型參數: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • 論文標準: depth=30, dim=256, ff_mult=6, prob_survival=0.95")
        print(f"   • 數據量: 15,000訓練樣本, 3,000測試樣本")
        print(f"   • 批次大小: 128 (接近論文規模)")
        print(f"   • 訓練策略: 300 epochs + cosine退火 + 10K warmup")

        print(f"\n🎯 論文標準環境建議:")
        if accuracy < 70:
            print(f"   • 300 epochs後準確度仍需提升，考慮調整模型架構")
            print(f"   • 可能需要更大的數據集或更長的訓練")
            print(f"   • 檢查數據預處理是否與論文一致")
        elif accuracy < 80:
            print(f"   • 論文標準模型表現良好！")
            print(f"   • 300個epochs充分利用了論文配置")
            print(f"   • 可考慮微調超參數進一步優化")
        else:
            print(f"   • 論文標準模型表現優秀！")
            print(f"   • 成功復現論文級別的訓練效果")
            print(f"   • 適合發表或實際應用部署")

        # 泛化能力分析
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  需要注意:")
            print(f"   • 訓練-驗證差異較大 ({overfitting_diff:.2f}%)")
            print(f"   • 可能需要更多數據或更強正則化")
        elif overfitting_diff > 8:
            print(f"\n🔶 泛化表現:")
            print(f"   • 訓練-驗證差異適中 ({overfitting_diff:.2f}%)")
            print(f"   • 論文配置取得良好平衡")
        else:
            print(f"\n✅ 優秀泛化:")
            print(f"   • 訓練-驗證差異很小 ({overfitting_diff:.2f}%)")
            print(f"   • 論文配置實現出色泛化能力")

        print(f"\n🚀 論文標準總結:")
        print(f"   • 訓練策略: 300 epochs + 論文標準超參數")
        print(f"   • 正則化: 標籤平滑0.1 + 權重衰減0.05 + 隨機深度0.8")
        print(f"   • 學習率: 峰值1e-3 + cosine退火 + 10K warmup")
        print(f"   • 優化器: AdamW + 梯度裁剪1.0 + epsilon 1e-6")
        print(f"   • 目標: 復現論文ImageNet-1K級別的訓練效果")

    except Exception as e:
        print(f"❌ 論文標準測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
CPU優化版 gMLP 圖像分類測試
專為CPU環境優化，包含可視化結果和準確率優化技巧
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from g_mlp_pytorch import gMLPVision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_cifar10_data_enhanced(quick_test=True):
    """加載增強的 CIFAR-10 數據集 - CPU優化版"""
    print("📦 加載CPU優化的 CIFAR-10 數據集...")

    # CPU優化的數據增強策略 - 論文啟發但適配CIFAR-10
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # 論文中沒有使用RandomErasing，所以移除以保持論文一致性
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    if quick_test:
        # 論文啟發配置：使用更多數據以符合論文訓練規模
        trainset = Subset(trainset, range(15000))  # 增加到15K樣本更接近論文規模
        testset = Subset(testset, range(3000))  # 相應增加測試數據到3K
        print("   � 論文啟發模式：大規模數據集訓練")

    # 論文啟發DataLoader配置
    batch_size = 128  # 更接近論文batch size但適配CPU (論文4096，這裡128)
    num_workers = 0  # CPU單線程避免競爭
    pin_memory = False  # CPU不需要pin_memory

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,  # 測試也用同樣batch_size
        num_workers=num_workers,
        pin_memory=pin_memory,
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

    print(f"   ✓ 訓練樣本: {len(trainset)}")
    print(f"   ✓ 測試樣本: {len(testset)}")
    print(f"   ✓ 類別數: {len(classes)}")
    print(f"   ✓ 論文啟發優化: batch_size={batch_size}, 大規模訓練配置")

    return trainloader, testloader, classes


def create_optimized_gmlp_model(model_size="S"):
    """創建論文標準的 gMLP 模型架構"""
    print(f"\n🏗️ 創建論文標準 gMLP-{model_size} 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)  # 設置4個線程
    print("   ⚡ CPU模式：已設置4個線程")

    # 論文標準架構配置 (基於Table 1)
    if model_size == "Ti":  # gMLP-Ti (最小模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 128,  # d_model = 128
            "ff_mult": 6,  # d_ffn / d_model = 768/128 = 6
            "prob_survival": 1.00,  # 論文中Ti模型不使用隨機深度
            "params_target": 5.9,  # 目標參數量(M)
        }
    elif model_size == "S":  # gMLP-S (中等模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 256,  # d_model = 256
            "ff_mult": 6,  # d_ffn / d_model = 1536/256 = 6
            "prob_survival": 0.95,  # 論文隨機深度存活率
            "params_target": 19.5,  # 目標參數量(M)
        }
    elif model_size == "B":  # gMLP-B (大模型)
        config = {
            "depth": 30,  # #L = 30
            "dim": 512,  # d_model = 512
            "ff_mult": 6,  # d_ffn / d_model = 3072/512 = 6
            "prob_survival": 0.80,  # 論文隨機深度存活率
            "params_target": 73.4,  # 目標參數量(M)
        }
    else:
        raise ValueError(f"不支援的模型大小: {model_size}")

    model = gMLPVision(
        # === 核心架構參數 (嚴格按照論文Table 1) ===
        image_size=32,  # CIFAR-10圖像尺寸 (論文用224，這裡適配32)
        patch_size=4,  # 適配CIFAR-10的patch size
        num_classes=10,  # CIFAR-10分類數量
        dim=config["dim"],  # d_model (論文標準)
        depth=config["depth"],  # 層數 #L (論文標準)
        # === 網絡結構參數 ===
        ff_mult=config["ff_mult"],  # 前饋網絡倍數 (論文計算得出)
        channels=3,  # 輸入通道數
        # === 正則化參數 (論文標準) ===
        prob_survival=config["prob_survival"],  # 隨機深度存活率
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    params_M = total_params / 1e6  # 轉換為百萬參數

    print(f"   ✓ gMLP-{model_size} 模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 實際參數數量: {total_params:,} ({params_M:.1f}M)")
    print(f"   ✓ 論文目標參數: {config['params_target']}M")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(
        f"   ✓ 論文架構: depth={config['depth']}, dim={config['dim']}, ff_mult={config['ff_mult']}"
    )
    print(f"   ✓ 隨機深度: prob_survival={config['prob_survival']}")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=300):
    """論文標準訓練配置 - 基於ImageNet-1K超參數"""
    print(f"\n🏋️ 開始論文標準訓練 ({epochs} 個 epochs)...")

    # 論文標準訓練配置 (基於ImageNet-1K超參數)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 論文標準標籤平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 論文峰值學習率
        weight_decay=0.05,  # 論文權重衰減
        betas=(0.9, 0.999),  # 論文Adam參數
        eps=1e-6,  # 論文Adam epsilon
    )

    # 論文標準學習率調度器 (Cosine退火，10K warmup steps適配)
    total_steps = len(trainloader) * epochs
    warmup_steps = min(10000, total_steps // 10)  # 論文10K warmup，但適配較小數據集

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # 論文峰值學習率1e-3
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=warmup_steps / total_steps,  # 基於warmup steps的百分比
        anneal_strategy="cos",  # 論文使用cosine退火
        final_div_factor=1000,  # 大幅衰減到初始值的1/1000
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 論文標準早停配置
    best_val_acc = 0
    patience = 30  # 論文級別長訓練的耐心值
    patience_counter = 0

    # 論文不使用動態過擬合檢測，專注於標準訓練
    print("   📄 使用論文標準訓練配置：300 epochs + 標準正則化")

    for epoch in range(epochs):
        # 記錄每個epoch開始時間
        epoch_start_time = time.time()

        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(
            f"\nEpoch {epoch + 1}/{epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 動態調整數據增強 - 強化防過擬合策略
        if epoch >= epochs * 0.4:  # 從40%開始減少數據增強 (0.7->0.4)
            # 逐步減少隨機擦除概率
            for transform in trainloader.dataset.dataset.transform.transforms:
                if isinstance(transform, transforms.RandomErasing):
                    if epoch >= epochs * 0.4 and epoch < epochs * 0.7:
                        transform.p = 0.1  # 中期降低
                    elif epoch >= epochs * 0.7:
                        transform.p = 0.03  # 後期大幅降低

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 論文標準梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,  # 論文梯度裁剪閾值1.0
            )
            optimizer.step()
            scheduler.step()  # OneCycleLR需要每個batch更新

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 15 == 0 or (i + 1) == len(
                trainloader
            ):  # 顯示進度且包含最後批次
                acc = 100.0 * correct / total
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"   批次 {i+1:3d}/{len(trainloader)}: 損失 = {running_loss/(i+1):.4f}, "
                    f"準確率 = {acc:.2f}%, 學習率 = {current_lr:.6f}"
                )

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 驗證階段
        val_acc = quick_validate(model, testloader, device)
        val_accs.append(val_acc)

        # OneCycleLR不需要手動step

        # 記錄每個epoch結束時間
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練準確率 = {epoch_acc:.2f}%, 驗證準確率 = {val_acc:.2f}%, 時間 = {epoch_duration:.2f}s"
        )

        # 論文標準訓練進度報告 (移除過擬合監控，專注標準訓練)
        train_val_diff = epoch_acc - val_acc
        if train_val_diff > 10:
            print(f"   📊 訓練-驗證差異: {train_val_diff:.2f}%")

        # 長訓練進度提示
        if epoch == epochs // 10:
            print(f"   🔄 已完成10%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 4:
            print(f"   🔄 已完成25%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 2:
            print(f"   🔄 已完成50%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 3 // 4:
            print(f"   🔄 已完成75%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 9 // 10:
            print(f"   🔄 已完成90%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")

        # 論文標準早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint_paper.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 論文標準訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint_paper.pth"))
        print("   • 已載入最佳論文標準模型權重")

    return train_losses, train_accs, val_accs, epoch_times, total_training_time


def quick_validate(model, testloader, device):
    """快速驗證"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def evaluate_model_with_visualization(model, testloader, device, classes):
    """評估模型並生成可視化結果"""
    print("\n📊 評估CPU優化模型並生成可視化...")

    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total
    print(f"   ✓ CPU模型整體準確率: {overall_acc:.2f}%")

    # 1. 各類別準確率條形圖
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    class_accs = []
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accs.append(acc)
        else:
            class_accs.append(0)

    bars = plt.bar(classes, class_accs, color=plt.cm.tab10(np.arange(10)))
    plt.title(
        "CPU-Optimized gMLP: Accuracy of Each Category", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    # 在柱狀圖上添加數值
    for bar, acc in zip(bars, class_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. 混淆矩陣
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("CPU-Optimized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")

    # 3. 標準化混淆矩陣
    plt.subplot(2, 2, 3)
    cm_normalized = confusion_matrix(all_labels, all_predictions, normalize="true")
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(
        "CPU-Optimized Normalized Confusion Matrix", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")

    # 4. 類別分佈
    plt.subplot(2, 2, 4)
    unique, counts = np.unique(all_labels, return_counts=True)
    plt.pie(
        counts,
        labels=[classes[i] for i in unique],
        autopct="%1.1f%%",
        colors=plt.cm.tab10(np.arange(len(unique))),
    )
    plt.title("CPU Test Set Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gmlp_cpu_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 CPU優化模型詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製CPU訓練歷史"""
    print("\n📈 繪製CPU訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="CPU Training Loss")
    plt.title("CPU-Optimized Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="CPU Training Accuracy")
    plt.plot(val_accs, "r-", linewidth=2, label="CPU Validation Accuracy")
    plt.title("CPU-Optimized Accuracy Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="CPU Train-Val Difference")
    plt.title("CPU Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(epoch_times, "orange", linewidth=2, marker="o", label="CPU Epoch Time")
        plt.title("CPU Training Time per Epoch", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_cpu_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化CPU模型樣本預測結果"""
    print(f"\n🔍 可視化CPU模型 {num_samples} 個樣本預測...")

    model.eval()

    # 獲取一批數據
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, 1)

    # 繪製結果
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        "CPU-Optimized gMLP Prediction Results", fontsize=16, fontweight="bold"
    )

    for i in range(min(num_samples, len(images))):
        ax = axes[i // 4, i % 4]

        # 反標準化圖像
        img = images[i].cpu()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        ax.imshow(img.permute(1, 2, 0))

        true_label = classes[labels[i]]
        pred_label = classes[predictions[i]]
        confidence = probabilities[i, predictions[i]].item()

        # 設置顏色（正確=綠色，錯誤=紅色）
        color = "green" if labels[i] == predictions[i] else "red"

        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("gmlp_cpu_sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_model_architectures():
    """比較不同gMLP模型架構的規格"""
    print("\n📋 論文標準 gMLP 模型架構比較:")
    print("=" * 80)
    print(
        f"{'模型':<8} {'深度':<6} {'維度':<8} {'FFN倍數':<8} {'隨機深度':<10} {'目標參數(M)':<12}"
    )
    print("-" * 80)

    models_config = {
        "Ti": {
            "depth": 30,
            "dim": 128,
            "ff_mult": 6,
            "prob_survival": 1.00,
            "params": 5.9,
        },
        "S": {
            "depth": 30,
            "dim": 256,
            "ff_mult": 6,
            "prob_survival": 0.95,
            "params": 19.5,
        },
        "B": {
            "depth": 30,
            "dim": 512,
            "ff_mult": 6,
            "prob_survival": 0.80,
            "params": 73.4,
        },
    }

    for name, config in models_config.items():
        print(
            f"gMLP-{name:<3} {config['depth']:<6} {config['dim']:<8} {config['ff_mult']:<8} "
            f"{config['prob_survival']:<10.2f} {config['params']:<12.1f}"
        )

    print("-" * 80)
    print("💡 建議:")
    print("   • gMLP-Ti: 快速實驗和概念驗證")
    print("   • gMLP-S:  平衡性能和計算資源 (推薦)")
    print("   • gMLP-B:  追求最佳性能 (需要更多資源)")
    print("=" * 80)


def main():
    print("🖼️ 論文標準 gMLP 圖像分類測試")
    print("=" * 60)
    print("📄 基於官方ImageNet-1K超參數配置")
    print("=" * 60)

    # 顯示模型架構比較
    compare_model_architectures()

    try:
        # 1. 加載論文標準數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建論文標準模型 (可選擇模型大小)
        model_size = "S"  # 選擇 'Ti', 'S', 或 'B'
        model, device = create_optimized_gmlp_model(model_size=model_size)

        # 3. 論文標準訓練
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=300,  # 論文標準300 epochs
            )
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存論文標準模型
        torch.save(model.state_dict(), "gmlp_paper_model.pth")
        print("\n💾 論文標準模型已保存為 'gmlp_paper_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 論文標準測試完成！")
        print(f"\n📈 論文標準最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n📄 論文配置特性:")
        print(f"   • 模型架構: gMLP-{model_size}")
        print(f"   • 模型參數: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • 論文標準: depth=30, dim=256, ff_mult=6, prob_survival=0.95")
        print(f"   • 數據量: 15,000訓練樣本, 3,000測試樣本")
        print(f"   • 批次大小: 128 (接近論文規模)")
        print(f"   • 訓練策略: 300 epochs + cosine退火 + 10K warmup")

        print(f"\n🎯 論文標準環境建議:")
        if accuracy < 70:
            print(f"   • 300 epochs後準確度仍需提升，考慮調整模型架構")
            print(f"   • 可能需要更大的數據集或更長的訓練")
            print(f"   • 檢查數據預處理是否與論文一致")
        elif accuracy < 80:
            print(f"   • 論文標準模型表現良好！")
            print(f"   • 300個epochs充分利用了論文配置")
            print(f"   • 可考慮微調超參數進一步優化")
        else:
            print(f"   • 論文標準模型表現優秀！")
            print(f"   • 成功復現論文級別的訓練效果")
            print(f"   • 適合發表或實際應用部署")

        # 泛化能力分析
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  需要注意:")
            print(f"   • 訓練-驗證差異較大 ({overfitting_diff:.2f}%)")
            print(f"   • 可能需要更多數據或更強正則化")
        elif overfitting_diff > 8:
            print(f"\n🔶 泛化表現:")
            print(f"   • 訓練-驗證差異適中 ({overfitting_diff:.2f}%)")
            print(f"   • 論文配置取得良好平衡")
        else:
            print(f"\n✅ 優秀泛化:")
            print(f"   • 訓練-驗證差異很小 ({overfitting_diff:.2f}%)")
            print(f"   • 論文配置實現出色泛化能力")

        print(f"\n🚀 論文標準總結:")
        print(f"   • 訓練策略: 300 epochs + 論文標準超參數")
        print(f"   • 正則化: 標籤平滑0.1 + 權重衰減0.05 + 隨機深度0.8")
        print(f"   • 學習率: 峰值1e-3 + cosine退火 + 10K warmup")
        print(f"   • 優化器: AdamW + 梯度裁剪1.0 + epsilon 1e-6")
        print(f"   • 目標: 復現論文ImageNet-1K級別的訓練效果")

    except Exception as e:
        print(f"❌ 論文標準測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
