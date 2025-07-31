<<<<<<< HEAD
"""
增強版 gMLP 圖像分類測試
包含可視化結果和準確率優化技巧
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
    """加載增強的 CIFAR-10 數據集"""
    print("📦 加載增強的 CIFAR-10 數據集...")

    # 更精細的數據增強策略 - 減輕增強強度
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 反射填充
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # 減少旋轉角度
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1  # 減輕顏色變化
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1
            ),  # 減少模糊概率
            # 暫時移除仿射變換以加速訓練
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.10)),  # 減輕隨機擦除
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
        trainset = Subset(trainset, range(5000))  # 增加到5000樣本提升性能
        testset = Subset(testset, range(1000))  # 增加到1000樣本提升評估穩定性
        print("   ⚡ 快速測試模式：使用更多數據提升準確度")

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # 優化批次大小和數據加載
    )  # 更大批次提升訓練效率
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
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

    return trainloader, testloader, classes


def create_optimized_gmlp_model():
    """創建優化的 gMLP 模型"""
    print("\n🏗️ 創建優化的 gMLP 模型...")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 最佳patch size平衡細節和效率
        num_classes=10,  # 分類數量
        dim=256,  # 特徵維度：適中大小平衡性能和速度
        depth=6,  # 模型深度：減少到6層提升訓練效率
        # === 網絡結構參數 ===
        ff_mult=4,  # 前饋倍數：保持4維持計算效率
        channels=3,  # 輸入通道數
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 優化模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 參數數量: {total_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=10):
    """使用學習率調度器的優化訓練"""
    print(f"\n🏋️ 開始優化訓練 ({epochs} 個 epochs)...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 適度標籤平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # 提高初始學習率加速收斂
        weight_decay=0.01,  # 適度權重衰減
        betas=(0.9, 0.999),  # 優化的動量參數
        eps=1e-6,  # 數值穩定性
    )  # 平衡學習速度和穩定性

    # 使用更高效的學習率調度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 第一次重啟的週期（epochs）
        T_mult=1,  # 週期倍數
        eta_min=1e-6,  # 最小學習率
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 早停機制變量
    best_val_acc = 0
    patience = 6  # 適度patience平衡訓練時間和性能
    patience_counter = 0

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

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0  # 調整梯度裁剪閾值
            )  # 更適中的梯度裁剪
            optimizer.step()

            # 每個epoch結束後更新學習率（而不是每個batch）

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 20 == 0:
                acc = 100.0 * correct / total
                current_lr = scheduler.get_last_lr()[0]  # 獲取當前學習率
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

        # 更新學習率（每個epoch）
        scheduler.step()

        # 記錄每個epoch結束時間
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練準確率 = {epoch_acc:.2f}%, 驗證準確率 = {val_acc:.2f}%, 時間 = {epoch_duration:.2f}s"
        )

        # 早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint.pth"))
        print("   • 已載入最佳模型權重")

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
    print("\n📊 評估模型並生成可視化...")

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
    print(f"   ✓ overall accuracy: {overall_acc:.2f}%")  # 總體準確率

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
    plt.title("Accuracy of Each Category", fontsize=14, fontweight="bold")
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
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
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
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
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
    plt.title("Test Set Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gmlp_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製訓練歷史"""
    print("\n📈 繪製訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="Training Loss")
    plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="Training Accuracy")
    plt.plot(val_accs, "r-", linewidth=2, label="Validation Accuracy")
    plt.title("Accuracy Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="Train-Val Difference")
    plt.title("Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(epoch_times, "orange", linewidth=2, marker="o", label="Epoch Time")
        plt.title("Training Time per Epoch", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化樣本預測結果"""
    print(f"\n🔍 可視化 {num_samples} 個樣本預測...")

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
    fig.suptitle("gMLP Prediction Results", fontsize=16, fontweight="bold")

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
    plt.savefig("gmlp_sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("🖼️ 增強版 gMLP 圖像分類測試")
    print("=" * 60)

    try:
        # 1. 加載增強數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建優化模型
        model, device = create_optimized_gmlp_model()

        # 3. 優化訓練
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=20,  # 適度減少epochs提升效率
            )  # 平衡訓練時間和性能
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存模型
        torch.save(model.state_dict(), "gmlp_model.pth")
        print("\n💾 模型已保存為 'gmlp_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 增強測試完成！")
        print(f"\n📈 最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n🎯 改進建議:")
        if accuracy < 70:
            print(f"   • 考慮增加訓練時間和數據量")
            print(f"   • 嘗試更強的數據增強")
            print(f"   • 調整學習率和優化器參數")
            print(f"   • 檢查數據質量和標籤正確性")
        elif accuracy < 80:
            print(f"   • 表現良好！可嘗試更深的模型")
            print(f"   • 考慮使用學習率預熱")
            print(f"   • 實驗不同的正則化技巧")
            print(f"   • 嘗試集成學習方法")
        elif accuracy < 90:
            print(f"   • 很好的表現！已達到目標80%以上")
            print(f"   • 可以考慮模型蒸餾來壓縮大小")
            print(f"   • 實驗測試時間增強(TTA)")
            print(f"   • 可以用於生產環境")
        else:
            print(f"   • 優秀的表現！模型已經很好")
            print(f"   • 可以用於實際應用")
            print(f"   • 考慮在其他數據集上測試泛化能力")

        # 過擬合檢測
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  過擬合警告:")
            print(f"   • 訓練-驗證差異過大 ({overfitting_diff:.2f}%)")
            print(f"   • 建議增加正則化或減少模型複雜度")
        elif overfitting_diff > 10:
            print(f"\n🔶 輕微過擬合:")
            print(f"   • 訓練-驗證差異較大 ({overfitting_diff:.2f}%)")
            print(f"   • 可以適當調整正則化參數")
        else:
            print(f"\n✅ 模型泛化良好:")
            print(f"   • 訓練-驗證差異合理 ({overfitting_diff:.2f}%)")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
=======
"""
增強版 gMLP 圖像分類測試
包含可視化結果和準確率優化技巧
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
    """加載增強的 CIFAR-10 數據集"""
    print("📦 加載增強的 CIFAR-10 數據集...")

    # 更精細的數據增強策略 - 減輕增強強度
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # 反射填充
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # 減少旋轉角度
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1  # 減輕顏色變化
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1
            ),  # 減少模糊概率
            # 暫時移除仿射變換以加速訓練
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.10)),  # 減輕隨機擦除
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
        trainset = Subset(trainset, range(5000))  # 增加到5000樣本提升性能
        testset = Subset(testset, range(1000))  # 增加到1000樣本提升評估穩定性
        print("   ⚡ 快速測試模式：使用更多數據提升準確度")

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # 優化批次大小和數據加載
    )  # 更大批次提升訓練效率
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
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

    return trainloader, testloader, classes


def create_optimized_gmlp_model():
    """創建優化的 gMLP 模型"""
    print("\n🏗️ 創建優化的 gMLP 模型...")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 最佳patch size平衡細節和效率
        num_classes=10,  # 分類數量
        dim=256,  # 特徵維度：適中大小平衡性能和速度
        depth=6,  # 模型深度：減少到6層提升訓練效率
        # === 網絡結構參數 ===
        ff_mult=4,  # 前饋倍數：保持4維持計算效率
        channels=3,  # 輸入通道數
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 優化模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 參數數量: {total_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=10):
    """使用學習率調度器的優化訓練"""
    print(f"\n🏋️ 開始優化訓練 ({epochs} 個 epochs)...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 適度標籤平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # 提高初始學習率加速收斂
        weight_decay=0.01,  # 適度權重衰減
        betas=(0.9, 0.999),  # 優化的動量參數
        eps=1e-6,  # 數值穩定性
    )  # 平衡學習速度和穩定性

    # 使用更高效的學習率調度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 第一次重啟的週期（epochs）
        T_mult=1,  # 週期倍數
        eta_min=1e-6,  # 最小學習率
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 早停機制變量
    best_val_acc = 0
    patience = 6  # 適度patience平衡訓練時間和性能
    patience_counter = 0

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

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0  # 調整梯度裁剪閾值
            )  # 更適中的梯度裁剪
            optimizer.step()

            # 每個epoch結束後更新學習率（而不是每個batch）

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 20 == 0:
                acc = 100.0 * correct / total
                current_lr = scheduler.get_last_lr()[0]  # 獲取當前學習率
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

        # 更新學習率（每個epoch）
        scheduler.step()

        # 記錄每個epoch結束時間
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練準確率 = {epoch_acc:.2f}%, 驗證準確率 = {val_acc:.2f}%, 時間 = {epoch_duration:.2f}s"
        )

        # 早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint.pth"))
        print("   • 已載入最佳模型權重")

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
    print("\n📊 評估模型並生成可視化...")

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
    print(f"   ✓ overall accuracy: {overall_acc:.2f}%")  # 總體準確率

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
    plt.title("Accuracy of Each Category", fontsize=14, fontweight="bold")
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
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
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
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
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
    plt.title("Test Set Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gmlp_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製訓練歷史"""
    print("\n📈 繪製訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="Training Loss")
    plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="Training Accuracy")
    plt.plot(val_accs, "r-", linewidth=2, label="Validation Accuracy")
    plt.title("Accuracy Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="Train-Val Difference")
    plt.title("Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(epoch_times, "orange", linewidth=2, marker="o", label="Epoch Time")
        plt.title("Training Time per Epoch", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化樣本預測結果"""
    print(f"\n🔍 可視化 {num_samples} 個樣本預測...")

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
    fig.suptitle("gMLP Prediction Results", fontsize=16, fontweight="bold")

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
    plt.savefig("gmlp_sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    print("🖼️ 增強版 gMLP 圖像分類測試")
    print("=" * 60)

    try:
        # 1. 加載增強數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建優化模型
        model, device = create_optimized_gmlp_model()

        # 3. 優化訓練
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=20,  # 適度減少epochs提升效率
            )  # 平衡訓練時間和性能
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存模型
        torch.save(model.state_dict(), "gmlp_model.pth")
        print("\n💾 模型已保存為 'gmlp_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 增強測試完成！")
        print(f"\n📈 最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n🎯 改進建議:")
        if accuracy < 70:
            print(f"   • 考慮增加訓練時間和數據量")
            print(f"   • 嘗試更強的數據增強")
            print(f"   • 調整學習率和優化器參數")
            print(f"   • 檢查數據質量和標籤正確性")
        elif accuracy < 80:
            print(f"   • 表現良好！可嘗試更深的模型")
            print(f"   • 考慮使用學習率預熱")
            print(f"   • 實驗不同的正則化技巧")
            print(f"   • 嘗試集成學習方法")
        elif accuracy < 90:
            print(f"   • 很好的表現！已達到目標80%以上")
            print(f"   • 可以考慮模型蒸餾來壓縮大小")
            print(f"   • 實驗測試時間增強(TTA)")
            print(f"   • 可以用於生產環境")
        else:
            print(f"   • 優秀的表現！模型已經很好")
            print(f"   • 可以用於實際應用")
            print(f"   • 考慮在其他數據集上測試泛化能力")

        # 過擬合檢測
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  過擬合警告:")
            print(f"   • 訓練-驗證差異過大 ({overfitting_diff:.2f}%)")
            print(f"   • 建議增加正則化或減少模型複雜度")
        elif overfitting_diff > 10:
            print(f"\n🔶 輕微過擬合:")
            print(f"   • 訓練-驗證差異較大 ({overfitting_diff:.2f}%)")
            print(f"   • 可以適當調整正則化參數")
        else:
            print(f"\n✅ 模型泛化良好:")
            print(f"   • 訓練-驗證差異合理 ({overfitting_diff:.2f}%)")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
>>>>>>> 420764095488647da1ecd1309c810893dfec8ea4
