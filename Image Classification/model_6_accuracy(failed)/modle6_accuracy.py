<<<<<<< HEAD
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

    # CPU優化的數據增強策略 - 平衡效率與準確度
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),  # 稍微增加旋轉角度
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1
            ),  # 增強顏色變換
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3
            ),  # 添加輕量級仿射變換
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(
                p=0.15, scale=(0.02, 0.08)
            ),  # 添加隨機擦除提升泛化
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
        # 高準確度模式：大幅增加數據量達到90%目標
        trainset = Subset(trainset, range(5000))  # 大幅增加至5k樣本
        testset = Subset(testset, range(1000))  # 增加至1k測試樣本
        print("   🎯 高準確度模式：大量數據支撐90%目標")

    # CPU專用DataLoader優化 - 高準確度配置
    batch_size = 64  # 增加batch_size提升訓練穩定性
    num_workers = 2  # 適度增加workers加速數據載入
    pin_memory = False  # CPU環境不需要pin_memory

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
    print(
        f"   ✓ 高準確度優化: batch_size={batch_size}, num_workers=2, pin_memory=False"
    )

    return trainloader, testloader, classes


def create_optimized_gmlp_model():
    """創建CPU優化的 gMLP 模型"""
    print("\n🏗️ 創建CPU優化的 gMLP 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)  # 設置4個線程
    print("   ⚡ CPU模式：已設置4個線程")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 保持較小patch_size提升精度
        num_classes=10,  # 分類數量
        dim=384,  # 增加特徵維度提升表達能力 (256->384)
        depth=8,  # 增加模型深度提升學習能力 (5->8)
        # === 網絡結構參數 ===
        ff_mult=4,  # 保持前饋倍數
        channels=3,  # 輸入通道數
        # === 正則化參數 ===
        prob_survival=0.85,  # 適度調整隨機深度 (0.8->0.85)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 高準確度模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 參數數量: {total_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"   ✓ 高準確度配置: patch_size={4}, dim={384}, depth={8}")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=10):
    """高準確度訓練 - 目標90%準確率"""
    print(f"\n🏋️ 開始高準確度訓練 ({epochs} 個 epochs)...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 降低標籤平滑防止欠擬合
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # 提高基礎學習率加快收斂
        weight_decay=0.05,  # 增加權重衰減防止過擬合
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 高準確度學習率調度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # 提高最大學習率加快學習
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,  # 恢復30%升溫時間
        anneal_strategy="cos",
        final_div_factor=100,  # 大幅降低最終學習率確保收斂
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 早停機制變量 - 高準確度設置
    best_val_acc = 0
    patience = 15  # 大幅增加patience確保充分訓練
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

        # 動態調整數據增強 - 高準確度策略
        if epoch >= epochs * 0.8:  # 後20%的epochs減少數據增強
            # 減少隨機擦除和旋轉強度確保精確學習
            for transform in trainloader.dataset.dataset.transform.transforms:
                if isinstance(transform, transforms.RandomErasing):
                    transform.p = 0.02  # 大幅降低隨機擦除
                elif isinstance(transform, transforms.RandomRotation):
                    transform.degrees = 3  # 降低旋轉角度

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5  # 更嚴格的梯度裁剪
            )
            optimizer.step()
            scheduler.step()  # OneCycleLR需要每個batch更新

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 25 == 0 or (i + 1) == len(
                trainloader
            ):  # 更頻繁監控且顯示最後批次
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

        # 高準確度訓練進度提示
        if epoch == epochs // 5:
            print(f"   🎯 已完成20%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 2:
            print(f"   🎯 已完成50%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 4 // 5:
            print(f"   🎯 已完成80%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")

        # 90%目標檢查
        if val_acc >= 90.0:
            print(f"   🎉 達到90%目標！當前驗證準確率: {val_acc:.2f}%")

        # 早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint_cpu.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

        # 提前達到90%目標可選擇性結束
        if best_val_acc >= 92.0 and epoch >= epochs * 0.6:  # 60%進度後達到92%可考慮結束
            print(
                f"   🎯 已超越目標！最佳驗證準確率: {best_val_acc:.2f}%，可考慮提前結束"
            )

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 高準確度訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint_cpu.pth"))
        print("   • 已載入最佳高準確度模型權重")

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
    print(f"   ✓ 高準確度模型整體準確率: {overall_acc:.2f}%")

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
        "High-Accuracy gMLP: Accuracy of Each Category", fontsize=14, fontweight="bold"
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
    plt.title("High-Accuracy Confusion Matrix", fontsize=14, fontweight="bold")
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
        "High-Accuracy Normalized Confusion Matrix", fontsize=14, fontweight="bold"
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
    plt.title(
        "High-Accuracy Test Set Category Distribution", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("gmlp_cpu_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 高準確度模型詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製高準確度訓練歷史"""
    print("\n📈 繪製高準確度訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="High-Accuracy Training Loss")
    plt.title("High-Accuracy Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="High-Accuracy Training")
    plt.plot(val_accs, "r-", linewidth=2, label="High-Accuracy Validation")
    plt.title("High-Accuracy Training Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="High-Accuracy Train-Val Diff")
    plt.title("High-Accuracy Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(
            epoch_times,
            "orange",
            linewidth=2,
            marker="o",
            label="High-Accuracy Epoch Time",
        )
        plt.title(
            "High-Accuracy Training Time per Epoch", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_cpu_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化高準確度模型樣本預測結果"""
    print(f"\n🔍 可視化高準確度模型 {num_samples} 個樣本預測...")

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
        "High-Accuracy gMLP Prediction Results", fontsize=16, fontweight="bold"
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


def main():
    print("🎯 高準確度版 gMLP 圖像分類測試")
    print("=" * 60)
    print("🏆 目標90%準確率 - 可犧牲訓練時間換取高準確度")
    print("=" * 60)

    try:
        # 1. 加載高準確度數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建高準確度模型
        model, device = create_optimized_gmlp_model()

        # 3. 高準確度訓練 - 目標90%準確率
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=80,  # 大幅增加epochs確保達到90%目標
            )
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存高準確度模型
        torch.save(model.state_dict(), "gmlp_high_accuracy_model.pth")
        print("\n💾 高準確度模型已保存為 'gmlp_high_accuracy_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 高準確度測試完成！")
        print(f"\n📈 高準確度最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n🎯 高準確度特性:")
        print(f"   • 模型參數: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • 高準確度配置: patch_size=4, dim=384, depth=8")
        print(f"   • 數據量: 5000訓練樣本, 1000測試樣本")
        print(f"   • 批次大小: 64 (高效能)")
        print(f"   • 優化策略: 大模型 + 大數據 + 長訓練")

        print(f"\n🎯 90%目標達成評估:")
        if accuracy >= 90:
            print(f"   🎉 恭喜！成功達到90%目標！")
            print(f"   🏆 最終準確率: {accuracy:.2f}%")
            print(f"   ✅ 訓練策略完全成功")
        elif accuracy >= 85:
            print(f"   🔥 非常接近90%目標！")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   💡 建議：可嘗試進一步微調或增加訓練時間")
        elif accuracy >= 80:
            print(f"   📊 良好進展，但需要進一步優化")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   🔧 建議：檢查模型架構或數據增強策略")
        else:
            print(f"   ⚠️  準確率不達預期")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   🔧 建議：重新檢查所有超參數設置")

        # 過擬合檢測
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  過擬合警告:")
            print(f"   • 訓練-驗證差異過大 ({overfitting_diff:.2f}%)")
            print(f"   • 建議：增加正則化或使用更多數據")
        elif overfitting_diff > 8:
            print(f"\n🔶 輕微過擬合:")
            print(f"   • 訓練-驗證差異適中 ({overfitting_diff:.2f}%)")
            print(f"   • 這在高準確度訓練中是可接受的")
        else:
            print(f"\n✅ 模型泛化優秀:")
            print(f"   • 訓練-驗證差異很小 ({overfitting_diff:.2f}%)")
            print(f"   • 模型具有良好的泛化能力")

        print(f"\n🚀 高準確度訓練總結:")
        print(f"   • 訓練策略: 80 epochs + 大模型 + 適量數據")
        print(f"   • 模型規模: {sum(p.numel() for p in model.parameters()):,} 參數")
        print(f"   • 數據規模: 5k訓練樣本")
        print(f"   • 時間投入: {total_training_time/3600:.1f} 小時")
        print(f"   • 準確度收益: 相比基礎版本大幅提升")

    except Exception as e:
        print(f"❌ 高準確度測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
=======
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

    # CPU優化的數據增強策略 - 平衡效率與準確度
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),  # 稍微增加旋轉角度
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.1
            ),  # 增強顏色變換
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3
            ),  # 添加輕量級仿射變換
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(
                p=0.15, scale=(0.02, 0.08)
            ),  # 添加隨機擦除提升泛化
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
        # 高準確度模式：大幅增加數據量達到90%目標
        trainset = Subset(trainset, range(5000))  # 大幅增加至5k樣本
        testset = Subset(testset, range(1000))  # 增加至1k測試樣本
        print("   🎯 高準確度模式：大量數據支撐90%目標")

    # CPU專用DataLoader優化 - 高準確度配置
    batch_size = 64  # 增加batch_size提升訓練穩定性
    num_workers = 2  # 適度增加workers加速數據載入
    pin_memory = False  # CPU環境不需要pin_memory

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
    print(
        f"   ✓ 高準確度優化: batch_size={batch_size}, num_workers=2, pin_memory=False"
    )

    return trainloader, testloader, classes


def create_optimized_gmlp_model():
    """創建CPU優化的 gMLP 模型"""
    print("\n🏗️ 創建CPU優化的 gMLP 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)  # 設置4個線程
    print("   ⚡ CPU模式：已設置4個線程")

    model = gMLPVision(
        # === 核心架構參數 ===
        image_size=32,  # 圖像尺寸
        patch_size=4,  # 保持較小patch_size提升精度
        num_classes=10,  # 分類數量
        dim=384,  # 增加特徵維度提升表達能力 (256->384)
        depth=8,  # 增加模型深度提升學習能力 (5->8)
        # === 網絡結構參數 ===
        ff_mult=4,  # 保持前饋倍數
        channels=3,  # 輸入通道數
        # === 正則化參數 ===
        prob_survival=0.85,  # 適度調整隨機深度 (0.8->0.85)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 高準確度模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 參數數量: {total_params:,}")
    print(f"   ✓ 模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"   ✓ 高準確度配置: patch_size={4}, dim={384}, depth={8}")

    return model, device


def train_model_with_scheduler(model, trainloader, testloader, device, epochs=10):
    """高準確度訓練 - 目標90%準確率"""
    print(f"\n🏋️ 開始高準確度訓練 ({epochs} 個 epochs)...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 降低標籤平滑防止欠擬合
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # 提高基礎學習率加快收斂
        weight_decay=0.05,  # 增加權重衰減防止過擬合
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 高準確度學習率調度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # 提高最大學習率加快學習
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,  # 恢復30%升溫時間
        anneal_strategy="cos",
        final_div_factor=100,  # 大幅降低最終學習率確保收斂
    )

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []  # 記錄每個epoch的時間

    # 記錄總訓練開始時間
    total_start_time = time.time()

    # 早停機制變量 - 高準確度設置
    best_val_acc = 0
    patience = 15  # 大幅增加patience確保充分訓練
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

        # 動態調整數據增強 - 高準確度策略
        if epoch >= epochs * 0.8:  # 後20%的epochs減少數據增強
            # 減少隨機擦除和旋轉強度確保精確學習
            for transform in trainloader.dataset.dataset.transform.transforms:
                if isinstance(transform, transforms.RandomErasing):
                    transform.p = 0.02  # 大幅降低隨機擦除
                elif isinstance(transform, transforms.RandomRotation):
                    transform.degrees = 3  # 降低旋轉角度

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5  # 更嚴格的梯度裁剪
            )
            optimizer.step()
            scheduler.step()  # OneCycleLR需要每個batch更新

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 25 == 0 or (i + 1) == len(
                trainloader
            ):  # 更頻繁監控且顯示最後批次
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

        # 高準確度訓練進度提示
        if epoch == epochs // 5:
            print(f"   🎯 已完成20%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs // 2:
            print(f"   🎯 已完成50%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")
        elif epoch == epochs * 4 // 5:
            print(f"   🎯 已完成80%訓練，目前最佳驗證準確率: {best_val_acc:.2f}%")

        # 90%目標檢查
        if val_acc >= 90.0:
            print(f"   🎉 達到90%目標！當前驗證準確率: {val_acc:.2f}%")

        # 早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model_checkpoint_cpu.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停：驗證準確率 {patience} 個epoch未提升")
                break

        # 提前達到90%目標可選擇性結束
        if best_val_acc >= 92.0 and epoch >= epochs * 0.6:  # 60%進度後達到92%可考慮結束
            print(
                f"   🎯 已超越目標！最佳驗證準確率: {best_val_acc:.2f}%，可考慮提前結束"
            )

    # 計算總訓練時間
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 高準確度訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
    )
    print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")
    print(f"   • 最快epoch時間: {np.min(epoch_times):.2f}s")
    print(f"   • 最慢epoch時間: {np.max(epoch_times):.2f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_model_checkpoint_cpu.pth"))
        print("   • 已載入最佳高準確度模型權重")

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
    print(f"   ✓ 高準確度模型整體準確率: {overall_acc:.2f}%")

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
        "High-Accuracy gMLP: Accuracy of Each Category", fontsize=14, fontweight="bold"
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
    plt.title("High-Accuracy Confusion Matrix", fontsize=14, fontweight="bold")
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
        "High-Accuracy Normalized Confusion Matrix", fontsize=14, fontweight="bold"
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
    plt.title(
        "High-Accuracy Test Set Category Distribution", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("gmlp_cpu_evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印詳細報告
    print(f"\n📋 高準確度模型詳細分類報告:")
    target_names = [f"{i}_{classes[i]}" for i in range(10)]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=3
    )
    print(report)

    return overall_acc


def plot_training_history(train_losses, train_accs, val_accs, epoch_times=None):
    """繪製高準確度訓練歷史"""
    print("\n📈 繪製高準確度訓練歷史...")

    # 調整圖片大小以容納時間圖表
    if epoch_times is not None:
        plt.figure(figsize=(20, 5))
        subplot_count = 4
    else:
        plt.figure(figsize=(15, 5))
        subplot_count = 3

    # 損失曲線
    plt.subplot(1, subplot_count, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="High-Accuracy Training Loss")
    plt.title("High-Accuracy Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, subplot_count, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="High-Accuracy Training")
    plt.plot(val_accs, "r-", linewidth=2, label="High-Accuracy Validation")
    plt.title("High-Accuracy Training Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率差異
    plt.subplot(1, subplot_count, 3)
    diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(diff, "purple", linewidth=2, label="High-Accuracy Train-Val Diff")
    plt.title("High-Accuracy Overfitting Monitor", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Difference (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 時間統計圖（如果提供了時間數據）
    if epoch_times is not None:
        plt.subplot(1, subplot_count, 4)
        plt.plot(
            epoch_times,
            "orange",
            linewidth=2,
            marker="o",
            label="High-Accuracy Epoch Time",
        )
        plt.title(
            "High-Accuracy Training Time per Epoch", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig("gmlp_cpu_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model, testloader, device, classes, num_samples=12):
    """可視化高準確度模型樣本預測結果"""
    print(f"\n🔍 可視化高準確度模型 {num_samples} 個樣本預測...")

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
        "High-Accuracy gMLP Prediction Results", fontsize=16, fontweight="bold"
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


def main():
    print("🎯 高準確度版 gMLP 圖像分類測試")
    print("=" * 60)
    print("🏆 目標90%準確率 - 可犧牲訓練時間換取高準確度")
    print("=" * 60)

    try:
        # 1. 加載高準確度數據
        trainloader, testloader, classes = load_cifar10_data_enhanced(quick_test=True)

        # 2. 創建高準確度模型
        model, device = create_optimized_gmlp_model()

        # 3. 高準確度訓練 - 目標90%準確率
        train_losses, train_accs, val_accs, epoch_times, total_training_time = (
            train_model_with_scheduler(
                model,
                trainloader,
                testloader,
                device,
                epochs=80,  # 大幅增加epochs確保達到90%目標
            )
        )

        # 4. 繪製訓練歷史
        plot_training_history(train_losses, train_accs, val_accs, epoch_times)

        # 5. 詳細評估與可視化
        accuracy = evaluate_model_with_visualization(model, testloader, device, classes)

        # 6. 可視化預測樣本
        visualize_sample_predictions(model, testloader, device, classes)

        # 7. 保存高準確度模型
        torch.save(model.state_dict(), "gmlp_high_accuracy_model.pth")
        print("\n💾 高準確度模型已保存為 'gmlp_high_accuracy_model.pth'")

        print("\n" + "=" * 60)
        print("✅ 高準確度測試完成！")
        print(f"\n📈 高準確度最終結果:")
        print(f"   • 最終測試準確率: {accuracy:.2f}%")
        print(f"   • 最佳驗證準確率: {max(val_accs):.2f}%")
        print(f"   • 訓練-驗證差異: {train_accs[-1] - val_accs[-1]:.2f}%")
        print(
            f"   • 總訓練時間: {total_training_time:.2f}s ({total_training_time/60:.2f}min)"
        )
        print(f"   • 平均每epoch時間: {np.mean(epoch_times):.2f}s")

        print(f"\n🎯 高準確度特性:")
        print(f"   • 模型參數: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • 高準確度配置: patch_size=4, dim=384, depth=8")
        print(f"   • 數據量: 5000訓練樣本, 1000測試樣本")
        print(f"   • 批次大小: 64 (高效能)")
        print(f"   • 優化策略: 大模型 + 大數據 + 長訓練")

        print(f"\n🎯 90%目標達成評估:")
        if accuracy >= 90:
            print(f"   🎉 恭喜！成功達到90%目標！")
            print(f"   🏆 最終準確率: {accuracy:.2f}%")
            print(f"   ✅ 訓練策略完全成功")
        elif accuracy >= 85:
            print(f"   🔥 非常接近90%目標！")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   💡 建議：可嘗試進一步微調或增加訓練時間")
        elif accuracy >= 80:
            print(f"   📊 良好進展，但需要進一步優化")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   🔧 建議：檢查模型架構或數據增強策略")
        else:
            print(f"   ⚠️  準確率不達預期")
            print(f"   📈 當前準確率: {accuracy:.2f}%")
            print(f"   🔧 建議：重新檢查所有超參數設置")

        # 過擬合檢測
        overfitting_diff = train_accs[-1] - val_accs[-1]
        if overfitting_diff > 15:
            print(f"\n⚠️  過擬合警告:")
            print(f"   • 訓練-驗證差異過大 ({overfitting_diff:.2f}%)")
            print(f"   • 建議：增加正則化或使用更多數據")
        elif overfitting_diff > 8:
            print(f"\n🔶 輕微過擬合:")
            print(f"   • 訓練-驗證差異適中 ({overfitting_diff:.2f}%)")
            print(f"   • 這在高準確度訓練中是可接受的")
        else:
            print(f"\n✅ 模型泛化優秀:")
            print(f"   • 訓練-驗證差異很小 ({overfitting_diff:.2f}%)")
            print(f"   • 模型具有良好的泛化能力")

        print(f"\n🚀 高準確度訓練總結:")
        print(f"   • 訓練策略: 80 epochs + 大模型 + 適量數據")
        print(f"   • 模型規模: {sum(p.numel() for p in model.parameters()):,} 參數")
        print(f"   • 數據規模: 5k訓練樣本")
        print(f"   • 時間投入: {total_training_time/3600:.1f} 小時")
        print(f"   • 準確度收益: 相比基礎版本大幅提升")

    except Exception as e:
        print(f"❌ 高準確度測試失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
>>>>>>> 420764095488647da1ecd1309c810893dfec8ea4
