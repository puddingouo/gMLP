"""
超縮小版 gMLP 圖像分類模型
基於論文架構但大幅縮小規模以提高訓練效率
針對快速原型開發和資源受限環境優化
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


def load_cifar10_data_ultrafast(quick_test=True):
    """加載超快速 CIFAR-10 數據集 - 針對快速訓練優化"""
    print("📦 加載超快速 CIFAR-10 數據集...")

    # 簡化的數據增強策略 - 減少計算開銷
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=2),  # 減少padding
            transforms.RandomHorizontalFlip(p=0.3),  # 降低概率
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
        # 超快速模式：使用更少數據進行快速原型
        trainset = Subset(trainset, range(50000))  # 減少到30K樣本
        testset = Subset(testset, range(10000))  # 減少到5K樣本
        print("   🚀 超快速模式：小規模數據集訓練")

    # 優化DataLoader配置
    batch_size = 64  # 減少batch size以適配小模型
    num_workers = 0
    pin_memory = False

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
        shuffle=False,
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
    print(f"   ✓ Batch大小: {batch_size}")

    return trainloader, testloader, classes


def create_ultra_small_gmlp_model(model_size="L"):
    """創建超縮小版 gMLP 模型架構"""
    print(f"\n🏗️ 創建超縮小版 gMLP-{model_size} 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)
    print("   ⚡ CPU模式：已設置4個線程")

    # 超縮小版架構配置 - 大幅降低複雜度
    if model_size == "Test":  # 測試模型 - 新增
        config = {
            "depth": 4,  # 極少層數
            "dim": 64,  # 極小維度
            "ff_mult": 2,  # 最小FFN倍數
            "prob_survival": 1.00,
            "params_target": 0.1,
        }
    elif model_size == "Nano":  # 極小模型 - 新增
        config = {
            "depth": 6,  # 極少層數
            "dim": 64,  # 極小維度
            "ff_mult": 2,  # 最小FFN倍數
            "prob_survival": 1.00,
            "params_target": 0.3,
        }
    elif model_size == "XS":  # 超小模型
        config = {
            "depth": 8,  # 減少層數
            "dim": 80,  # 小維度
            "ff_mult": 3,  # 小FFN倍數
            "prob_survival": 1.00,
            "params_target": 0.8,
        }
    elif model_size == "S":  # 小模型
        config = {
            "depth": 12,  # 中等層數
            "dim": 128,  # 中等維度
            "ff_mult": 3,  # 中等FFN倍數
            "prob_survival": 0.98,
            "params_target": 2.0,
        }
    elif model_size == "M":  # 中等模型
        config = {
            "depth": 16,  # 適中層數
            "dim": 160,  # 適中維度
            "ff_mult": 4,  # 適中FFN倍數
            "prob_survival": 0.95,
            "params_target": 4.5,
        }
    elif model_size == "L":  # 大模型 - 新增
        config = {
            "depth": 30,  # #L = 30
            "dim": 128,  # d_model = 128
            "ff_mult": 6,  # d_ffn / d_model = 768/128 = 6
            "prob_survival": 1.00,  # 論文中Ti模型不使用隨機深度
            "params_target": 5.9,  # 目標參數量(M)
        }
    else:
        raise ValueError(
            f"不支援的模型大小: {model_size}。支援: Test, Nano, XS, S, M, L"
        )

    model = gMLPVision(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=config["dim"],
        depth=config["depth"],
        ff_mult=config["ff_mult"],
        channels=3,
        prob_survival=config["prob_survival"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    params_M = total_params / 1e6

    # 詳細參數分析
    print(f"\n📊 詳細參數分析:")
    total_analyzed = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_analyzed += param_count
        if param_count > 1000:  # 只顯示主要參數
            print(f"   • {name:<35}: {param_count:>8,} ({param_count/1e6:.3f}M)")

    print(f"   {'='*50}")
    print(f"   • {'總計':<35}: {total_analyzed:>8,} ({total_analyzed/1e6:.3f}M)")

    print(f"\n✅ 超縮小版 gMLP-{model_size} 模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 實際參數數量: {total_params:,} ({params_M:.2f}M)")
    print(f"   ✓ 目標參數預期: {config['params_target']}M")
    print(
        f"   ✓ 參數差異分析: 實際比預期 {'多' if params_M > config['params_target'] else '少'} {abs(params_M - config['params_target']):.2f}M"
    )
    print(f"   ✓ 模型檔案大小: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    print(
        f"   ✓ 架構配置: depth={config['depth']}, dim={config['dim']}, ff_mult={config['ff_mult']}"
    )

    return model, device


def train_ultra_fast(model, trainloader, testloader, device, epochs=50):
    """超快速訓練配置 - 針對快速原型開發 + 過擬合早停"""
    print(f"\n🏋️ 開始超快速訓練 ({epochs} 個 epochs)...")
    print("   🛡️  啟用過擬合早停保護")

    # 快速訓練配置
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # 減少標籤平滑
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-3,  # 提高學習率加速收斂
        weight_decay=0.01,  # 減少權重衰減
        betas=(0.9, 0.95),  # 調整momentum
    )

    # 簡化學習率調度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []  # 新增：記錄驗證損失
    epoch_times = []

    best_val_acc = 0
    patience = 15  # 減少耐心值
    patience_counter = 0

    # 過擬合早停配置 - 適配超小模型（更嚴格）
    overfitting_patience = 6  # 更短的過擬合容忍期（從8降到6）
    overfitting_counter = 0
    overfitting_threshold = 8.0  # 更低的過擬合閾值（從10.0降到8.0）
    min_epochs_before_overfitting_check = 8  # 更早開始檢測（從10降到8）

    total_start_time = time.time()

    for epoch in range(epochs):
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

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 每50個批次或最後一個批次顯示進度
            if (i + 1) % 50 == 0 or (i + 1) == len(trainloader):
                acc = 100.0 * correct / total
                print(
                    f"   批次 {i+1:3d}/{len(trainloader)}: 損失={running_loss/(i+1):.4f}, 準確率={acc:.2f}%"
                )

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 驗證階段
        val_acc, val_loss = quick_validate_with_loss(
            model, testloader, device, criterion
        )
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練={epoch_acc:.2f}%, 驗證={val_acc:.2f}%, 時間={epoch_duration:.1f}s"
        )

        # 過擬合早停檢測
        train_val_diff = epoch_acc - val_acc
        if epoch >= min_epochs_before_overfitting_check:
            if train_val_diff > overfitting_threshold:
                overfitting_counter += 1
                print(
                    f"   ⚠️  過擬合警告: 差異 {train_val_diff:.2f}% > 閾值 {overfitting_threshold}% ({overfitting_counter}/{overfitting_patience})"
                )

                if overfitting_counter >= overfitting_patience:
                    print(
                        f"   🛑 過擬合早停: 連續 {overfitting_patience} epochs 訓練-驗證差異超過 {overfitting_threshold}%"
                    )
                    break
            else:
                overfitting_counter = 0  # 重置計數器

        # 顯示訓練-驗證差異（用於監控）
        if train_val_diff > 5:
            print(f"   📊 訓練-驗證差異: {train_val_diff:.2f}%")

        # 早停機制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_ultra_small_model.pth")
            print(f"   💾 新最佳模型已保存: 驗證準確率 {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏰ 性能早停：驗證準確率 {patience} 個epoch未提升")
                break

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 超快速訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.1f}s ({total_training_time/60:.1f}min)"
    )
    print(f"   • 實際訓練epochs: {len(train_losses)} / {epochs}")
    print(f"   • 平均每epoch: {np.mean(epoch_times):.1f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

    # 早停原因分析
    if len(train_losses) < epochs:
        final_train_val_diff = (
            train_accs[-1] - val_accs[-1] if train_accs and val_accs else 0
        )
        if overfitting_counter >= overfitting_patience:
            print(f"   • 早停原因: 過擬合檢測 (差異: {final_train_val_diff:.2f}%)")
        elif patience_counter >= patience:
            print(f"   • 早停原因: 性能停滯 ({patience} epochs無提升)")
    else:
        print(f"   • 訓練狀態: 完整訓練 ({epochs} epochs)")

    # 載入最佳模型
    if best_val_acc > 0:
        model.load_state_dict(torch.load("best_ultra_small_model.pth"))
        print("   • 已載入最佳模型權重")

    return (
        train_losses,
        train_accs,
        val_accs,
        val_losses,
        epoch_times,
        total_training_time,
    )


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


def quick_validate_with_loss(model, testloader, device, criterion):
    """快速驗證（同時計算準確率和損失）"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    return accuracy, avg_loss


def evaluate_ultra_model(model, testloader, device, classes):
    """評估超縮小版模型"""
    print("\n📊 評估超縮小版模型...")

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
    print(f"   ✓ 整體準確率: {overall_acc:.2f}%")

    # 簡化的結果可視化
    plt.figure(figsize=(12, 8))

    # 各類別準確率
    plt.subplot(2, 2, 1)
    class_accs = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(10)
    ]
    bars = plt.bar(classes, class_accs, color=plt.cm.tab10(np.arange(10)))
    plt.title("Ultra-Small gMLP: Class Accuracy", fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, class_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 混淆矩陣
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # 準確率分佈
    plt.subplot(2, 2, 3)
    plt.hist(class_accs, bins=8, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title("Class Accuracy Distribution", fontweight="bold")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    # 模型統計
    plt.subplot(2, 2, 4)
    stats_text = f"""Model Statistics:
• Overall Accuracy: {overall_acc:.2f}%
• Best Class: {classes[np.argmax(class_accs)]} ({max(class_accs):.1f}%)
• Worst Class: {classes[np.argmin(class_accs)]} ({min(class_accs):.1f}%)
• Average Class Accuracy: {np.mean(class_accs):.1f}%
• Standard Deviation: {np.std(class_accs):.1f}%"""

    plt.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )
    plt.axis("off")
    plt.title("Model Statistics", fontweight="bold")

    plt.tight_layout()
    plt.savefig("ultra_small_gmlp_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return overall_acc


def plot_ultra_training_history(
    train_losses, train_accs, val_accs, val_losses, epoch_times
):
    """繪製超快速訓練歷史（包含訓練和驗證損失比較）"""
    print("\n📈 繪製超快速訓練歷史...")

    plt.figure(figsize=(16, 4))

    # 損失曲線比較（訓練 vs 驗證）
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="Training Loss")
    plt.plot(val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.title("Loss Comparison", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, 4, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="Training Acc")
    plt.plot(val_accs, "r-", linewidth=2, label="Validation Acc")
    plt.title("Accuracy Curves", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 過擬合監控（準確率差異）
    plt.subplot(1, 4, 3)
    acc_diff = np.array(train_accs) - np.array(val_accs)
    plt.plot(acc_diff, "purple", linewidth=2, label="Accuracy Diff")
    plt.title("Overfitting Monitor", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Training - Validation (%)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()

    # 訓練時間
    plt.subplot(1, 4, 4)
    plt.plot(epoch_times, "orange", linewidth=2, marker="o", markersize=3)
    plt.title("Training Time per Epoch", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ultra_small_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_ultra_architectures():
    """比較超縮小版模型架構"""
    print("\n📋 超縮小版 gMLP 模型架構比較:")
    print("=" * 85)
    print(
        f"{'模型':<8} {'深度':<6} {'維度':<8} {'FFN倍數':<8} {'參數(M)':<10} {'過擬合風險':<12}"
    )
    print("-" * 85)

    configs = {
        "Test": {"depth": 4, "dim": 64, "ff_mult": 2, "params": 0.1, "risk": "極低"},
        "Nano": {"depth": 6, "dim": 64, "ff_mult": 2, "params": 0.3, "risk": "很低"},
        "XS": {"depth": 8, "dim": 80, "ff_mult": 3, "params": 0.8, "risk": "低"},
        "S": {"depth": 12, "dim": 128, "ff_mult": 3, "params": 2.0, "risk": "中等"},
        "M": {"depth": 16, "dim": 160, "ff_mult": 4, "params": 4.5, "risk": "較高"},
        "L": {"depth": 30, "dim": 128, "ff_mult": 6, "params": 5.9, "risk": "很高"},
    }

    for model, config in configs.items():
        print(
            f"{model:<8} {config['depth']:<6} {config['dim']:<8} "
            f"{config['ff_mult']:<8} {config['params']:<10} {config['risk']:<12}"
        )

    print("-" * 85)
    print("🎯 選擇建議（針對不同數據量）:")
    print("   📊 數據量導向選擇:")
    print("     • < 25K 樣本: Test/Nano (極低過擬合風險)")
    print("     • 25K-40K 樣本: XS (推薦，平衡性能) ⭐")
    print("     • 40K-50K 樣本: S (較好性能)")
    print("     • > 50K 樣本: M (高性能，需監控過擬合)")
    print("\n   ⏱️ 訓練時間導向選擇:")
    print("     • Test: 超極速測試 (<30秒) 🚀")
    print("     • Nano: 極速原型開發 (<1分鐘)")
    print("     • XS: 快速實驗 (~2分鐘) ⭐推薦")
    print("     • S: 平衡訓練 (~5分鐘)")
    print("     • M/L: 長時間訓練 (>10分鐘)")
    print("\n   🛡️ 過擬合風險控制:")
    print("     • 嚴格控制: Test/Nano")
    print("     • 平衡控制: XS/S ⭐推薦")
    print("     • 需要監控: M/L (配合早停)")


def get_model_recommendation(train_samples, target_time_min=5, risk_tolerance="medium"):
    """智能模型推薦系統"""
    print(f"\n🤖 智能模型推薦系統")
    print(f"   📊 訓練樣本: {train_samples:,}")
    print(f"   ⏱️ 目標時間: {target_time_min} 分鐘")
    print(f"   🛡️ 風險容忍: {risk_tolerance}")
    print("-" * 50)

    # 基於數據量的基礎推薦
    if train_samples < 25000:
        base_recommendation = "Nano"
        risk_level = "very_low"
    elif train_samples < 40000:
        base_recommendation = "XS"
        risk_level = "low"
    elif train_samples < 50000:
        base_recommendation = "S"
        risk_level = "medium"
    else:
        base_recommendation = "M"
        risk_level = "high"

    # 時間約束調整
    time_map = {"Test": 0.5, "Nano": 1, "XS": 2, "S": 5, "M": 10, "L": 15}
    if target_time_min < 2:
        time_recommendation = "Nano"
    elif target_time_min < 4:
        time_recommendation = "XS"
    elif target_time_min < 8:
        time_recommendation = "S"
    else:
        time_recommendation = "M"

    # 風險容忍調整
    risk_map = {"low": ["Test", "Nano"], "medium": ["XS", "S"], "high": ["M", "L"]}

    risk_recommendations = risk_map.get(risk_tolerance, ["XS", "S"])

    # 綜合決策
    recommendations = [base_recommendation, time_recommendation] + risk_recommendations
    final_recommendation = max(set(recommendations), key=recommendations.count)

    print(f"   🎯 基於數據量: {base_recommendation}")
    print(f"   ⏱️ 基於時間約束: {time_recommendation}")
    print(f"   🛡️ 基於風險容忍: {', '.join(risk_recommendations)}")
    print(f"   ✅ 最終推薦: {final_recommendation}")

    return final_recommendation


def main():
    """主函數 - 超快速版本"""
    print("🚀 超縮小版 gMLP 圖像分類訓練開始")
    print("=" * 60)

    # 架構比較
    compare_ultra_architectures()

    # 數據加載
    trainloader, testloader, classes = load_cifar10_data_ultrafast(quick_test=True)

    # 智能模型選擇：基於數據集規模自動選擇最佳架構
    train_samples = len(trainloader.dataset)

    # 使用智能推薦系統
    model_size = get_model_recommendation(
        train_samples=train_samples,
        target_time_min=5,  # 目標訓練時間（分鐘）
        risk_tolerance="medium",  # 過擬合風險容忍度: "low", "medium", "high"
    )

    # 可手動覆蓋自動選擇
    # model_size = "XS"  # 取消註解以手動選擇: "Test", "Nano", "XS", "S", "M", "L"
    model, device = create_ultra_small_gmlp_model(model_size=model_size)

    # 快速訓練
    train_losses, train_accs, val_accs, val_losses, epoch_times, total_time = (
        train_ultra_fast(
            model, trainloader, testloader, device, epochs=100  # 可調整epochs
        )
    )

    # 結果可視化（包含訓練和驗證損失比較）
    plot_ultra_training_history(
        train_losses, train_accs, val_accs, val_losses, epoch_times
    )

    # 模型評估
    final_acc = evaluate_ultra_model(model, testloader, device, classes)

    print(f"\n🎉 超快速訓練完成!")
    print(f"   • 模型大小: {model_size}")
    print(f"   • 最終準確率: {final_acc:.2f}%")
    print(f"   • 總訓練時間: {total_time/60:.1f} 分鐘")
    print(f"   • 平均每epoch: {np.mean(epoch_times):.1f} 秒")


if __name__ == "__main__":
    main()
