"""
超縮小版 gMLP 圖像分類模型 - 自由選擇版本
基於論文架構但大幅縮小規模以提高訓練效率
針對快速原型開發和資源受限環境優化
支援互動式模型選擇
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
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(p=0.3),
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
        trainset = Subset(trainset, range(50000))
        testset = Subset(testset, range(10000))
        print("   🚀 超快速模式：小規模數據集訓練")

    batch_size = 128
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


def display_model_options():
    """顯示所有可用的模型選項"""
    print("\n" + "=" * 80)
    print("🏗️  可用的 gMLP 模型架構")
    print("=" * 80)

    models_info = {
        "Test": {
            "depth": 4,
            "dim": 64,
            "ff_mult": 2,
            "params": "0.15M",
            "time": "<30秒",
            "risk": "極低",
            "desc": "超極速測試模型",
        },
        "Nano": {
            "depth": 6,
            "dim": 64,
            "ff_mult": 2,
            "params": "0.20M",
            "time": "~1分鐘",
            "risk": "很低",
            "desc": "極小快速模型",
        },
        "XS": {
            "depth": 8,
            "dim": 80,
            "ff_mult": 3,
            "params": "0.27M",
            "time": "~2分鐘",
            "risk": "低",
            "desc": "超小平衡模型",
        },
        "S": {
            "depth": 12,
            "dim": 128,
            "ff_mult": 3,
            "params": "0.65M",
            "time": "~5分鐘",
            "risk": "中等",
            "desc": "小型性能模型",
        },
        "M": {
            "depth": 16,
            "dim": 160,
            "ff_mult": 4,
            "params": "1.35M",
            "time": "~10分鐘",
            "risk": "較高",
            "desc": "中型高性能模型",
        },
        "L": {
            "depth": 30,
            "dim": 128,
            "ff_mult": 6,
            "params": "1.85M",
            "time": "~15分鐘",
            "risk": "很高",
            "desc": "大型頂級模型",
        },
    }

    print(
        f"{'編號':<4} {'名稱':<6} {'深度':<6} {'維度':<6} {'FFN':<5} {'參數':<8} {'時間':<10} {'過擬合風險':<10} {'描述':<20}"
    )
    print("-" * 80)

    for i, (name, info) in enumerate(models_info.items(), 1):
        print(
            f"{i:<4} {name:<6} {info['depth']:<6} {info['dim']:<6} {info['ff_mult']:<5} "
            f"{info['params']:<8} {info['time']:<10} {info['risk']:<10} {info['desc']:<20}"
        )

    print("-" * 80)
    print("💡 推薦選擇:")
    print("   🚀 快速測試: Test (1) 或 Nano (2)")
    print("   ⚖️  平衡性能: XS (3) 或 S (4) - 最推薦")
    print("   🎯 高性能: M (5) 或 L (6) - 需要更多時間")
    print("=" * 80)

    return models_info


def get_user_model_choice():
    """獲取用戶的模型選擇"""
    models_info = display_model_options()
    model_names = list(models_info.keys())

    while True:
        try:
            print("\n🤖 請選擇要使用的模型:")
            choice = input(
                "   輸入編號 (1-6) 或模型名稱 (Test/Nano/XS/S/M/L): "
            ).strip()

            # 處理數字輸入
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= 6:
                    selected_model = model_names[choice_num - 1]
                    break
                else:
                    print("   ❌ 請輸入 1-6 之間的數字")
                    continue

            # 處理名稱輸入
            elif choice.upper() in model_names:
                selected_model = choice.upper()
                break
            elif choice.lower() in [name.lower() for name in model_names]:
                # 大小寫不敏感匹配
                selected_model = next(
                    name for name in model_names if name.lower() == choice.lower()
                )
                break
            else:
                print("   ❌ 無效輸入，請輸入正確的編號或模型名稱")
                continue

        except (ValueError, KeyboardInterrupt):
            print("   ❌ 輸入錯誤，請重新輸入")
            continue

    # 顯示選擇確認
    selected_info = models_info[selected_model]
    print(f"\n✅ 您選擇了: {selected_model} 模型")
    print(f"   📋 模型詳情:")
    print(f"      • 深度: {selected_info['depth']} 層")
    print(f"      • 維度: {selected_info['dim']}")
    print(f"      • FFN倍數: {selected_info['ff_mult']}")
    print(f"      • 預估參數: {selected_info['params']}")
    print(f"      • 預估時間: {selected_info['time']}")
    print(f"      • 過擬合風險: {selected_info['risk']}")
    print(f"      • 描述: {selected_info['desc']}")

    # 確認選擇
    confirm = (
        input(f"\n   確認使用 {selected_model} 模型嗎? (y/n, 預設=y): ").strip().lower()
    )
    if confirm in ["n", "no"]:
        print("   🔄 重新選擇...")
        return get_user_model_choice()  # 遞迴重新選擇

    return selected_model


def get_training_parameters():
    """獲取訓練參數設置"""
    print("\n" + "=" * 60)
    print("⚙️  訓練參數設置")
    print("=" * 60)

    # 預設參數
    default_epochs = 50
    default_quick_test = True

    try:
        # 選擇數據集模式
        print("\n📦 數據集模式選擇:")
        print("   1. 快速模式 (50K訓練 + 10K測試) - 推薦")
        print("   2. 完整模式 (50K訓練 + 10K測試)")

        data_choice = input("   選擇模式 (1/2, 預設=1): ").strip()
        quick_test = True if data_choice != "2" else False

        # 設置訓練輪數
        epochs_input = input(f"\n🏋️  訓練輪數 (預設={default_epochs}): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

        # 選擇是否啟用早停
        print("\n🛡️  過擬合保護:")
        print("   1. 啟用早停機制 (推薦)")
        print("   2. 關閉早停機制")

        early_stop_choice = input("   選擇 (1/2, 預設=1): ").strip()
        enable_early_stop = True if early_stop_choice != "2" else False

        print(f"\n✅ 訓練參數確認:")
        print(f"   📦 數據模式: {'快速模式' if quick_test else '完整模式'}")
        print(f"   🏋️  訓練輪數: {epochs}")
        print(f"   🛡️  早停機制: {'啟用' if enable_early_stop else '關閉'}")

        return {
            "epochs": epochs,
            "quick_test": quick_test,
            "enable_early_stop": enable_early_stop,
        }

    except (ValueError, KeyboardInterrupt):
        print("   ⚠️  使用預設參數")
        return {
            "epochs": default_epochs,
            "quick_test": default_quick_test,
            "enable_early_stop": True,
        }


def create_ultra_small_gmlp_model(model_size="L"):
    """創建超縮小版 gMLP 模型架構"""
    print(f"\n🏗️ 創建超縮小版 gMLP-{model_size} 模型...")

    # CPU專用優化設置
    torch.set_num_threads(4)
    print("   ⚡ CPU模式：已設置4個線程")

    # 超縮小版架構配置
    if model_size == "Test":
        config = {
            "depth": 4,
            "dim": 64,
            "ff_mult": 2,
            "prob_survival": 1.00,
            "params_target": 0.15,
        }
    elif model_size == "Nano":
        config = {
            "depth": 6,
            "dim": 64,
            "ff_mult": 2,
            "prob_survival": 1.00,
            "params_target": 0.20,
        }
    elif model_size == "XS":
        config = {
            "depth": 8,
            "dim": 80,
            "ff_mult": 3,
            "prob_survival": 1.00,
            "params_target": 0.27,
        }
    elif model_size == "S":
        config = {
            "depth": 12,
            "dim": 128,
            "ff_mult": 3,
            "prob_survival": 0.98,
            "params_target": 0.65,
        }
    elif model_size == "M":
        config = {
            "depth": 16,
            "dim": 160,
            "ff_mult": 4,
            "prob_survival": 0.95,
            "params_target": 1.35,
        }
    elif model_size == "L":
        config = {
            "depth": 30,
            "dim": 128,
            "ff_mult": 6,
            "prob_survival": 1.00,
            "params_target": 1.85,
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

    print(f"\n✅ 超縮小版 gMLP-{model_size} 模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 實際參數數量: {total_params:,} ({params_M:.2f}M)")
    print(f"   ✓ 目標參數預期: {config['params_target']}M")
    print(
        f"   ✓ 架構配置: depth={config['depth']}, dim={config['dim']}, ff_mult={config['ff_mult']}"
    )

    return model, device


def train_ultra_fast(
    model, trainloader, testloader, device, epochs=50, enable_early_stop=True
):
    """超快速訓練配置 - 支援自定義早停設置"""
    print(f"\n🏋️ 開始超快速訓練 ({epochs} 個 epochs)...")
    if enable_early_stop:
        print("   🛡️  啟用過擬合早停保護")
    else:
        print("   ⚠️  早停保護已關閉")

    # 快速訓練配置
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []
    epoch_times = []

    best_val_acc = 0
    patience = 15 if enable_early_stop else epochs + 1
    patience_counter = 0

    # 過擬合早停配置
    overfitting_patience = 6 if enable_early_stop else epochs + 1
    overfitting_counter = 0
    overfitting_threshold = 8.0
    min_epochs_before_overfitting_check = 8

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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

        # 早停檢測 (只在啟用時執行)
        if enable_early_stop:
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
                    overfitting_counter = 0

            if train_val_diff > 5:
                print(f"   📊 訓練-驗證差異: {train_val_diff:.2f}%")

            # 性能早停機制
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
        else:
            # 不啟用早停時仍然保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_ultra_small_model.pth")
                print(f"   💾 新最佳模型已保存: 驗證準確率 {best_val_acc:.2f}%")

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 超快速訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.1f}s ({total_training_time/60:.1f}min)"
    )
    print(f"   • 實際訓練epochs: {len(train_losses)} / {epochs}")
    print(f"   • 平均每epoch: {np.mean(epoch_times):.1f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}%")

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
    """繪製超快速訓練歷史"""
    print("\n📈 繪製超快速訓練歷史...")

    plt.figure(figsize=(16, 4))

    # 損失曲線比較
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

    # 過擬合監控
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


def main():
    """主函數 - 自由選擇版本"""
    print("🚀 超縮小版 gMLP 圖像分類訓練 - 自由選擇版本")
    print("=" * 70)

    try:
        # 用戶選擇模型
        model_size = get_user_model_choice()

        # 用戶設置訓練參數
        training_params = get_training_parameters()

        # 數據加載
        trainloader, testloader, classes = load_cifar10_data_ultrafast(
            quick_test=training_params["quick_test"]
        )

        # 創建選定的模型
        model, device = create_ultra_small_gmlp_model(model_size=model_size)

        # 開始訓練
        print(f"\n🎬 開始訓練 {model_size} 模型...")
        train_losses, train_accs, val_accs, val_losses, epoch_times, total_time = (
            train_ultra_fast(
                model,
                trainloader,
                testloader,
                device,
                epochs=training_params["epochs"],
                enable_early_stop=training_params["enable_early_stop"],
            )
        )

        # 結果可視化
        plot_ultra_training_history(
            train_losses, train_accs, val_accs, val_losses, epoch_times
        )

        # 模型評估
        final_acc = evaluate_ultra_model(model, testloader, device, classes)

        # 訓練總結
        print(f"\n🎉 訓練完成總結:")
        print(f"   • 選擇模型: {model_size}")
        print(f"   • 最終準確率: {final_acc:.2f}%")
        print(f"   • 總訓練時間: {total_time/60:.1f} 分鐘")
        print(f"   • 平均每epoch: {np.mean(epoch_times):.1f} 秒")
        print(f"   • 實際訓練輪數: {len(train_losses)}/{training_params['epochs']}")
        print(
            f"   • 早停狀態: {'啟用' if training_params['enable_early_stop'] else '關閉'}"
        )

        # 詢問是否繼續訓練其他模型
        print(f"\n" + "=" * 70)
        continue_choice = input("🔄 是否要訓練其他模型? (y/n): ").strip().lower()
        if continue_choice in ["y", "yes"]:
            print("\n" + "🔄 重新開始..." + "\n")
            main()  # 遞迴調用重新開始

    except KeyboardInterrupt:
        print("\n\n❌ 用戶中斷程序")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("   請檢查輸入並重試")


if __name__ == "__main__":
    main()
