<<<<<<< HEAD
"""
訓練策略對比實驗
比較「少量epoch+大數據量」vs「多量epoch+小數據量」的效果
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
from sklearn.metrics import classification_report


def load_data_for_strategy(strategy="large_data"):
    """根據策略加載不同數據配置"""
    print(f"📦 加載數據 - 策略: {strategy}")

    # 簡化的數據增強
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
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

    if strategy == "large_data":
        # 策略A：大數據量 + 少epoch
        trainset = Subset(trainset, range(4000))  # 更多訓練數據
        testset = Subset(testset, range(800))
        batch_size = 64  # 較大批次
        print("   ✓ 策略A：大數據量策略")
        print(f"   ✓ 訓練樣本: {len(trainset)}, 批次大小: {batch_size}")
    else:
        # 策略B：小數據量 + 多epoch
        trainset = Subset(trainset, range(1200))  # 較少訓練數據
        testset = Subset(testset, range(300))
        batch_size = 32  # 較小批次
        print("   ✓ 策略B：小數據量策略")
        print(f"   ✓ 訓練樣本: {len(trainset)}, 批次大小: {batch_size}")

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
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

    return trainloader, testloader, classes


def create_model():
    """創建統一的模型配置"""
    torch.set_num_threads(4)

    model = gMLPVision(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=128,  # 較小模型加速對比
        depth=3,
        ff_mult=3,
        channels=3,
        prob_survival=0.8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device


def train_with_strategy(model, trainloader, testloader, device, strategy="large_data"):
    """根據策略進行訓練"""

    if strategy == "large_data":
        epochs = 6  # 少量epoch
        lr = 0.003  # 稍高學習率
        print(f"\n🏋️ 策略A訓練：{epochs} epochs，大數據量")
    else:
        epochs = 18  # 多量epoch
        lr = 0.002  # 稍低學習率
        print(f"\n🏋️ 策略B訓練：{epochs} epochs，小數據量")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []

    start_time = time.time()
    best_val_acc = 0
    patience = 4
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # 訓練
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        scheduler.step()

        print(
            f"Epoch {epoch+1:2d}/{epochs}: "
            f"訓練準確率={train_acc:5.2f}%, "
            f"驗證準確率={val_acc:5.2f}%, "
            f"時間={epoch_time:4.1f}s"
        )

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停於第 {epoch+1} epoch")
                break

    total_time = time.time() - start_time

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "total_time": total_time,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "epochs_used": len(train_accs),
    }


def compare_strategies():
    """執行策略對比實驗"""
    print("🔬 開始訓練策略對比實驗")
    print("=" * 60)

    results = {}

    # 策略A：大數據量 + 少epoch
    print("\n" + "=" * 30 + " 策略A " + "=" * 30)
    trainloader_a, testloader_a, classes = load_data_for_strategy("large_data")
    model_a, device = create_model()
    results["strategy_a"] = train_with_strategy(
        model_a, trainloader_a, testloader_a, device, "large_data"
    )

    # 策略B：小數據量 + 多epoch
    print("\n" + "=" * 30 + " 策略B " + "=" * 30)
    trainloader_b, testloader_b, classes = load_data_for_strategy("small_data")
    model_b, device = create_model()
    results["strategy_b"] = train_with_strategy(
        model_b, trainloader_b, testloader_b, device, "small_data"
    )

    return results


def plot_comparison(results):
    """繪製對比結果"""
    print("\n📈 繪製對比結果...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "訓練策略對比：大數據量+少epoch vs 小數據量+多epoch",
        fontsize=16,
        fontweight="bold",
    )

    # 獲取數據
    strategy_a = results["strategy_a"]
    strategy_b = results["strategy_b"]

    # 1. 訓練損失對比
    ax = axes[0, 0]
    epochs_a = range(1, len(strategy_a["train_losses"]) + 1)
    epochs_b = range(1, len(strategy_b["train_losses"]) + 1)
    ax.plot(
        epochs_a,
        strategy_a["train_losses"],
        "b-",
        linewidth=2,
        label="策略A: 大數據+少epoch",
    )
    ax.plot(
        epochs_b,
        strategy_b["train_losses"],
        "r-",
        linewidth=2,
        label="策略B: 小數據+多epoch",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("訓練損失對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 訓練準確率對比
    ax = axes[0, 1]
    ax.plot(
        epochs_a, strategy_a["train_accs"], "b-", linewidth=2, label="策略A: 訓練準確率"
    )
    ax.plot(
        epochs_a, strategy_a["val_accs"], "b--", linewidth=2, label="策略A: 驗證準確率"
    )
    ax.plot(
        epochs_b, strategy_b["train_accs"], "r-", linewidth=2, label="策略B: 訓練準確率"
    )
    ax.plot(
        epochs_b, strategy_b["val_accs"], "r--", linewidth=2, label="策略B: 驗證準確率"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("準確率對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 每epoch時間對比
    ax = axes[0, 2]
    ax.plot(
        epochs_a,
        strategy_a["epoch_times"],
        "b-",
        linewidth=2,
        marker="o",
        label="策略A",
    )
    ax.plot(
        epochs_b,
        strategy_b["epoch_times"],
        "r-",
        linewidth=2,
        marker="s",
        label="策略B",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("每Epoch訓練時間")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 過擬合程度對比
    ax = axes[1, 0]
    diff_a = np.array(strategy_a["train_accs"]) - np.array(strategy_a["val_accs"])
    diff_b = np.array(strategy_b["train_accs"]) - np.array(strategy_b["val_accs"])
    ax.plot(epochs_a, diff_a, "b-", linewidth=2, label="策略A: 過擬合程度")
    ax.plot(epochs_b, diff_b, "r-", linewidth=2, label="策略B: 過擬合程度")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train-Val Accuracy Difference (%)")
    ax.set_title("過擬合程度對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 效率對比條形圖
    ax = axes[1, 1]
    strategies = ["策略A\n(大數據+少epoch)", "策略B\n(小數據+多epoch)"]
    times = [strategy_a["total_time"], strategy_b["total_time"]]
    accuracies = [strategy_a["best_val_acc"], strategy_b["best_val_acc"]]

    x = np.arange(len(strategies))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width / 2, times, width, label="總訓練時間(s)", color="skyblue")
    bars2 = ax2.bar(
        x + width / 2, accuracies, width, label="最佳驗證準確率(%)", color="lightcoral"
    )

    ax.set_xlabel("策略")
    ax.set_ylabel("時間 (seconds)", color="blue")
    ax2.set_ylabel("準確率 (%)", color="red")
    ax.set_title("效率 vs 準確率")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)

    # 添加數值標籤
    for bar, time in zip(bars1, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
        )
    for bar, acc in zip(bars2, accuracies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    # 6. 詳細數據對比表
    ax = axes[1, 2]
    ax.axis("off")

    # 創建對比表格
    comparison_data = [
        ["指標", "策略A (大數據)", "策略B (小數據)"],
        ["訓練樣本數", "4000", "1200"],
        ["訓練epochs", f"{strategy_a['epochs_used']}", f"{strategy_b['epochs_used']}"],
        [
            "總訓練時間",
            f"{strategy_a['total_time']:.1f}s",
            f"{strategy_b['total_time']:.1f}s",
        ],
        [
            "最佳驗證準確率",
            f"{strategy_a['best_val_acc']:.2f}%",
            f"{strategy_b['best_val_acc']:.2f}%",
        ],
        [
            "最終訓練準確率",
            f"{strategy_a['final_train_acc']:.2f}%",
            f"{strategy_b['final_train_acc']:.2f}%",
        ],
        [
            "最終驗證準確率",
            f"{strategy_a['final_val_acc']:.2f}%",
            f"{strategy_b['final_val_acc']:.2f}%",
        ],
        [
            "過擬合程度",
            f"{strategy_a['final_train_acc']-strategy_a['final_val_acc']:.2f}%",
            f"{strategy_b['final_train_acc']-strategy_b['final_val_acc']:.2f}%",
        ],
        [
            "平均每epoch時間",
            f"{np.mean(strategy_a['epoch_times']):.1f}s",
            f"{np.mean(strategy_b['epoch_times']):.1f}s",
        ],
    ]

    table = ax.table(
        cellText=comparison_data[1:],
        colLabels=comparison_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 設置表格樣式
    for i in range(len(comparison_data)):
        for j in range(3):
            if i == 0:  # 標題行
                table[(i, j)].set_facecolor("#4CAF50")
                table[(i, j)].set_text_props(weight="bold", color="white")
            elif j == 0:  # 指標列
                table[(i, j)].set_facecolor("#E8F5E8")
            else:
                table[(i, j)].set_facecolor("#F9F9F9")

    ax.set_title("詳細對比數據", fontweight="bold")

    plt.tight_layout()
    plt.savefig("training_strategy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_analysis(results):
    """打印分析結果"""
    print("\n" + "=" * 60)
    print("📊 訓練策略分析報告")
    print("=" * 60)

    strategy_a = results["strategy_a"]
    strategy_b = results["strategy_b"]

    print(f"\n🔍 策略A（大數據+少epoch）:")
    print(f"   • 訓練樣本: 4000")
    print(f"   • 使用epochs: {strategy_a['epochs_used']}")
    print(f"   • 總訓練時間: {strategy_a['total_time']:.1f}s")
    print(f"   • 最佳驗證準確率: {strategy_a['best_val_acc']:.2f}%")
    print(
        f"   • 過擬合程度: {strategy_a['final_train_acc']-strategy_a['final_val_acc']:.2f}%"
    )

    print(f"\n🔍 策略B（小數據+多epoch）:")
    print(f"   • 訓練樣本: 1200")
    print(f"   • 使用epochs: {strategy_b['epochs_used']}")
    print(f"   • 總訓練時間: {strategy_b['total_time']:.1f}s")
    print(f"   • 最佳驗證準確率: {strategy_b['best_val_acc']:.2f}%")
    print(
        f"   • 過擬合程度: {strategy_b['final_train_acc']-strategy_b['final_val_acc']:.2f}%"
    )

    print(f"\n💡 策略優劣分析:")

    # 準確率比較
    if strategy_a["best_val_acc"] > strategy_b["best_val_acc"]:
        acc_winner = "策略A（大數據）"
        acc_diff = strategy_a["best_val_acc"] - strategy_b["best_val_acc"]
    else:
        acc_winner = "策略B（小數據）"
        acc_diff = strategy_b["best_val_acc"] - strategy_a["best_val_acc"]

    # 時間比較
    if strategy_a["total_time"] < strategy_b["total_time"]:
        time_winner = "策略A（大數據）"
        time_diff = strategy_b["total_time"] - strategy_a["total_time"]
    else:
        time_winner = "策略B（小數據）"
        time_diff = strategy_a["total_time"] - strategy_b["total_time"]

    # 過擬合比較
    overfitting_a = abs(strategy_a["final_train_acc"] - strategy_a["final_val_acc"])
    overfitting_b = abs(strategy_b["final_train_acc"] - strategy_b["final_val_acc"])

    if overfitting_a < overfitting_b:
        overfitting_winner = "策略A（大數據）"
        overfitting_diff = overfitting_b - overfitting_a
    else:
        overfitting_winner = "策略B（小數據）"
        overfitting_diff = overfitting_a - overfitting_b

    print(f"\n🏆 綜合比較:")
    print(f"   • 準確率優勝: {acc_winner} (領先 {acc_diff:.2f}%)")
    print(f"   • 訓練速度優勝: {time_winner} (快 {time_diff:.1f}s)")
    print(f"   • 泛化能力優勝: {overfitting_winner} (過擬合少 {overfitting_diff:.2f}%)")

    print(f"\n🎯 實用建議:")
    print(f"   • 如果追求最高準確率: 選擇{acc_winner}")
    print(f"   • 如果注重訓練效率: 選擇{time_winner}")
    print(f"   • 如果要求泛化能力: 選擇{overfitting_winner}")

    # 根據結果給出具體建議
    if (
        strategy_a["best_val_acc"] > strategy_b["best_val_acc"]
        and strategy_a["total_time"] < strategy_b["total_time"]
    ):
        print(f"\n✅ 結論: 策略A（大數據+少epoch）在準確率和效率上都更優秀！")
    elif (
        strategy_b["best_val_acc"] > strategy_a["best_val_acc"]
        and strategy_b["total_time"] < strategy_a["total_time"]
    ):
        print(f"\n✅ 結論: 策略B（小數據+多epoch）在準確率和效率上都更優秀！")
    else:
        print(f"\n⚖️ 結論: 兩種策略各有優劣，需要根據具體需求選擇")
        if abs(strategy_a["best_val_acc"] - strategy_b["best_val_acc"]) < 2:
            print(f"   準確率相近，建議選擇訓練時間更短的策略")
        else:
            print(f"   如果準確率差距較大，建議優先考慮準確率")


def main():
    """主函數"""
    print("🔬 gMLP訓練策略對比實驗")
    print("比較「大數據量+少epoch」vs「小數據量+多epoch」")
    print("=" * 60)

    try:
        # 執行對比實驗
        results = compare_strategies()

        # 繪製對比圖表
        plot_comparison(results)

        # 打印分析報告
        print_analysis(results)

        print(f"\n✅ 對比實驗完成！")
        print(f"   • 對比圖表已保存: training_strategy_comparison.png")
        print(f"   • 詳細分析報告已顯示")

    except Exception as e:
        print(f"❌ 實驗失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
=======
"""
訓練策略對比實驗
比較「少量epoch+大數據量」vs「多量epoch+小數據量」的效果
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
from sklearn.metrics import classification_report


def load_data_for_strategy(strategy="large_data"):
    """根據策略加載不同數據配置"""
    print(f"📦 加載數據 - 策略: {strategy}")

    # 簡化的數據增強
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
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

    if strategy == "large_data":
        # 策略A：大數據量 + 少epoch
        trainset = Subset(trainset, range(4000))  # 更多訓練數據
        testset = Subset(testset, range(800))
        batch_size = 64  # 較大批次
        print("   ✓ 策略A：大數據量策略")
        print(f"   ✓ 訓練樣本: {len(trainset)}, 批次大小: {batch_size}")
    else:
        # 策略B：小數據量 + 多epoch
        trainset = Subset(trainset, range(1200))  # 較少訓練數據
        testset = Subset(testset, range(300))
        batch_size = 32  # 較小批次
        print("   ✓ 策略B：小數據量策略")
        print(f"   ✓ 訓練樣本: {len(trainset)}, 批次大小: {batch_size}")

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
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

    return trainloader, testloader, classes


def create_model():
    """創建統一的模型配置"""
    torch.set_num_threads(4)

    model = gMLPVision(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=128,  # 較小模型加速對比
        depth=3,
        ff_mult=3,
        channels=3,
        prob_survival=0.8,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device


def train_with_strategy(model, trainloader, testloader, device, strategy="large_data"):
    """根據策略進行訓練"""

    if strategy == "large_data":
        epochs = 6  # 少量epoch
        lr = 0.003  # 稍高學習率
        print(f"\n🏋️ 策略A訓練：{epochs} epochs，大數據量")
    else:
        epochs = 18  # 多量epoch
        lr = 0.002  # 稍低學習率
        print(f"\n🏋️ 策略B訓練：{epochs} epochs，小數據量")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

    train_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []

    start_time = time.time()
    best_val_acc = 0
    patience = 4
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # 訓練
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        scheduler.step()

        print(
            f"Epoch {epoch+1:2d}/{epochs}: "
            f"訓練準確率={train_acc:5.2f}%, "
            f"驗證準確率={val_acc:5.2f}%, "
            f"時間={epoch_time:4.1f}s"
        )

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   早停於第 {epoch+1} epoch")
                break

    total_time = time.time() - start_time

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "total_time": total_time,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "epochs_used": len(train_accs),
    }


def compare_strategies():
    """執行策略對比實驗"""
    print("🔬 開始訓練策略對比實驗")
    print("=" * 60)

    results = {}

    # 策略A：大數據量 + 少epoch
    print("\n" + "=" * 30 + " 策略A " + "=" * 30)
    trainloader_a, testloader_a, classes = load_data_for_strategy("large_data")
    model_a, device = create_model()
    results["strategy_a"] = train_with_strategy(
        model_a, trainloader_a, testloader_a, device, "large_data"
    )

    # 策略B：小數據量 + 多epoch
    print("\n" + "=" * 30 + " 策略B " + "=" * 30)
    trainloader_b, testloader_b, classes = load_data_for_strategy("small_data")
    model_b, device = create_model()
    results["strategy_b"] = train_with_strategy(
        model_b, trainloader_b, testloader_b, device, "small_data"
    )

    return results


def plot_comparison(results):
    """繪製對比結果"""
    print("\n📈 繪製對比結果...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "訓練策略對比：大數據量+少epoch vs 小數據量+多epoch",
        fontsize=16,
        fontweight="bold",
    )

    # 獲取數據
    strategy_a = results["strategy_a"]
    strategy_b = results["strategy_b"]

    # 1. 訓練損失對比
    ax = axes[0, 0]
    epochs_a = range(1, len(strategy_a["train_losses"]) + 1)
    epochs_b = range(1, len(strategy_b["train_losses"]) + 1)
    ax.plot(
        epochs_a,
        strategy_a["train_losses"],
        "b-",
        linewidth=2,
        label="策略A: 大數據+少epoch",
    )
    ax.plot(
        epochs_b,
        strategy_b["train_losses"],
        "r-",
        linewidth=2,
        label="策略B: 小數據+多epoch",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("訓練損失對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 訓練準確率對比
    ax = axes[0, 1]
    ax.plot(
        epochs_a, strategy_a["train_accs"], "b-", linewidth=2, label="策略A: 訓練準確率"
    )
    ax.plot(
        epochs_a, strategy_a["val_accs"], "b--", linewidth=2, label="策略A: 驗證準確率"
    )
    ax.plot(
        epochs_b, strategy_b["train_accs"], "r-", linewidth=2, label="策略B: 訓練準確率"
    )
    ax.plot(
        epochs_b, strategy_b["val_accs"], "r--", linewidth=2, label="策略B: 驗證準確率"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("準確率對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 每epoch時間對比
    ax = axes[0, 2]
    ax.plot(
        epochs_a,
        strategy_a["epoch_times"],
        "b-",
        linewidth=2,
        marker="o",
        label="策略A",
    )
    ax.plot(
        epochs_b,
        strategy_b["epoch_times"],
        "r-",
        linewidth=2,
        marker="s",
        label="策略B",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("每Epoch訓練時間")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 過擬合程度對比
    ax = axes[1, 0]
    diff_a = np.array(strategy_a["train_accs"]) - np.array(strategy_a["val_accs"])
    diff_b = np.array(strategy_b["train_accs"]) - np.array(strategy_b["val_accs"])
    ax.plot(epochs_a, diff_a, "b-", linewidth=2, label="策略A: 過擬合程度")
    ax.plot(epochs_b, diff_b, "r-", linewidth=2, label="策略B: 過擬合程度")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train-Val Accuracy Difference (%)")
    ax.set_title("過擬合程度對比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 效率對比條形圖
    ax = axes[1, 1]
    strategies = ["策略A\n(大數據+少epoch)", "策略B\n(小數據+多epoch)"]
    times = [strategy_a["total_time"], strategy_b["total_time"]]
    accuracies = [strategy_a["best_val_acc"], strategy_b["best_val_acc"]]

    x = np.arange(len(strategies))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width / 2, times, width, label="總訓練時間(s)", color="skyblue")
    bars2 = ax2.bar(
        x + width / 2, accuracies, width, label="最佳驗證準確率(%)", color="lightcoral"
    )

    ax.set_xlabel("策略")
    ax.set_ylabel("時間 (seconds)", color="blue")
    ax2.set_ylabel("準確率 (%)", color="red")
    ax.set_title("效率 vs 準確率")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)

    # 添加數值標籤
    for bar, time in zip(bars1, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
        )
    for bar, acc in zip(bars2, accuracies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    # 6. 詳細數據對比表
    ax = axes[1, 2]
    ax.axis("off")

    # 創建對比表格
    comparison_data = [
        ["指標", "策略A (大數據)", "策略B (小數據)"],
        ["訓練樣本數", "4000", "1200"],
        ["訓練epochs", f"{strategy_a['epochs_used']}", f"{strategy_b['epochs_used']}"],
        [
            "總訓練時間",
            f"{strategy_a['total_time']:.1f}s",
            f"{strategy_b['total_time']:.1f}s",
        ],
        [
            "最佳驗證準確率",
            f"{strategy_a['best_val_acc']:.2f}%",
            f"{strategy_b['best_val_acc']:.2f}%",
        ],
        [
            "最終訓練準確率",
            f"{strategy_a['final_train_acc']:.2f}%",
            f"{strategy_b['final_train_acc']:.2f}%",
        ],
        [
            "最終驗證準確率",
            f"{strategy_a['final_val_acc']:.2f}%",
            f"{strategy_b['final_val_acc']:.2f}%",
        ],
        [
            "過擬合程度",
            f"{strategy_a['final_train_acc']-strategy_a['final_val_acc']:.2f}%",
            f"{strategy_b['final_train_acc']-strategy_b['final_val_acc']:.2f}%",
        ],
        [
            "平均每epoch時間",
            f"{np.mean(strategy_a['epoch_times']):.1f}s",
            f"{np.mean(strategy_b['epoch_times']):.1f}s",
        ],
    ]

    table = ax.table(
        cellText=comparison_data[1:],
        colLabels=comparison_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 設置表格樣式
    for i in range(len(comparison_data)):
        for j in range(3):
            if i == 0:  # 標題行
                table[(i, j)].set_facecolor("#4CAF50")
                table[(i, j)].set_text_props(weight="bold", color="white")
            elif j == 0:  # 指標列
                table[(i, j)].set_facecolor("#E8F5E8")
            else:
                table[(i, j)].set_facecolor("#F9F9F9")

    ax.set_title("詳細對比數據", fontweight="bold")

    plt.tight_layout()
    plt.savefig("training_strategy_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_analysis(results):
    """打印分析結果"""
    print("\n" + "=" * 60)
    print("📊 訓練策略分析報告")
    print("=" * 60)

    strategy_a = results["strategy_a"]
    strategy_b = results["strategy_b"]

    print(f"\n🔍 策略A（大數據+少epoch）:")
    print(f"   • 訓練樣本: 4000")
    print(f"   • 使用epochs: {strategy_a['epochs_used']}")
    print(f"   • 總訓練時間: {strategy_a['total_time']:.1f}s")
    print(f"   • 最佳驗證準確率: {strategy_a['best_val_acc']:.2f}%")
    print(
        f"   • 過擬合程度: {strategy_a['final_train_acc']-strategy_a['final_val_acc']:.2f}%"
    )

    print(f"\n🔍 策略B（小數據+多epoch）:")
    print(f"   • 訓練樣本: 1200")
    print(f"   • 使用epochs: {strategy_b['epochs_used']}")
    print(f"   • 總訓練時間: {strategy_b['total_time']:.1f}s")
    print(f"   • 最佳驗證準確率: {strategy_b['best_val_acc']:.2f}%")
    print(
        f"   • 過擬合程度: {strategy_b['final_train_acc']-strategy_b['final_val_acc']:.2f}%"
    )

    print(f"\n💡 策略優劣分析:")

    # 準確率比較
    if strategy_a["best_val_acc"] > strategy_b["best_val_acc"]:
        acc_winner = "策略A（大數據）"
        acc_diff = strategy_a["best_val_acc"] - strategy_b["best_val_acc"]
    else:
        acc_winner = "策略B（小數據）"
        acc_diff = strategy_b["best_val_acc"] - strategy_a["best_val_acc"]

    # 時間比較
    if strategy_a["total_time"] < strategy_b["total_time"]:
        time_winner = "策略A（大數據）"
        time_diff = strategy_b["total_time"] - strategy_a["total_time"]
    else:
        time_winner = "策略B（小數據）"
        time_diff = strategy_a["total_time"] - strategy_b["total_time"]

    # 過擬合比較
    overfitting_a = abs(strategy_a["final_train_acc"] - strategy_a["final_val_acc"])
    overfitting_b = abs(strategy_b["final_train_acc"] - strategy_b["final_val_acc"])

    if overfitting_a < overfitting_b:
        overfitting_winner = "策略A（大數據）"
        overfitting_diff = overfitting_b - overfitting_a
    else:
        overfitting_winner = "策略B（小數據）"
        overfitting_diff = overfitting_a - overfitting_b

    print(f"\n🏆 綜合比較:")
    print(f"   • 準確率優勝: {acc_winner} (領先 {acc_diff:.2f}%)")
    print(f"   • 訓練速度優勝: {time_winner} (快 {time_diff:.1f}s)")
    print(f"   • 泛化能力優勝: {overfitting_winner} (過擬合少 {overfitting_diff:.2f}%)")

    print(f"\n🎯 實用建議:")
    print(f"   • 如果追求最高準確率: 選擇{acc_winner}")
    print(f"   • 如果注重訓練效率: 選擇{time_winner}")
    print(f"   • 如果要求泛化能力: 選擇{overfitting_winner}")

    # 根據結果給出具體建議
    if (
        strategy_a["best_val_acc"] > strategy_b["best_val_acc"]
        and strategy_a["total_time"] < strategy_b["total_time"]
    ):
        print(f"\n✅ 結論: 策略A（大數據+少epoch）在準確率和效率上都更優秀！")
    elif (
        strategy_b["best_val_acc"] > strategy_a["best_val_acc"]
        and strategy_b["total_time"] < strategy_a["total_time"]
    ):
        print(f"\n✅ 結論: 策略B（小數據+多epoch）在準確率和效率上都更優秀！")
    else:
        print(f"\n⚖️ 結論: 兩種策略各有優劣，需要根據具體需求選擇")
        if abs(strategy_a["best_val_acc"] - strategy_b["best_val_acc"]) < 2:
            print(f"   準確率相近，建議選擇訓練時間更短的策略")
        else:
            print(f"   如果準確率差距較大，建議優先考慮準確率")


def main():
    """主函數"""
    print("🔬 gMLP訓練策略對比實驗")
    print("比較「大數據量+少epoch」vs「小數據量+多epoch」")
    print("=" * 60)

    try:
        # 執行對比實驗
        results = compare_strategies()

        # 繪製對比圖表
        plot_comparison(results)

        # 打印分析報告
        print_analysis(results)

        print(f"\n✅ 對比實驗完成！")
        print(f"   • 對比圖表已保存: training_strategy_comparison.png")
        print(f"   • 詳細分析報告已顯示")

    except Exception as e:
        print(f"❌ 實驗失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
>>>>>>> 420764095488647da1ecd1309c810893dfec8ea4
