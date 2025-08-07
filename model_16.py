"""
自定義 gMLP 圖像分類模型 - 簡化版本
基於論文架構，支援完全自定義模型配置
針對快速原型開發和資源受限環境優化
僅使用 AdamW + CosineAnnealingLR 策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from g_mlp_pytorch import gMLPVision
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
import os

# 設定隨機種子以確保實驗可重現
torch.manual_seed(0)
np.random.seed(0)


def progress_bar(batch_idx, total_batches, msg):
    """進度條顯示"""
    progress = (batch_idx + 1) / total_batches
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r[{bar}] {progress:.0%} {msg}", end="", flush=True)
    if batch_idx == total_batches - 1:
        print()  # 換行


def get_lr(optimizer):
    """獲取當前學習率"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def mixup_data(x, y, alpha=1.0, lam=1.0, count=0, device="cpu"):
    """Mixup 數據增強"""
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
    """Mixup 損失函數"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_cifar10_data_enhanced(quick_test=True, use_mixup_transform=False):
    """加載增強版 CIFAR-10 數據集"""
    print("📦 加載增強版 CIFAR-10 數據集...")

    if use_mixup_transform:
        # 使用增強版變換策略
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=32, scale=(0.6, 1.0)
                ),  # 調整為 CIFAR-10 尺寸
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(40),  # 調整為適合 CIFAR-10
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        print("   🎯 使用增強版數據變換 (ImageNet normalization)")
    else:
        # 使用標準的 CIFAR-10 變換策略
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        print("   📊 使用標準 CIFAR-10 數據變換")

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    if quick_test:
        trainset = Subset(trainset, range(50000))
        testset = Subset(testset, range(10000))
        print("   🚀 快速模式：完整數據集訓練")

    batch_size = 64
    num_workers = 1
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
        batch_size=10,
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
    print(f"   ✓ 訓練批次大小: {batch_size}")
    print(f"   ✓ 測試批次大小: 10")

    return trainloader, testloader, classes


def get_training_parameters_enhanced():
    """獲取增強版訓練參數設置 - 僅 AdamW + CosineAnnealingLR"""
    print("\n" + "=" * 60)
    print("⚙️  增強版訓練參數設置")
    print("=" * 60)

    # 預設參數
    default_lr = 0.01
    default_wd = 0.012
    default_epochs = 50
    default_alpha = 0.1
    default_batch_split = 1

    try:
        print(f"\n📚 使用 AdamW + CosineAnnealingLR 策略 (唯一選項)")

        # 基本參數設置
        lr_input = input(f"   📚 學習率 (預設={default_lr}): ").strip()
        lr = float(lr_input) if lr_input else default_lr

        wd_input = input(f"   ⚖️  權重衰減 (預設={default_wd}): ").strip()
        wd = float(wd_input) if wd_input else default_wd

        epochs_input = input(f"   🏋️  訓練輪數 (預設={default_epochs}): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

        # Mixup 參數
        print("\n🎨 Mixup 數據增強:")
        print("   1. 啟用 Mixup (推薦)")
        print("   2. 關閉 Mixup")
        mixup_choice = input("   選擇 (1/2, 預設=1): ").strip()
        use_mixup = True if mixup_choice != "2" else False

        alpha = default_alpha
        if use_mixup:
            alpha_input = input(
                f"   🎭 Mixup alpha 參數 (預設={default_alpha}): "
            ).strip()
            alpha = float(alpha_input) if alpha_input else default_alpha

        # 批次分割
        batch_split_input = input(
            f"\n🔢 批次分割因子 (預設={default_batch_split}): "
        ).strip()
        batch_split = (
            int(batch_split_input)
            if batch_split_input.isdigit()
            else default_batch_split
        )

        # 數據變換策略
        print("\n🔄 數據變換策略:")
        print("   1. 標準 CIFAR-10 變換")
        print("   2. 增強版變換 (ImageNet 風格)")
        transform_choice = input("   選擇 (1/2, 預設=1): ").strip()
        use_enhanced_transform = True if transform_choice == "2" else False

        # 早停機制設置
        print("\n⏹️ 早停機制設置:")
        print("   1. 啟用早停機制 (推薦)")
        print("   2. 關閉早停機制")
        early_stop_choice = input("   選擇 (1/2, 預設=1): ").strip()
        use_early_stopping = True if early_stop_choice != "2" else False

        patience = 10
        min_delta = 0.001
        if use_early_stopping:
            patience_input = input(
                "   ⏰ 耐心值 - 多少輪無改善後停止 (預設=10): "
            ).strip()
            patience = int(patience_input) if patience_input.isdigit() else 10

            min_delta_input = input("   📏 最小改善幅度 (預設=0.001): ").strip()
            min_delta = float(min_delta_input) if min_delta_input else 0.001

        print(f"\n✅ 訓練參數確認:")
        print(f"   📚 優化器: AdamW")
        print(f"   📚 學習率: {lr}")
        print(f"   📈 調度器: CosineAnnealingLR")
        print(f"   ⚖️  權重衰減: {wd}")
        print(f"   🏋️  訓練輪數: {epochs}")
        print(f"   🎨 Mixup: {'啟用' if use_mixup else '關閉'}")
        if use_mixup:
            print(f"   🎭 Alpha: {alpha}")
        print(f"   🔢 批次分割: {batch_split}")
        print(f"   🔄 變換策略: {'增強版' if use_enhanced_transform else '標準'}")
        print(f"   ⏹️ 早停機制: {'啟用' if use_early_stopping else '關閉'}")
        if use_early_stopping:
            print(f"   ⏰ 耐心值: {patience}")
            print(f"   📏 最小改善: {min_delta}")

        return {
            "lr": lr,
            "weight_decay": wd,
            "epochs": epochs,
            "use_mixup": use_mixup,
            "alpha": alpha,
            "batch_split": batch_split,
            "use_enhanced_transform": use_enhanced_transform,
            "optimizer_type": "AdamW",
            "scheduler_type": "CosineAnnealingLR",
            "use_early_stopping": use_early_stopping,
            "patience": patience,
            "min_delta": min_delta,
        }

    except (ValueError, KeyboardInterrupt):
        print("   ⚠️  使用預設參數 (AdamW + CosineAnnealingLR 策略)")
        return {
            "lr": default_lr,
            "weight_decay": default_wd,
            "epochs": default_epochs,
            "use_mixup": True,
            "alpha": default_alpha,
            "batch_split": default_batch_split,
            "use_enhanced_transform": False,
            "optimizer_type": "AdamW",
            "scheduler_type": "CosineAnnealingLR",
            "use_early_stopping": True,
            "patience": 10,
            "min_delta": 0.001,
        }


def get_user_model_choice():
    """獲取用戶的自定義模型配置"""
    print("\n" + "=" * 80)
    print("🏗️  自定義 gMLP 模型配置")
    print("=" * 80)

    print("📋 參數建議範圍:")
    print("   • 深度 (depth): 4-30 層")
    print("   • 維度 (dim): 64-256")
    print("   • FFN倍數 (ff_mult): 2-8")
    print("   • 存活機率 (prob_survival): 0.8-1.0")
    print("   • 注意力維度 (attn_dim): 64-128")

    while True:
        try:
            print("\n🔧 請輸入模型參數:")

            depth = int(input("   📏 深度 (推薦 8-16): "))
            if depth < 1 or depth > 50:
                print("   ⚠️ 深度建議在 1-50 之間")
                continue

            dim = int(input("   📐 維度 (推薦 64-256): "))
            if dim < 32 or dim > 512:
                print("   ⚠️ 維度建議在 32-512 之間")
                continue

            ff_mult = int(input("   🔢 FFN倍數 (推薦 2-6): "))
            if ff_mult < 1 or ff_mult > 12:
                print("   ⚠️ FFN倍數建議在 1-12 之間")
                continue

            prob_survival_input = input("   🎯 存活機率 (預設 1.0): ").strip()
            prob_survival = float(prob_survival_input) if prob_survival_input else 1.0
            if prob_survival < 0.1 or prob_survival > 1.0:
                print("   ⚠️ 存活機率必須在 0.1-1.0 之間")
                continue

            attn_dim_input = input("   🧠 注意力維度 (預設 64): ").strip()
            attn_dim = int(attn_dim_input) if attn_dim_input else 64
            if attn_dim < 32 or attn_dim > 256:
                print("   ⚠️ 注意力維度建議在 32-256 之間")
                continue

            # 估算參數數量
            estimated_params = estimate_gmlp_params(depth, dim, ff_mult, attn_dim)

            print(f"\n📊 模型配置預覽:")
            print(f"   • 深度: {depth} 層")
            print(f"   • 維度: {dim}")
            print(f"   • FFN倍數: {ff_mult}")
            print(f"   • 存活機率: {prob_survival}")
            print(f"   • 注意力維度: {attn_dim}")
            print(f"   • 預估參數: {estimated_params:.2f}M")

            # 預估訓練時間和記憶體
            if estimated_params < 0.5:
                time_est = "1-3分鐘"
                memory_est = "低"
            elif estimated_params < 1.0:
                time_est = "3-8分鐘"
                memory_est = "中等"
            elif estimated_params < 2.0:
                time_est = "8-15分鐘"
                memory_est = "較高"
            else:
                time_est = "15分鐘以上"
                memory_est = "很高"

            print(f"   • 預估訓練時間: {time_est}")
            print(f"   • 記憶體需求: {memory_est}")

            confirm = input(f"\n   確認使用此配置嗎? (y/n, 預設=y): ").strip().lower()
            if confirm in ["n", "no"]:
                print("   🔄 重新配置...")
                continue

            return {
                "depth": depth,
                "dim": dim,
                "ff_mult": ff_mult,
                "prob_survival": prob_survival,
                "attn_dim": attn_dim,
                "estimated_params": estimated_params,
            }

        except ValueError:
            print("   ❌ 請輸入有效的數字")
            continue
        except KeyboardInterrupt:
            print("\n   ❌ 用戶中斷，使用預設配置")
            return {
                "depth": 12,
                "dim": 128,
                "ff_mult": 3,
                "prob_survival": 1.0,
                "attn_dim": 64,
                "estimated_params": 0.65,
            }


def estimate_gmlp_params(depth, dim, ff_mult, attn_dim):
    """估算 gMLP 模型參數數量"""
    # Patch embedding: (patch_size^2 * channels * dim) + dim
    patch_embedding = (4 * 4 * 3 * dim) + dim

    # Each gMLP block
    block_params = (
        dim * 2  # Layer norm
        + dim * (dim * ff_mult * 2)
        + (dim * ff_mult * 2)  # Input projection
        + (32 // 4) ** 2
        + (32 // 4)  # Spatial gating (num_patches^2 + num_patches)
        + (dim * ff_mult) * dim
        + dim  # Output projection
    )

    total_blocks = depth * block_params

    # Final layer norm
    final_norm = dim * 2

    # Classifier
    classifier = dim * 10 + 10

    total_params = patch_embedding + total_blocks + final_norm + classifier
    return total_params / 1e6  # 轉換為百萬


def create_custom_gmlp_model(model_config):
    """創建自定義 gMLP 模型架構"""
    print(f"\n🏗️ 創建自定義 gMLP 模型...")

    torch.set_num_threads(4)
    print("   ⚡ CPU模式：已設置4個線程")

    model = gMLPVision(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=model_config["dim"],
        depth=model_config["depth"],
        ff_mult=model_config["ff_mult"],
        channels=3,
        prob_survival=model_config["prob_survival"],
        attn_dim=model_config["attn_dim"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    params_M = total_params / 1e6

    print(f"\n✅ 自定義 gMLP 模型創建完成")
    print(f"   ✓ 設備: {device}")
    print(f"   ✓ 實際參數數量: {total_params:,} ({params_M:.2f}M)")
    print(f"   ✓ 預估參數數量: {model_config['estimated_params']:.2f}M")
    print(
        f"   ✓ 架構配置: depth={model_config['depth']}, dim={model_config['dim']}, ff_mult={model_config['ff_mult']}, attn_dim={model_config['attn_dim']}"
    )

    return model, device


def train_enhanced(model, trainloader, testloader, device, training_params):
    """增強版訓練函數 - 僅 AdamW + CosineAnnealingLR 策略"""
    print(f"\n🏋️ 開始增強版訓練 ({training_params['epochs']} 個 epochs)...")
    print(f"   🎨 Mixup: {'啟用' if training_params['use_mixup'] else '關閉'}")
    print(f"   📚 優化器: AdamW")
    print(f"   📈 調度器: CosineAnnealingLR")
    print(
        f"   ⏹️ 早停機制: {'啟用' if training_params['use_early_stopping'] else '關閉'}"
    )
    if training_params["use_early_stopping"]:
        print(f"   ⏰ 耐心值: {training_params['patience']} epochs")
        print(f"   📏 最小改善: {training_params['min_delta']}")

    # 🎯 使用 AdamW + CosineAnnealingLR 配置
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
        betas=(0.9, 0.95),
    )
    lr_scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_params["epochs"], eta_min=8e-6
    )

    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []
    epoch_times = []
    best_val_acc = 0

    # 早停機制變數
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    total_start_time = time.time()

    for epoch in range(training_params["epochs"]):
        epoch_start_time = time.time()

        # 訓練階段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        count = 0
        lam = 1.0

        print(
            f'\nEpoch: {epoch + 1}/{training_params["epochs"]}, LR: {get_lr(optimizer):.6f}'
        )
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if count == training_params["batch_split"]:
                # 🎯 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
                optimizer.step()
                optimizer.zero_grad()
                count = 0

            inputs, targets = inputs.to(device), targets.to(device)

            if training_params["use_mixup"]:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, training_params["alpha"], lam, count, device
                )
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss = loss / training_params["batch_split"]
            loss.backward()
            count += 1

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if training_params["use_mixup"]:
                correct += (
                    lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
                )
            else:
                correct += predicted.eq(targets).sum().item()

            # 進度條顯示
            if batch_idx % 20 == 0 or batch_idx == len(trainloader) - 1:
                msg = f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})"
                progress_bar(batch_idx, len(trainloader), msg)

        epoch_loss = train_loss / len(trainloader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 驗證階段
        val_acc, val_loss = test_model(model, testloader, device, criterion)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # 更新學習率
        lr_scheduler_obj.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch {epoch + 1} 完成: 訓練={epoch_acc:.2f}%, 驗證={val_acc:.2f}%, 時間={epoch_duration:.1f}s"
        )

        # 保存最佳模型和早停機制
        improved = False
        prev_best = best_val_acc

        if val_acc > best_val_acc + training_params["min_delta"]:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            improved = True

            model_name = "best_custom_gmlp_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "acc": val_acc,
                    "epoch": epoch,
                    "optimizer_type": "AdamW",
                    "scheduler_type": "CosineAnnealingLR",
                },
                model_name,
            )
            improvement = val_acc - prev_best
            print(
                f"   💾 新最佳模型已保存: 驗證準確率 {best_val_acc:.2f}% (+{improvement:.3f}%)"
            )
        else:
            patience_counter += 1

        # 早停檢查
        if training_params["use_early_stopping"]:
            if patience_counter >= training_params["patience"]:
                early_stopped = True
                print(
                    f"\n⏹️ 早停觸發！連續 {training_params['patience']} 個 epochs 無改善"
                )
                print(
                    f"   📈 最佳驗證準確率: {best_val_acc:.2f}% (第 {best_epoch + 1} epoch)"
                )
                print(
                    f"   ⏰ 當前耐心計數: {patience_counter}/{training_params['patience']}"
                )
                break
            elif patience_counter > 0:
                print(
                    f"   ⏰ 早停計數: {patience_counter}/{training_params['patience']} (準確率未改善 ≥ {training_params['min_delta']})"
                )

        # 顯示當前狀態
        if not improved:
            print(
                f"   📊 當前準確率: {val_acc:.2f}% (最佳: {best_val_acc:.2f}%, 差距: {best_val_acc - val_acc:.3f}%)"
            )

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"\n⏱️ 增強版訓練時間統計:")
    print(
        f"   • 總訓練時間: {total_training_time:.1f}s ({total_training_time/60:.1f}min)"
    )
    print(f"   • 平均每epoch: {np.mean(epoch_times):.1f}s")
    print(f"   • 最佳驗證準確率: {best_val_acc:.2f}% (第 {best_epoch + 1} epoch)")
    print(f"   • 使用策略: AdamW + CosineAnnealingLR")

    # 早停機制統計
    if training_params["use_early_stopping"]:
        if early_stopped:
            print(f"   • 早停狀態: ⏹️ 早停觸發 (第 {epoch + 1} epoch)")
            print(f"   • 實際訓練輪數: {epoch + 1}/{training_params['epochs']}")
            print(
                f"   • 節省時間: {(training_params['epochs'] - epoch - 1) * np.mean(epoch_times):.1f}s"
            )
        else:
            print(f"   • 早停狀態: ✅ 完整訓練完成")
            print(f"   • 實際訓練輪數: {len(train_losses)}/{training_params['epochs']}")
    else:
        print(f"   • 實際訓練輪數: {len(train_losses)}/{training_params['epochs']}")

    # 載入最佳模型
    if best_val_acc > 0:
        checkpoint = torch.load("best_custom_gmlp_model.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   • 已載入最佳模型權重 (AdamW 訓練)")

    return (
        train_losses,
        train_accs,
        val_accs,
        val_losses,
        epoch_times,
        total_training_time,
        early_stopped,
        best_epoch,
    )


def test_model(model, testloader, device, criterion):
    """測試模型性能"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(testloader)

    return accuracy, avg_loss


def evaluate_custom_model(model, testloader, device, classes):
    """評估自定義模型"""
    print("\n📊 評估自定義模型...")

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

    # 結果可視化
    plt.figure(figsize=(12, 8))

    # 各類別準確率
    plt.subplot(2, 2, 1)
    class_accs = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(10)
    ]
    bars = plt.bar(classes, class_accs, color=plt.cm.tab10(np.arange(10)))
    plt.title("Custom gMLP: Class Accuracy", fontweight="bold")
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
    stats_text = f"""Custom Model Statistics:
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
    plt.savefig("custom_gmlp_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return overall_acc


def plot_enhanced_training_history(
    train_losses, train_accs, val_accs, val_losses, epoch_times
):
    """繪製增強版訓練歷史"""
    print("\n📈 繪製增強版訓練歷史...")

    plt.figure(figsize=(16, 4))

    # 損失曲線比較
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, "b-", linewidth=2, label="Training Loss")
    plt.plot(val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.title("Custom gMLP Loss Curves", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 準確率曲線
    plt.subplot(1, 4, 2)
    plt.plot(train_accs, "g-", linewidth=2, label="Training Acc")
    plt.plot(val_accs, "r-", linewidth=2, label="Validation Acc")
    plt.title("Custom gMLP Accuracy Curves", fontweight="bold")
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
    plt.savefig("custom_gmlp_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def genetic_algorithm_optimization():
    """遺傳算法優化入口"""
    from genetic_optimizer import run_genetic_optimization

    run_genetic_optimization()


def main():
    """主函數 - 添加遺傳算法選項"""
    print("🚀 自定義 gMLP 圖像分類訓練")
    print("🎯 使用 AdamW + CosineAnnealingLR 策略")
    print("=" * 70)

    print("\n選擇訓練模式:")
    print("1. 手動配置訓練")
    print("2. 遺傳算法自動優化 🧬")

    choice = input("請選擇 (1/2, 預設=1): ").strip()

    if choice == "2":
        genetic_algorithm_optimization()
        return

    try:
        # 原有的手動配置流程
        model_config = get_user_model_choice()
        training_params = get_training_parameters_enhanced()

        trainloader, testloader, classes = load_cifar10_data_enhanced(
            quick_test=True,
            use_mixup_transform=training_params["use_enhanced_transform"],
        )

        model, device = create_custom_gmlp_model(model_config)

        print(f"\n🎬 開始訓練自定義模型...")
        train_result = train_enhanced(
            model, trainloader, testloader, device, training_params
        )
        (
            train_losses,
            train_accs,
            val_accs,
            val_losses,
            epoch_times,
            total_time,
            early_stopped,
            best_epoch,
        ) = train_result

        plot_enhanced_training_history(
            train_losses, train_accs, val_accs, val_losses, epoch_times
        )

        final_acc = evaluate_custom_model(model, testloader, device, classes)

        print(f"\n🎉 訓練完成總結:")
        print(
            f"   • 模型配置: {model_config['depth']}層, {model_config['dim']}維度, FFN×{model_config['ff_mult']}"
        )
        print(f"   • 最終準確率: {final_acc:.2f}%")
        print(f"   • 總訓練時間: {total_time/60:.1f} 分鐘")
        print(f"   • 平均每epoch: {np.mean(epoch_times):.1f} 秒")
        print(f"   • 實際訓練輪數: {len(train_losses)}/{training_params['epochs']}")
        print(f"   • 優化策略: AdamW + CosineAnnealingLR")
        print(f"   • Mixup狀態: {'啟用' if training_params['use_mixup'] else '關閉'}")

        if training_params["use_early_stopping"]:
            if early_stopped:
                print(
                    f"   • 早停狀態: ⏹️ 提前停止 (節省 {training_params['epochs'] - len(train_losses)} epochs)"
                )
                print(f"   • 最佳epoch: 第 {best_epoch + 1} epoch")
            else:
                print(f"   • 早停狀態: ✅ 訓練完成 (未觸發早停)")
        else:
            print(f"   • 早停狀態: ❌ 未啟用")

        print(f"\n" + "=" * 70)
        continue_choice = input("🔄 是否要訓練其他模型? (y/n): ").strip().lower()
        if continue_choice in ["y", "yes"]:
            print("\n" + "🔄 重新開始..." + "\n")
            main()

    except KeyboardInterrupt:
        print("\n\n❌ 用戶中斷程序")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("   請檢查輸入並重試")


if __name__ == "__main__":
    main()
