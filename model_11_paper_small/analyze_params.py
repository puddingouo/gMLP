"""
gMLP 參數分析工具
深入分析實際參數與目標參數的差異
"""

from g_mlp_pytorch import gMLPVision
import torch
import math


def analyze_gmlp_parameters(model_size="XS"):
    """詳細分析gMLP模型參數"""

    # 模型配置
    configs = {
        "Test": {"depth": 4, "dim": 64, "ff_mult": 2, "target": 0.1},
        "Nano": {"depth": 6, "dim": 64, "ff_mult": 2, "target": 0.3},
        "XS": {"depth": 8, "dim": 80, "ff_mult": 3, "target": 0.8},
        "S": {"depth": 12, "dim": 128, "ff_mult": 3, "target": 2.0},
        "M": {"depth": 16, "dim": 160, "ff_mult": 4, "target": 4.5},
        "L": {"depth": 30, "dim": 128, "ff_mult": 6, "target": 5.9},
    }

    config = configs[model_size]

    print(f"🔍 {model_size} 模型參數深度分析")
    print("=" * 70)
    print(
        f"配置: depth={config['depth']}, dim={config['dim']}, ff_mult={config['ff_mult']}"
    )
    print(f"目標參數: {config['target']}M")
    print("-" * 70)

    # 創建模型
    model = gMLPVision(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=config["dim"],
        depth=config["depth"],
        ff_mult=config["ff_mult"],
        channels=3,
        prob_survival=1.00,
    )

    # 分析每一層參數
    print("\n📊 逐層參數分析:")
    total_params = 0
    layer_groups = {}

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        # 分組統計
        if "to_patch_embedding" in name:
            group = "Patch Embedding"
        elif "pos_emb" in name:
            group = "Position Embedding"
        elif "layers" in name:
            if "norm" in name:
                group = "Layer Normalization"
            elif "proj_in" in name or "proj_out" in name:
                group = "Linear Projections"
            elif "sgu" in name:
                group = "Spatial Gating Unit"
            else:
                group = "Other Layers"
        elif "to_logits" in name:
            group = "Classification Head"
        else:
            group = "Other"

        if group not in layer_groups:
            layer_groups[group] = 0
        layer_groups[group] += param_count

        if param_count > 500:  # 只顯示較大的參數
            print(f"   • {name:<40}: {param_count:>8,} ({param_count/1e6:.3f}M)")

    print("\n📈 參數分組統計:")
    for group, count in sorted(layer_groups.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"   • {group:<25}: {count:>8,} ({count/1e6:.3f}M, {percentage:.1f}%)")

    print("\n🧮 參數量理論計算:")

    # 計算各部分理論參數量
    patch_size = 4
    image_size = 32
    channels = 3
    num_classes = 10
    dim = config["dim"]
    depth = config["depth"]
    ff_mult = config["ff_mult"]

    # Patch embedding: (patch_size^2 * channels) * dim
    patch_emb_params = (patch_size * patch_size * channels) * dim
    print(
        f"   • Patch Embedding: {patch_size}²×{channels}×{dim} = {patch_emb_params:,}"
    )

    # Position embedding: (image_size/patch_size)^2 * dim
    num_patches = (image_size // patch_size) ** 2
    pos_emb_params = num_patches * dim
    print(f"   • Position Embedding: {num_patches}×{dim} = {pos_emb_params:,}")

    # 每層gMLP參數
    # Layer Norm: dim (weight) + dim (bias) = 2*dim
    norm_params_per_layer = 2 * dim

    # Linear projections: dim*dim*ff_mult + dim*ff_mult (proj_in) + dim*ff_mult*dim + dim (proj_out)
    proj_params_per_layer = (
        dim * (dim * ff_mult) + (dim * ff_mult) + (dim * ff_mult) * dim + dim
    )

    # SGU: 複雜度較低，主要是權重矩陣
    sgu_params_per_layer = num_patches * num_patches + num_patches  # 簡化估算

    layer_params = (
        norm_params_per_layer + proj_params_per_layer + sgu_params_per_layer
    ) * depth
    print(f"   • gMLP層 (×{depth}): ~{layer_params:,}")

    # Classification head: dim * num_classes
    head_params = dim * num_classes
    print(f"   • Classification Head: {dim}×{num_classes} = {head_params:,}")

    # 理論總計
    theoretical_total = patch_emb_params + pos_emb_params + layer_params + head_params
    print(f"   • 理論總計: {theoretical_total:,} ({theoretical_total/1e6:.3f}M)")

    print("\n📊 最終對比:")
    actual_M = total_params / 1e6
    target_M = config["target"]
    theoretical_M = theoretical_total / 1e6

    print(f"   • 實際參數: {total_params:,} ({actual_M:.3f}M)")
    print(f"   • 目標參數: {target_M:.1f}M")
    print(f"   • 理論估算: {theoretical_total:,} ({theoretical_M:.3f}M)")
    print(f"   • 實際 vs 目標: {((actual_M/target_M-1)*100):+.1f}%")
    print(f"   • 實際 vs 理論: {((actual_M/theoretical_M-1)*100):+.1f}%")

    print("\n💡 差異原因分析:")
    if actual_M < target_M:
        print("   ✅ 實際參數少於目標 - 這是好事！")
        print("   📉 可能原因:")
        print("      • gMLPVision架構比預期更精簡")
        print("      • 某些層使用了參數共享或更高效的實現")
        print("      • 目標估算公式過於保守")
        print("      • patch_size=4 產生的patches較少")
    else:
        print("   ⚠️ 實際參數多於目標")
        print("   📈 可能需要優化模型架構")


if __name__ == "__main__":
    # 分析不同規模的模型
    for size in ["Test", "Nano", "XS", "S"]:
        analyze_gmlp_parameters(size)
        print("\n" + "=" * 70 + "\n")
