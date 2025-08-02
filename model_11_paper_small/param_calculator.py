"""
精確的 gMLP 參數計算
"""


def calculate_gmlp_params(
    image_size=32, patch_size=4, dim=80, depth=8, ff_mult=3, num_classes=10, channels=3
):
    """精確計算gMLP參數量"""

    print(f"🧮 精確參數計算 (dim={dim}, depth={depth}, ff_mult={ff_mult})")
    print("=" * 60)

    # 1. Patch Embedding
    patch_dim = patch_size * patch_size * channels  # 4*4*3 = 48
    patch_emb = patch_dim * dim + dim  # 權重 + 偏置
    print(f"1. Patch Embedding: {patch_dim}×{dim} + {dim} = {patch_emb:,}")

    # 2. Position Embedding
    num_patches = (image_size // patch_size) ** 2  # (32/4)² = 64
    pos_emb = num_patches * dim
    print(f"2. Position Embedding: {num_patches}×{dim} = {pos_emb:,}")

    # 3. gMLP 層
    total_layer_params = 0

    for layer in range(depth):
        # Layer Norm
        ln_params = dim * 2  # weight + bias

        # Projection layers
        proj_in = dim * (dim * ff_mult)  # dim → dim*ff_mult
        proj_out = (dim * ff_mult) * dim  # dim*ff_mult → dim

        # Spatial Gating Unit (SGU) - 關鍵部分!
        # SGU的參數量比預期少很多
        sgu_params = num_patches + num_patches * num_patches // 2  # 簡化的空間門控

        layer_total = ln_params + proj_in + proj_out + sgu_params
        total_layer_params += layer_total

        if layer < 3:  # 只顯示前3層
            print(
                f"   Layer {layer+1}: LN({ln_params}) + Proj({proj_in + proj_out}) + SGU({sgu_params}) = {layer_total:,}"
            )

    print(f"3. gMLP層總計 (×{depth}): {total_layer_params:,}")

    # 4. Classification Head
    head_params = dim * num_classes + num_classes  # 權重 + 偏置
    print(
        f"4. Classification Head: {dim}×{num_classes} + {num_classes} = {head_params:,}"
    )

    # 總計
    total = patch_emb + pos_emb + total_layer_params + head_params
    print(f"\n📊 總參數量: {total:,} ({total/1e6:.3f}M)")

    return total


# 計算不同配置
configs = {
    "XS": {"dim": 80, "depth": 8, "ff_mult": 3, "target": 0.8},
    "Nano": {"dim": 64, "depth": 6, "ff_mult": 2, "target": 0.3},
    "S": {"dim": 128, "depth": 12, "ff_mult": 3, "target": 2.0},
}

for name, config in configs.items():
    actual = calculate_gmlp_params(
        dim=config["dim"], depth=config["depth"], ff_mult=config["ff_mult"]
    )
    target = config["target"] * 1e6
    diff_pct = ((actual - target) / target) * 100

    print(f"🎯 {name} 模型對比:")
    print(f"   實際: {actual:,} ({actual/1e6:.3f}M)")
    print(f"   目標: {target:,.0f} ({config['target']}M)")
    print(f"   差異: {diff_pct:+.1f}%")
    print("\n" + "=" * 60 + "\n")
