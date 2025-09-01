import torch
import pandas as pd
from g_mlp_pytorch import gMLPVision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_gmlp_model():
    model = gMLPVision(
        image_size=28,  # MNIST圖像大小
        patch_size=4,  # 適合28x28的patch大小
        num_classes=10,  # MNIST類別數
        dim=16,  # MNIST較簡單，維度可調低
        depth=2,  # 深度可調低
        ff_mult=2,  # 特徵維度擴展倍數
        channels=1,  # MNIST為單通道
        prob_survival=1.0,
    )
    return model


if __name__ == "__main__":
    model = create_gmlp_model()
    model.load_state_dict(torch.load("98.43_mnist_RELU_1_16_2.pt", map_location=device))
    model.to(device)

    # 終端機只顯示表格（不含完整權重）
    param_list = []
    for name, param in model.named_parameters():
        param_list.append(
            {
                "層名稱": name,
                "形狀": list(param.shape),
                "參數數量": param.numel(),
            }
        )
    df = pd.DataFrame(param_list)
    # print("\n模型參數表格：")
    # print(df.to_string(index=False))

    # 輸出完整權重到 .txt
    with open("gmlp_params_full_weights.txt", "w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            f.write(f"層名稱: {name}\n")
            f.write(f"形狀: {list(param.shape)}\n")
            f.write(f"參數數量: {param.numel()}\n")
            f.write("權重:\n")
            f.write(str(param.detach().cpu().numpy()))
            f.write("\n" + "-" * 60 + "\n")
    print("\n已將完整權重輸出至 gmlp_params_full_weights.txt")
