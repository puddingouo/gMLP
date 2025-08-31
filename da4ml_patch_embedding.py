import numpy as np
import torch
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import einsum, quantize, relu
from da4ml.trace.ops import reduce
from da4ml.codegen import VerilogModel

# 參數設定
image_size = 28
patch_size = 4
channels = 1
dim = 16

num_patches = (image_size // patch_size) ** 2  # 49
patch_dim = patch_size * patch_size * channels  # 16


# 權重載入（僅 patch embedding 部分）
def load_patch_embed_weights(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    w_embed = state_dict["to_patch_embed.1.weight"].detach().cpu().numpy().T
    b_embed = state_dict["to_patch_embed.1.bias"].detach().cpu().numpy()
    return w_embed, b_embed


model_path = "98.43_mnist_RELU_2_16_2.pt"
w_embed, b_embed = load_patch_embed_weights(model_path)


# 測試 patch embedding 運算
def patch_embedding_forward(inp):
    inp = quantize(inp, 1, 7, 0)
    # inp: (batch, num_patches, patch_dim)
    x = einsum("bnd,de->bne", inp, w_embed)  # (1, 49, 16) -> (1, 49, 16)
    x = x + b_embed
    return x


# 測試用隨機輸入
batch = 1
inp = FixedVariableArrayInput((batch, num_patches, patch_dim))
out = patch_embedding_forward(inp)
print("Patch embedding 輸出 shape:", out.shape)
print("Patch embedding 輸出範例:", out[0, 0, :])

# 產生 Verilog
comb_logic = comb_trace(inp, out)
verilog_model = VerilogModel(comb_logic, "vmodel", "./verilog_output", latency_cutoff=5)
verilog_model.write()

print("Verilog 檔案已產生至 ./verilog_output 資料夾")
