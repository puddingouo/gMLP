import numpy as np
import torch
from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import einsum, quantize, relu
from da4ml.codegen import VerilogModel

# 參數設定 (與你的 gMLPVision 模型對應)
image_size = 28
patch_size = 4
num_classes = 10
dim = 128
depth = 3
ff_mult = 2
channels = 1

num_patches = (image_size // patch_size) ** 2  # 49
patch_dim = patch_size * patch_size * channels  # 16
inner_dim = dim * ff_mult  # 256


# --- 新增：從 PyTorch .pt 檔案載入權重 ---
def load_weights_from_pytorch(model_path, depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    state_dict = torch.load(model_path, map_location=device)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Patch Embedding
    w_embed = to_numpy(state_dict["to_patch_embed.1.weight"]).T
    b_embed = to_numpy(state_dict["to_patch_embed.1.bias"])

    # gMLP block 的權重和偏置
    w_proj_in, b_proj_in = [], []
    w_gate, b_gate = [], []
    w_proj_out, b_proj_out = [], []
    ln_gamma, ln_beta = [], []

    for i in range(depth):
        # LayerNorm
        ln_gamma.append(to_numpy(state_dict[f"layers.{i}.fn.norm.weight"]))
        ln_beta.append(to_numpy(state_dict[f"layers.{i}.fn.norm.bias"]))

        # Projections and Gating
        w_proj_in.append(to_numpy(state_dict[f"layers.{i}.fn.fn.proj_in.0.weight"]).T)
        b_proj_in.append(to_numpy(state_dict[f"layers.{i}.fn.fn.proj_in.0.bias"]))

        # SGU
        w_gate.append(to_numpy(state_dict[f"layers.{i}.fn.fn.sgu.weight"]).squeeze(0).T)
        b_gate.append(to_numpy(state_dict[f"layers.{i}.fn.fn.sgu.bias"]).squeeze(0))

        w_proj_out.append(to_numpy(state_dict[f"layers.{i}.fn.fn.proj_out.weight"]).T)
        b_proj_out.append(to_numpy(state_dict[f"layers.{i}.fn.fn.proj_out.bias"]))

    # 輸出層前的 LayerNorm 和最終輸出層
    final_ln_gamma = to_numpy(state_dict["to_logits.0.weight"])
    final_ln_beta = to_numpy(state_dict["to_logits.0.bias"])
    w_out = to_numpy(state_dict["to_logits.2.weight"]).T
    b_out = to_numpy(state_dict["to_logits.2.bias"])

    return (
        w_embed,
        b_embed,
        w_proj_in,
        b_proj_in,
        w_gate,
        b_gate,
        w_proj_out,
        b_proj_out,
        w_out,
        b_out,
        ln_gamma,
        ln_beta,
        final_ln_gamma,
        final_ln_beta,
    )


# --- 新增：LayerNorm 的 NumPy 實現 ---
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta


# --- 載入權重 ---
# 將 'your_model.pt' 換成你實際的檔案路徑
model_path = "99.5_mnist_RELU_3_128_2.pt"
(
    w_embed,
    b_embed,
    w_proj_in,
    b_proj_in,
    w_gate,
    b_gate,
    w_proj_out,
    b_proj_out,
    w_out,
    b_out,
    ln_gamma,
    ln_beta,
    final_ln_gamma,
    final_ln_beta,
) = load_weights_from_pytorch(model_path, depth)


def operation(inp):
    # --- 對輸入進行量化 ---
    inp = quantize(inp, 1, 7, 0)

    # 1. Patch Embedding
    x = einsum("bnd,de->bne", inp, w_embed)  # (1, 49, 16) -> (1, 49, 128)
    x = x + b_embed

    # 2. 疊代 gMLP blocks
    for i in range(depth):
        residual = x
        x = layer_norm(x, ln_gamma[i], ln_beta[i])

        # Proj in
        x = einsum("bnd,de->bne", x, w_proj_in[i])  # (1, 49, 128) -> (1, 49, 256)
        x = x + b_proj_in[i]
        x = relu(x)

        # Spatial Gating Unit (SGU)
        u, v = np.split(x, 2, axis=-1)
        v = einsum("bnd,nm->bmd", v, w_gate[i])  # (1, 49, 128) -> (1, 49, 128)
        v = v + b_gate[i]
        x = u * v

        # Proj out
        x = einsum("bnd,de->bne", x, w_proj_out[i])  # (1, 49, 128) -> (1, 49, 128)
        x = x + b_proj_out[i]

        # 殘差連接
        x = x + residual

    # 3. 平均所有 patch
    x = np.mean(x, axis=1)  # (1, 49, 128) -> (1, 128)

    # 4. 最終 LayerNorm 和輸出層
    x = layer_norm(x, final_ln_gamma, final_ln_beta)
    out = einsum("bd,dc->bc", x, w_out)  # (1, 128) -> (1, 10)
    out = out + b_out
    return out


# Symbolic input
batch = 1
inp = FixedVariableArrayInput((batch, num_patches, patch_dim))
out = operation(inp)

# 產生 Verilog
comb_logic = comb_trace(inp, out)
verilog_model = VerilogModel(comb_logic, "vmodel", "./verilog_output", latency_cutoff=5)
verilog_model.write()

print("Verilog 檔案已產生至 ./verilog_output 資料夾")
