import torch
import numpy as np
import os

# ===== 參數設定 =====
MODEL_PATH = "99.3_gmlp_model.pt"  # 你的模型檔案
OUTPUT_DIR = "mem_files"  # 輸出的資料夾，改為 mem_files
BIT_WIDTH = 16  # 量化位數（與 Verilog 的 DATA_WIDTH 匹配）
SCALE_FACTOR = 32767  # 對應 16-bit signed [-32767, 32767]

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 載入模型 =====
# 由於模型是 state_dict，直接載入即可
state_dict = torch.load(MODEL_PATH, map_location="cpu")
print(f"成功載入模型參數從: {MODEL_PATH}")


# ===== 量化函數 =====
def quantize_tensor(tensor, scale_factor=SCALE_FACTOR):
    t = tensor.detach().cpu().numpy()
    t = np.clip(t, -1.0, 1.0)  # 限制範圍，避免溢位
    t_q = np.round(t * scale_factor).astype(np.int16)  # 16-bit 量化
    return t_q


# ===== 輔助函數：寫入 MEM 檔 (Hex格式) =====
def write_mem_file(filepath, data_array):
    """將 numpy array 寫入 MEM 檔案，每行一個16進制數值，供 $readmemh 使用"""
    with open(filepath, "w") as f:
        flat_data = data_array.flatten()
        for val in flat_data:
            # 將 val 轉為 Python 原生 int 再進行位元運算，避免 NumPy 的 OverflowError
            hex_val = f"{int(val) & 0xFFFF:04x}"
            f.write(f"{hex_val}\n")
    print(f"✅ 已輸出量化參數到 {filepath} (Hex格式)，共 {len(flat_data)} 筆數據。")


# ===== 處理所有權重和偏置 =====
print("\n--- 開始處理模型參數 ---")
for name, param in state_dict.items():
    # 將層名稱中的 '.' 替換為 '_'，使其成為有效的檔案名稱
    safe_name = name.replace(".", "_")

    # 決定輸出路徑，副檔名改為 .mem
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.mem")

    # 判斷是權重還是偏置
    if "weight" in name:
        # PyTorch weight 是 (out_dim, in_dim)
        # Verilog memory 是 [in_dim-1:0][out_dim-1:0]
        # 需要轉置 (transpose)
        q_param = quantize_tensor(param.T, SCALE_FACTOR)
        write_mem_file(output_path, q_param)
    elif "bias" in name:
        q_param = quantize_tensor(param, SCALE_FACTOR)
        write_mem_file(output_path, q_param)
    else:
        print(f"⚠️ 跳過參數: {name} (非權重或偏置)")

print("\n🎉 所有參數已成功轉換並儲存至 'mem_files' 資料夾。")
