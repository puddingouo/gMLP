`timescale 1ns / 1ps

module matrix_multiplier #(
    // 假設 A 是 M x K 矩陣, B 是 K x N 矩陣
    parameter DATA_WIDTH    = 16,
    parameter M             = 1, // 矩陣 A 的行數
    parameter K             = 32, // 矩陣 A 的列數 / 矩陣 B 的行數
    parameter N             = 10, // 矩陣 B 的列數
    parameter A_MEM_INIT_FILE = "matrix_c.mem",
    parameter B_MEM_INIT_FILE = "matrix_d.mem",
    // --- 新增偏置檔案參數 ---
    parameter BIAS_MEM_INIT_FILE = "bias.mem"
  ) (
    input clk,
    input rstn,
    input start,
    output reg done,
    // --- 新增以下埠 ---
    input [$clog2(M*N)-1:0] result_addr, // 讀取結果的位址
    output [ACC_WIDTH-1:0] result_out     // ★★★ 關鍵修改：將寬度改為 ACC_WIDTH ★★★
  );

  // --- 狀態定義 (新增 ADD_BIAS) ---
  localparam IDLE         = 4'd0;
  localparam FETCH        = 4'd1;
  localparam MULTIPLY     = 4'd2;
  localparam ACCUMULATE   = 4'd3;
  localparam ADD_BIAS     = 4'd4; // <-- 新增狀態
  localparam STORE        = 4'd5;
  localparam FINISH       = 4'd6;

  // --- 內部信號 ---
  reg [3:0] state, next_state; // 狀態機寬度增加

  // --- 索引計數器 ---
  reg [$clog2(M)-1:0] i_reg, i_next; // 矩陣 A 的行索引
  reg [$clog2(K)-1:0] k_reg, k_next; // 內積索引
  reg [$clog2(N)-1:0] j_reg, j_next; // 矩陣 B 的列索引

  // --- 累加器 ---
  // 乘積為 2*DATA_WIDTH，累加 K 次可能需要更寬的位元
  localparam ACC_WIDTH = 2 * DATA_WIDTH + $clog2(K);
  reg signed [ACC_WIDTH-1:0] acc_reg, acc_next;

  // --- 記憶體位址與資料 ---
  reg  [$clog2(M*K)-1:0] a_addr;
  wire [DATA_WIDTH-1:0] a_data;
  reg  [$clog2(K*N)-1:0] b_addr;
  wire [DATA_WIDTH-1:0] b_data;
  // --- 修改：偏置記憶體相關信號 ---
  reg  [$clog2(N)-1:0]   bias_addr;
  wire [DATA_WIDTH-1:0]   bias_data_16b; // 從 ROM 讀取的 16-bit 原始偏置
  // ---
  reg  [$clog2(M*N)-1:0] c_addr;
  reg  [ACC_WIDTH-1:0]   c_data_in;
  reg                    c_we; // 寫入致能

  // --- 實例化記憶體 ---
  // 矩陣 A (輸入)
  mlp_weights_rom #(
                    .ADDR_WIDTH($clog2(M*K)),
                    .DATA_WIDTH(DATA_WIDTH),
                    .MEM_SIZE((M*K) * DATA_WIDTH), // <-- 新增此行
                    .MEM_INIT_FILE(A_MEM_INIT_FILE)
                  ) matrix_a_ram (
                    .clk(clk), .enb(1'b1), .addrb(a_addr), .doutb(a_data)
                  );

  // 矩陣 B (權重)
  mlp_weights_rom #(
                    .ADDR_WIDTH($clog2(K*N)),
                    .DATA_WIDTH(DATA_WIDTH),
                    .MEM_SIZE((K*N) * DATA_WIDTH), // <-- 新增此行
                    .MEM_INIT_FILE(B_MEM_INIT_FILE)
                  ) matrix_b_rom (
                    .clk(clk), .enb(1'b1), .addrb(b_addr), .doutb(b_data)
                  );

  // --- 修改：實例化偏置記憶體 (Bias ROM) ---
  // 將 DATA_WIDTH 改為 16 (DATA_WIDTH)，以匹配 16-bit 的 .mem 檔案
  mlp_weights_rom #(
                    .ADDR_WIDTH($clog2(N)),
                    .DATA_WIDTH(DATA_WIDTH), // <-- 修改: 從 ACC_WIDTH 改為 DATA_WIDTH
                    .MEM_SIZE(N * DATA_WIDTH),   // <-- 修改: 從 ACC_WIDTH 改為 DATA_WIDTH
                    .MEM_INIT_FILE(BIAS_MEM_INIT_FILE)
                  ) bias_rom (
                    .clk(clk), .enb(1'b1), .addrb(bias_addr), .doutb(bias_data_16b) // <-- 修改: 輸出到新的 16-bit wire
                  );


  // 矩陣 C (輸出結果) - 使用 simple_dual_port_ram
  simple_dual_port_ram #(
                         .DATA_WIDTH(ACC_WIDTH),
                         .ADDR_WIDTH($clog2(M*N))
                       ) matrix_c_ram (
                         .clk(clk), .wea(c_we), .addra(c_addr), .dina(c_data_in),
                         // --- 修改以下連線 ---
                         .enb(1'b1), .addrb(result_addr), .doutb(result_out)
                       );

  // --- 實例化乘法器核心 ---
  reg  booth_en;
  wire booth_done;
  wire [2*DATA_WIDTH-1:0] booth_product;

  // 注意：這裡直接使用 booth_fsm，因為 top_booth 包含 ROM，而我們需要更靈活的記憶體存取
  booth_fsm #(
              .DATAWIDTH(DATA_WIDTH)
            ) booth_fsm_inst (
              .clk(clk), .rstn(rstn), .en(booth_en),
              .multiplier(a_data),   // 來自矩陣 A
              .multiplicand(b_data), // 來自矩陣 B
              .done(booth_done), .product(booth_product)
            );

  // --- 狀態機時序邏輯 ---
  always @(posedge clk)
  begin
    if (!rstn)
    begin
      state <= IDLE;
      i_reg <= 0;
      j_reg <= 0;
      k_reg <= 0;
      acc_reg <= 0;
    end
    else
    begin
      state <= next_state;
      i_reg <= i_next;
      j_reg <= j_next;
      k_reg <= k_next;
      acc_reg <= acc_next;
    end
  end

  // --- 狀態機組合邏輯 ---
  always @(*)
  begin
    // 預設值
    next_state = state;
    done = 1'b0;
    booth_en = 1'b0;
    c_we = 1'b0;
    c_data_in = 0;
    acc_next = acc_reg;
    i_next = i_reg;
    j_next = j_reg;
    k_next = k_reg;

    // 記憶體位址計算 (A[i,k], B[k,j], C[i,j], Bias[j])
    a_addr = i_reg * K + k_reg;
    b_addr = k_reg * N + j_reg;
    c_addr = i_reg * N + j_reg;
    bias_addr = j_reg; // 偏置位址只跟 j 有關

    case (state)
      IDLE:
      begin
        acc_next = 0;
        i_next = 0;
        j_next = 0;
        k_next = 0;
        if (start)
        begin
          next_state = FETCH;
        end
      end
      FETCH:
      begin
        // 位址已在前一個週期設定好，等待資料穩定
        // 假設 ROM 讀取延遲為 1 個週期
        next_state = MULTIPLY;
      end
      MULTIPLY:
      begin
        booth_en = 1'b1;
        next_state = ACCUMULATE;
      end
      ACCUMULATE:
      begin
        if (booth_done)
        begin
          // ★★★ 關鍵修改：修正 booth_product 的符號擴展 ★★★
          // 將 32-bit 的 booth_product 符號擴展至 ACC_WIDTH，再與 acc_reg 相加
          acc_next = acc_reg + {{ACC_WIDTH - (2*DATA_WIDTH){booth_product[2*DATA_WIDTH-1]}}, booth_product};

          if (k_reg == K - 1)
          begin
            // 內積完成，準備去加上偏置
            // ★★★ 關鍵修改：不要在這裡跳到 ADD_BIAS ★★★
            // 先停留在 ACCUMULATE，下個週期 acc_reg 更新後再跳轉
            next_state = ADD_BIAS;
          end
          else
          begin
            k_next = k_reg + 1;
            next_state = FETCH;
          end
        end
      end
      // --- 修改 ADD_BIAS 狀態的邏輯 ---
      ADD_BIAS:
      begin
        // ★★★ 關鍵修改：此時 acc_reg 已經是 k=0 到 k-1 的累加總和 ★★★
        // 現在才加上偏置
        // 將 16-bit 的 Q8.8 偏置左移 8 位並符號擴展至 ACC_WIDTH，使其變為 Q16.16 格式
        acc_next = acc_reg + {{ACC_WIDTH-24{bias_data_16b[15]}}, bias_data_16b, 8'h00};
        next_state = STORE;
      end
      STORE:
      begin
        c_we = 1'b1;
        c_data_in = acc_next; // 將 (累加結果+偏置) 寫入
        acc_next = 0; // 清空累加器
        k_next = 0;   // 重置內積索引

        if (i_reg == M - 1 && j_reg == N - 1)
        begin
          next_state = FINISH;
        end
        else if (j_reg == N - 1)
        begin
          i_next = i_reg + 1;
          j_next = 0;
          next_state = FETCH;
        end
        else
        begin
          j_next = j_reg + 1;
          next_state = FETCH;
        end
      end
      FINISH:
      begin
        done = 1'b1;
        next_state = IDLE;
      end
      default:
        next_state = IDLE;
    endcase
  end

endmodule
