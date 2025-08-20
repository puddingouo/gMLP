// 模組：管線化線性層 (向量-矩陣乘法)
// 功能：out_vector = in_vector * weights + bias
// 特色：採用深度管線化架構以最大化吞吐量
module linear_layer #(
    // --- 參數定義 ---
    parameter integer DATA_WIDTH = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer IN_DIM = 2,
    parameter integer OUT_DIM = 4
) (
    // --- 連接埠 ---
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    input  wire [ IN_DIM*DATA_WIDTH-1:0] in_vector,
    output reg  [OUT_DIM*DATA_WIDTH-1:0] out_vector,
    output reg                           done,
    output wire                          busy
);

    // --- 內部儲存與信號 ---
    reg signed [DATA_WIDTH-1:0] weights[0:IN_DIM-1][0:OUT_DIM-1];
    reg signed [DATA_WIDTH-1:0] bias[0:OUT_DIM-1];

    // --- 記憶體初始化 (用於合成) ---
    // Vivado 會在合成期間讀取這些檔案，並將資料嵌入到 FPGA 的 Block RAM 中。
    // 請確保路徑相對於 Vivado 專案的 .srcs 目錄是正確的，或者使用絕對路徑。
    initial begin
        $readmemh("F:/Lab/gMLP/gMLP_FPGA/gMLP.srcs/sources_1/imports/test/test_weight.mem",
                  weights);
        $readmemh("F:/Lab/gMLP/gMLP_FPGA/gMLP.srcs/sources_1/imports/test/test_bias.mem", bias);
    end

    wire signed [DATA_WIDTH-1:0] in_vector_unpacked [ 0:IN_DIM-1];
    reg signed  [DATA_WIDTH-1:0] out_vector_unpacked[0:OUT_DIM-1];

    // --- 輸入向量解包 ---
    genvar i_unpack;
    generate
        for (i_unpack = 0; i_unpack < IN_DIM; i_unpack = i_unpack + 1) begin : UNPACK_INPUT
            assign in_vector_unpacked[i_unpack] = in_vector[(i_unpack+1)*DATA_WIDTH-1-:DATA_WIDTH];
        end
    endgenerate

    // --- 輸出向量打包 ---
    integer j_pack;
    always @(*) begin
        for (j_pack = 0; j_pack < OUT_DIM; j_pack = j_pack + 1) begin
            out_vector[(j_pack+1)*DATA_WIDTH-1-:DATA_WIDTH] = out_vector_unpacked[j_pack];
        end
    end

    // --- 管線化計算邏輯 ---
    localparam S_IDLE = 2'b01;
    localparam S_COMPUTE = 2'b10;
    reg [1:0] state, next_state;

    // 管線計數器
    reg [$clog2(IN_DIM+1):0] i_count;  // 計數器需要多一位來標識完成
    reg start_reg;

    // 階段 1: 乘法結果暫存器 (OUT_DIM 個並行乘法)
    reg signed [2*DATA_WIDTH-1:0] product_reg[0:OUT_DIM-1];

    // 階段 2: 累加器 (OUT_DIM 個並行累加器)
    reg signed [2*DATA_WIDTH-1:0] mac_accumulator[0:OUT_DIM-1];

    // 將忙碌信號與計算狀態綁定
    assign busy = (state == S_COMPUTE);

    // 狀態機時序邏輯
    always @(posedge clk or posedge rst) begin
        if (rst) state <= S_IDLE;
        else state <= next_state;
    end
    // 狀態機組合邏輯
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: if (start) next_state = S_COMPUTE;
            S_COMPUTE: if (i_count == IN_DIM) next_state = S_IDLE;  // 計算完成後回到 IDLE
            default: next_state = S_IDLE;
        endcase
    end

    // 主要管線邏輯
    integer j;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // --- 重置所有暫存器 ---
            // 這是避免鎖存器的第一步：確保系統有明確的初始狀態
            i_count <= 0;
            done <= 1'b0;
            start_reg <= 1'b0;
            state <= S_IDLE;
            for (j = 0; j < OUT_DIM; j = j + 1) begin
                mac_accumulator[j] <= 0;
                product_reg[j] <= 0;
                out_vector_unpacked[j] <= 0;
            end
        end else begin
            // --- 預設賦值 ---
            // 這是避免鎖存器的關鍵技巧：在每個時脈週期的開始，
            // 先假設信號保持不變或設為預設值。
            // 這樣即使在後續的 if/case 中沒有賦值，它們也有明確的來源。
            done <= 1'b0;
            start_reg <= start;

            // --- 根據狀態執行 ---
            case (state)
                S_IDLE: begin
                    i_count <= 0;
                    // 當 start 信號來臨時，清除累加器，為下一次計算做準備
                    if (start && !start_reg) begin
                        for (j = 0; j < OUT_DIM; j = j + 1) begin
                            mac_accumulator[j] <= 0;
                        end
                    end
                end
                S_COMPUTE: begin
                    // --- 階段 1: 乘法 ---
                    // **修正**: 增加 i_count < IN_DIM 判斷，避免讀取陣列越界
                    if (i_count < IN_DIM) begin
                        for (j = 0; j < OUT_DIM; j = j + 1) begin
                            product_reg[j] <= $signed(in_vector_unpacked[i_count]) *
                                $signed(weights[i_count][j]);
                        end
                    end

                    // --- 階段 2: 累加 ---
                    // 將上一個週期的乘法結果累加起來
                    if (i_count > 0) begin
                        for (j = 0; j < OUT_DIM; j = j + 1) begin
                            mac_accumulator[j] <= mac_accumulator[j] + product_reg[j];
                        end
                    end

                    // --- 階段 3: 偏置與輸出 ---
                    // 當所有乘法累加完成時 (i_count == IN_DIM)
                    if (i_count == IN_DIM) begin
                        for (j = 0; j < OUT_DIM; j = j + 1) begin
                            // 此時 mac_accumulator 包含 p0+...+p(n-2)
                            // product_reg 包含 p(n-1)
                            out_vector_unpacked[j] <= ($signed(
                                mac_accumulator[j]
                            ) + $signed(
                                product_reg[j]
                            ) + ($signed(
                                bias[j]
                            ) <<< FRAC_BITS)) >>> FRAC_BITS;
                        end
                        done <= 1'b1;  // 計算完成
                    end

                    i_count <= i_count + 1;
                end
            endcase

            // 更新狀態機
            state <= next_state;
        end
    end

endmodule
