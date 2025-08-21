`timescale 1ns / 1ps

// --- 記憶體檔案路徑 (使用 `define 宏以相容 Verilog-2001) ---
`define WEIGHT_FILE "F:/Lab/gMLP/gMLP_FPGA/gMLP.srcs/sources_1/imports/test/test_weight.mem"
`define BIAS_FILE   "F:/Lab/gMLP/gMLP_FPGA/gMLP.srcs/sources_1/imports/test/test_bias.mem"

module linear_layer_proj_in_tb;

    // --- 參數設定 ---
    localparam integer DATA_WIDTH = 16;
    localparam integer FRAC_BITS  = 8;  // 假設小數部分為 8 位
    localparam integer IN_DIM     = 2;
    localparam integer OUT_DIM    = 4;
    localparam integer CLK_PERIOD = 10; // 100MHz

    // --- Testbench 信號 ---
    reg                           clk;
    reg                           rst;
    reg                           start;
    reg  signed [IN_DIM*DATA_WIDTH-1:0]  tb_in_vector_flat;
    wire signed [OUT_DIM*DATA_WIDTH-1:0] tb_out_vector_flat;
    wire                          tb_done;

    // --- 實例化待測模組 (DUT) ---
    linear_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_DIM(IN_DIM),
        .OUT_DIM(OUT_DIM)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .in_vector(tb_in_vector_flat),
        .out_vector(tb_out_vector_flat),
        .done(tb_done)
    );

    // --- 時脈產生 ---
    always #(CLK_PERIOD / 2) clk = ~clk;

    // --- Testbench 內部記憶體，用於驗證 ---
    reg signed [DATA_WIDTH-1:0] tb_weights[0:IN_DIM*OUT_DIM-1];
    reg signed [DATA_WIDTH-1:0] tb_bias[0:OUT_DIM-1];
    reg signed [DATA_WIDTH-1:0] tb_in_vector[0:IN_DIM-1];
    reg signed [DATA_WIDTH-1:0] tb_expected_out[0:OUT_DIM-1];

    // --- 主要測試流程 ---
    initial begin
        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;
        tb_in_vector_flat = 0;

        $display("--- Testbench Started ---");

        // 1. 從 .mem 檔案載入權重和偏置到 Testbench 的記憶體中
        $readmemh(`WEIGHT_FILE, tb_weights);
        $readmemh(`BIAS_FILE,   tb_bias);
        $display("[%0t] Loaded weight and bias files into testbench memory.", $time);

        // 2. 將權重和偏置載入到 DUT 中 (透過階層式參考)
        load_data_to_dut();

        // 3. 產生預設的輸入向量
        generate_predefined_input();

        // 4. 重置 DUT
        #(CLK_PERIOD * 2);
        rst = 1'b0;
        $display("[%0t] Reset released.", $time);

        // 5. 啟動 DUT
        #(CLK_PERIOD);
        start = 1'b1;
        $display("[%0t] Start signal asserted.", $time);
        #(CLK_PERIOD);
        start = 1'b0;

        // 6. 等待 DUT 完成
        $display("[%0t] Waiting for 'done' signal...", $time);
        wait (tb_done == 1'b1);
        $display("[%0t] 'done' signal received. Verification will start.", $time);

        // 7. 驗證結果
        #1;  // 等待一小段時間確保信號穩定
        verify_result();

        $display("--- Testbench Finished ---");
        $finish;
    end

    // --- 任務：將資料載入 DUT ---
    task load_data_to_dut;
        // 變數宣告必須在 task 之後, begin 之前
        integer i;
    begin
        $display("Loading weights and biases into DUT...");
        // 載入偏置
        for (i = 0; i < OUT_DIM; i = i + 1) begin
            dut.bias[i] = tb_bias[i];
        end
        // 載入權重 (一維陣列直接複製)
        for (i = 0; i < IN_DIM * OUT_DIM; i = i + 1) begin
            dut.weights[i] = tb_weights[i];
        end
    end
    endtask

    // --- 任務：產生預設的輸入向量 ---
    task generate_predefined_input;
        // 變數宣告必須在 task 之後, begin 之前
        integer i;
        real scale_factor;
    begin
        scale_factor = 1 << FRAC_BITS; // 或 scale_factor = 2.0 ** FRAC_BITS;
        $display("Generating predefined input vector with a ramp pattern...");

        // 步驟 1: 產生一個完整的、可預測的輸入向量 (斜坡信號)
        tb_in_vector[0] = 16'h0100; // 1.0
        tb_in_vector[1] = 16'h0080; // 0.5

        // 步驟 2: 將設定好的 tb_in_vector 陣列打包成一個扁平的向量
        for (i = 0; i < IN_DIM; i = i + 1) begin
            tb_in_vector_flat[(i+1)*DATA_WIDTH-1-:DATA_WIDTH] = tb_in_vector[i];
        end
    end
    endtask

    // --- 任務：驗證結果 ---
    task verify_result;
        // 變數宣告必須在 task 之後, begin 之前
        integer i, j;
        integer error_count;
        reg signed [2*DATA_WIDTH-1:0] mac_accumulator;
        reg [DATA_WIDTH-1:0] dut_out_unpacked[0:OUT_DIM-1];
    begin
        error_count = 0; // 初始化移到 begin 內部
        $display("Verifying results...");

        // 解包 DUT 的輸出向量
        for (j = 0; j < OUT_DIM; j = j + 1) begin
            dut_out_unpacked[j] = tb_out_vector_flat[(j+1)*DATA_WIDTH-1-:DATA_WIDTH];
        end

        // 在 Testbench 中計算預期輸出
        for (j = 0; j < OUT_DIM; j = j + 1) begin
            mac_accumulator = 0;
            for (i = 0; i < IN_DIM; i = i + 1) begin
                // 修正：使用 Row-Major 索引 (i * OUT_DIM + j)
                mac_accumulator = mac_accumulator +
                    $signed(tb_in_vector[i]) * $signed(tb_weights[i*OUT_DIM+j]);
            end
            // 修正：在加入累加器之前，將偏置左移以對齊小數點
            tb_expected_out[j] = ($signed(mac_accumulator) + ($signed(tb_bias[j]) <<< FRAC_BITS)) >>> FRAC_BITS;
        end

        // 比較 DUT 輸出和預期輸出
        for (j = 0; j < OUT_DIM; j = j + 1) begin
            if (dut_out_unpacked[j] != tb_expected_out[j]) begin
                $display("ERROR at Output[%0d]: DUT = %h, Expected = %h", j, dut_out_unpacked[j],
                         tb_expected_out[j]);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0) begin
            $display("\n*** TEST PASSED ***");
        end else begin
            $display("\n*** TEST FAILED: Found %0d mismatches. ***", error_count);
        end
    end
    endtask

endmodule
