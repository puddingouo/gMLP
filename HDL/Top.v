`timescale 1ns / 1ps

// --- 頂層模組：用於 Nexys A7 FPGA ---
module fpga_top (
    // --- 板載 I/O ---
    input  wire        CLK100MHZ,
    input  wire        CPU_RESETN, // 通常是板上的重置按鈕，低電位有效
    input  wire [15:0] SW,         // 16個開關 (未使用)
    input  wire        BTNU,       // Up Button for Start
    input  wire        BTNR,       // Right Button for output element selection
    input  wire        BTND,       // Down Button for input vector selection
    output wire [15:0] LED,
    output wire        LED16_R,    // LED for 'done' signal
    output wire        LED17_R,    // LED for 'busy' signal

    // --- 七段顯示器 I/O ---
    output wire [7:0]  SEG, // Cathodes
    output wire [7:0]  AN   // Anodes
);

    // --- 參數設定 ---
    localparam integer DATA_WIDTH = 16;
    localparam integer FRAC_BITS  = 8;
    localparam integer IN_DIM     = 2;
    localparam integer OUT_DIM    = 4;

    // --- 內部信號 ---
    reg                           rst;
    reg                           start;
    reg  signed [IN_DIM*DATA_WIDTH-1:0]  dut_in_vector;
    wire signed [OUT_DIM*DATA_WIDTH-1:0] out_vector_flat;
    wire                          done;
    wire                          busy;

    // --- 預設輸入向量陣列 (共3組) ---
    reg signed [IN_DIM*DATA_WIDTH-1:0] input_vectors [0:2];
    initial begin
        // 0: [ 1.0, -1.0] -> {in1, in0}
        input_vectors[0] = {16'hff00, 16'h0100};
        // 1: [ 0.5,  -0.5]
        input_vectors[1] = {16'hff80, 16'h0080};
        // 2: [2.5, -1.5]
        input_vectors[2] = {16'hfe80, 16'h0280};
    end

    // --- 輸入/輸出元素選擇索引 ---
    reg [1:0] input_select_idx;
    reg [1:0] output_select_idx;
    wire signed [DATA_WIDTH-1:0] selected_output_element;

    // --- 實例化待測模組 (DUT) ---
    linear_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_DIM(IN_DIM),
        .OUT_DIM(OUT_DIM)
    ) dut (
        .clk(CLK100MHZ),
        .rst(rst),
        .start(start_d), // 傳遞延遲一拍的 start
        .in_vector(dut_in_vector),
        .out_vector(out_vector_flat),
        .done(done),
        .busy(busy)
    );

    // --- 實例化七段顯示器驅動模組 ---
    seven_segment_driver seg_driver (
        .clk(CLK100MHZ),
        .reset(rst),
        .data_in(selected_output_element),
        .SEG(SEG),
        .AN(AN)
    );

    // --- 處理重置信號 ---
    always @(posedge CLK100MHZ) begin
        rst <= ~CPU_RESETN;
    end

    // --- 按鈕防彈跳與邊緣檢測 ---
    reg btnu_reg1, btnu_reg2, btnu_posedge;
    reg btnr_reg1, btnr_reg2, btnr_posedge;
    reg btnd_reg1, btnd_reg2, btnd_posedge;

    always @(posedge CLK100MHZ) begin
        // BTNU (Start)
        btnu_reg1 <= BTNU;
        btnu_reg2 <= btnu_reg1;
        btnu_posedge <= btnu_reg1 & ~btnu_reg2;

        // BTNR (Output Select)
        btnr_reg1 <= BTNR;
        btnr_reg2 <= btnr_reg1;
        btnr_posedge <= btnr_reg1 & ~btnr_reg2;

        // BTND (Input Select)
        btnd_reg1 <= BTND;
        btnd_reg2 <= btnd_reg1;
        btnd_posedge <= btnd_reg1 & ~btnd_reg2;
    end

    reg start_d; // 延遲一拍的 start

    // --- 處理啟動信號 ---
    always @(posedge CLK100MHZ) begin
        if (rst) begin
            start <= 1'b0;
            start_d <= 1'b0;
        end else if (btnu_posedge && !busy) begin
            start <= 1'b1;
            start_d <= 1'b0;
        end else begin
            start <= 1'b0;
            start_d <= start;
        end
    end

    // --- 輸入向量鎖存 ---
    always @(posedge CLK100MHZ) begin
        if (rst) begin
            dut_in_vector <= 0;
        end else if (start) begin
            // 從陣列中根據索引選擇輸入向量
            dut_in_vector <= input_vectors[input_select_idx];
        end
        // 保持值直到下次 start
    end

    // --- 處理輸入向量選擇 ---
    always @(posedge CLK100MHZ) begin
        if (rst) begin
            input_select_idx <= 2'b0;
        end else if (btnd_posedge) begin
            input_select_idx <= (input_select_idx == 2) ? 2'b0 : input_select_idx + 1;
        end
    end

    // --- 處理輸出元素選擇 ---
    always @(posedge CLK100MHZ) begin
        if (rst) begin
            output_select_idx <= 2'b0;
        end else if (btnr_posedge) begin
            output_select_idx <= (output_select_idx == OUT_DIM - 1) ? 2'b0 : output_select_idx + 1;
        end
    end

    // --- 根據選擇索引設定輸出 ---
    reg signed [DATA_WIDTH-1:0] selected_output_element_reg;
    always @(*) begin
        case (output_select_idx)
            2'd0: selected_output_element_reg = out_vector_flat[1*DATA_WIDTH-1 : 0*DATA_WIDTH];
            2'd1: selected_output_element_reg = out_vector_flat[2*DATA_WIDTH-1 : 1*DATA_WIDTH];
            2'd2: selected_output_element_reg = out_vector_flat[3*DATA_WIDTH-1 : 2*DATA_WIDTH];
            2'd3: selected_output_element_reg = out_vector_flat[4*DATA_WIDTH-1 : 3*DATA_WIDTH];
            default: selected_output_element_reg = 16'h0;
        endcase
    end
    assign selected_output_element = selected_output_element_reg;

    // --- 連接輸出到 LED ---
    assign LED[1:0]   = input_select_idx;
    assign LED[3:2]   = output_select_idx;
    assign LED[15:4]  = 12'b0;
    assign LED16_R = done;
    assign LED17_R = busy;

endmodule
