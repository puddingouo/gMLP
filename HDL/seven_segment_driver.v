`timescale 1ns / 1ps

// 模組：七段顯示器驅動
// 功能：將一個 16 位元的數字，以 4 位十六進位數的形式顯示在 4 個七段顯示器上。
//      採用時分複用（掃描）的方式驅動。
module seven_segment_driver (
    input wire clk, // 100MHz 系統時脈
    input wire reset,
    input wire [15:0] data_in, // 要顯示的 16 位元資料
    output reg [7:0] SEG, // 7 段陰極 (低電位有效) [dp,g,f,e,d,c,b,a]
    output reg [7:0] AN // 8 個陽極 (低電位有效)
);

    // 產生一個刷新時脈 (約 763 Hz)，用於切換不同的顯示位
    // 100,000,000 / (2^17) = 762.93 Hz
    reg [16:0] refresh_counter;
    wire refresh_tick;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            refresh_counter <= 0;
        end else begin
            refresh_counter <= refresh_counter + 1;
        end
    end
    // 在計數器歸零時產生一個脈衝
    assign refresh_tick = (refresh_counter == 17'd131071);

    // 2 位元計數器，用於選擇要啟動的顯示位 (0 到 3)
    reg [1:0] digit_select;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            digit_select <= 0;
        end else if (refresh_tick) begin
            digit_select <= digit_select + 1;
        end
    end

    // 根據當前選擇的位，從 16 位元資料中選出對應的 4 位元十六進位數
    wire [3:0] hex_digit;
    assign hex_digit = data_in >> (digit_select * 4);

    // 十六進位轉七段顯示碼 (共陽極 -> 段選為低電位有效)
    // seg = {dp, g, f, e, d, c, b, a}
    always @(*) begin
        case (hex_digit)
            4'h0: SEG = 8'b11000000; // 0
            4'h1: SEG = 8'b11111001; // 1
            4'h2: SEG = 8'b10100100; // 2
            4'h3: SEG = 8'b10110000; // 3
            4'h4: SEG = 8'b10011001; // 4
            4'h5: SEG = 8'b10010010; // 5
            4'h6: SEG = 8'b10000010; // 6
            4'h7: SEG = 8'b11111000; // 7
            4'h8: SEG = 8'b10000000; // 8
            4'h9: SEG = 8'b10010000; // 9
            4'hA: SEG = 8'b10001000; // A
            4'hB: SEG = 8'b10000011; // b
            4'hC: SEG = 8'b11000110; // C
            4'hD: SEG = 8'b10100001; // d
            4'hE: SEG = 8'b10000110; // E
            4'hF: SEG = 8'b10001110; // F
            default: SEG = 8'b11111111; // 熄滅
        endcase
    end

    // 陽極位選控制 (低電位有效)
    // 我們使用最右邊的 4 個顯示器 (AN0 到 AN3)
    always @(*) begin
        case (digit_select)
            2'd0: AN = 8'b11111110; // 啟動 AN0 (最右邊)
            2'd1: AN = 8'b11111101; // 啟動 AN1
            2'd2: AN = 8'b11111011; // 啟動 AN2
            2'd3: AN = 8'b11110111; // 啟動 AN3
            default: AN = 8'b11111111; // 全部關閉
        endcase
    end

endmodule
