`timescale 1ns/1ps

module top_booth_tb;

  // Parameters
  localparam DATA_WIDTH = 16;
  localparam ADDR_WIDTH = 3;

  // Testbench signals
  reg clk;
  reg rstn;
  reg en;
  reg [DATA_WIDTH-1:0] multiplier;
  reg [ADDR_WIDTH-1:0] rom_addr;
  wire done;
  wire [2*DATA_WIDTH-1:0] product;

  // Instantiate the Device Under Test (DUT)
  top_booth #(
              .DATA_WIDTH(DATA_WIDTH)
            ) dut (
              .clk(clk),
              .rstn(rstn),
              .en(en),
              .multiplier(multiplier),
              .rom_addr(rom_addr),
              .done(done),
              .product(product)
            );

  // Clock generation
  always #5 clk = ~clk;

  // Test sequence
  integer i;
  // 宣告臨時變數來儲存顯示用的值
  reg [2*DATA_WIDTH-1:0] temp_product;
  reg [ADDR_WIDTH-1:0] temp_rom_addr;
  reg [DATA_WIDTH-1:0] temp_multiplier;
  real real_product; // <-- 新增：宣告一個實數變數
  initial
  begin
    // Initialize signals
    clk = 1;
    rstn = 1;
    en = 0;
    multiplier = 0;
    rom_addr = 0;

    // Apply reset
    #10;
    rstn = 0;
    #10;
    rstn = 1;
    #10;

    $display("Simulation Start: Iterating through weights in ROM...");

    // Loop through all addresses in the ROM
    for (i = 0; i < (1 << ADDR_WIDTH); i = i + 1)
    begin
      // 1. 設定位址以從 ROM 讀取資料
      rom_addr <= i;
      // 假設 multiplier 代表 1.0，其 Q8.8 表示法為 256
      multiplier <= 256; // 將 multiplier 設為 1 到 8
      en <= 0;             // 確保 booth fsm 保持在 IDLE

      // 2. 等待 ROM 的讀取延遲 (Read Latency = 2)
      //    至少等待 2 個時脈週期，讓 multiplicand 穩定
      repeat (2) @(posedge clk);

      // 3. 啟動 Booth 乘法器
      en <= 1;
      // 等待一個時脈邊緣，確保 FSM 進入運算狀態且 done 已被拉低
      @(posedge clk);
      en <= 0;
      // 4. 等待 FSM 發出完成信號
      wait (done == 1);

      // 6. 顯示儲存的結果
      @(posedge clk);
      // ... 儲存結果到 temp_... 變數 ...
      temp_product = product;
      temp_rom_addr = rom_addr;
      temp_multiplier = multiplier;

      // 將 Q16.16 結果轉換為實數
      real_product = $signed(temp_product) / 65536.0;

      // 顯示所有結果
      $display("Test Case %0d: ROM Addr=%h, Multiplier=%h, Product=%h, Decimal_Result=%f",
               i, temp_rom_addr, temp_multiplier, temp_product, real_product);
    end

    $display("Simulation Finished.");
    #100;
    $finish;
  end

endmodule
