`timescale 1ns/1ps

module booth_mult_tb;

  // 參數設定 (需與 booth_multiplier_pure_hw.v 保持一致)
  parameter DATAWIDTH = 16;
  parameter CYCLE = 10;     // 設定時脈週期

  // 1. 宣告訊號
  // 輸入用 reg
  reg clk;
  reg rstn;
  reg en;

  // 使用 signed 方便隨機生成負數與計算 Golden Answer 進行比對
  // 注意：DUT 的 port 雖然沒寫 signed，但 Verilog 會自動對接 bit
  reg signed [DATAWIDTH-1:0] multiplier;
  reg signed [DATAWIDTH-1:0] multiplicand;

  // 輸出用 wire
  wire done;
  wire signed [2*DATAWIDTH-1:0] product;

  // 用於驗證的黃金值 (Golden Answer)
  reg signed [2*DATAWIDTH-1:0] expected_product;

  integer i;

  // 2. 實例化 (Instantiate) 你的 "純硬體架構" 模組
  booth_multiplier_pure_hw #(
                             .DATAWIDTH(DATAWIDTH)
                           ) u_dut (
                             .clk(clk),
                             .rstn(rstn),
                             .en(en),
                             .multiplier(multiplier),
                             .multiplicand(multiplicand),
                             .done(done),
                             .product(product)
                           );

  // 時脈產生 (Clock Generation)
  initial
    clk = 0;
  always #(CYCLE/2) clk = ~clk;

  // 3. 測試流程
  initial
  begin
    // --- 波形錄製設定 (針對 PrimeTime/Verdi 優化) ---
    // 修改檔名以區別於舊的 FSM 版本
    $fsdbDumpfile("booth_pure_hw.fsdb");
    // ★ 重點: 這裡直接 Dump 整個 TB 層級
    $fsdbDumpvars(0, booth_pure_hw_tb);

    // --- *** SDF 反標設定 (Gate-level 用) *** ---
    // 若有做 Gate-level simulation 再解開以下註解
    // $sdf_annotate("Netlist/booth_pure_hw.sdf", u_dut);

    // --- 初始化訊號 ---
    rstn = 1;
    en = 0;
    multiplier = 0;
    multiplicand = 0;

    // --- Reset 系統 ---
    #(CYCLE) rstn = 0;  // 拉低 Reset
    #(CYCLE) rstn = 1;  // 釋放 Reset
    #(CYCLE);

    // --- 開始測試 ---
    $display("=== Simulation Start (Pure Hardware Architecture) ===");

    for(i = 0; i < 100; i = i + 1)
    begin
      // 隨機產生輸入資料 (限制範圍避免 overflow，雖然 2*DATAWIDTH 不會爆，但方便觀察)
      // 使用 $random 生成有正有負的數值
      multiplier   = $random % 2048;
      multiplicand = $random % 2048;

      // 計算預期結果 (Golden) - Testbench 裡的 * 是軟體行為，作為標準答案
      expected_product = multiplier * multiplicand;

      // 啟動訊號 (Handshake: Start)
      @(negedge clk); // 對齊時脈下降緣送資料 (Best Practice)
      en = 1;

      @(negedge clk); // 一個 cycle 後把 enable 拉低
      en = 0;

      // 等待運算完成 (Handshake: Wait for Done)
      // 這裡會自動等待硬體算出結果
      wait(done);

      // 稍作延遲確保數據穩定 (Optional，但在波形上比較好讀)
      #(CYCLE/2);

      // 顯示與檢查結果
      // 這裡使用 %d 顯示十進位，%h 顯示十六進位
      if (product !== expected_product)
      begin
        $display("[ERROR] Test %0d Failed!", i);
        $display("  Inputs: %d * %d", multiplier, multiplicand);
        $display("  Output: %d (Hex: %h)", product, product);
        $display("  Expect: %d (Hex: %h)", expected_product, expected_product);
        // 遇到錯誤可以選擇 $stop 暫停查看波形
        // $stop;
      end
      else
      begin
        $display("Test %0d Pass: %d * %d = %d", i, multiplier, multiplicand, product);
      end

      // 等待幾個 Cycle 再進行下一筆測試，讓波形間隔開來，比較好讀
      #(CYCLE*2);
    end

    $display("=== Simulation End ===");
    $finish; // 結束模擬
  end

  // Timeout 保護機制 (防止硬體邏輯錯誤導致 Done 永遠不拉起，卡死模擬)
  initial
  begin
    #100000;
    $display("=== Timeout: Simulation forced to stop (Check your 'done' signal) ===");
    $finish;
  end

endmodule
