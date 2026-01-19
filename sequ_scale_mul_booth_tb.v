`timescale 1ns/1ps

module sequ_scale_mul_booth_tb();

  // ==========================================
  // 1. 訊號宣告
  // ==========================================
  reg                  clk;
  reg                  rst_n;
  reg  signed [15:0]   a;
  reg  signed [15:0]   b;
  reg                  valid_i;

  wire signed [31:0]   ans;
  wire                 valid_o;

  // 對照組與統計
  reg  signed [31:0]   expected_ans;
  integer              i;
  integer              pass_cnt;
  integer              fail_cnt;

  // ==========================================
  // 2. 實例化 DUT (Device Under Test)
  // ==========================================
  // 請確認這裡的 module 名稱對應你現在要測的版本
  // 如果是 Radix-4，請改成 sequ_scale_mul_booth
  sequ_scale_mul_booth uut (
                         .clk(clk),
                         .rst_n(rst_n),
                         .a(a),
                         .b(b),
                         .valid_i(valid_i),
                         .ans(ans),
                         .valid_o(valid_o)
                       );

  // ==========================================
  // 3. 產生時鐘 (100MHz)
  // ==========================================
  initial
  begin
    clk = 0;
    forever
      #5 clk = ~clk;
  end

  // ==========================================
  // 4. 波形紀錄 (可選)
  // ==========================================
  //initial
  //begin
  //  $fsdbDumpfile("sequ_scale_mul_random.fsdb");
  // $fsdbDumpvars(0, sequ_scale_mul_booth_tb);
  //end

  // ==========================================
  // 5. 主測試流程
  // ==========================================
  initial
  begin
    // --- 系統初始化 ---
    rst_n = 0;
    a = 0;
    b = 0;
    valid_i = 0;
    pass_cnt = 0;
    fail_cnt = 0;

    // --- 重置釋放 ---
    #20;
    @(posedge clk);
    #3;
    rst_n = 1;
    #20;

    $display("==================================================");
    $display("START: Running 100 Random Tests for Booth Multiplier");
    $display("==================================================");

    // --- 迴圈執行 100 次隨機測試 ---
    for (i = 1; i <= 100; i = i + 1)
    begin
      // 1. 產生隨機輸入
      // $random 會產生 32-bit 有號整數，直接 assign 給 16-bit 會自動截斷，
      // 這樣可以自然地測試到正數與負數。
      run_test_case(i, $random, $random);
    end

    // --- 測試結束報告 ---
    $display("==================================================");
    $display("FINAL REPORT");
    $display("Total Tests : 100");
    $display("Passed      : %0d", pass_cnt);
    $display("Failed      : %0d", fail_cnt);
    $display("==================================================");

    if (fail_cnt == 0)
      $display("PERFECT! All random tests passed.");
    else
      $display("WARNING: There were errors.");

    $finish;
  end

  // ==========================================
  // Task: 單次測試流程
  // ==========================================
  task run_test_case;
    input integer idx;          // 測試編號
    input signed [15:0] in_a;   // 輸入 A
    input signed [15:0] in_b;   // 輸入 B
    begin
      // A. 設定輸入
      @(posedge clk);
      #3; // 模擬輸入延遲
      a = in_a;
      b = in_b;
      valid_i = 1;

      // 計算黃金值 (Verilog 原生乘法)
      expected_ans = in_a * in_b;

      // B. 送出 Pulse
      @(posedge clk);
      #3;
      valid_i = 0;

      // C. 等待計算完成
      wait(valid_o);
      #2; // 等待數據穩定

      // D. 檢查結果
      if (ans !== expected_ans)
      begin
        $display("[FAIL] Test #%0d: %d * %d = %d (Expected: %d)", idx, in_a, in_b, ans, expected_ans);
        fail_cnt = fail_cnt + 1;
      end
      else
      begin
        // 若不想看太多訊息，可以把下面這行註解掉
        $display("[PASS] Test #%0d: %d * %d = %d", idx, in_a, in_b, ans);
        pass_cnt = pass_cnt + 1;
      end

      // E. 測試間隔 (讓波形分開一點，方便觀察)
      #10;
    end
  endtask

endmodule
