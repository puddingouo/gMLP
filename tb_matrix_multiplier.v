`timescale 1ns / 1ps

module tb_matrix_multiplier;

  // --- 參數定義 ---
  localparam DATA_WIDTH    = 16;
  localparam M             = 1;
  localparam K             = 32;
  localparam N             = 10;
  // ★★★ 關鍵修改：ACC_WIDTH 必須與 DUT 的定義匹配 ★★★
  localparam ACC_WIDTH     = 2 * DATA_WIDTH + $clog2(K);
  localparam CLK_PERIOD    = 10; // 時脈週期為 10ns (100MHz)

  // --- 測試平台信號 ---
  reg  clk;
  reg  rstn;
  reg  start;
  wire done;
  // --- 修改：移除 result_addr_reg，宣告一個寬向量來接收整個結果 ---
  wire [M*N*ACC_WIDTH-1:0] result_out_vector;


  // --- 實例化待測模組 (DUT) ---
  // --- 修改：移除 MEM_INIT_FILE 參數，更新 result_out 連線 ---
  matrix_multiplier #(
                      .DATA_WIDTH(DATA_WIDTH),
                      .M(M),
                      .K(K),
                      .N(N)
                    ) dut (
                      .clk(clk),
                      .rstn(rstn),
                      .start(start),
                      .done(done),
                      .result_out(result_out_vector)
                    );

  // --- 時脈產生 ---
  always
  begin
    clk = 1'b0;
    #(CLK_PERIOD / 2);
    clk = 1'b1;
    #(CLK_PERIOD / 2);
  end

  integer i; // 用於迴圈的整數
  reg signed [ACC_WIDTH-1:0] current_result;
  // --- 測試流程 ---
  initial
  begin
    // 1. 初始化 & 重置
    rstn = 1'b0;
    start = 1'b0;
    $display("T=%0t: System reset asserted.", $time);
    #(CLK_PERIOD * 5);

    rstn = 1'b1;
    $display("T=%0t: System reset released.", $time);
    #(CLK_PERIOD);

    // 2. 發出 start 訊號
    start = 1'b1;
    $display("T=%0t: Start signal asserted.", $time);
    #(CLK_PERIOD);
    start = 1'b0;

    // 3. 等待運算完成
    $display("T=%0t: Waiting for done signal...", $time);
    wait (done == 1'b1);
    $display("T=%0t: Matrix multiplication done.", $time);

    #(CLK_PERIOD); // 等待一個週期確保訊號穩定

    // --- 修改: 從寬向量 result_out_vector 中提取並顯示結果 ---
    $display("--- Reading Result Matrix C (Size: %0d x %0d) ---", M, N);
    for (i = 0; i < M * N; i = i + 1)
    begin
      // 從 result_out_vector 中提取第 i 個元素
      // 宣告一個暫存器來儲存提取出的值

      current_result = result_out_vector[i*ACC_WIDTH +: ACC_WIDTH];

      // 顯示十六進位原始值和轉換後的浮點數值 (假設為 Q16.16)
      $display("C[%0d] = %h (%f)", i, current_result, current_result / 65536.0);
    end
    $display("------------------------------------");


    #(CLK_PERIOD * 10);

    // 4. 結束模擬
    $display("T=%0t: Simulation finished.", $time);
    $finish;
  end

  // --- 修改：詳細過程監控 ---
  reg [8*10:1] current_state_str; // 用於顯示狀態名稱的字串

  always @(*)
  begin
    // 將狀態機的數值轉換為可讀的字串
    case (dut.state)
      4'd0:
        current_state_str = "IDLE";
      4'd1:
        current_state_str = "FETCH";
      4'd2:
        current_state_str = "MULTIPLY";
      4'd3:
        current_state_str = "ACCUMULATE";
      4'd4:
        current_state_str = "ADD_BIAS"; // <-- 新增
      4'd5:
        current_state_str = "STORE";
      4'd6:
        current_state_str = "FINISH";
      default:
        current_state_str = "UNKNOWN";
    endcase
  end

  always @(posedge clk)
  begin
    // 在重置結束後開始顯示詳細資訊
    if (rstn && !done)
    begin
      $display("T=%0t | State: %s | i,j,k: %d,%d,%d | a_addr:%h b_addr:%h | a_data:%h b_data:%h | product:%h | acc_reg:%h | bias_data:%h",
               $time,
               current_state_str,
               dut.i_reg, dut.j_reg, dut.k_reg,
               dut.a_addr, dut.b_addr,
               dut.a_data, dut.b_data,
               dut.booth_product,
               dut.acc_reg,
               dut.bias_data_16b // <-- 新增監控 bias_data
              );
    end
  end

endmodule
