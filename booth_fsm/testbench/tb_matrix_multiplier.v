`timescale 1ns / 1ps

module tb_matrix_multiplier;

  // --- 參數定義 ---
  localparam DATA_WIDTH    = 16;
  localparam M             = 1;
  localparam K             = 32;
  localparam N             = 10;
  localparam ACC_WIDTH     = 2 * DATA_WIDTH;
  localparam CLK_PERIOD    = 10; // 時脈週期為 10ns (100MHz)

  // --- 測試平台信號 ---
  reg  clk;
  reg  rstn;
  reg  start;
  wire done;
  // --- 新增：用於讀取結果的信號 ---
  reg  [$clog2(M*N)-1:0] result_addr_reg;
  wire [ACC_WIDTH-1:0]  result_out_wire;


  // --- 實例化待測模組 (DUT) ---
  matrix_multiplier #(
                      .DATA_WIDTH(DATA_WIDTH),
                      .M(M),
                      .K(K),
                      .N(N),
                      .A_MEM_INIT_FILE("to_logits_2_input.mem"), // 建議使用 matrix_a.mem
                      .B_MEM_INIT_FILE("to_logits_2_weight.mem"),
                      .BIAS_MEM_INIT_FILE("to_logits_2_bias.mem") // <-- 新增此參數
                    ) dut (
                      .clk(clk),
                      .rstn(rstn),
                      .start(start),
                      .done(done),
                      // --- 新增以下連線 ---
                      .result_addr(result_addr_reg),
                      .result_out(result_out_wire)
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
  // --- 測試流程 ---
  initial
  begin
    // 1. 初始化 & 重置
    rstn = 1'b0;
    start = 1'b0;
    result_addr_reg = 0; // 初始化讀取位址
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

    #(CLK_PERIOD); // 等待一個週期確保最後的寫入完成

    // --- 修改: 使用讀取埠來讀取並顯示結果 ---
    $display("--- Reading Result Matrix C (Size: %0d x %0d) ---", M, N);
    for (i = 0; i < M * N; i = i + 1)
    begin
      result_addr_reg = i;
      #(CLK_PERIOD); // 等待一個週期讓 RAM 讀取資料
      // 顯示十六進位原始值和轉換後的浮點數值 (Q16.16)
      $display("C[%0d] = %h (%f)", i, result_out_wire, $signed(result_out_wire) / 65536.0);
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
