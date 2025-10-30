`timescale 1ns / 1ps

module tb_matrix_multiplier;

  // --- 參數定義 ---
  localparam DATA_WIDTH    = 16;
  localparam M             = 2;
  localparam K             = 4;
  localparam N             = 8;
  localparam CLK_PERIOD    = 10; // 時脈週期為 10ns (100MHz)

  // --- 測試平台信號 ---
  reg  clk;
  reg  rstn;
  reg  start;
  wire done;

  // --- 實例化待測模組 (DUT) ---
  matrix_multiplier #(
                      .DATA_WIDTH(DATA_WIDTH),
                      .M(M),
                      .K(K),
                      .N(N),
                      .A_MEM_INIT_FILE("matrix_a.mem"),
                      .B_MEM_INIT_FILE("matrix_b.mem")
                    ) dut (
                      .clk(clk),
                      .rstn(rstn),
                      .start(start),
                      .done(done)
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

    // 新增: 讀取並顯示結果
    $display("--- Reading Result Matrix C (Size: %0d x %0d) ---", M, N);
    // 使用階層式參考直接存取 DUT 內部的 RAM
    // 路徑: dut -> matrix_c_ram -> xpm_memory_inst -> xpm_memory_base_inst -> mem
    for (i = 0; i < M * N; i = i + 1)
    begin
      // 顯示十六進位原始值和轉換後的浮點數值
      $display("C[%0d] = %h (%f)", i, dut.matrix_c_ram.xpm_memory_inst.xpm_memory_base_inst.mem[i], $signed(dut.matrix_c_ram.xpm_memory_inst.xpm_memory_base_inst.mem[i]) / 65536.0);
    end
    $display("------------------------------------");


    #(CLK_PERIOD * 10);

    // 4. 結束模擬
    $display("T=%0t: Simulation finished.", $time);
    $finish;
  end

  // --- 新增：詳細過程監控 ---
  reg [8*10:1] current_state_str; // 用於顯示狀態名稱的字串

  always @(*)
  begin
    // 將狀態機的數值轉換為可讀的字串
    case (dut.state)
      3'd0:
        current_state_str = "IDLE";
      3'd1:
        current_state_str = "FETCH";
      3'd2:
        current_state_str = "MULTIPLY";
      3'd3:
        current_state_str = "ACCUMULATE";
      3'd4:
        current_state_str = "STORE";
      3'd5:
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
      $display("T=%0t | State: %s | i,j,k: %d,%d,%d | a_addr:%h b_addr:%h | a_data:%h b_data:%h | product:%h | acc_reg:%h",
               $time,
               current_state_str,
               dut.i_reg, dut.j_reg, dut.k_reg,
               dut.a_addr, dut.b_addr,
               dut.a_data, dut.b_data,
               dut.booth_product,
               dut.acc_reg
              );
    end
  end

endmodule
