module booth_mult #(
    parameter DATAWIDTH = 16
  )(
    input                        clk,
    input                        rstn,
    input                        en,
    input        [DATAWIDTH-1:0] multiplier,
    input        [DATAWIDTH-1:0] multiplicand,
    output reg                   done,
    output reg [2*DATAWIDTH-1:0] product
  );

  //================================================================
  // 1. 定義暫存器 (Registers)
  //================================================================
  reg [2*DATAWIDTH:0] p_reg;
  reg [DATAWIDTH-1:0] m_reg;
  reg [4:0]           cnt;     // Current Count
  reg                 busy;

  //================================================================
  // 2. 定義組合邏輯訊號 (Wires)
  //================================================================
  wire [DATAWIDTH:0]   acc;
  wire [DATAWIDTH:0]   sum_out;
  wire [DATAWIDTH:0]   operand;
  wire [2*DATAWIDTH:0] p_next;

  // --- 關鍵修改：將計數器的加法移出來 ---
  wire [4:0]           next_cnt;
  wire                 cnt_finished;

  //================================================================
  // 3. 組合邏輯區 (Combinational Logic) - 所有的運算都在這
  //================================================================

  // 3.1 Booth 演算法運算
  assign acc = p_reg[2*DATAWIDTH : DATAWIDTH];

  assign operand =
         (p_reg[1:0] == 2'b01) ? {m_reg[DATAWIDTH-1], m_reg} :
         (p_reg[1:0] == 2'b10) ? (~{m_reg[DATAWIDTH-1], m_reg} + 1'b1) :
         {(DATAWIDTH+1){1'b0}};

  assign sum_out = acc + operand;

  assign p_next = {sum_out[DATAWIDTH], sum_out, p_reg[DATAWIDTH:1]};

  // 3.2 計數器運算 (Counter Logic)
  // 這就是硬體上的 "Adder" 或 "Incrementer"
  assign next_cnt = cnt + 1'b1;

  // 判斷是否結束 (Comparator)
  assign cnt_finished = (cnt == DATAWIDTH - 1);

  //================================================================
  // 4. 循序邏輯區 (Sequential Logic) - 只做 D Flip-Flop 更新
  //================================================================
  always @(posedge clk or negedge rstn)
  begin
    if (!rstn)
    begin
      p_reg   <= 0;
      m_reg   <= 0;
      cnt     <= 0;
      busy    <= 0;
      done    <= 0;
      product <= 0;
    end
    else
    begin
      if (busy)
      begin
        // 這裡沒有任何運算符號 (+ - * /)，只有訊號傳遞
        p_reg <= p_next;     // Update Datapath
        cnt   <= next_cnt;   // Update Counter

        if (cnt_finished)
        begin
          busy    <= 0;
          done    <= 1;
          product <= p_next[2*DATAWIDTH:1];
        end
      end
      else
      begin
        done <= 0;
        if (en)
        begin
          busy    <= 1;
          cnt     <= 0;     // Load initial value
          m_reg   <= multiplicand;
          p_reg   <= { {(DATAWIDTH){1'b0}}, multiplier, 1'b0 };
        end
      end
    end
  end

endmodule
