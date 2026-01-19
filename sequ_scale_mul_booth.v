`timescale 1ns/1ps

module sequ_scale_mul_booth(
    input  wire clk,
    input  wire rst_n,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    input  wire       valid_i,
    output wire signed [31:0] ans,
    output reg        valid_o
  );

  localparam HIGH_VOLT = 1'b1;
  localparam LOW_VOLT  = 1'b0;

  // =========================================================================
  // 暫存器定義
  // =========================================================================
  reg [32:0] p_reg; // {Accumulator, Multiplier, Ghost}
  reg [15:0] a_reg;
  reg [4:0]  cnt;

  // =========================================================================
  // FSM 狀態
  // =========================================================================
  reg [1:0] current_state, next_state;
  localparam IDLE = 2'd0, CAL = 2'd1, DONE = 2'd2;

  // =========================================================================
  // 1. 實例化 Booth Encoder
  // =========================================================================

  // 抓取編碼所需的 bits (最低兩位)
  wire [1:0] encoder_in = p_reg[1:0];

  // 宣告控制訊號
  wire op_add, op_sub, op_nop;

  // 呼叫 booth_encoder 模組 (邏輯與之前相同)
  sequ_scale_mul_booth_encoder u_encoder (
                  .code   (encoder_in),
                  .do_add (op_add),
                  .do_sub (op_sub),
                  .do_nop (op_nop)
                );

  // =========================================================================
  // 2. ALU (算術邏輯單元) - 單一加法器架構
  // =========================================================================
  reg signed [15:0] sum_out;
  wire signed [15:0] current_acc = p_reg[32:17];

  // 準備加法器的第二個運算元 (Operand 2)
  reg signed [15:0] alu_op2;

  always @(*)
  begin
    // 這裡不做運算，只做「數據選擇」
    if (op_nop)
      alu_op2 = 16'd0;       // 若是不動作，加 0
    else if (op_sub)
      alu_op2 = ~a_reg;      // 若是減法，取反相 (1's complement)
    else
      alu_op2 = a_reg;       // 若是加法，取原值
  end

  always @(*)
  begin
    // **核心修改**：只有一個加法器
    // 減法原理： A - B = A + (~B) + 1
    // 當 op_sub 為 1 時，它同時充當了 Carry-in 的角色，補足了那個 "+1"
    sum_out = current_acc + alu_op2 + op_sub;
  end

  // =========================================================================
  // 3. 下一階移位邏輯 (Datapath)
  // =========================================================================
  wire [32:0] next_p_reg_val = { sum_out[15], sum_out, p_reg[16:1] };
  assign ans = p_reg[32:1];

  // =========================================================================
  // 4. 下一狀態邏輯 (FSM)
  // =========================================================================
  always @(*)
  begin
    next_state = IDLE;
    case (current_state)
      IDLE:
        next_state = (valid_i) ? CAL : IDLE;
      CAL:
        next_state = (cnt == 5'd15) ? DONE : CAL; // 確保跑滿 16 cycles
      DONE:
        next_state = IDLE;
      default:
        next_state = IDLE;
    endcase
  end

  // =========================================================================
  // 5. 循序邏輯 (Registers)
  // =========================================================================
  always @(posedge clk or negedge rst_n)
  begin
    if(!rst_n)
    begin
      p_reg   <= 33'b0;
      a_reg   <= 16'b0;
      cnt     <= 5'b0;
      valid_o <= LOW_VOLT;
      current_state <= IDLE;
    end
    else
    begin
      current_state <= next_state;

      case (current_state)
        IDLE:
        begin
          if (valid_i)
          begin
            p_reg   <= {16'd0, b, 1'b0};
            a_reg   <= a;
            cnt     <= 5'b0;
            valid_o <= LOW_VOLT;
          end
          else
          begin
            valid_o <= LOW_VOLT;
          end
        end

        CAL:
        begin
          p_reg <= next_p_reg_val;
          cnt   <= cnt + 1'b1;
        end

        DONE:
        begin
          valid_o <= HIGH_VOLT;
          cnt     <= 5'b0;
        end

        default:
          cnt <= 5'b0;
      endcase
    end
  end

endmodule
