module sequ_scale_mul_booth_encoder (
    input  wire [1:0] code,    // {current_bit, ghost_bit}
    output reg        do_add,
    output reg        do_sub,
    output reg        do_nop
  );
  always @(*)
  begin
    do_add = 1'b0;
    do_sub = 1'b0;
    do_nop = 1'b0;
    case (code)
      2'b00:
        do_nop = 1'b1; // 0 -> 0
      2'b01:
        do_add = 1'b1; // 0 -> 1 (+A)
      2'b10:
        do_sub = 1'b1; // 1 -> 0 (-A)
      2'b11:
        do_nop = 1'b1; // 1 -> 1
      default:
        do_nop = 1'b1;
    endcase
  end
endmodule
