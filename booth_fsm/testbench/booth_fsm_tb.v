`timescale 1ns/1ps

// Basic exhaustive self checking test bench.
`define TEST_WIDTH 16  //<-- 修改?�� 16 以匹??��?��?��?��?�寬�?
`define MEM_DEPTH 8    //<-- 定義記憶體深�? (檔�?�中??��?��?��?�數)

module booth_fsm_tb;

  reg clk;
  reg rstn;
  reg en;
  reg [`TEST_WIDTH-1:0] multiplier;
  reg [`TEST_WIDTH-1:0] multiplicand;
  wire    done;

  // Memory to store weights from file
  reg [`TEST_WIDTH-1:0] weights_mem[0:`MEM_DEPTH-1];

  //输入 ：�?��?��?��?�符?��??�符?��，�?�出：�?��?��??
  wire signed [2*`TEST_WIDTH-1:0] product;
  wire signed [`TEST_WIDTH-1:0]   m1_in;
  wire signed [`TEST_WIDTH-1:0]   m2_in;

  reg  signed [2*`TEST_WIDTH-1:0] product_ref;

  assign m1_in = multiplier;
  assign m2_in = multiplicand;

  booth_fsm #(.DATAWIDTH(`TEST_WIDTH)) booth
            (
              .clk(clk),
              .rstn(rstn),
              .en(en),
              .multiplier(multiplier),
              .multiplicand(multiplicand),
              .done  (done),
              .product(product)
            );

  always #5 clk = ~clk; // <-- 建議使用 5ns 以產??? 100MHz ??��??

  integer i;
  initial
  begin
    // Load memory from file
    // ??�設 test_weight.mem ?��??��??��?��?�夾
    $readmemh("E:/Lab/gMLP/Image_Classification/fpga_gmlp_mnist/HDL/booth_fsm/test_weight.mem", weights_mem);

    clk = 1;
    en = 0;
    rstn = 1;
    #10 rstn = 0;
    #10 rstn = 1;

    multiplier=0;
    multiplicand=0;
    #10;

    for(i=0; i < `MEM_DEPTH; i = i + 1)
    begin
      en = 1;
      multiplier   <= i; // 給�?��??? 0 ?�� 7 ??? multiplier
      multiplicand <= weights_mem[i]; // 從�?�憶體�???? multiplicand

      @(posedge clk);
      en <= 0;

      wait (done == 1);

      product_ref = $signed(multiplier) * $signed(multiplicand);

      if (product_ref !== product)
      begin
        $display("ERROR: i=%0d, multiplier=%d, multiplicand=%d, product=%d, expected=%d",
                 i, $signed(multiplier), $signed(multiplicand), $signed(product), product_ref);
      end
      else
      begin
        $display("PASS: i=%0d, multiplier=%d, multiplicand=%d, product=%d",
                 i, $signed(multiplier), $signed(multiplicand), $signed(product));
      end

      @(posedge clk);
    end
    $display("Simulation done.");
    #100;
    $finish;
  end

endmodule
