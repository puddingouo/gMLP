`timescale 1ns / 1ps

module mean_tb;

  // Parameters
  localparam CLK_PERIOD = 10;
  localparam PATCH_COUNT = 49;
  localparam FEATURE_COUNT = 32;
  localparam TOTAL_INPUTS = PATCH_COUNT * FEATURE_COUNT;

  // DUT signals
  reg clk;
  reg rst;
  reg start;
  reg data_valid_in;
  reg [15:0] data_in;
  wire done;
  wire data_valid_out;
  wire [15:0] mean_out;

  // Testbench internal signals
  reg [15:0] mem [0:TOTAL_INPUTS-1];
  integer input_idx;
  integer output_file;

  // Instantiate the DUT
  mean dut (
         .clk(clk),
         .rst(rst),
         .start(start),
         .data_valid_in(data_valid_in),
         .data_in(data_in),
         .done(done),
         .data_valid_out(data_valid_out),
         .mean_out(mean_out)
       );

  // Clock generator
  always #(CLK_PERIOD / 2) clk = ~clk;
  integer i,j;
  // Main test sequence
  initial
  begin
    // 1. Initialization and file loading
    clk = 0;
    rst = 1;
    start = 0;
    data_valid_in = 0;
    data_in = 0;
    input_idx = 0;

    $readmemh("input.mem", mem);
    output_file = $fopen("output_mean.txt", "w");
    if (output_file == 0)
    begin
      $display("Error: Could not open output file.");
      $finish;
    end

    // 2. Reset sequence
    # (CLK_PERIOD * 2);
    rst = 0;
    # (CLK_PERIOD);

    // 3. Start the process
    start = 1;
    #CLK_PERIOD;
    start = 0;

    $display("TB: Starting data input stream...");

    // 4. Feed input data
    // The data in memory is stored as [patch0_feat0, patch0_feat1, ...],
    // but the DUT expects [patch0_feat0, patch1_feat0, ...].
    // We need to reorder it.
    for (j = 0; j < FEATURE_COUNT; j = j + 1)
    begin
      for (i = 0; i < PATCH_COUNT; i = i + 1)
      begin
        wait(dut.state_reg == 1); // Wait until DUT is in ACCUM state
        data_valid_in = 1;
        data_in = mem[i * FEATURE_COUNT + j];
        #CLK_PERIOD;
      end
    end
    data_valid_in = 0;

    $display("TB: All input data sent. Waiting for completion...");

    // 5. Wait for 'done' signal
    wait(done);
    $display("TB: 'done' signal received. Simulation finished.");

    // 6. Finalize
    # (CLK_PERIOD * 5);
    $fclose(output_file);
    $finish;
  end

  // Monitor and save output
  always @(posedge clk)
  begin
    if (data_valid_out)
    begin
      // Display Q8.8 value as both hex and real number for verification
      $display("TB: Received mean value: %h (Real: %f)", mean_out, $signed(mean_out) / 256.0);
      $fdisplay(output_file, "%h", mean_out);
    end
  end

endmodule
