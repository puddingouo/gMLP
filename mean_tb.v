`timescale 1ns / 1ps

module mean_tb;

  // Parameters from the DUT
  localparam PATCH_COUNT   = 49;
  localparam FEATURE_COUNT = 32;
  localparam DATA_WIDTH    = 16;
  localparam TOTAL_INPUTS  = PATCH_COUNT * FEATURE_COUNT;

  // Testbench signals
  reg clk;
  reg rst;
  reg start;
  reg data_valid_in;
  reg [DATA_WIDTH-1:0] data_in;

  wire done;
  wire data_valid_out;
  wire [DATA_WIDTH-1:0] mean_out;

  // Instantiate the Device Under Test (DUT)
  mean uut (
         .clk(clk),
         .rst(rst),
         .start(start),
         .data_valid_in(data_valid_in),
         .data_in(data_in),
         .done(done),
         .data_valid_out(data_valid_out),
         .mean_out(mean_out)
       );

  // Clock generation
  initial
  begin
    clk = 0;
    forever
      #5 clk = ~clk; // 100MHz clock
  end

  // Memory to hold input data from file
  reg [DATA_WIDTH-1:0] input_memory [0:TOTAL_INPUTS-1];

  // Array to store output results
  reg signed [DATA_WIDTH-1:0] result_memory [0:FEATURE_COUNT-1];
  integer result_idx;
  integer output_file;

  // Main test sequence
  initial
  begin
    // 1. Load input data from file
    $readmemh("input.mem", input_memory);

    // 2. Initialize signals and apply reset
    $display("[$time] Starting Testbench...");
    rst = 1;
    start = 0;
    data_valid_in = 0;
    data_in = 0;
    result_idx = 0;
    #20;
    rst = 0;
    #10;
    $display("[$time] Reset released.");

    // 3. Start the DUT
    start = 1;
    @(posedge clk);
    start = 0;
    $display("[$time] Start signal asserted. Feeding data...");

    // 4. Feed data to the DUT
    // The DUT expects data in feature-major order:
    // (feature 0, patch 0), (feature 0, patch 1), ..., (feature 0, patch 48)
    // (feature 1, patch 0), (feature 1, patch 1), ..., (feature 1, patch 48)
    // ...
    // The input.mem file is in patch-major order. We need to re-order it.
    for (integer f = 0; f < FEATURE_COUNT; f = f + 1)
    begin
      for (integer p = 0; p < PATCH_COUNT; p = p + 1)
      begin
        // Wait for the read state before providing data
        wait (uut.state_reg == uut.S_ACCUM_READ);
        @(posedge clk);

        // Now in write state, provide data
        data_valid_in <= 1;
        // The index is calculated based on how input.mem is structured (patch-major)
        data_in <= input_memory[p * FEATURE_COUNT + f];
        @(posedge clk);
        data_valid_in <= 0;
      end
      $display("[$time] Finished feeding data for feature %0d.", f);
    end

    $display("[$time] All data has been sent. Waiting for 'done' signal.");

    // 5. Wait for calculation to complete
    wait (done);
    $display("[$time] 'done' signal received. Simulation finished.");

    // 6. Display results and write to file
    output_file = $fopen("output_mean.txt", "w");
    $display("\n--- Calculated Mean Values ---");
    $fdisplay(output_file, "--- Calculated Mean Values ---");
    for (integer i = 0; i < FEATURE_COUNT; i = i + 1)
    begin
      $display("Feature %2d Mean: %h (%d)", i, result_memory[i], result_memory[i]);
      $fdisplay(output_file, "Feature %2d Mean: %h (%d)", i, result_memory[i], result_memory[i]);
    end
    $fclose(output_file);


    // 7. Stop simulation
    $stop;
  end

  // Task to capture output data
  always @(posedge clk)
  begin
    if (data_valid_out)
    begin
      if (result_idx < FEATURE_COUNT)
      begin
        result_memory[result_idx] = mean_out;
        result_idx = result_idx + 1;
      end
    end
  end

endmodule
