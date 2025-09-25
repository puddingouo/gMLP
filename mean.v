module mean (
    input wire          clk,
    input wire          rst,
    input wire          start,          // Start the mean calculation
    input wire          data_valid_in,  // Indicates data_in is valid
    input wire  [15:0]  data_in,        // Input data (Q8.8 fixed-point format)
    output wire         done,           // Calculation finished
    output wire         data_valid_out, // Indicates mean_out is valid
    output wire [15:0]  mean_out        // Output mean value (Q8.8 fixed-point format)
  );

  // Parameters
  localparam PATCH_COUNT = 49;
  localparam FEATURE_COUNT = 32;
  // Accumulator width for Q16.8 format.
  // 16 (integer bits) = 7 (original integer) + 6 (log2(49)) + 3 (safety margin)
  // 8 (fractional bits) to match input. Total = 24.
  localparam ACCUM_WIDTH = 24;

  // State machine states
  localparam S_IDLE     = 3'd0;
  localparam S_ACCUM    = 3'd1;
  localparam S_DIV      = 3'd2;
  localparam S_OUTPUT   = 3'd3;
  localparam S_DONE     = 3'd4;

  // Registers
  reg [2:0] state_reg, state_next;
  // Sum accumulator in Q16.8 format
  reg signed [ACCUM_WIDTH-1:0] sum [0:FEATURE_COUNT-1];
  reg [5:0] patch_cnt_reg, patch_cnt_next;     // Counter for patches (0-48)
  reg [4:0] feature_cnt_reg, feature_cnt_next; // Counter for features (0-31)
  // Mean register in Q8.8 format
  reg signed [15:0] mean_reg [0:FEATURE_COUNT-1];
  reg signed [15:0] mean_out_reg;
  reg data_valid_out_reg;
  reg done_reg;

  // Sequential logic for state and counters
  always @(posedge clk or posedge rst)
  begin
    if (rst)
    begin
      state_reg <= S_IDLE;
      patch_cnt_reg <= 0;
      feature_cnt_reg <= 0;
    end
    else
    begin
      state_reg <= state_next;
      patch_cnt_reg <= patch_cnt_next;
      feature_cnt_reg <= feature_cnt_next;
    end
  end

  // Sequential logic for data processing
  always @(posedge clk)
  begin
    if (rst)
    begin
      for (integer i = 0; i < FEATURE_COUNT; i = i + 1)
      begin
        sum[i] <= 0;
      end
    end
    else
    begin
      // S_IDLE: Clear sums when starting
      if (state_next == S_IDLE)
      begin
        for (integer i = 0; i < FEATURE_COUNT; i = i + 1)
        begin
          sum[i] <= 0;
        end
      end
      // S_ACCUM: Accumulate input data. Fixed-point addition is same as integer addition.
      else if (state_reg == S_ACCUM && data_valid_in)
      begin
        sum[feature_cnt_reg] <= sum[feature_cnt_reg] + $signed(data_in);
      end
      // S_DIV: Calculate all means.
      else if (state_reg == S_DIV)
      begin
        for (integer i = 0; i < FEATURE_COUNT; i = i + 1)
        begin
          // For fixed-point Qm.n, division by an integer constant k is simply (value / k).
          // Here, sum (Q16.8) / 49 results in a mean value in Q16.8 format.
          // The result is then truncated to 16 bits to fit into mean_reg (Q8.8).
          mean_reg[i] <= sum[i] / PATCH_COUNT;
        end
      end
    end

    // Output registers
    mean_out_reg <= (state_reg == S_OUTPUT) ? mean_reg[feature_cnt_reg] : 16'd0;
    data_valid_out_reg <= (state_reg == S_OUTPUT);
    done_reg <= (state_reg == S_DONE);
  end

  // Combinational logic for state transitions and counter updates
  always @(*)
  begin
    state_next = state_reg;
    patch_cnt_next = patch_cnt_reg;
    feature_cnt_next = feature_cnt_reg;

    case (state_reg)
      S_IDLE:
      begin
        if (start)
        begin
          state_next = S_ACCUM;
        end
      end
      S_ACCUM:
      begin
        if (data_valid_in)
        begin
          if (patch_cnt_reg == PATCH_COUNT - 1 && feature_cnt_reg == FEATURE_COUNT - 1)
          begin
            state_next = S_DIV;
            patch_cnt_next = 0;
            feature_cnt_next = 0;
          end
          else if (patch_cnt_reg == PATCH_COUNT - 1)
          begin
            patch_cnt_next = 0;
            feature_cnt_next = feature_cnt_reg + 1;
          end
          else
          begin
            patch_cnt_next = patch_cnt_reg + 1;
          end
        end
      end
      S_DIV:
      begin
        // This state takes one cycle to calculate all means
        state_next = S_OUTPUT;
      end
      S_OUTPUT:
      begin
        if (feature_cnt_reg == FEATURE_COUNT - 1)
        begin
          state_next = S_DONE;
        end
        else
        begin
          feature_cnt_next = feature_cnt_reg + 1;
        end
      end
      S_DONE:
      begin
        // Stay in DONE until reset
        state_next = S_DONE;
      end
      default:
      begin
        state_next = S_IDLE;
      end
    endcase
  end

  // Assign outputs
  assign mean_out = mean_out_reg;
  assign data_valid_out = data_valid_out_reg;
  assign done = done_reg;

endmodule
