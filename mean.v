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
  localparam ACCUM_WIDTH = 24;

  // State machine states
  localparam S_IDLE         = 4'd0;
  localparam S_CLEAR        = 4'd1; // New state to clear the BRAM
  localparam S_ACCUM_READ   = 4'd2; // Read from BRAM for accumulation
  localparam S_ACCUM_WRITE  = 4'd3; // Write back the new sum to BRAM
  localparam S_DIV_READ     = 4'd4; // Read sum from BRAM for division
  localparam S_DIV_CALC     = 4'd5; // Perform division
  localparam S_OUTPUT       = 4'd6; // Output the result
  localparam S_DONE         = 4'd7;

  // Registers
  reg [3:0] state_reg, state_next;

  // Use BRAM style for sum array to save resources
  // Synthesis tools will infer this as a Block RAM
  (* ram_style = "block" *)
  reg signed [ACCUM_WIDTH-1:0] sum [0:FEATURE_COUNT-1];
  reg signed [ACCUM_WIDTH-1:0] sum_read_reg; // Register to hold value read from BRAM

  reg [5:0] patch_cnt_reg, patch_cnt_next;     // Counter for patches (0-48)
  reg [4:0] feature_cnt_reg, feature_cnt_next; // Counter for features (0-31)

  reg signed [15:0] mean_out_reg;
  reg data_valid_out_reg;
  reg done_reg;
  integer i=0;

  // Sequential logic for state and counters
  always @(posedge clk)
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

  // BRAM Write Logic: This block handles all writes to the 'sum' memory.
  // This is the write port of the BRAM.
  always @(posedge clk)
  begin
    if (state_reg == S_CLEAR)
    begin
      sum[feature_cnt_reg] <= 0;
    end
    else if (state_reg == S_ACCUM_WRITE && data_valid_in)
    begin
      sum[feature_cnt_reg] <= sum_read_reg + $signed(data_in);
    end
  end

  // BRAM Read Logic: This block ONLY handles reads from the 'sum' memory.
  // This is the read port of the BRAM. It is kept separate to ensure inference.
  always @(posedge clk)
  begin
    // The read address is feature_cnt_reg. The output is registered into sum_read_reg.
    // This read happens whenever the state is ACCUM_READ or DIV_READ.
    if (state_reg == S_ACCUM_READ || state_reg == S_DIV_READ)
    begin
      sum_read_reg <= sum[feature_cnt_reg];
    end
  end

  // Other Sequential Logic (datapath and output registers)
  always @(posedge clk)
  begin
    // S_DIV_CALC: Perform division on the value read in the previous cycle
    if (state_reg == S_DIV_CALC)
    begin
      // Division is performed serially, using only one hardware divider
      mean_out_reg <= sum_read_reg / PATCH_COUNT;
    end

    // Output registers
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
          // Go to CLEAR state first to zero-out the BRAM
          state_next = S_CLEAR;
          feature_cnt_next = 0; // Start clearing from the first address
        end
      end
      S_CLEAR:
      begin
        // This loop takes FEATURE_COUNT cycles to clear the BRAM
        if (feature_cnt_reg == FEATURE_COUNT - 1)
        begin
          // Clearing finished, now start accumulation
          state_next = S_ACCUM_READ;
          feature_cnt_next = 0; // Reset for accumulation phase
          patch_cnt_next = 0;
        end
        else
        begin
          // Move to the next address to clear
          feature_cnt_next = feature_cnt_reg + 1;
          state_next = S_CLEAR;
        end
      end
      S_ACCUM_READ:
      begin
        // Takes one cycle to read from BRAM.
        // The data_in corresponding to this read is expected in the next cycle.
        state_next = S_ACCUM_WRITE;
      end
      S_ACCUM_WRITE:
      begin
        if (data_valid_in)
        begin
          if (patch_cnt_reg == PATCH_COUNT - 1)
          begin
            patch_cnt_next = 0;
            if (feature_cnt_reg == FEATURE_COUNT - 1)
            begin
              state_next = S_DIV_READ; // All data accumulated, start division phase
              feature_cnt_next = 0;
            end
            else
            begin
              feature_cnt_next = feature_cnt_reg + 1;
              state_next = S_ACCUM_READ; // Move to next feature
            end
          end
          else
          begin
            patch_cnt_next = patch_cnt_reg + 1;
            state_next = S_ACCUM_READ; // Move to next patch
          end
        end
      end
      S_DIV_READ:
      begin
        // Takes one cycle to read from BRAM
        state_next = S_DIV_CALC;
      end
      S_DIV_CALC:
      begin
        // Takes one cycle to calculate
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
          state_next = S_DIV_READ; // Go back to read the next sum
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
