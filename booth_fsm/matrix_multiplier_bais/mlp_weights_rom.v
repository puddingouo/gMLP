`timescale 1ns / 1ps

// This module acts as a Read-Only Memory (ROM) to store pre-trained weights.
module mlp_weights_rom #(
    parameter ADDR_WIDTH    = 3,
    parameter DATA_WIDTH    = 16,
    parameter MEM_SIZE      = 128,
    parameter MEM_INIT_FILE = "none"  // Memory initialization file
  )
  (
    input                      clk,
    input                      enb, // Read enable
    input  [ADDR_WIDTH-1:0]    addrb, // Read address
    output [DATA_WIDTH-1:0]    doutb  // Read data output
  );

  // Unused ports are tied off
  wire dbiterrb;
  wire sbiterrb;

  xpm_memory_sdpram #(
                      .ADDR_WIDTH_A(ADDR_WIDTH),
                      .ADDR_WIDTH_B(ADDR_WIDTH),
                      .CLOCKING_MODE("common_clock"),
                      .ECC_MODE("no_ecc"),

                      // --- Memory Initialization Configuration ---
                      .MEMORY_INIT_FILE(MEM_INIT_FILE), // Specify the .mem file
                      .USE_MEM_INIT(1),                 // Enable memory initialization
                      .MEMORY_INIT_PARAM("0"),          // Not used when MEMORY_INIT_FILE is set
                      // --- Force Synthesis to use Block RAM ---
                      .MEMORY_PRIMITIVE("block"), // Changed from "auto" to "block"

                      .MEMORY_SIZE(MEM_SIZE),
                      .READ_DATA_WIDTH_B(DATA_WIDTH),
                      .READ_LATENCY_B(1),               // Data will be available 1 clock after address
                      .READ_RESET_VALUE_B("0"),
                      .RST_MODE_B("SYNC"),

                      // --- Make it a ROM ---
                      .WRITE_MODE_B("no_change"),
                      .WRITE_PROTECT(1),                // Set to 1 to protect memory from writes
                      .WRITE_DATA_WIDTH_A(DATA_WIDTH)
                      // Other parameters can be left as default
                    )
                    xpm_memory_sdpram_inst (
                      // Output Ports
                      .doutb(doutb),
                      .dbiterrb(dbiterrb),
                      .sbiterrb(sbiterrb),

                      // Port B: Read Port
                      .clkb(1'b0),       // Unused in common_clock mode
                      .clka(clk),        // Use common clock for both ports
                      .enb(enb),
                      .addrb(addrb),
                      .rstb(1'b0),       // Tie reset to low if not used
                      .regceb(1'b1),     // Always enable output register

                      // Port A: Write Port (Unused for ROM)
                      .ena(1'b0),
                      .wea(1'b0),
                      .addra('b0),
                      .dina('b0),

                      // Other unused inputs
                      .sleep(1'b0),
                      .injectdbiterra(1'b0),
                      .injectsbiterra(1'b0)
                    );

endmodule
