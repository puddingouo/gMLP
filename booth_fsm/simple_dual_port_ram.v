// A generic wrapper for the Xilinx Parameterized Macro (XPM) for a Simple Dual-Port RAM.
// This wrapper simplifies the instantiation by exposing only the most common parameters
// and hiding the complex configuration details of the XPM primitive.
// Port A: Write-Only
// Port B: Read-Only
module simple_dual_port_ram #(
    parameter DATA_WIDTH    = 16,                                                               // Width of the data bus
    parameter ADDR_WIDTH    = 3,                                                                // Number of address bits (determines memory depth)
    parameter MEM_INIT_FILE = "d:/Lab/gMLP/Image_Classification/fpga_gmlp_mnist/HDL/linear_layer/mem/test_weight.mem" // Memory initialization file (.mem)
  ) (
    // Common
    input clk,

    // Write Port (Port A)
    input                       wea,       // Write Enable
    input [ADDR_WIDTH-1:0]      addra,     // Write Address
    input [DATA_WIDTH-1:0]      dina,      // Data In

    // Read Port (Port B)
    input                       enb,       // Read Enable
    input [ADDR_WIDTH-1:0]      addrb,     // Read Address
    output [DATA_WIDTH-1:0]     doutb      // Data Out
  );

  localparam MEM_DEPTH = 1 << ADDR_WIDTH; // Calculate memory depth from address width

  // Instantiate the Xilinx Simple Dual-Port RAM primitive
  xpm_memory_sdpram #(
                      // Essential Parameters derived from wrapper parameters
                      .MEMORY_SIZE(MEM_DEPTH * DATA_WIDTH), // Total memory size in bits
                      .ADDR_WIDTH_A(ADDR_WIDTH),
                      .ADDR_WIDTH_B(ADDR_WIDTH),
                      .WRITE_DATA_WIDTH_A(DATA_WIDTH),
                      .READ_DATA_WIDTH_B(DATA_WIDTH),

                      // Memory Initialization
                      .MEMORY_INIT_FILE(MEM_INIT_FILE), // File to initialize memory from
                      .MEMORY_OPTIMIZATION("true"),     // Power and area optimization
                      .MEMORY_PRIMITIVE("block"),       // Force implementation using BRAM
                      .USE_MEM_INIT(1),                 // Enable memory initialization from file
                      .MEMORY_INIT_PARAM(""),           // Must be empty when using MEM_INIT_FILE

                      // Fixed Configuration for this wrapper
                      .CLOCKING_MODE("common_clock"),   // Use a single clock for both ports
                      .READ_LATENCY_B(1),               // Minimum latency for BRAM
                      .WRITE_MODE_B("no_change"),       // Port B is read-only
                      .RST_MODE_B("SYNC"),
                      .ECC_MODE("no_ecc")
                    )
                    xpm_memory_inst (
                      // Connect ports
                      .clka(clk),
                      .clkb(clk),

                      // Port A (Write)
                      .ena(1'b1), // Port A is always listening, controlled by wea
                      .wea(wea),
                      .addra(addra),
                      .dina(dina),

                      // Port B (Read)
                      .enb(enb),
                      .addrb(addrb),
                      .doutb(doutb),

                      // Unused ports
                      .dbiterrb(),
                      .sbiterrb(),
                      .injectdbiterra(1'b0),
                      .injectsbiterra(1'b0),
                      .regceb(1'b1),
                      .rstb(1'b0),
                      .sleep(1'b0)
                    );

endmodule
