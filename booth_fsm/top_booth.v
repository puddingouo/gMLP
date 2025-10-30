module top_booth #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 3,
    parameter MEM_INIT_FILE = "test_weight.mem"
  ) (
    input clk,
    input rstn,
    input en,
    input [DATA_WIDTH-1:0] multiplier,
    input [ADDR_WIDTH-1:0] rom_addr,
    output done,
    output [2*DATA_WIDTH-1:0] product
  );

  localparam MEM_SIZE = (1 << ADDR_WIDTH) * DATA_WIDTH;

  // Wires for connecting the ROM output to the booth_fsm input
  wire [DATA_WIDTH-1:0] multiplicand;
  wire rom_enb;

  reg [1:0] en_delay_reg; // 用 2 段 D 鎖存器來延遲 en 訊號
  wire      fsm_en;       // 使能給 FSM 使用的正確訊號

  always @(posedge clk)
  begin
    if (!rstn)
    begin
      en_delay_reg <= 2'b00;
    end
    else
    begin
      en_delay_reg <= {en_delay_reg[0], en}; // 將 en 訊號移入延遲暫存器
    end
  end

  // en 訊號被延遲 2 個時脈週期產生給 FSM 使用的訊號
  // ROM 讀取也需要延遲 2 個時脈週期
  assign fsm_en = en_delay_reg[1];
  assign rom_enb = en; // ROM 使能仍然由外部 en 信號觸發

  // Instantiate the mlp_weights_rom module
  // 將權重記憶體映射到 top_booth 模組中
  mlp_weights_rom #(
                    .ADDR_WIDTH(ADDR_WIDTH),
                    .DATA_WIDTH(DATA_WIDTH),
                    .MEM_SIZE(MEM_SIZE),
                    .MEM_INIT_FILE(MEM_INIT_FILE)
                  ) rom_inst (
                    .clk(clk),
                    .enb(rom_enb),
                    .addrb(rom_addr),
                    .doutb(multiplicand)
                  );

  // Instantiate the booth_fsm module
  booth_fsm #(
              .DATAWIDTH(DATA_WIDTH)
            ) booth_fsm_inst (
              .clk(clk),
              .rstn(rstn),
              .en(fsm_en),
              .multiplier(multiplier),
              .multiplicand(multiplicand),
              .done(done),
              .product(product)
            );

  // Logic to control the ROM enable and address (example logic)
  

endmodule
