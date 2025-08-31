`timescale 1 ns / 1 ps

module vmodel (
    input clk,
    input [6271:0] inp,
    output reg [31653:0] out
);

    reg [6272-1:0] stage0_inp;
    reg [224812-1:0] stage1_inp;
    wire [224812-1:0] stage0_out;
    wire [31654-1:0] stage1_out;

    vmodel_stage0 stage0 (.inp(stage0_inp), .out(stage0_out));
    vmodel_stage1 stage1 (.inp(stage1_inp), .out(stage1_out));

    always @(posedge clk) begin
        stage0_inp <= inp;
        stage1_inp <= stage0_out;
        out <= stage1_out;
    end
endmodule
