`timescale 1ns/1ps

module matrix_mult_tb;

    parameter DATAWIDTH = 16;
    // 設定維度: A(3x2) * B(2x3) = C(3x3)
    parameter M = 3; 
    parameter K = 2;
    parameter N = 3;
    parameter CYCLE = 10; // 100MHz

    reg clk, rstn, start;
    reg  [M*K*DATAWIDTH-1:0] A_flat;
    reg  [K*N*DATAWIDTH-1:0] B_flat;
    wire [M*N*2*DATAWIDTH-1:0] C_flat;
    wire done;

    // Testbench 用的 2D 陣列
    reg signed [DATAWIDTH-1:0] A_tb [0:M-1][0:K-1];
    reg signed [DATAWIDTH-1:0] B_tb [0:K-1][0:N-1];
    
    // 驗證用
    reg signed [2*DATAWIDTH-1:0] C_expected [0:M-1][0:N-1];
    wire signed [2*DATAWIDTH-1:0] C_dut [0:M-1][0:N-1]; 

    integer r, c, k;

    // 實例化 DUT
    matrix_mult #(
        .DATAWIDTH(DATAWIDTH), .M(M), .K(K), .N(N)
    ) u_dut (
        .clk(clk), .rstn(rstn), .start(start),
        .A_flat(A_flat), .B_flat(B_flat),
        .done(done), .C_flat(C_flat)
    );

    // Clock
    initial clk = 0;
    always #(CYCLE/2) clk = ~clk;

    // Pack Inputs
    always @(*) begin
        for (r=0; r<M; r=r+1)
            for (c=0; c<K; c=c+1)
                A_flat[(r*K+c)*DATAWIDTH +: DATAWIDTH] = A_tb[r][c];
        
        for (r=0; r<K; r=r+1)
            for (c=0; c<N; c=c+1)
                B_flat[(r*N+c)*DATAWIDTH +: DATAWIDTH] = B_tb[r][c];
    end

    // Unpack Output
    genvar i_gen, j_gen;
    generate
        for (i_gen=0; i_gen<M; i_gen=i_gen+1) begin
            for (j_gen=0; j_gen<N; j_gen=j_gen+1) begin
                assign C_dut[i_gen][j_gen] = C_flat[(i_gen*N+j_gen)*2*DATAWIDTH +: 2*DATAWIDTH];
            end
        end
    endgenerate

    initial begin
        $fsdbDumpfile("matrix_mult.fsdb");
        $fsdbDumpvars(0, matrix_mult_tb);

        rstn = 1; start = 0;

        // 1. 產生隨機資料
        $display("=== Generating Random Matrices A(%0dx%0d) and B(%0dx%0d) ===", M, K, K, N);
        for(r=0; r<M; r=r+1)
            for(c=0; c<K; c=c+1)
                A_tb[r][c] = $random % 50; 

        for(r=0; r<K; r=r+1)
            for(c=0; c<N; c=c+1)
                B_tb[r][c] = $random % 50;

        // 2. 計算 Golden Answer
        for(r=0; r<M; r=r+1) begin
            for(c=0; c<N; c=c+1) begin
                C_expected[r][c] = 0;
                for(k=0; k<K; k=k+1) begin
                    C_expected[r][c] = C_expected[r][c] + (A_tb[r][k] * B_tb[k][c]);
                end
            end
        end

        // 3. 執行模擬
        #(CYCLE*2);
        // 使用 negedge 操作 reset，確保 Setup Time
        @(negedge clk) rstn = 0;
        @(negedge clk) rstn = 1;

        @(negedge clk) start = 1;
        @(negedge clk) start = 0;

        $display("Processing... Please wait.");
        wait(done);
        #(CYCLE*2);

        // 4. 檢查結果
        $display("=== Checking Results ===");
        for(r=0; r<M; r=r+1) begin
            for(c=0; c<N; c=c+1) begin
                if (C_dut[r][c] !== C_expected[r][c]) begin
                    $display("[ERROR] C[%0d][%0d] Mismatch! DUT: %d, Exp: %d", r, c, C_dut[r][c], C_expected[r][c]);
                end else begin
                    $display("    C[%0d][%0d] Pass: %d", r, c, C_dut[r][c]);
                end
            end
        end
        $display("=== Test Finished ===");
        $finish;
    end

endmodule