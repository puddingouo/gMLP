`timescale 1ns/1ps

module matrix_mult #(
    parameter DATAWIDTH = 16,
    parameter M = 4, // A Rows
    parameter K = 4, // A Cols / B Rows
    parameter N = 4  // B Cols
)(
    input                                clk,
    input                                rstn,
    input                                start,
    input      [M*K*DATAWIDTH-1:0]       A_flat, 
    input      [K*N*DATAWIDTH-1:0]       B_flat,
    output reg                           done,
    output reg [M*N*2*DATAWIDTH-1:0]     C_flat
);

    //================================================================
    // 1. 內部訊號宣告
    //================================================================
    wire signed [DATAWIDTH-1:0]   A [0:M-1][0:K-1];
    wire signed [DATAWIDTH-1:0]   B [0:K-1][0:N-1];
    reg  signed [2*DATAWIDTH-1:0] C [0:M-1][0:N-1]; 

    // 計數器
    reg [31:0] i_cnt, j_cnt, k_cnt;
    
    // 累加器
    reg signed [2*DATAWIDTH-1:0] accumulator;

    // --- 嚴格硬體定義：運算用的 Wire ---
    wire [31:0] i_next_val, j_next_val, k_next_val;
    wire signed [2*DATAWIDTH-1:0] acc_sum_val;

    // Booth 介面
    reg  mult_en;
    reg  signed [DATAWIDTH-1:0] mult_a_in;
    reg  signed [DATAWIDTH-1:0] mult_b_in;
    wire mult_done;
    wire signed [2*DATAWIDTH-1:0] mult_product;

    // FSM
    localparam IDLE      = 3'b000;
    localparam PREP_MULT = 3'b001;
    localparam RUN_MULT  = 3'b010;
    localparam ACCUM     = 3'b011;
    localparam NEXT_STEP = 3'b100;
    localparam FINISH    = 3'b101;
    reg [2:0] state, next_state;

    //================================================================
    // 2. 資料解包 (Unpack)
    //================================================================
    genvar r, c_idx; 
    generate
        for (r = 0; r < M; r = r + 1) begin : UPK_A_R
            for (c_idx = 0; c_idx < K; c_idx = c_idx + 1) begin : UPK_A_C
                assign A[r][c_idx] = A_flat[(r*K + c_idx)*DATAWIDTH +: DATAWIDTH];
            end
        end
        for (r = 0; r < K; r = r + 1) begin : UPK_B_R
            for (c_idx = 0; c_idx < N; c_idx = c_idx + 1) begin : UPK_B_C
                assign B[r][c_idx] = B_flat[(r*N + c_idx)*DATAWIDTH +: DATAWIDTH];
            end
        end
    endgenerate

    //================================================================
    // 3. 實例化 Booth Multiplier
    //================================================================
    booth_multiplier_pure_hw #(
        .DATAWIDTH(DATAWIDTH)
    ) u_booth_core (
        .clk(clk),
        .rstn(rstn),
        .en(mult_en),
        .multiplier(mult_a_in),
        .multiplicand(mult_b_in),
        .done(mult_done),
        .product(mult_product)
    );

    //================================================================
    // 4. 組合邏輯運算區 (The "Math" Section)
    //================================================================
    // 所有的加法運算都在這裡完成
    assign k_next_val = k_cnt + 1;
    assign j_next_val = j_cnt + 1;
    assign i_next_val = i_cnt + 1;
    
    // 累加器加法
    assign acc_sum_val = accumulator + mult_product;

    // FSM Next State Logic
    always @(*) begin
        case (state)
            IDLE:      next_state = start ? PREP_MULT : IDLE;
            PREP_MULT: next_state = RUN_MULT;
            RUN_MULT:  next_state = mult_done ? ACCUM : RUN_MULT;
            ACCUM:     next_state = NEXT_STEP;
            NEXT_STEP: begin
                if (i_cnt == M-1 && j_cnt == N-1 && k_cnt == K-1)
                    next_state = FINISH;
                else
                    next_state = PREP_MULT;
            end
            FINISH:    next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end

    //================================================================
    // 5. 循序邏輯區 (The "Storage" Section)
    //================================================================
    always @(posedge clk or negedge rstn) begin
        if (!rstn) state <= IDLE;
        else       state <= next_state;
    end

    integer row, col;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            i_cnt <= 0; j_cnt <= 0; k_cnt <= 0;
            accumulator <= 0;
            mult_en <= 0;
            mult_a_in <= 0; mult_b_in <= 0;
            done <= 0;
            for (row=0; row<M; row=row+1)
                for (col=0; col<N; col=col+1)
                    C[row][col] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    i_cnt <= 0; j_cnt <= 0; k_cnt <= 0;
                    accumulator <= 0;
                end

                PREP_MULT: begin
                    mult_a_in <= A[i_cnt][k_cnt];
                    mult_b_in <= B[k_cnt][j_cnt];
                    mult_en <= 1; 
                end

                RUN_MULT: begin
                    mult_en <= 0;
                end

                ACCUM: begin
                    // 使用 wire 的運算結果
                    accumulator <= acc_sum_val; 
                end

                NEXT_STEP: begin
                    if (k_cnt == K-1) begin
                        C[i_cnt][j_cnt] <= accumulator;
                        accumulator <= 0;
                        k_cnt <= 0;
                        
                        if (j_cnt == N-1) begin
                            j_cnt <= 0;
                            if (i_cnt != M-1) begin
                                i_cnt <= i_next_val; // 使用 wire
                            end
                        end else begin
                            j_cnt <= j_next_val; // 使用 wire
                        end
                    end else begin
                        k_cnt <= k_next_val; // 使用 wire
                    end
                end

                FINISH: begin
                    done <= 1;
                end
            endcase
        end
    end

    // 輸出打包
    generate
        for (r = 0; r < M; r = r + 1) begin : PK_C_R
            for (c_idx = 0; c_idx < N; c_idx = c_idx + 1) begin : PK_C_C
                always @(*) begin
                    C_flat[(r*N + c_idx)*2*DATAWIDTH +: 2*DATAWIDTH] = C[r][c_idx];
                end
            end
        end
    endgenerate

endmodule