`timescale 1ns / 1ps

module tb_IZ_pipeline_16;

    // IZ_pipeline_16 Parameters
    parameter PERIOD = 10;
    parameter precision = 16;
    parameter para_t_step = 16'h0148;
    parameter para_k = 16'h0a3d;
    parameter para_a = 16'h0014;
    parameter para_b = 16'h47ae;
    parameter para_Vmin = 16'haaeb;
    parameter para_d = 16'h5eb8;
    parameter para_C_dived_1 = 16'h042f;
    parameter para_Vr = 16'h9f3b;
    parameter para_Vt = 16'hc937;
    parameter para_Vpeak = 16'h30e5;

    // IZ_pipeline_16 Inputs
    reg         clk = 0;
    reg         rst_n = 0;
    reg  [15:0] i_mul_h = 16'h0fff;  // i=100

    initial begin
        forever #(PERIOD / 2) clk = ~clk;
    end

    initial begin
        #(PERIOD * 2) rst_n = 1;
    end

    initial begin
        #(PERIOD * 5000);
        $finish;
    end

    IZ_pipeline_16 #(
        .precision     (precision),
        .para_t_step   (para_t_step),
        .para_k        (para_k),
        .para_a        (para_a),
        .para_b        (para_b),
        .para_Vmin     (para_Vmin),
        .para_d        (para_d),
        .para_C_dived_1(para_C_dived_1),
        .para_Vr       (para_Vr),
        .para_Vt       (para_Vt),
        .para_Vpeak    (para_Vpeak)
    ) u_IZ_pipeline_16 (
        .clk    (clk),
        .rst_n  (rst_n),
        .i_mul_h(i_mul_h),
        .spike(spike)
    );


endmodule
