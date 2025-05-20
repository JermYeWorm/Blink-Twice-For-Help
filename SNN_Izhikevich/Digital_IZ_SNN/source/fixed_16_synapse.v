`timescale 1ns / 1ps

module fixed_16_synapse #(
    parameter WEIGHT = 16'h05ff
) (
    input clk,
    input rst_n,
    input spike,
    output reg [15:0] i_mul_h
);

    reg  [15:0] i_mul_h_next;

    always @(posedge clk, negedge rst_n) begin
        if (~rst_n) begin
            i_mul_h <= 16'h0000;
        end else begin
            if (i_mul_h_next[15]) begin
                i_mul_h <= 16'h7fff;
            end else begin
                i_mul_h <= i_mul_h_next;
            end
        end
    end

    always @(*) begin
        if (spike) begin
            i_mul_h_next = WEIGHT + fixed_16_mul(i_mul_h, 16'h7eb8);
        end else begin
            i_mul_h_next = fixed_16_mul(i_mul_h, 16'h7eb8);
        end
    end

    function [15:0] fixed_16_mul;
        input [15:0] a;
        input [15:0] b;

        reg [29:0] temp;
        reg [14:0] a_temp, b_temp;

        begin
            if (a[15]) begin
                a_temp = ~(a[14:0] - 1);
            end else begin
                a_temp = a[14:0];
            end

            if (b[15]) begin
                b_temp = ~(b[14:0] - 1);
            end else begin
                b_temp = b[14:0];
            end

            temp = a_temp[14:0] * b_temp[14:0];
            if (a[15] ^ b[15]) begin
                fixed_16_mul = ~{1'b0, temp[29:15]} + 1;
            end else begin
                fixed_16_mul = {1'b0, temp[29:15]};
            end
        end

    endfunction
endmodule
