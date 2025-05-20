module snn_fixed_16_2_4_4_2 (
    input rst_n,
    input clk,
    input [1:0] spike_in,
    output [1:0] spike_out
);

    wire [15:0] i_mul_h_layer_input_to_layer_1[7:0];
    wire [15:0] i_mul_h_layer_1_to_layer_2[15:0];
    wire [15:0] i_mul_h_layer_2_to_layer_output[7:0];
    wire [1:0] spike_layer_input;
    wire [3:0] spike_layer_1;
    wire [3:0] spike_layer_2;
    wire [1:0] spike_layer_output;
    genvar i, j;

    assign spike_layer_input = spike_in;
    assign spike_out = spike_layer_output;

    //neurons instantiation

    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_neurons_layer_1
            wire [15:0] sum_layer_input_to_layer_1 = i_mul_h_layer_input_to_layer_1[i * 2] + 
                                                 i_mul_h_layer_input_to_layer_1[i * 2 + 1];
            IZ_pipeline_16 u_layer_1_IZ_pipeline_16 (
                .clk(clk),
                .rst_n(rst_n),
                .i_mul_h(sum_layer_input_to_layer_1[15] ? 16'h7fff : sum_layer_input_to_layer_1),
                .spike(spike_layer_1[i])
            );
        end
    endgenerate
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_neurons_layer_2
            wire [15:0] sum_layer_1_to_layer_2 = i_mul_h_layer_1_to_layer_2[i * 4] + 
                                                 i_mul_h_layer_1_to_layer_2[i * 4 + 1] + 
                                                 i_mul_h_layer_1_to_layer_2[i * 4 + 2] + 
                                                 i_mul_h_layer_1_to_layer_2[i * 4 + 3];
            IZ_pipeline_16 u_layer_2_IZ_pipeline_16 (
                .clk(clk),
                .rst_n(rst_n),
                .i_mul_h(sum_layer_1_to_layer_2[15] ? 16'h7fff : sum_layer_1_to_layer_2),
                .spike(spike_layer_2[i])
            );
        end
    endgenerate
    generate
        for (i = 0; i < 2; i = i + 1) begin : gen_neurons_layer_output
            wire [15:0] sum_layer_2_to_output = i_mul_h_layer_2_to_layer_output[i * 4] +
                                                i_mul_h_layer_2_to_layer_output[i * 4 + 1] +
                                                i_mul_h_layer_2_to_layer_output[i * 4 + 2] +
                                                i_mul_h_layer_2_to_layer_output[i * 4 + 3];
            IZ_pipeline_16 u_layer_output_IZ_pipeline_16 (
                .clk(clk),
                .rst_n(rst_n),
                .i_mul_h(sum_layer_2_to_output[15] ? 16'h7fff : sum_layer_2_to_output),
                .spike(spike_layer_output[i])
            );
        end
    endgenerate

    //synapse instantiation
    generate
        for (i = 0; i < 2; i = i + 1) begin : gen_synapse_layer_input_to_layer_1
            for (j = 0; j < 4; j = j + 1) begin
                fixed_16_synapse u_layer_input_to_layer_1_fixed_16_synapse (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .spike  (spike_layer_input[i]),
                    .i_mul_h(i_mul_h_layer_input_to_layer_1[i*4+j])
                );
            end
        end
    endgenerate

    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_synapse_layer_1_to_layer_2
            for (j = 0; j < 4; j = j + 1) begin
                fixed_16_synapse u_layer_1_to_layer_2_fixed_16_synapse (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .spike  (spike_layer_1[i]),
                    .i_mul_h(i_mul_h_layer_1_to_layer_2[i*4+j])
                );
            end
        end
    endgenerate

    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_synapse_layer_2_to_layer_output
            for (j = 0; j < 2; j = j + 1) begin
                fixed_16_synapse u_layer_2_to_layer_output_fixed_16_synapse (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .spike  (spike_layer_2[i]),
                    .i_mul_h(i_mul_h_layer_2_to_layer_output[i*2+j])
                );
            end
        end
    endgenerate

endmodule
