`timescale 1ns / 1ps

module tb_snn_fixed_16_2_4_4_2;

    // snn_fixed_16_2_4_4_2 Parameters
    parameter PERIOD = 2;


    // snn_fixed_16_2_4_4_2 Inputs
    reg        rst_n = 0;
    reg        clk = 0;
    reg        clk_input_spike = 0;
    reg  [1:0] spike_in = 2'b11;

    // snn_fixed_16_2_4_4_2 Outputs
    wire [1:0] spike_out;

    initial begin
        forever #(PERIOD / 2) clk = ~clk;
    end

    // always @(posedge clk) begin
    //     $display("@%t clk = %b", $time, clk);
    // end

    initial begin
        forever #(100*(PERIOD / 2)) clk_input_spike = ~clk_input_spike;
    end

    initial begin
        #(PERIOD * 2) rst_n = 1;
    end

    snn_fixed_16_2_4_4_2 u_snn_fixed_16_2_4_4_2 (
        .rst_n   (rst_n),
        .clk     (clk),
        .spike_in(spike_in[1:0]),

        .spike_out(spike_out[1:0])
    );

    always @(posedge clk) begin
        if (~rst_n) begin
            spike_in <= 2'b00;
        end else begin
            if (clk_input_spike) begin
                spike_in <= $urandom_range(0, 3);
            end else begin
                spike_in <= 2'b00;
            end
        end
    end
    initial begin
        #(500*(PERIOD * 2));
        $finish;
    end

    initial begin
        $dumpfile("tb_snn_fixed_16_2_4_4_2.vcd");
        $dumpvars;
        $recordfile("tb_snn_fixed_16_2_4_4_2.trn");
        $recordvars();
    end

endmodule
