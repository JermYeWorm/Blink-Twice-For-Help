# ####################################################################

#  Created by Genus(TM) Synthesis Solution 21.19-s055_1 on Tue May 20 12:29:30 EDT 2025

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1000fF
set_units -time 1000ps

# Set the current design
current_design snn_fixed_16_2_4_4_2

create_clock -name "clk" -period 200000.0 -waveform {0.0 100000.0} [get_ports clk]
set_load -pin_load 1.0 [get_ports {spike_out[1]}]
set_load -pin_load 1.0 [get_ports {spike_out[0]}]
set_clock_gating_check -setup 0.0 
set_input_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_in[1]}]
set_input_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_in[0]}]
set_output_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_out[1]}]
set_output_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_out[0]}]
set_driving_cell -lib_cell scs130lp_buf_0 -library scs130lp_tt_1.62_25 -pin "X" [get_ports rst_n]
set_driving_cell -lib_cell scs130lp_buf_0 -library scs130lp_tt_1.62_25 -pin "X" [get_ports clk]
set_driving_cell -lib_cell scs130lp_buf_0 -library scs130lp_tt_1.62_25 -pin "X" [get_ports {spike_in[1]}]
set_driving_cell -lib_cell scs130lp_buf_0 -library scs130lp_tt_1.62_25 -pin "X" [get_ports {spike_in[0]}]
set_wire_load_mode "enclosed"
set_clock_uncertainty -setup 0.1 [get_clocks clk]
set_clock_uncertainty -hold 0.1 [get_clocks clk]
