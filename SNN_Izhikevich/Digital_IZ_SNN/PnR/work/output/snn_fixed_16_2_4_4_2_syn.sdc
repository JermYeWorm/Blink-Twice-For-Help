# ####################################################################

#  Created by Genus(TM) Synthesis Solution 21.19-s055_1 on Mon May 05 20:02:03 EDT 2025

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1000fF
set_units -time 1000ps

# Set the current design
current_design snn_fixed_16_2_4_4_2

create_clock -name "clk" -period 50.0 -waveform {0.0 25.0} [get_ports clk]
set_clock_gating_check -setup 0.0 
set_input_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_in[1]}]
set_input_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_in[0]}]
set_output_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_out[1]}]
set_output_delay -clock [get_clocks clk] -add_delay 0.5 [get_ports {spike_out[0]}]
set_wire_load_mode "enclosed"
set_clock_uncertainty -setup 0.1 [get_clocks clk]
set_clock_uncertainty -hold 0.1 [get_clocks clk]
