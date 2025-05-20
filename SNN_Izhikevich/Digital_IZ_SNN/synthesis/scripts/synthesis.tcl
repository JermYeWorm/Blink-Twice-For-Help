# Set the search paths to the libraries and the HDL files
# Remember that "." means your current directory. Add more directories
# after the . if you like. 
set_db log_file log/genus.log
set_db cmd_file log/genus.cmd
set_db init_hdl_search_path {../../source} 
# set_db init_lib_search_path {../lib}
set_db init_lib_search_path {../../lib}
# top_module is defined in the command line with -execute "set top_module <module_name>"
# set top_module "MCEL" 
# set target_library "sky130_fd_sc_hd__tt_025C_1v80"
set target_library "scs130lp_tt_1.62_25_nldm"

read_libs ${target_library}.lib
# check_library -lib_cell [get_lib_cells]

read_hdl [glob ../../source/*v]

elaborate $top_module

read_power_intent ../scripts/upf
apply_power_intent
commit_power_intent

# Define the main clock
create_clock -name clk -period 200000 [get_ports clk]
# Set input and output delay assumptions (you can tune them based on your I/O interface)
# Assume external input signals arrive 0.5ns after the clock edge
set_input_delay 0.5 -clock clk [get_ports spike_in]
# Assume external output must be valid 0.5ns before the next clock edge
set_output_delay 0.5 -clock clk [get_ports spike_out]
# Optional: Set clock uncertainty/margin (e.g., 0.1ns)
set_clock_uncertainty 0.1 [get_clocks clk]
set_driving_cell -lib_cell scs130lp_buf_0 -pin X [all_inputs]
set_load 1 [all_outputs]

set_db syn_generic_effort medium
set_db syn_map_effort medium
set_db syn_opt_effort medium

syn_generic
syn_map
syn_opt

report timing > report/timing_3KHz.rep
report gates  > report/cell_3KHz.rep
report power  > report/power_3KHz.rep

write_netlist -mapped >  output/${top_module}_syn.v
write_netlist -pg -mapped >  output/${top_module}_pg_syn.v
write_sdf > output/${top_module}_syn.sdf
write_sdc >  output/${top_module}_syn.sdc

exit