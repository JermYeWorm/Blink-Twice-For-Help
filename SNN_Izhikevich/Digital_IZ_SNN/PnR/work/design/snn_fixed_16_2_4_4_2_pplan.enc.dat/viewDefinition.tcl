if {![namespace exists ::IMEX]} { namespace eval ::IMEX {} }
set ::IMEX::dataVar [file dirname [file normalize [info script]]]
set ::IMEX::libVar ${::IMEX::dataVar}/libs

create_library_set -name libSTD\
   -timing\
    [list ${::IMEX::libVar}/mmmc/scs130lp_tt_1.62_25_nldm.lib]
create_rc_corner -name rc\
   -cap_table ${::IMEX::libVar}/mmmc/sky130_cap.captable\
   -preRoute_res 1\
   -postRoute_res 1\
   -preRoute_cap 1\
   -postRoute_cap 1\
   -postRoute_xcap 1\
   -preRoute_clkres 0\
   -preRoute_clkcap 0
create_delay_corner -name Delay\
   -library_set libSTD\
   -rc_corner rc
create_constraint_mode -name Constraints\
   -sdc_files\
    [list ${::IMEX::libVar}/mmmc/snn_fixed_16_2_4_4_2_syn.sdc]
create_analysis_view -name Default -constraint_mode Constraints -delay_corner Delay
set_analysis_view -setup [list Default] -hold [list Default]
