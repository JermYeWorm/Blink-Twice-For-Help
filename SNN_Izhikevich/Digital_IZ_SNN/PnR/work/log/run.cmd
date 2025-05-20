#######################################################
#                                                     
#  Innovus Command Logging File                     
#  Created on Tue May 20 13:43:26 2025                
#                                                     
#######################################################

#@(#)CDS: Innovus v21.19-s058_1 (64bit) 04/04/2024 09:59 (Linux 3.10.0-693.el7.x86_64)
#@(#)CDS: NanoRoute 21.19-s058_1 NR231113-0413/21_19-UB (database version 18.20.605) {superthreading v2.17}
#@(#)CDS: AAE 21.19-s004 (64bit) 04/04/2024 (Linux 3.10.0-693.el7.x86_64)
#@(#)CDS: CTE 21.19-s010_1 () Mar 27 2024 01:55:37 ( )
#@(#)CDS: SYNTECH 21.19-s002_1 () Sep  6 2023 22:17:00 ( )
#@(#)CDS: CPE v21.19-s026
#@(#)CDS: IQuantus/TQuantus 21.1.1-s966 (64bit) Wed Mar 8 10:22:20 PST 2023 (Linux 3.10.0-693.el7.x86_64)

set_global _enable_mmmc_by_default_flow      $CTE::mmmc_default
suppressMessage ENCEXT-2799
getVersion
define_proc_arguments ViaFillQor -info {This procedure extracts Viafill details from innovus db} -define_args {
        {-window "window coordinates" "" list optional}
        {-window_size "window size in microns" "" string optional}
    
    }
define_proc_arguments ProcessFills -info {This procedure processes Fill types} -define_args {
    {-fillInfo "Design Fill data" "" list required}
				{-csvName "File path for Fill Data csv file" "Path of CSV file" string required}
				{-selectFill "type of fill to be selected in session" "list of BRIDGE/EXTENSION/STAMP/FLOATING" list required}
    {-output_data "Boolean Flag to output Fill Data for further processing" "" string required}
}
define_proc_arguments FillQor -info {This procedure extracts fill details from innovus db} -define_args {
    {-layers "Fills Cleanup on which all layers" "list of Metal/Routing layers" list optional}
				{-selectFill "type of fill to be selected in session" "list of BRIDGE/EXTENSION/STAMP/FLOATING" list optional}
				{-outData "Boolean Flag to output Fill Data for further processing" "" boolean optional}
    {-outDataFile "File path for Fill Data csv file" "Path of CSV file" string optional}
}
define_proc_arguments ProcessFills_fast -info {This procedure processes Fill types} -define_args {
    {-fillInfo "Design Fill data" "" list required}
				{-csvName "File path for Fill Data csv file" "Path of CSV file" string required}
				{-selectFill "type of fill to be selected in session" "list of BRIDGE/EXTENSION/STAMP/FLOATING" list required}
    {-output_data "Boolean Flag to output Fill Data for further processing" "" string required}
}
define_proc_arguments FillQor_fast -info {This procedure extracts fill details from innovus db} -define_args {
    {-layers "Fills Cleanup on which all layers" "list of Metal/Routing layers" list optional}
				{-selectFill "type of fill to be selected in session" "list of BRIDGE/EXTENSION/STAMP/FLOATING" list optional}
				{-outData "Boolean Flag to output Fill Data for further processing" "" boolean optional}
    {-outDataFile "File path for Fill Data csv file" "Path of CSV file" string optional}
}
define_proc_arguments ProcessFills_fast_stampOnly -info {This procedure processes Fill types} -define_args {
    {-fillInfo "Design Fill data" "" list required}
	
}
define_proc_arguments FillQor_fast_stampOnly -info {This procedure extracts fill details from innovus db} -define_args {
    {-layers "Fills Cleanup on which all layers" "list of Metal/Routing layers" list optional}
}
win
setMultiCpuUsage -localCpu max
set init_gnd_net vgnd
set init_pwr_net vpwr
set init_lef_file {../../lib/s130.tlef
/afs/glue.umd.edu/department/enee/software/cadskywaterpdk/S130IP/DIG/current/scs130lp/lef/scs130lp.lef}
set init_verilog ../../synthesis/work/output/snn_fixed_16_2_4_4_2_pg_syn.v
set init_mmmc_file ../scripts/Default.view
init_design
setDrawView fplan
getIoFlowFlag
setFPlanRowSpacingAndType 0 2
setIoFlowFlag 0
floorPlan -flip s -r 0.7 0.4 4.0 4.0 4.0 4.0
uiSetTool select
getIoFlowFlag
fit
saveDesign ../work/design/snn_fixed_16_2_4_4_2_fplan.enc
saveFPlan ../work/design/snn_fixed_16_2_4_4_2.fp
addRing -nets {vpwr vgnd} -type core_rings -follow core -layer {top met1 bottom met1 left met2 right met2} -width 0.5 -spacing 1 -offset 1.0 -center 0 -threshold 0 -jog_distance 0 -snap_wire_center_to_grid None
set sprCreateIeStripeNets {}
set sprCreateIeStripeLayers {}
set sprCreateIeStripeWidth 1
set sprCreateIeStripeSpacing 1
set sprCreateIeStripeThreshold 1
sroute -jogControl { preferWithChanges differentLayer } -nets {vgnd vpwr}
saveDesign ../work/design/snn_fixed_16_2_4_4_2_pplan.enc
addWellTap -cell scs130lp_tapvpwrvgnd_1 -cellInterval 50 -checkerBoard
setPlaceMode -place_global_place_io_pins true
setPlaceMode -fp false
getPlaceMode -place_hierarchical_flow -quiet
report_message -start_cmd
getRouteMode -maxRouteLayer -quiet
getRouteMode -user -maxRouteLayer
getPlaceMode -place_global_place_io_pins -quiet
getPlaceMode -user -maxRouteLayer
getPlaceMode -quiet -adaptiveFlowMode
getPlaceMode -timingDriven -quiet
getPlaceMode -adaptive -quiet
getPlaceMode -relaxSoftBlockageMode -quiet
getPlaceMode -user -relaxSoftBlockageMode
getPlaceMode -ignoreScan -quiet
getPlaceMode -user -ignoreScan
getPlaceMode -repairPlace -quiet
getPlaceMode -user -repairPlace
getPlaceMode -inPlaceOptMode -quiet
getPlaceMode -quiet -bypassFlowEffortHighChecking
getDesignMode -quiet -siPrevention
getPlaceMode -quiet -place_global_exp_enable_3d
getPlaceMode -exp_slack_driven -quiet
um::push_snapshot_stack
getDesignMode -quiet -flowEffort
getDesignMode -highSpeedCore -quiet
getPlaceMode -quiet -adaptive
set spgFlowInInitialPlace 1
getPlaceMode -sdpAlignment -quiet
getPlaceMode -softGuide -quiet
getPlaceMode -useSdpGroup -quiet
getPlaceMode -sdpAlignment -quiet
getPlaceMode -enableDbSaveAreaPadding -quiet
getPlaceMode -quiet -wireLenOptEffort
getPlaceMode -sdpPlace -quiet
getPlaceMode -exp_slack_driven -quiet
getPlaceMode -sdpPlace -quiet
getPlaceMode -groupHighLevelClkGate -quiet
setvar spgRptErrorForScanConnection 0
getPlaceMode -place_global_exp_allow_missing_scan_chain -quiet
getPlaceMode -place_check_library -quiet
getPlaceMode -trimView -quiet
getPlaceMode -expTrimOptBeforeTDGP -quiet
getPlaceMode -quiet -useNonTimingDeleteBufferTree
getPlaceMode -congEffort -quiet
getPlaceMode -relaxSoftBlockageMode -quiet
getPlaceMode -user -relaxSoftBlockageMode
getPlaceMode -ignoreScan -quiet
getPlaceMode -user -ignoreScan
getPlaceMode -repairPlace -quiet
getPlaceMode -user -repairPlace
getPlaceMode -congEffort -quiet
getPlaceMode -fp -quiet
getPlaceMode -timingDriven -quiet
getPlaceMode -user -timingDriven
getPlaceMode -fastFp -quiet
getPlaceMode -clusterMode -quiet
get_proto_model -type_match {flex_module flex_instgroup} -committed -name -tcl
getPlaceMode -inPlaceOptMode -quiet
getPlaceMode -quiet -bypassFlowEffortHighChecking
getPlaceMode -ultraCongEffortFlow -quiet
getPlaceMode -forceTiming -quiet
getPlaceMode -fp -quiet
getPlaceMode -fastfp -quiet
getPlaceMode -timingDriven -quiet
getPlaceMode -fp -quiet
getPlaceMode -fastfp -quiet
getPlaceMode -powerDriven -quiet
getExtractRCMode -quiet -engine
getAnalysisMode -quiet -clkSrcPath
getAnalysisMode -quiet -clockPropagation
getAnalysisMode -quiet -cppr
setExtractRCMode -engine preRoute
setAnalysisMode -clkSrcPath false -clockPropagation forcedIdeal
getPlaceMode -exp_slack_driven -quiet
isAnalysisModeSetup
getPlaceMode -quiet -place_global_exp_solve_unbalance_path
getPlaceMode -quiet -NMPsuppressInfo
getPlaceMode -quiet -place_global_exp_wns_focus_v2
getPlaceMode -quiet -place_incr_exp_isolation_flow
getPlaceMode -enableDistPlace -quiet
getPlaceMode -quiet -clusterMode
getPlaceMode -wl_budget_mode -quiet
setPlaceMode -reset -place_global_exp_balance_buffer_chain
getPlaceMode -wl_budget_mode -quiet
setPlaceMode -reset -place_global_exp_balance_pipeline
getPlaceMode -place_global_exp_balance_buffer_chain -quiet
getPlaceMode -place_global_exp_balance_pipeline -quiet
getPlaceMode -tdgpMemFlow -quiet
getPlaceMode -user -resetCombineRFLevel
getPlaceMode -quiet -resetCombineRFLevel
setPlaceMode -resetCombineRFLevel 1000
setvar spgSpeedupBuildVSM 1
getPlaceMode -tdgpResetCteTG -quiet
getPlaceMode -macroPlaceMode -quiet
getPlaceMode -place_global_replace_QP -quiet
getPlaceMode -macroPlaceMode -quiet
getPlaceMode -enableDistPlace -quiet
getPlaceMode -exp_slack_driven -quiet
getPlaceMode -place_global_ignore_spare -quiet
getPlaceMode -enableDistPlace -quiet
getPlaceMode -quiet -expNewFastMode
setPlaceMode -expHiddenFastMode 1
setPlaceMode -reset -ignoreScan
getPlaceMode -quiet -place_global_exp_auto_finish_floorplan
getPlaceMode -quiet -IOSlackAdjust
getPlaceMode -tdgpCteZeroDelayModeDelBuf -quiet
set_global timing_enable_zero_delay_analysis_mode true
getPlaceMode -quiet -useNonTimingDeleteBufferTree
getPlaceMode -quiet -prePlaceOptSimplifyNetlist
getPlaceMode -quiet -enablePrePlaceOptimizations
getPlaceMode -quiet -prePlaceOptDecloneInv
deleteBufferTree -decloneInv
getPlaceMode -tdgpCteZeroDelayModeDelBuf -quiet
set_global timing_enable_zero_delay_analysis_mode false
getAnalysisMode -quiet -honorClockDomains
getPlaceMode -honorUserPathGroup -quiet
getAnalysisMode -quiet -honorClockDomains
set delaycal_use_default_delay_limit 101
set delaycal_default_net_delay 0
set delaycal_default_net_load 0
set delaycal_default_net_load_ignore_for_ilm 0
set delaycal_input_transition_delay 1ps
getAnalysisMode -clkSrcPath -quiet
getAnalysisMode -clockPropagation -quiet
getAnalysisMode -checkType -quiet
buildTimingGraph
getDelayCalMode -ignoreNetLoad -quiet
getDelayCalMode -ignoreNetLoad -quiet
setDelayCalMode -ignoreNetLoad true -quiet
get_global timing_enable_path_group_priority
get_global timing_constraint_enable_group_path_resetting
set_global timing_enable_path_group_priority false
set_global timing_constraint_enable_group_path_resetting false
getOptMode -allowPreCTSClkSrcPaths -quiet
set_global _is_ipo_interactive_path_groups 1
group_path -name in2reg_tmp.783287 -from {0x2d 0x30} -to 0x31 -ignore_source_of_trigger_arc
getOptMode -allowPreCTSClkSrcPaths -quiet
set_global _is_ipo_interactive_path_groups 1
group_path -name in2out_tmp.783287 -from {0x4c 0x4f} -to 0x50 -ignore_source_of_trigger_arc
set_global _is_ipo_interactive_path_groups 1
group_path -name reg2reg_tmp.783287 -from 0x6a -to 0x83
set_global _is_ipo_interactive_path_groups 1
group_path -name reg2out_tmp.783287 -from 0xb6 -to 0xcf
setPathGroupOptions reg2reg_tmp.783287 -effortLevel high
getNanoRouteMode -routeStrictlyHonorNonDefaultRule -quiet
getNanoRouteMode -routeBottomRoutingLayer -quiet
getNanoRouteMode -routeTopRoutingLayer -quiet
isAnalysisModeSetup
getAnalysisMode -analysisType -quiet
isAnalysisModeSetup
all_setup_analysis_views
all_hold_analysis_views
get_analysis_view $view -delay_corner
get_delay_corner $dcCorner -power_domain_list
get_delay_corner $dcCorner -library_set
get_library_set $libSetName -si
get_delay_corner $dcCorner -late_library_set
get_delay_corner $dcCorner -early_library_set
reset_path_group -name reg2out_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name in2reg_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name in2out_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name reg2reg_tmp.783287
set_global _is_ipo_interactive_path_groups 0
setDelayCalMode -ignoreNetLoad false
set delaycal_use_default_delay_limit 1000
set delaycal_default_net_delay 1000ps
set delaycal_input_transition_delay 0ps
set delaycal_default_net_load 0.5pf
set delaycal_default_net_load_ignore_for_ilm 0
all_setup_analysis_views
getPlaceMode -place_global_exp_ignore_low_effort_path_groups -quiet
getPlaceMode -exp_slack_driven -quiet
getAnalysisMode -quiet -honorClockDomains
getPlaceMode -quiet -place_global_exp_inverter_rewiring
getPlaceMode -ignoreUnproperPowerInit -quiet
getPlaceMode -quiet -expSkipGP
setDelayCalMode -engine feDc
createBasicPathGroups -quiet
psp::embedded_egr_init_
psp::embedded_egr_term_
psp::embedded_egr_init_
psp::embedded_egr_term_
psp::embedded_egr_init_
psp::embedded_egr_term_
reset_path_group
set_global _is_ipo_interactive_path_groups 0
scanReorder
setDelayCalMode -engine aae
all_setup_analysis_views
getPlaceMode -exp_slack_driven -quiet
set_global timing_enable_path_group_priority $gpsPrivate::optSave_ctePGPriority
set_global timing_constraint_enable_group_path_resetting $gpsPrivate::optSave_ctePGResetting
getPlaceMode -quiet -tdgpAdjustNetWeightBySlack
get_ccopt_clock_trees *
getPlaceMode -exp_insert_guidance_clock_tree -quiet
getPlaceMode -exp_cluster_based_high_fanout_buffering -quiet
getPlaceMode -place_global_exp_incr_skp_preserve_mode_v2 -quiet
getPlaceMode -quiet -place_global_exp_netlist_balance_flow
getPlaceMode -quiet -timingEffort
getAnalysisMode -quiet -honorClockDomains
getPlaceMode -honorUserPathGroup -quiet
getAnalysisMode -quiet -honorClockDomains
set delaycal_use_default_delay_limit 101
set delaycal_default_net_delay 0
set delaycal_default_net_load 0
set delaycal_default_net_load_ignore_for_ilm 0
getAnalysisMode -clkSrcPath -quiet
getAnalysisMode -clockPropagation -quiet
getAnalysisMode -checkType -quiet
buildTimingGraph
getDelayCalMode -ignoreNetLoad -quiet
getDelayCalMode -ignoreNetLoad -quiet
setDelayCalMode -ignoreNetLoad true -quiet
get_global timing_enable_path_group_priority
get_global timing_constraint_enable_group_path_resetting
set_global timing_enable_path_group_priority false
set_global timing_constraint_enable_group_path_resetting false
getOptMode -allowPreCTSClkSrcPaths -quiet
set_global _is_ipo_interactive_path_groups 1
group_path -name in2reg_tmp.783287 -from {0x1c8 0x1cb} -to 0x1cc -ignore_source_of_trigger_arc
getOptMode -allowPreCTSClkSrcPaths -quiet
set_global _is_ipo_interactive_path_groups 1
group_path -name in2out_tmp.783287 -from {0x1e7 0x1ea} -to 0x1eb -ignore_source_of_trigger_arc
set_global _is_ipo_interactive_path_groups 1
group_path -name reg2reg_tmp.783287 -from 0x205 -to 0x21e
set_global _is_ipo_interactive_path_groups 1
group_path -name reg2out_tmp.783287 -from 0x251 -to 0x26a
setPathGroupOptions reg2reg_tmp.783287 -effortLevel high
reset_path_group -name reg2out_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name in2reg_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name in2out_tmp.783287
set_global _is_ipo_interactive_path_groups 0
reset_path_group -name reg2reg_tmp.783287
set_global _is_ipo_interactive_path_groups 0
setDelayCalMode -ignoreNetLoad false
set delaycal_use_default_delay_limit 1000
set delaycal_default_net_delay 1000ps
set delaycal_default_net_load 0.5pf
set delaycal_default_net_load_ignore_for_ilm 0
all_setup_analysis_views
getPlaceMode -place_global_exp_ignore_low_effort_path_groups -quiet
getPlaceMode -exp_slack_driven -quiet
getPlaceMode -quiet -cong_repair_commit_clock_net_route_attr
getPlaceMode -enableDbSaveAreaPadding -quiet
getPlaceMode -quiet -wireLenOptEffort
setPlaceMode -reset -improveWithPsp
getPlaceMode -quiet -debugGlobalPlace
getPlaceMode -congRepair -quiet
getPlaceMode -fp -quiet
getPlaceMode -user -rplaceIncrNPClkGateAwareMode
getPlaceMode -user -congRepairMaxIter
getPlaceMode -quiet -congRepairPDClkGateMode4
setPlaceMode -rplaceIncrNPClkGateAwareMode 4
getPlaceMode -quiet -expCongRepairPDOneLoop
setPlaceMode -congRepairMaxIter 1
getPlaceMode -quickCTS -quiet
get_proto_model -type_match {flex_module flex_instgroup} -committed -name -tcl
getPlaceMode -congRepairForceTrialRoute -quiet
getPlaceMode -user -congRepairForceTrialRoute
setPlaceMode -congRepairForceTrialRoute true
::goMC::is_advanced_metrics_collection_running
congRepair
::goMC::is_advanced_metrics_collection_running
::goMC::is_advanced_metrics_collection_running
::goMC::is_advanced_metrics_collection_running
setPlaceMode -reset -congRepairForceTrialRoute
getPlaceMode -quiet -congRepairPDClkGateMode4
setPlaceMode -reset -rplaceIncrNPClkGateAwareMode
setPlaceMode -reset -congRepairMaxIter
getPlaceMode -congRepairCleanupPadding -quiet
getPlaceMode -quiet -wireLenOptEffort
all_setup_analysis_views
getPlaceMode -exp_slack_driven -quiet
set_global timing_enable_path_group_priority $gpsPrivate::optSave_ctePGPriority
set_global timing_constraint_enable_group_path_resetting $gpsPrivate::optSave_ctePGResetting
getPlaceMode -place_global_exp_incr_skp_preserve_mode_v2 -quiet
getPlaceMode -quiet -place_global_exp_netlist_balance_flow
getPlaceMode -quiet -timingEffort
getPlaceMode -tdgpDumpStageTiming -quiet
getPlaceMode -quiet -tdgpAdjustNetWeightBySlack
getPlaceMode -trimView -quiet
getOptMode -quiet -viewOptPolishing
getOptMode -quiet -fastViewOpt
spInternalUse deleteViewOptManager
spInternalUse tdgp clearSkpData
setAnalysisMode -clkSrcPath true -clockPropagation sdcControl
getPlaceMode -exp_slack_driven -quiet
setExtractRCMode -engine preRoute
setPlaceMode -reset -relaxSoftBlockageMode
setPlaceMode -reset -ignoreScan
setPlaceMode -reset -repairPlace
getPlaceMode -quiet -NMPsuppressInfo
setvar spgSpeedupBuildVSM 0
getPlaceMode -macroPlaceMode -quiet
getPlaceMode -place_global_replace_QP -quiet
getPlaceMode -macroPlaceMode -quiet
getPlaceMode -exp_slack_driven -quiet
getPlaceMode -enableDistPlace -quiet
getPlaceMode -place_global_ignore_spare -quiet
getPlaceMode -tdgpMemFlow -quiet
setPlaceMode -reset -resetCombineRFLevel
getPlaceMode -enableDistPlace -quiet
getPlaceMode -quiet -clusterMode
getPlaceMode -quiet -place_global_exp_solve_unbalance_path
getPlaceMode -enableDistPlace -quiet
setPlaceMode -reset -expHiddenFastMode
getPlaceMode -tcg2Pass -quiet
getPlaceMode -quiet -wireLenOptEffort
getPlaceMode -fp -quiet
getPlaceMode -fastfp -quiet
getPlaceMode -doRPlace -quiet
getPlaceMode -RTCPlaceDesignFlow -quiet
getPlaceMode -quickCTS -quiet
set spgFlowInInitialPlace 0
getPlaceMode -user -maxRouteLayer
spInternalUse TDGP resetIgnoreNetLoad
getPlaceMode -place_global_exp_balance_pipeline -quiet
getDesignMode -quiet -flowEffort
report_message -end_cmd
um::create_snapshot -name final -auto min
um::pop_snapshot_stack
um::create_snapshot -name place_design
getPlaceMode -exp_slack_driven -quiet
setDrawView place
fit
setOptMode -yieldEffort none
setOptMode -effort high
setOptMode -maxDensity 0.9
setOptMode -drcMargin 0.0
setOptMode -holdTargetSlack 0.0 -setupTargetSlack 0.0
setOptMode -SimplifyNetlist false
setOptMode -fixCap true -fixTran true -fixFanoutLoad true
optDesign -preCTS -outDir work/report/
fit
saveDesign work/design/snn_fixed_16_2_4_4_2_placed.enc
addTieHiLo -cell scs130lp_conb_1 -prefix tieOff
globalNetConnect vpwr -type pgpin -pin vpwr -inst * -all -override
globalNetConnect vgnd -type pgpin -pin vgnd -inst * -all -override
globalNetConnect vpwr -type pgpin -pin vpb -inst * -all -override
globalNetConnect vgnd -type pgpin -pin vnb -inst * -all -override
setNanoRouteMode -quiet -routeInsertAntennaDiode 0
setNanoRouteMode -quiet -timingEngine {}
setNanoRouteMode -quiet -routeWithTimingDriven 0
setNanoRouteMode -quiet -routeWithSiPostRouteFix 0
setNanoRouteMode -quiet -drouteStartIteration default
setNanoRouteMode -quiet -routeTopRoutingLayer 6
setNanoRouteMode -quiet -routeBottomRoutingLayer 3
setNanoRouteMode -quiet -drouteEndIteration 150
setNanoRouteMode -quiet -routeWithTimingDriven true
setNanoRouteMode -quiet -routeWithSiDriven false
routeDesign -globalDetail
getNanoRouteMode -quiet
getNanoRouteMode -quiet envSuperthreading
getNanoRouteMode -quiet drouteFixAntenna
getNanoRouteMode -quiet routeInsertAntennaDiode
getNanoRouteMode -quiet routeAntennaCellName
getNanoRouteMode -quiet timingEngine
getNanoRouteMode -quiet routeWithTimingDriven
setNanoRouteMode -quiet routeWithTimingDriven true
getNanoRouteMode -quiet routeWithEco
getNanoRouteMode -quiet routeWithSiDriven
getNanoRouteMode -quiet routeTdrEffort
setNanoRouteMode -quiet routeTdrEffort 10
getNanoRouteMode -quiet routeWithSiPostRouteFix
getNanoRouteMode -quiet drouteAutoStop
getNanoRouteMode -quiet routeSelectedNetOnly
getNanoRouteMode -quiet drouteStartIteration
getNanoRouteMode -quiet envNumberProcessor
getNanoRouteMode -quiet envSuperthreading
getNanoRouteMode -quiet routeTopRoutingLayer
getNanoRouteMode -quiet routeBottomRoutingLayer
getNanoRouteMode -quiet drouteEndIteration
getNanoRouteMode -quiet routeEcoOnlyInLayers
timeDesign -postRoute -outDir ../work/report/
setOptMode -yieldEffort none
setOptMode -highEffort
setOptMode -maxDensity 0.95
setOptMode -drcMargin 0.0
setOptMode -holdTargetSlack 0.0 -setupTargetSlack 0.0
setOptMode -SimplifyNetlist false
setOptMode -fixCap true -fixTran true -fixFanoutLoad false
setDelayCalMode -engine aae -SIAware true
setAnalysisMode -analysisType onChipVariation -cppr both
optDesign -postRoute -outDir ../work/report/
setFillerMode -corePrefix snn_fixed_16_2_4_4_2_FILL -core {scs130lp_fill_1 scs130lp_fill_2 scs130lp_fill_4 scs130lp_fill_8}
addFiller -cell {scs130lp_fill_1 scs130lp_fill_2 scs130lp_fill_4 scs130lp_fill_8} -prefix snn_fixed_16_2_4_4_2FILL -markFixed
saveDesign ../work/design/snn_fixed_16_2_4_4_2_routed.enc
verifyConnectivity -type regular -error 50 -warning 50 -reportfile ../work/report/snn_fixed_16_2_4_4_2_Conn_regular.rpt
verifyConnectivity -type special -error 50 -warning 50 -reportfile ../work/report/snn_fixed_16_2_4_4_2_Conn_special.rpt
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_src.drc
clearDrc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_0.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_3.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_11.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_1.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_2.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_7.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_4.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_8.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_6.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_9.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_10.drc
saveDrc /tmp/innovus_temp_783287_compute3.eng.umd.edu_yihuiw_bNqZ2f/vergQTmpJxGmXT/qthread_5.drc
saveDesign ../work/design/snn_fixed_16_2_4_4_2_done.enc -def
streamOut ../work/output/snn_fixed_16_2_4_4_2_soc.gds -mapFile /afs/glue.umd.edu/department/enee/software/cadskywaterpdk/pdk/V2.1.306/LIBS/S130/s130_innovus.layermap -libName Test -structureName snn_fixed_16_2_4_4_2 -units 2000 -mode ALL
saveNetlist ../work/output/snn_fixed_16_2_4_4_2_soc.v -includePowerGround
extractRC -outfile ../work/output/snn_fixed_16_2_4_4_2.cap
rcOut -spef ../work/output/snn_fixed_16_2_4_4_2.spef
writeTimingCon -pt -filePrefix ../work/timing/snn_fixed_16_2_4_4_2_done
