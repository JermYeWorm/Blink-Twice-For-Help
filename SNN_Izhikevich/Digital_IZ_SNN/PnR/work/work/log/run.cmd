#######################################################
#                                                     
#  Innovus Command Logging File                     
#  Created on Tue May 20 13:32:53 2025                
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
