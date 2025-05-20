
//input ports
add mapped point clk clk -type PI PI
add mapped point reset reset -type PI PI

//output ports
add mapped point out[1] out[1] -type PO PO
add mapped point out[0] out[0] -type PO PO

//inout ports




//Sequential Pins
add mapped point count_out[1]/q count_out_reg[1]/Q -type DFF DFF
add mapped point count_out[0]/q count_out_reg[0]/Q -type DFF DFF



//Black Boxes



//Empty Modules as Blackboxes
