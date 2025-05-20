
//input ports
add mapped point reset reset -type PI PI
add mapped point req_l_event[7] req_l_event[7] -type PI PI
add mapped point req_l_event[6] req_l_event[6] -type PI PI
add mapped point req_l_event[5] req_l_event[5] -type PI PI
add mapped point req_l_event[4] req_l_event[4] -type PI PI
add mapped point req_l_event[3] req_l_event[3] -type PI PI
add mapped point req_l_event[2] req_l_event[2] -type PI PI
add mapped point req_l_event[1] req_l_event[1] -type PI PI
add mapped point req_l_event[0] req_l_event[0] -type PI PI
add mapped point ack_r ack_r -type PI PI

//output ports
add mapped point ack_l_event[7] ack_l_event[7] -type PO PO
add mapped point ack_l_event[6] ack_l_event[6] -type PO PO
add mapped point ack_l_event[5] ack_l_event[5] -type PO PO
add mapped point ack_l_event[4] ack_l_event[4] -type PO PO
add mapped point ack_l_event[3] ack_l_event[3] -type PO PO
add mapped point ack_l_event[2] ack_l_event[2] -type PO PO
add mapped point ack_l_event[1] ack_l_event[1] -type PO PO
add mapped point ack_l_event[0] ack_l_event[0] -type PO PO
add mapped point req_r req_r -type PO PO
add mapped point addr_event[2] addr_event[2] -type PO PO
add mapped point addr_event[1] addr_event[1] -type PO PO
add mapped point addr_event[0] addr_event[0] -type PO PO

//inout ports




//Sequential Pins
add mapped point layer1_to_layer2_0/dout_a[0]/q layer1_to_layer2_0_dout_a_reg[0]/Q -type DFF DFF
add mapped point layer1_to_layer2_0/dout_b[0]/q layer1_to_layer2_0_dout_b_reg[0]/Q -type DFF DFF
add mapped point layer0_to_layer1_0/dout_b[0]/q layer0_to_layer1_0_dout_b_reg[0]/Q -type DFF DFF
add mapped point layer1_to_layer2_0/dout_a[1]/q layer1_to_layer2_0_dout_a_reg[1]/Q -type DFF DFF
add mapped point layer0_to_layer1_0/dout_a[0]/q layer0_to_layer1_0_dout_a_reg[0]/Q -type DFF DFF
add mapped point layer1_to_layer2_0/dout_b[1]/q layer1_to_layer2_0_dout_b_reg[1]/Q -type DFF DFF
add mapped point layer0_to_layer1_1/dout_b[0]/q layer0_to_layer1_1_dout_b_reg[0]/Q -type DFF DFF
add mapped point layer0_to_layer1_1/dout_a[0]/q layer0_to_layer1_1_dout_a_reg[0]/Q -type DFF DFF



//Black Boxes



//Empty Modules as Blackboxes
