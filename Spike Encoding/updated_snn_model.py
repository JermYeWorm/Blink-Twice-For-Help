class SNNModelTraining(nn.Module):
    def __init__(self, input_dim, beta=0.5, mem_threshold=0.5, spike_grad2=surrogate.atan(alpha=2),
                 layer1=64, layer2=16, output_dim=1, dropout_rate=0.1, num_step=6):
        super().__init__()

        self.num_step = num_step
        self.input_dim = input_dim
        self.beta = beta
        self.spike_grad = spike_grad2
        self.mem_threshold = mem_threshold

        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)

        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=0.5, learn_beta=False, 
                              learn_threshold=False, init_hidden=False, reset_mechanism="none") # 0.8
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=0.5, learn_beta=False,
                              learn_threshold=False, init_hidden=False, reset_mechanism="none") # 0.4
        # self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=0.5, learn_beta=False,
        #                       learn_threshold=False, init_hidden=False, reset_mechanism="none")
        self.dropout = nn.Dropout(dropout_rate)

        self.norm_layer = nn.LayerNorm([self.num_step, self.input_dim])
        self.reset_mem = False
        self.relu = nn.ReLU()
        self.Q_bit = 8
        self.quant_max = 1


    def forward(self, x):
        if self.reset_mem:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            # mem3 = self.lif3.init_leaky()

        # x = self.norm_layer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        scale_f=(2**self.Q_bit-1)/self.quant_max
            
        for step in range(self.num_step):
            
            input_ = x[:, step, :]
            cur1 = self.dropout(self.fc1(input_))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.dropout(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # cur3 = self.fc3(spk2)
            # spk3, mem3 = self.lif3(cur3, mem3)

            # mem1 = torch.round(scale_f*torch.clamp(mem1, -self.quant_max, self.quant_max))/scale_f
            # mem2 = torch.round(scale_f*torch.clamp(mem2, -self.quant_max, self.quant_max))/scale_f
            # mem3 = torch.round(scale_f*torch.clamp(mem3, -self.quant_max, self.quant_max))/scale_f

        # return spk3   # Training Mode only need to return the output
        return spk2