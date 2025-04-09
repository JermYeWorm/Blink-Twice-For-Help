# SNN Training Using STDP Unsupervised Algorithm
# Most code taken from: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb#scrollTo=QXZ6Tuqc9Q-l

########################### IMPORTS ######################################
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv

########################## NEURON MODEL ####################################
# snntorch only has variation of LIF neurons, but we need Izhikevech model
# code and numbers from Yihui's matlab sim
class izhikevech(nn.Module):
  def __init__(self, u0, k=0.4471817006977834, a=0.0032799410036917333, b=24.478421990208606, Vmin=-66.46563513097735,
               d=50.0, C=38.0, Vr=-77.40291336465064, Vt=-44.90054428048817, Vpeak=15.489726771001997):
      super(izhikevech, self).__init__()

      # initialize parameters of IZ model
      self.u = u0
      self.v = Vr
      self.k = k
      self.a = a
      self.b = b
      self.Vmin = Vmin
      self.d = d
      self.C = C
      self.Vr = Vr
      self.Vt = Vt
      self.Vpeak = Vpeak
      
  
  # the forward function is called each time we call izhikevech
  def forward(self, v, u, I, mem, h):
    # calculate t+1 point
    # v = mem potential, u = recovery var
    if (v>= self.Vpeak):
        v = self.Vmin
        u = u + self.d
    else:
        v = v + h * ( self.k * ( v - self.Vr ) * ( v - self.Vt) - u + I)/self.C
        u = u + h * ( self.a * ( self.b * ( v-self.Vr ) - u ) )
    return v, u

############################### DATA PARSING AND FORMATTING ################################
# load and parse encoded data from Jeremy's sim
def importcsv(file_path, output_arr):
   with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        output_arr.append(row)

file_path='C:\\Users\\Sierra\\OneDrive - University of Maryland\\UMD\\Spring 2025\\ENEE408D\\EEG Simulation\\Spike Encoding\\spike_enc_output.csv'
spks = []
# with open(file_path, 'r') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         spks.append(row)
importcsv(file_path, spks)
spks[0] = [float(x) for x in spks[0]]
spks[1] = [float(x) for x in spks[1]]
file_path = 'EEG-EyeBlinks/EEG-IO/S16_labels.csv'
targetsRaw = []
importcsv(file_path, targetsRaw) # targets starts at index 3
# reformat target
targetsRaw = targetsRaw[3:-1]
targets = [[],[]]
for i in range(len(targetsRaw)):
   targets[0].append(float(targetsRaw[i][0]))
   targets[1].append(float(targetsRaw[i][1]))
labels = [x for x in targets[1]]

# parse blinks into windows
blink_window = int(0.5/(spks[0][1] - spks[0][0])) # +/- num of indices around center of spike
blink_sets = [[], []]
spks_pos = [[], []]
spks_neg = [[], []]
tmp_pos = []
tmp_neg = []
data = []
err = 0.002
for center in targets[0]:
   for i, s in enumerate(spks[0]):
    if (i >= blink_window and i <= len(spks[0])-blink_window) and (s == center or (s <= center+err and s >= center-err)):
        # total inputs 
        blink_sets[0].append(spks[0][i-blink_window:i+blink_window]) # time
        blink_sets[1].append(spks[1][i-blink_window:i+blink_window]) # spikes
        # separate into pos and neg spike inputs
        spks_pos[0].append(spks[0][i-blink_window:i+blink_window]) # time
        spks_neg[0].append(spks[0][i-blink_window:i+blink_window]) # time
        for x in spks[1][i-blink_window:i+blink_window]:
            y = x
            if x < 0:
              x = 0.0
            tmp_pos.append(x)
            if y > 0:
               y = 0.0
            tmp_neg.append(y)
        spks_pos[1].append(tmp_pos) # spikes
        spks_neg[1].append(tmp_neg) # spikes
        data.append([tmp_pos, tmp_neg]) # I don't think I need to include time right? <---------------------?
        tmp_pos = []
        tmp_neg = []
        break
        # this creates an array where each index is a singular blink with window 'blink_window'

# TODO create digital signal from stem spikes to represent actual spike inputs
# actually, I don't think I need to do this....
# snn_clk = 1e6
# period = 1./snn_clk
# input_window = np.linspace(0, 1, 1/period)

# now convert the data into a torch.Tensor object (ugh.)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float)  # shape: [2, window_size]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)  # shape: [2] or int
        #return self.data[idx], self.labels[idx]
        return data_tensor, label_tensor
    
batch_size = len(data)
dataset = CustomDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# I don't know what to do about the test_loader <-------------------------------------------------------------?

#################################### NETWORK DEFINITIION #######################################
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network Architecture
num_inputs = 2
num_hidden = 4
num_outputs = 2

# Temporal Dynamics
num_steps = len(spks)
u = 0
v = 0

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.iz1 = izhikevech(0)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.iz2 = izhikevech(0)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.iz1.v
        mem2 = self.iz2.v
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.iz1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.iz2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        
# Load the network onto CUDA if available
net = Net().to(device)

###################################### TRAINING #########################################
# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# one iteration of training
dataa, targetss = next(iter(train_loader))
dataa = dataa.to(device)
targetss = targetss.to(device)

spk_rec, mem_rec = net(dataa)
print(mem_rec.size())

# # initialize the total loss value
# loss_val = torch.zeros((1), dtype=dtype, device=device)

# # sum loss at every step
# for step in range(num_steps):
#   loss_val += loss(mem_rec[step], targets)

# print(f"Training loss: {loss_val.item():.3f}")
# print_batch_accuracy(data, targets, train=True)

# # clear previously stored gradients
# optimizer.zero_grad()

# # calculate the gradients
# loss_val.backward()

# # weight update
# optimizer.step()

# # calculate new network outputs using the same data
# spk_rec, mem_rec = net(data.view(batch_size, -1))

# # initialize the total loss value
# loss_val = torch.zeros((1), dtype=dtype, device=device)

# # sum loss at every step
# for step in range(num_steps):
#   loss_val += loss(mem_rec[step], targets)

# print(f"Training loss: {loss_val.item():.3f}")
# print_batch_accuracy(data, targets, train=True)