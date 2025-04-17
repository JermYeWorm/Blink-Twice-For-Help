# imports
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

############################### DATA PARSING AND FORMATTING ################################
# load and parse encoded data from Jeremy's sim
def importcsv(file_path, output_arr):
   with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        output_arr.append(row)

# TRAINING data -----------------------------------------------------------------------
file_path='C:\\Users\\Sierra\\OneDrive - University of Maryland\\UMD\\Spring 2025\\ENEE408D\\EEG Simulation\\Spike Encoding\\spike_enc_output.csv'
spks = []
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
labels = [1. if x >= 2 else x for x in targets[1]]

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

# TEST data--------------------------------------------------------------------------
file_path='C:\\Users\\Sierra\\OneDrive - University of Maryland\\UMD\\Spring 2025\\ENEE408D\\EEG Simulation\\Spike Encoding\\spike_enc_output_test.csv'
spks_test = []
importcsv(file_path, spks_test)
spks_test[0] = [float(x) for x in spks_test[0]]
spks_test[1] = [float(x) for x in spks_test[1]]
file_path = 'EEG-EyeBlinks/EEG-IO/S14_labels.csv'
targetsRaw = []
importcsv(file_path, targetsRaw) # targets starts at index 3
# reformat target
targetsRaw = targetsRaw[3:-1]
targets_test = [[],[]]
for i in range(len(targetsRaw)):
   targets_test[0].append(float(targetsRaw[i][0]))
   targets_test[1].append(float(targetsRaw[i][1]))
labels_test = [1. if x >= 2 else x for x in targets_test[1]]

# parse blinks into windows
blink_window_test = int(0.5/(spks_test[0][1] - spks_test[0][0])) # +/- num of indices around center of spike
blink_sets_test = [[], []]
spks_pos_test = [[], []]
spks_neg_test = [[], []]
tmp_pos = []
tmp_neg = []
data_test = []
err = 0.002
for center in targets_test[0]:
   for i, s in enumerate(spks_test[0]):
    if (i >= blink_window_test and i <= len(spks_test[0])-blink_window_test) and (s == center or (s <= center+err and s >= center-err)):
        # total inputs 
        blink_sets_test[0].append(spks_test[0][i-blink_window_test:i+blink_window_test]) # time
        blink_sets_test[1].append(spks_test[1][i-blink_window_test:i+blink_window_test]) # spikes
        # separate into pos and neg spike inputs
        spks_pos_test[0].append(spks_test[0][i-blink_window_test:i+blink_window_test]) # time
        spks_neg_test[0].append(spks_test[0][i-blink_window_test:i+blink_window_test]) # time
        for x in spks_test[1][i-blink_window_test:i+blink_window_test]:
            y = x
            if x < 0:
              x = 0.0
            tmp_pos.append(x)
            if y > 0:
               y = 0.0
            tmp_neg.append(y)
        spks_pos_test[1].append(tmp_pos) # spikes
        spks_neg_test[1].append(tmp_neg) # spikes
        data_test.append([tmp_pos, tmp_neg]) # I don't think I need to include time right? <---------------------?
        tmp_pos = []
        tmp_neg = []
        break
        # this creates an array where each index is a singular blink with window 'blink_window'

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
    
#batch_size = len(data)
batch_size = 30
# Convert to torch tensor and rearrange dimensions
data_tensor = torch.tensor(data, dtype=torch.float)  # shape: [30, 2, 256]
data_tensor = data_tensor.permute(2, 0, 1)  # now shape: [256, 30, 2]
labels_tensor = torch.tensor(labels, dtype=torch.long)  # shape: [30]
from torch.utils.data import TensorDataset
dataset = TensorDataset(data_tensor.permute(1, 0, 2), labels_tensor)  # [batch, time, input]
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Convert to torch tensor and rearrange dimensions
data_tensor_test = torch.tensor(data_test, dtype=torch.float)  # shape: [30, 2, 256]
data_tensor_test = data_tensor_test.permute(2, 0, 1)  # now shape: [256, 30, 2]
labels_tensor_test = torch.tensor(labels_test, dtype=torch.long)  # shape: [30]
dataset_test = TensorDataset(data_tensor_test.permute(1, 0, 2), labels_tensor_test)  # [batch, time, input]
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

# data = np.array(data).reshape(blink_window*2,batch_size, 2)
# dataset = CustomDataset(data, labels)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# I don't know what to do about the test_loader <-------------------------------------------------------------?

#################################### NETWORK DEFINITIION #######################################
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Network Architecture
num_inputs = 2
num_hidden = 4
num_outputs = 2

# Temporal Dynamics
num_steps = 256
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        
# Load the network onto CUDA if available
net = Net().to(device)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    #output, _ = net(data.view(batch_size, -1))
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

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        #data = data.to(device)
        data = data.permute(1, 0, 2).to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        #spk_rec, mem_rec = net(data.view(batch_size, -1))
        spk_rec, mem_rec = net(data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            #test_data = test_data.to(device)
            test_data = test_data.permute(1, 0, 2).to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer(
                    data, targets, epoch,
                    counter, iter_counter,
                    loss_hist, test_loss_hist,
                    test_data, test_targets)
            counter += 1
            iter_counter +=1



# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
#plt.plot(test_loss_hist)
plt.title("Loss Curves")
#plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
