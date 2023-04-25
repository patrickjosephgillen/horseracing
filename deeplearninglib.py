import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------

class RacesDataset(Dataset):
    def __init__(self, filename, X_columns, y_columns, parse_dates=None, vacant_stall_indicator=False, scalar=None):

        self.runners = pd.read_csv(filename, parse_dates=parse_dates, infer_datetime_format=True)

        # variable indicating whether stall is vacant (1) or not (0), e.g., 10 horses leave 6 stalls vacant
        if vacant_stall_indicator:
            self.runners["vacant"] = 1
            X_columns_aug = X_columns + ["vacant"]
        else:
            X_columns_aug = X_columns
        
        self.runners = self.runners[["race_id", "stall_number"] + X_columns_aug + y_columns]
        self.races = self.runners.pivot(index='race_id', columns='stall_number', values=self.runners.columns[2:]) # credit to Cullen Sun (https://cullensun.medium.com/)
        rearranged_columns = sorted(list(self.races.columns.values))
        self.races = self.races[rearranged_columns]
        self.races = self.races.fillna(0)

        if vacant_stall_indicator:
            self.races['vacant'] = np.logical_not(self.races['vacant']).astype(int)
        
        self.y_columns = [a == 'win' for (a, b) in self.races.columns]
        self.y = self.races.iloc[:, self.y_columns].values
        self.X_columns = np.logical_not(self.y_columns)
        self.X = self.races.iloc[:, self.X_columns].values

        # Feature Scaling: Standardization (Standard Scaling)
        
        # Tried scaling continuous variables only and not output variable but results were significantly worse

        if scalar is None:
            self.scalar = StandardScaler()
            self.X = self.scalar.fit_transform(self.X)
        else:
            self.scalar = scalar
            self.X = self.scalar.transform(self.X)
   
    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------------------------------------------

class LinSig(nn.Module):
    def __init__(self, input_layer_nodes, output_layer_nodes, bias=False):
        super().__init__()

        self.neural_network = nn.Sequential(
            nn.Linear(input_layer_nodes, output_layer_nodes, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.neural_network(x)
        return logits

class LinSoft(nn.Module):
    def __init__(self, input_layer_nodes, output_layer_nodes, bias=False):
        super().__init__()

        self.neural_network = nn.Sequential(
            nn.Linear(input_layer_nodes, output_layer_nodes, bias=bias),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.neural_network(x)
        return logits

class LinDropReluLinSoft(nn.Module):
    def __init__(self, input_layer_nodes, output_layer_nodes, bias=False):
        super().__init__()

        hidden_layer_nodes = round(math.sqrt(input_layer_nodes * output_layer_nodes)) # credit to Harpreet Singh Sachdev (https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev/)

        self.neural_network = nn.Sequential(
            nn.Linear(input_layer_nodes, hidden_layer_nodes, bias=bias),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hidden_layer_nodes, output_layer_nodes, bias=bias),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.neural_network(x)
        return logits

######################
# UNDER CONSTRUCTION #
######################

class MLR(nn.Module):
    """ Parsimonious version of LinSoft intended to replicate a multinomial logit regression with
    alternative specific variables and generic coefficients only """
    def __init__(self, input_layer_nodes, output_layer_nodes, bias=None): # bias is unused argument and will be ignored
        super().__init__()
        
        # Check if output_layer_nodes is an integer multiple of input_layer_nodes
        if input_layer_nodes % output_layer_nodes != 0:
            raise ValueError("inputt_layer_nodes must be an integer multiple of output_layer_nodes")
        
        self.input_size = input_layer_nodes
        self.output_size = output_layer_nodes
        
        self.coefficient_size = input_layer_nodes // output_layer_nodes

        weights = torch.zeros(self.coefficient_size)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
    
        # Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.weights.view(1, self.coefficient_size))

    def forward(self, x):
        # Reshape the weights tensor to (1, coefficient_size) and expand it to match the number of nodes in the input layer
        expanded_weights = self.weights.view(1, self.coefficient_size).expand(self.output_size, self.coefficient_size)

        # Reshape the input_layer_nodes tensor to (n, output_size, coefficient_size)
        n = x.shape[0]
        reshaped_input = x.view(n, self.output_size, self.coefficient_size)

        # Calculate the element-wise product and sum along the second dimension to compute the utilities
        # utilities = torch.sum(expanded_weights * reshaped_input, dim=2)

        # logits = torch.nn.functional.softmax(utilities, dim=1)

        # utilities = torch.mm(reshaped_input, expanded_weights.t())
        utilities = torch.matmul(reshaped_input, self.weights.view(1, self.coefficient_size).t())

        logits = torch.nn.functional.softmax(utilities.view(n, self.output_size), dim=1)

        return logits

######################
# FOR REFERENCE ONLY #
######################

class MyLinearLayer(nn.Module): # Source: https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
    """ Custom linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b

# ----------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"): # source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # load data to correct device
        X = X.to(device)
        y = y.to(device)
        
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device="cpu"): # source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # load data to correct device
            X = X.to(device)
            y = y.to(device)
        
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

