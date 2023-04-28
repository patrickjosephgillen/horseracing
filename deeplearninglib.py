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
    def __init__(self, fn1, fn2, XZ_columns, y_columns, parse_dates=None, vacant_stall_indicator=False, scalar=None, continuous_features=[]):

        # X = horse-specific features
        # Z = race-specific features

        self.runners_long = pd.read_csv(fn1, parse_dates=parse_dates, infer_datetime_format=True)
        self.races = pd.read_csv(fn2, parse_dates=parse_dates, infer_datetime_format=True)

        # Feature Scaling: Standardization (Standard Scaling), applied to continuous features only
        
        if scalar is None:
            self.scalar = StandardScaler()
        else:
            self.scalar = scalar

        if len(set(self.runners_long.columns).intersection(continuous_features)) > 0:
            self.runners_long.loc[:, continuous_features] = self.scalar.fit_transform(self.runners_long.loc[:, continuous_features])

        if len(set(self.races.columns).intersection(continuous_features)) > 0:
            self.races.loc[:, continuous_features] = self.scalar.fit_transform(self.races.loc[:, continuous_features])
   
        # variable indicating whether stall is vacant (1) or not (0), e.g., 10 horses leave 6 stalls vacant
        if vacant_stall_indicator:
            self.runners_long["vacant"] = 1
            XZ_columns_aug = XZ_columns + ["vacant"]
        else:
            XZ_columns_aug = XZ_columns
        
        self.runners_long = self.runners_long[["race_id", "stall_number"] + list(set(self.runners_long.columns).intersection(XZ_columns_aug)) + y_columns]
        self.runners_wide = self.runners_long.pivot(index='race_id', columns='stall_number', values=self.runners_long.columns[2:]) # credit to Cullen Sun (https://cullensun.medium.com/)
        rearranged_columns = sorted(list(self.runners_wide.columns.values))
        self.runners_wide = self.runners_wide[rearranged_columns]
        self.runners_wide = self.runners_wide.fillna(0)

        if vacant_stall_indicator:
            self.runners_wide['vacant'] = np.logical_not(self.runners_wide['vacant']).astype(int)
        
        self.y_columns = [a == 'win' for (a, b) in self.runners_wide.columns]
        self.y = self.runners_wide.iloc[:, self.y_columns].values
        self.X_columns = np.logical_not(self.y_columns)
        self.X = self.runners_wide.iloc[:, self.X_columns].values

        self.races = self.races[["race_id"] + list(set(self.races.columns).intersection(XZ_columns))]
        self.races = self.races.set_index('race_id')

        if len(set(self.races.columns).intersection(XZ_columns)) > 0:
            self.Z = self.races.values
            assert self.runners_wide.index.equals(self.races.index), "runners' and races' indices don't match up"
        else:
            self.Z = None

    def __len__(self):
        assert len(self.runners_wide) == len(self.races), "different lengths of runners_wide and races"
        return len(self.runners_wide)

    def __getitem__(self, idx):
        if self.Z is not None:
            return np.hstack((self.X[idx], self.Z[idx])), self.y[idx]
        else:
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

class ParsLin(nn.Module):
    """ Parsimonious version of Linear """
    def __init__(self, input_layer_nodes, output_layer_nodes):
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
        # Reshape races tensor to separate features and horses
        n = x.shape[0]
        reshaped_input = x.view(n, self.coefficient_size, self.output_size)

        # Transpose tensor to have each horse's features together
        transposed_input = reshaped_input.transpose(1, 2)

        # Multiply transposed tensor with coefficients tensor (broadcasted along last dimension)
        marginal_utilities = transposed_input * self.weights

        # Sum multiplied tensor along last dimension
        utilities = marginal_utilities.sum(dim=-1)
        
        return utilities

class MLR(nn.Module):
    """ Parsimonious version of LinSoft intended to replicate a multinomial logit regression with
    alternative specific variables and generic coefficients only """
    def __init__(self, input_layer_nodes, output_layer_nodes, bias=None): # bias is unused argument and will be ignored
        super().__init__()

        self.neural_network = nn.Sequential(
            ParsLin(input_layer_nodes, output_layer_nodes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.neural_network(x)
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