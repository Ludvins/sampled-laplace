
import torch
from torch import nn
import math

import torch.nn.functional as F


def get_mlp(name):
    torch.manual_seed(2147483647)
    
    layers = []
    activation = torch.nn.Tanh
    dims = [784, 200, 200, 10]
    for i, (_in, _out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(
            torch.nn.Linear(_in, _out)
        )

        if i != len(dims) -2:       
            layers.append(activation()) 
            layers.append(nn.Dropout(0.1))
    
    model = torch.nn.Sequential(*layers)

    # Load weights

    model.load_state_dict(torch.load("weights/" + name))
    model.eval()

    return model