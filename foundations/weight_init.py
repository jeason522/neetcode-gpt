import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in + fan_out))
        ini_weights = torch.randn(fan_out, fan_in) * std
        return torch.round(ini_weights, decimals=4).numpy().tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        ini_weights = torch.randn(fan_out, fan_in) * std
        return torch.round(ini_weights, decimals=4).numpy().tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        stds = []
        weights = []
        cur_dim = input_dim
        
        for _ in range(num_layers):
            fan_in = cur_dim
            fan_out = hidden_dim
            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)
            else:
                std = 1.0
            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)
            cur_dim = hidden_dim
        
        x = torch.randn(input_dim)
        for W in weights:
            z = torch.matmul(W, x)
            x = torch.relu(z)
            stds.append(round(float(torch.std(x)), 2))
        return stds