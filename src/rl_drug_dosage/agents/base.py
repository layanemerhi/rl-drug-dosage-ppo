"""Shared utilities for neural network agents."""

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
