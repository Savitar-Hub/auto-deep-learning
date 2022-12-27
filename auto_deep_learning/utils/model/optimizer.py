import torch.nn as nn
from torch import optim

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.0003)