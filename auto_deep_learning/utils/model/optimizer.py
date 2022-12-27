import torch.nn as nn
from torch import optim

def get_optimizer(
    model,
    lr
):
    return optim.Adam(model.parameters(), lr=lr)