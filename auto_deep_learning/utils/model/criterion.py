import torch.nn as nn


def get_criterion():
    criterion = nn.CrossEntropyLoss()

    return criterion
