import torch
import torch.nn as nn
from typing import Optional, Tuple
import layers


class CustomModel(nn.Module):
    def __init__(self) -> None:
        super(CustomModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

