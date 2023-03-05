import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import config


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super(CustomDataset, self).__init__()

        self.transform = A.Compose([
            A.Resize(config.image_size[0], config.image_size[1]),
            ToTensorV2()
        ])

    def __len__(self):
        pass

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        pass
