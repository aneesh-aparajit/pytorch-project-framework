import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import gc
import os, shutil
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

from src.config import config
from src.engine import run_training
from src.datasets.dataset import CustomDataset
from src.models.model import CustomModel
from src.losses import CustomLoss
from src.models.utils import get_optimizer, get_scheduler


if __name__ == '__main__':
    df = pd.read_csv(config.train_path)

    for fold in range(config.num_folds):
        print('#'*15)
        print(f'### Fold #{fold}')
        print('#'*15)

        train_df = df[df.kfold != fold]
        valid_df = df[df.kfold == fold]

        # TODO: Preprocess the data accordingly

        train_dataset = CustomDataset(df=train_df)
        valid_dataset = CustomDataset(df=valid_df)

        train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_bs)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.valid_bs)

        model, history = run_training(
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            fold=fold
        )

        gc.collect()
