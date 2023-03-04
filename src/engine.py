import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import warnings
warnings.simplefilter('ignore')

from typing import Tuple
from tqdm import tqdm
import copy
import wandb
from colorama import Fore, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

from src.config import config
from src.models.utils import get_optimizer, get_scheduler
from src.models.model import CustomModel
from src.losses import CustomLoss


def train_one_epoch(
    model: nn.Module,
    optimizer: optim, 
    scheduler: optim.lr_scheduler, 
    criterion: nn.Module, 
    train_loader: torch.utils.data.DataLoader
) -> float:
    '''
    This function trains the model for one epoch and logs the values to wandb.

    Args:
    ----
    * model  : nn.Module
        - The model to be trained
    * optimizer  : optim
        - The optimizer of the function
    * scheduler  : optim.lr_scheduler
        - The learning scheduler for the model
    * criterion  : nn.Module
        - The loss function to be optimized
    * train_loader  : torch.utils.data.DataLoader
        - A dataloader for the train dataset

    Return:
    ------
    * epoch_loss  : float    
        - Returns the loss for the current epoch
    '''
    model.train()
    running_loss = 0.0
    dataset_size = 0.0
    epoch_loss   = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for step, batch in pbar:
        batch_size = batch['X'].shape[0]
        X, y = batch['X'].to(config.device), batch['y'].to(config.device)

        yHat = model.forward(X)

        loss = criterion(yHat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            'train loss': loss, 
            'train current lr': current_lr
        })

        pbar.set_postfix(
            loss=f'{loss:.5f}', lr=f'{current_lr:.5f}')
        
    return epoch_loss


@torch.no_grad()
def validate_one_epochs(
    model: nn.Module, 
    criterion: nn.Module,
    valid_loader: torch.utils.data.DataLoader
) -> float:
    '''
    This function validates through the validation dataset

    Args:
    ----
    * model  : nn.Module
        - The model to be evaluated
    * criterion  : nn.Module
        - The optimize function we want to optimize
    * valid_loader  : torch.utils.data.DataLoader

    Return:
    ------
    * epoch_loss  : float
        - The aggregate from the entire epoch
    '''
    model.eval()
    running_loss = 0.0
    dataset_size = 0.0
    epoch_loss   = 0.0

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for step, batch in pbar:
        batch_size = batch['X'].shape[0]
        X, y = batch['X'].to(config.device), batch['y'].to(config.device)

        yHat = model.forward(X)

        loss = criterion(yHat, y)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        wandb.log({
            'validation loss': loss
        })

    return epoch_loss


def run_training(
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader,
    fold: int
) -> Tuple[nn.Module, dict]:
    wandb.init(
        project=config.project_name,
        config={k:v for k, v in vars(config).items() if '__' not in k},
        name=f'iteration-{config.iteration_num}',
        group=config.comment
    )

    model = CustomModel().to(device=config.device)
    optimizer = get_scheduler(model=model)
    scheduler = get_scheduler(optimizer=optimizer)
    criterion = CustomLoss()

    history = {
        'learning_rate': [], 
        'train_loss': [],
        'valid_loss': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_loss  = np.infty

    wandb.watch(models=[model], log_freq=100)

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            criterion=criterion, 
            train_loader=train_loader
        )

        valid_loss = validate_one_epochs(
            model=model, 
            criterion=criterion, 
            valid_loader=valid_loader
        )

        wandb.log({
            'learning_rate_at_epoch_end': optimizer.param_groups[0]['lr'],
            'train_loss_at_epoch_end': train_loss, 
            'valid_loss_at_epoch_end': valid_loss
        })

        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_loss:
            print(f'{c_}Validation loss decreased from {best_loss} to {valid_loss}')
            best_loss = valid_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

            path = f'../data/models/model-{fold}.bin'
            torch.save(best_model_wts, path)

            print(f'[MODEL SAVED]{sr_}')
    
    model.load_state_dict(torch.load(f'../data/models/model-{fold}.bin'))
    return model, history
