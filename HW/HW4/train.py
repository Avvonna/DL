import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    skip_warmup: int = 1
):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    :param skip_warmup: number of initial epochs to skip when plotting (default: 1)
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(skip_warmup + 1, len(train_losses) + 1), train_losses[skip_warmup:], label='train')
    axs[0].plot(range(skip_warmup + 1, len(val_losses) + 1), val_losses[skip_warmup:], label='val')
    axs[0].set_ylabel('loss')

    # """
    # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
    # Calculate train and validation perplexities given lists of losses
    # """
    train_perplexities = [np.exp(loss) for loss in train_losses]
    val_perplexities = [np.exp(loss) for loss in val_losses]

    axs[1].plot(range(skip_warmup + 1, len(train_perplexities) + 1), train_perplexities[skip_warmup:], label='train')
    axs[1].plot(range(skip_warmup + 1, len(val_perplexities) + 1), val_perplexities[skip_warmup:], label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = model.device
    train_loss = 0.0
    total_tokens = 0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Process one training step: calculate loss,
        # call backward and make one optimizer step.
        # Accumulate sum of losses for different batches in train_loss
        # """
        indices = indices.to(device, dtype=torch.long)
        lengths = lengths.to(device, dtype=torch.long)

        pred = model.forward(indices, lengths)  # логиты в формате (B * L * Vocab_size)

        pred_reshaped = pred[:, :-1, :].contiguous().view(-1, pred.size(-1))
        target_reshaped = indices[:, 1:].contiguous().view(-1)

        loss = criterion(pred_reshaped, target_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reduction="sum" - loss идет как сумма ошибок на токенах
        train_loss += loss.item()
        total_tokens += (lengths - 1).sum().item()

    train_loss /= total_tokens
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = model.device
    val_loss = 0.0
    total_tokens = 0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Process one validation step: calculate loss.
        # Accumulate sum of losses for different batches in val_loss
        # """
        indices = indices.to(device, dtype=torch.long)
        lengths = lengths.to(device, dtype=torch.long)

        pred = model.forward(indices, lengths)

        pred_reshaped = pred[:, :-1, :].contiguous().view(-1, pred.size(-1))
        target_reshaped = indices[:, 1:].contiguous().view(-1)

        loss = criterion(pred_reshaped, target_reshaped)

        val_loss += loss.item()
        total_tokens += (lengths - 1).sum().item()

    val_loss /= total_tokens
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(
        reduction="sum",
        ignore_index=model.tokenizer.pad_id()
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        # if scheduler is not None:
        #     scheduler.step()
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())
