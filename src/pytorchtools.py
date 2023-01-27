import os
import random
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch  # type: ignore
import torch_geometric  # type: ignore


def pyg_stratified_split(dataset: torch_geometric.data.Dataset,
                         labels: list,
                         fraction: float,
                         random_state: Optional[int] = None) -> Tuple[torch_geometric.data.Dataset, list,
torch_geometric.data.Dataset, list]:
    """Make stratified split of a Pytorch geometric dataset
    into train and test sets

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Dataset to split
    labels : list
        labels of the dataset, used for classification
    fraction : float
        fraction of the TRAIN dataset
    random_state : int, optional
        random state for the random.sampler

    Returns
    -------
    Tuple[torch_geometric.data.Dataset, list, torch_geometric.data.Dataset, list]
        Tuple containing train subset, train labels, test subset, test labels
    """

    if random_state:
        random.seed(random_state)

    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)

    train_indices_list = []  # type: ignore
    test_indices_list = []  # type: ignore
    for indices in indices_per_label.values():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        train_indices_list.extend(random_indices_sample)
        test_indices_list.extend(set(indices) - set(random_indices_sample))

    train_subset = dataset.index_select(train_indices_list)
    train_labels = list(map(labels.__getitem__, train_indices_list))
    test_subset = dataset.index_select(test_indices_list)
    test_labels = list(map(labels.__getitem__, test_indices_list))
    return train_subset, train_labels, test_subset, test_labels


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    original from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model) -> int:

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.counter

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
