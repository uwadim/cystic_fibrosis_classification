# coding: utf-8
"""
Script for hyperparameter searching and mlflow support
"""
import os

import hydra
import mlflow

import torch  # type: ignore
import torch_geometric  # type: ignore
from torch_geometric.loader import DataLoader, ImbalancedSampler  # type: ignore

from omegaconf import DictConfig
# from sklearn.model_selection import StratifiedKFold  # type: ignore

from dataset import PDataset  # type: ignore
from model import SkipGCN  # type: ignore
from pytorchtools import seed_everything, pyg_stratified_split  # type: ignore
from train import train_valid_model, test_model  # type: ignore


@hydra.main(version_base='1.3', config_path="hp_conf", config_name="config")
def main(cfg: DictConfig) -> float:
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    seed_everything(cfg['random_state'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # results_dirpath = str(helpers.get_create_path('../results'))

    # Prepare datasets
    dataset = PDataset(cfg=cfg).shuffle()
    train_valid_dataset, train_valid_labels, test_dataset, test_labels \
        = pyg_stratified_split(dataset=dataset, labels=dataset.labels.squeeze().tolist(), fraction=0.9,
                               random_state=cfg['random_state'])
    train_dataset, train_labels, valid_dataset, valid_labels \
        = pyg_stratified_split(dataset=train_valid_dataset, labels=train_valid_labels, fraction=0.7,
                               random_state=cfg['random_state'])

    train_sampler = ImbalancedSampler(train_dataset
                                      ,num_samples=cfg.training['batch_size']
                                      )
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training['batch_size'],
                              shuffle=False,
                              sampler=train_sampler
                              )
    # On GPU shuffle=True causes random results in ROC AUC for different calls of test_loader
    valid_loader = DataLoader(valid_dataset,
                              batch_size=len(valid_dataset),
                              shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=len(test_dataset),
                             shuffle=False)

    model = SkipGCN(config=cfg,
                    num_node_features=dataset.num_node_features)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.training['learning_rate'],
                                 weight_decay=cfg.training['weight_decay'])

    mlflow.set_tracking_uri(uri=cfg.mlflow['tracking_uri'])  # type: ignore
    mlflow.set_experiment(experiment_name=cfg['experiment_name'])  # type: ignore

    with mlflow.start_run():  # type: ignore
        hp_loss = train_valid_model(model=model,
                                    device=device,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    cfg=cfg,
                                    mlflow_object=mlflow)

        test_model(model=model,
                   device=device,
                   test_loader=test_loader,
                   criterion=criterion,
                   cfg=cfg)
    return hp_loss


if __name__ == "__main__":
    main()