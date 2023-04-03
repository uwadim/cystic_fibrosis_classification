import argparse
import os.path as osp
from collections import defaultdict

import numpy as np
import hydra
import mlflow
import torch  # type: ignore
from torch import sigmoid
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

# from sklearn.model_selection import StratifiedKFold  # type: ignore

from torch_geometric.datasets import TUDataset
from torch_geometric.logging import log
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gmp
from model import SkipGCN
from train import reset_model


class DummerClass:
    training = {'embedding_size': 8}


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = gmp(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    step = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.float().unsqueeze(-1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
        step += 1
    return running_loss / step


@torch.no_grad()
def test(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    step = 0
    batch_accuracy = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(out, data.y.float().unsqueeze(-1))  # Compute the loss.
        pred = sigmoid(out)
        y_true = data.y.detach().to('cpu').numpy()
        y_pred = pred.detach().to('cpu').numpy()
        batch_accuracy.append(accuracy_score(y_true, y_pred.round()))
        running_loss += loss.item()
        step += 1
    return np.mean(batch_accuracy), running_loss / step  # Derive ratio of correct predictions.


@hydra.main(version_base='1.3', config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(root='../data/TUDataset', name='MUTAG')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # model = GCN(hidden_channels=64, dataset=dataset)  # type: ignore
    model = SkipGCN(config=cfg, num_node_features=dataset.num_node_features)
    reset_model(model)  # Reinitialize layers
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training['learning_rate'])
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri(uri=cfg.mlflow['tracking_uri'])  # type: ignore
    mlflow.set_experiment(experiment_name=cfg['experiment_name'])  # type: ignore
    with mlflow.start_run():  # type: ignore
        for epoch in range(1, 201):
            train_loss = train(model, train_loader, criterion, optimizer)
            train_acc, _ = test(model, train_loader, criterion)
            test_acc, test_loss = test(model, test_loader, criterion)
            mlflow.log_metric('train loss', train_loss, step=epoch)  # type: ignore
            mlflow.log_metric('train accuracy', train_acc, step=epoch)  # type: ignore
            mlflow.log_metric('test loss', test_loss, step=epoch)  # type: ignore
            mlflow.log_metric('test accuracy', test_acc, step=epoch)  # type: ignore
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()  # type: ignore
