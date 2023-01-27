import argparse
import os.path as osp
from collections import defaultdict

import torch.nn.functional as F
import hydra
import torch  # type: ignore

# from sklearn.model_selection import StratifiedKFold  # type: ignore


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv
from model import SkipGCN



class DummerClass:
    training = {'embedding_size': 8}

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    return [
        int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        for mask in [data.train_mask, data.val_mask, data.test_mask]
    ]


def main(args):
    cfg = DummerClass()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
    #            hidden_channels=args.hidden_channels, device=device)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)

    # model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
    model = SkipGCN(config=cfg, num_node_features=dataset.num_features)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    # parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    main(args)  # type: ignore
