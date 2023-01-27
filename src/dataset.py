import contextlib
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd  # type: ignore
import torch  # type: ignore
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset  # type: ignore
from tqdm import tqdm


class PDataset(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 test=False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """Custom dataset class for graphs. Data are not split into train and test sets

        Parameters
        ----------
        cfg : DictConfig
            Dictionary of configuration parameters from Hydra
        test : bool
            flag indicating that Dataset is for testing
        transform : Callable, optional
        pre_transform : Callable, optional
        """
        self.cfg = cfg
        self.test = test
        self.ids: list = []
        self.labels: list = []
        super(PDataset, self).__init__(cfg.data['root_directory'], transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        # list of paths for network data and original data
        return [self.cfg.data['network_fpath'], self.cfg.data['original_fpath']]

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        original_data = pd.read_csv(self.cfg.data['original_fpath'])
        self.ids = original_data[self.cfg.data['id_colname']].to_list()
        return [f'data_{i}.pt' for i in self.ids]

    def download(self):
        pass

    def process(self):
        data = pd.read_csv(self.cfg.data['network_fpath'])
        original_data = pd.read_csv(self.cfg.data['original_fpath'])
        # Replace node indices 'N1', 'N2', ... 'X1', 'X2' to integers
        replace_dict = {val: idx for idx, val in
                        enumerate(pd.concat([data['p1'], data['p2']], axis=0, ignore_index=True).unique())}
        for col in ['p1', 'p2']:
            data[col].replace(replace_dict, inplace=True)
        edge_indices = torch.tensor(data[['p1', 'p2']].T.to_numpy())
        # If the nodes have no features
        #node_features = torch.tensor(np.ones(num_of_nodes).reshape(-1, 1))

        for index, colname in enumerate(tqdm(self.ids, total=len(self.ids))):
            # single patient
            original_data_elem = original_data.query(f'`{self.cfg.data["id_colname"]}` == "{colname}"')
            edge_weights = torch.tensor(data[str(colname)].to_numpy()).float()
            node_features = \
                torch.tensor(original_data_elem.drop(columns=[self.cfg.data['label_colname'],
                                                              self.cfg.data["id_colname"]]).T.to_numpy()).float()
            label = self._get_labels(
                original_data_elem[self.cfg.data['label_colname']])
            self.labels.append(label)
            # Create data object
            data_to_save = Data(x=node_features,
                                edge_index=edge_indices,
                                # edge_attr=None,
                                edge_weight=edge_weights,
                                y=label,
                                graph_index=colname)
            is_test = False
            with contextlib.suppress(AttributeError):
                if self.test:  # Check self.test exists and True
                    is_test = True
            if is_test:
                torch.save(data_to_save, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data_to_save, os.path.join(self.processed_dir, f'data_{index}.pt'))
        self.labels = torch.cat(self.labels)

    @staticmethod
    def _get_labels(label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.ids)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        is_test = False
        with contextlib.suppress(AttributeError):
            if self.test:  # Check self.test exists and True
                is_test = True
        return torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt')) if is_test else torch.load(
            os.path.join(self.processed_dir, f'data_{idx}.pt'))
