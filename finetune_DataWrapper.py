import json
import os

import numpy as np
import pandas as pd
from pathlib import Path

from finetune import finetune_Construct_dataset
from utils.DataEmbedding import DataEmbedding
from sklearn.model_selection import train_test_split
from utils.util import get_dataset_smiles_target
from utils.util import NumpyEncoder
from utils.util import filter_invalid_smiles

class DataWrapper:
    def __init__(self, data_root, features_dir='Data', dataset_name=None, seed=42):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.seed = seed

        self.load_data()

        self.features_dir = Path(features_dir)
        self.features_path = self.features_dir / f"{self.dataset_name}.pt"
        if not self.features_path.parent.exists():
            os.makedirs(self.features_path.parent)

        self._process()

        self.split_dir = Path("Split")
        self.splits_filename = self.split_dir / f"{self.dataset_name}_splits_seed{self.seed}.json"
        if not self.splits_filename.parent.exists():
            os.makedirs(self.splits_filename.parent)
        if not self.splits_filename.exists():
            self.splits = []
            self.make_splits()
        else:
            self.splits = json.load(open(self.splits_filename, "r"))




    @property
    def num_samples(self):
        return self.train_data_size


    @property
    def dim_features(self):
        return self._dim_features


    def load_data(self):
        self.smiles_col, self.target_col = get_dataset_smiles_target(self.dataset_name)

        dataset_path = Path(self.data_root)
        self.whole_data_df = pd.read_csv(dataset_path)

        valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
        print(len(valid_smiles))
        self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)
        self.train_data_size = len(self.whole_data_df)




    def _process(self):
        perpaper = DataEmbedding(data_df=self.whole_data_df,
                                 features_path=self.features_path,
                                 data_root=self.data_root,
                                 dataset_name=self.dataset_name)
        perpaper.process()

        self._dim_features = perpaper.dim_features


    def get_finetune_dataset(self, batch_size):
        indices = self.splits[0]
        train_indices = indices['train']
        valid_indices = indices['valid']
        test_indices = indices['test']

        train_dataset, valid_dataset, test_dataset, voc, test_xulei_dataset = finetune_Construct_dataset.constrcurt_dataset(self.features_path, train_idxs=train_indices, valid_idxs=valid_indices, test_idxs=test_indices)
        train_loader, valid_loader, test_loader, test_xulei_loader = finetune_Construct_dataset.construct_loader(train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, batch_size=batch_size, test_xulei_dataset=test_xulei_dataset)

        return train_loader, valid_loader, test_loader, voc, test_xulei_loader


    def make_splits(self):
        all_idxs = np.arange(self.train_data_size)
        # smiles = self.whole_data_df[:,self.smiles_col].values

        train_valid_split, test_split = train_test_split(all_idxs, test_size=0.1, random_state=self.seed)
        train_split, valid_split = train_test_split(train_valid_split, test_size=0.1, random_state=self.seed)
        split = {"train": [], "valid": [], "test": []}
        # split.append({"train": train_split, "valid": valid_split, "test": test_split})
        split["train"].append(train_split)
        split["valid"].append(valid_split)
        split["test"].append(test_split)
        self.splits.append(split)

        with open(self.splits_filename, "w") as f:
            json.dump(self.splits, f, cls=NumpyEncoder)






