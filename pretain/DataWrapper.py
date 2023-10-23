import os

import pandas as pd
from pathlib import Path

from pretain import Construct_dataset
from utils.DataEmbedding import DataEmbedding


class DataWrapper:
    def __init__(self, data_root, features_dir='Data', dataset_name=None, seed=42):
        self.data_root = data_root
        self.dataset_name = dataset_name

        self.load_data()

        self.features_dir = Path(features_dir)
        self.features_path = self.features_dir / f"{self.dataset_name}.pt"
        if not self.features_path.parent.exists():
            os.makedirs(self.features_path.parent)

        self._process()



    @property
    def num_samples(self):
        return self.train_data_size


    @property
    def dim_features(self):
        return self._dim_features


    def load_data(self):
        self.smiles_col = 'smiles'
        dataset_path = Path(self.data_root)
        self.whole_data_df = pd.read_csv(dataset_path)
        self.train_data_size = len(self.whole_data_df)


    def _process(self):
        perpaper = DataEmbedding(data_df=self.whole_data_df,
                                 features_path=self.features_path,
                                 data_root=self.data_root,
                                 dataset_name=self.dataset_name)
        perpaper.process()

        self._dim_features = perpaper.dim_features


    def get_pretain_dataset(self, batch_size):
        pretain_dataset = Construct_dataset.constrcurt_dataset(self.features_path)
        pretain_loader = Construct_dataset.construct_loader(pretain_dataset, batch_size=batch_size)

        return pretain_loader




