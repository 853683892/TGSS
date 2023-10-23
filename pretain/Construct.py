import os.path
from pathlib import Path
from pretain.DataWrapper import *
from pretain.DataLoader import *


def construct_dataset(dataset_dir, dataset_name, output_dir='Outputs/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = dataset_name
    data_dir = Path(dataset_dir).parent / f"Data"
    dataset = DataWrapper(data_root=dataset_dir, features_dir=data_dir, dataset_name=dataset_name)

    return dataset

def construct_dataloader(dataset):

    return DataLoader(dataset)
