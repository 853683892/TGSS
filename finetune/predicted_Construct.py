import os.path
from pathlib import Path
from DataWrapper import *
from DataLoader import *


def construct_dataset(dataset_dir = '/xulei/GSSL/datasets', output_dir='Outputs/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = Path(dataset_dir).parent / f"Data"
    dataset = DataWrapper(data_root=dataset_dir, features_dir=data_dir, dataset_name='BACE')

    return dataset

def construct_dataloader(dataset):

    return DataLoader(dataset)
