import os.path
from finetune.finetune_DataWrapper import *
from finetune.finetune_DataLoader import DataLoader


def construct_dataset(dataset_dir, dataset_name, output_dir='Outputs/',):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_root = dataset_dir
    dataset_name = dataset_name
    data_dir = Path(dataset_dir).parent / f"Data"
    dataset = DataWrapper(data_root=data_root, features_dir=data_dir, dataset_name=dataset_name)

    return dataset

def construct_dataloader(dataset):

    return DataLoader(dataset)
