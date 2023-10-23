import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, lines, mask, edge_index, x, voc, label):
        self.lines = lines
        self.mask = mask
        self.edge_index = edge_index
        self.x = x
        self.voc = voc
        self.label = label

    def __getitem__(self, index):
        return self.lines[index], self.mask[index], self.edge_index[index], self.x[index], self.voc, self.label[index]
        # return self.lines, self.mask, self.edge_index, self.x

    def __len__(self):
        return len(self.label)


def constrcurt_dataset(features_path):
    # input_dataset = torch.load(features_path)
    # seq_data, masks, voc, x, edge_index = input_dataset["seq_data"], input_dataset["mask"], input_dataset["voc"], input_dataset["x"], input_dataset["edge_index"]
    seq_data, masks, edge_index, x, voc, label = pickle.load(open(features_path, "rb"))
    print(voc)

    # torchvoc = []
    # for i in range(len(voc)):
    #     torchvoc.append(i)

    VOCindex = []
    for i in range(len(voc)):
        if voc[i] == '=':
            VOCindex.append(i)
        elif voc[i] == '(':
            VOCindex.append(i)
        elif voc[i] == ')':
            VOCindex.append(i)
        elif voc[i] == '[':
            VOCindex.append(i)
        elif voc[i] == ']':
            VOCindex.append(i)
        elif voc[i] == '1':
            VOCindex.append(i)
        elif voc[i] == '2':
            VOCindex.append(i)
        elif voc[i] == '3':
            VOCindex.append(i)
        elif voc[i] == '4':
            VOCindex.append(i)
        elif voc[i] == '5':
            VOCindex.append(i)
        elif voc[i] == '6':
            VOCindex.append(i)
        elif voc[i] == '7':
            VOCindex.append(i)
        elif voc[i] == '8':
            VOCindex.append(i)
        elif voc[i] == '9':
            VOCindex.append(i)
        elif voc[i] == '0':
            VOCindex.append(i)
        elif voc[i] == '#':
            VOCindex.append(i)
        elif voc[i] == '+':
            VOCindex.append(i)
        elif voc[i] == '-':
            VOCindex.append(i)
        elif voc[i] == 'H':
            VOCindex.append(i)
        elif voc[i] == '/':
            VOCindex.append(i)
        elif voc[i] == '\\':
            VOCindex.append(i)
        elif voc[i] == '@':
            VOCindex.append(i)
        elif voc[i] == '%':
            VOCindex.append(i)
        elif voc[i] == '.':
            VOCindex.append(i)
    print(VOCindex)

    voc_index = torch.tensor(VOCindex)

    # voc = torch.tensor(torchvoc)


    model_dataset = MoleculeDataset(np.array(seq_data)[:200000], np.array(masks)[:200000], np.array(edge_index)[:200000], np.array(x)[:200000],  np.array(voc_index), np.array(label)[:200000])


    return model_dataset

def construct_loader(model_dataset, batch_size):
    model_loader = DataLoader(model_dataset, batch_size, shuffle=True, drop_last=True)

    return model_loader



