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


def constrcurt_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    # input_dataset = torch.load(features_path)
    # seq_data, masks, voc, x, edge_index = input_dataset["seq_data"], input_dataset["mask"], input_dataset["voc"], input_dataset["x"], input_dataset["edge_index"]
    seq_data, masks, edge_index, x, voc, y_all = pickle.load(open(features_path, "rb"))
    print(voc)

    # torchvoc = []
    # for i in range(len(voc)):
    #     torchvoc.append(i)
    #
    # voc = torch.tensor(torchvoc)

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

    train_idxs = train_idxs[0]
    valid_idxs = valid_idxs[0]
    test_idxs = test_idxs[0]
    '0:63,102:165'
    test_xulei_idxs = []
    for i in range(64):
        test_xulei_idxs.append(i)
    train_dataset = MoleculeDataset(np.array(seq_data)[train_idxs], np.array(masks)[train_idxs], np.array(edge_index)[train_idxs], np.array(x)[train_idxs],  np.array(voc_index), np.array(y_all)[train_idxs])
    valid_dataset = MoleculeDataset(np.array(seq_data)[valid_idxs], np.array(masks)[valid_idxs], np.array(edge_index)[valid_idxs], np.array(x)[valid_idxs],  np.array(voc_index), np.array(y_all)[valid_idxs])
    test_dataset = MoleculeDataset(np.array(seq_data)[test_idxs], np.array(masks)[test_idxs], np.array(edge_index)[test_idxs], np.array(x)[test_idxs],  np.array(voc_index), np.array(y_all)[test_idxs])
    test_xulei_dataset = MoleculeDataset(np.array(seq_data)[test_xulei_idxs], np.array(masks)[test_xulei_idxs], np.array(edge_index)[test_xulei_idxs], np.array(x)[test_xulei_idxs],  np.array(voc_index), np.array(y_all)[test_xulei_idxs])


    return train_dataset, valid_dataset, test_dataset, voc, test_xulei_dataset

def construct_loader(train_dataset=None, valid_dataset=None,test_dataset=None, batch_size=None, test_xulei_dataset=None):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
    test_xulei_loader = DataLoader(test_xulei_dataset, batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader, test_xulei_loader



