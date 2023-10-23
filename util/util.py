import logging
import random
from math import sqrt

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from scipy import stats
import json
from rdkit import Chem



def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset in ['BACE', 'Mutagenesis', 'BBBP', 'Tox21', 'ClinTox', 'BENZENE', 'HIV', 'MUV', 'SIDER']:
        return 0
    elif dataset in ['FreeSolv', 'ESOL', 'Lipophilicity', 'hERG']:
        return 1

    raise ValueError('Invalid dataset name.')

def get_dataset_smiles_target(dataset_name):
    if dataset_name == 'BACE':
        smiles_name = 'mol'
        target_name = 'Class'
    elif dataset_name == 'Mutagenesis':
        smiles_name = 'SMILES'
        target_name = 'label'
    elif dataset_name == 'Lipophilicity':
        smiles_name = 'smiles'
        target_name = 'exp'
    elif dataset_name == 'ESOL':
        smiles_name = 'smiles'
        target_name = 'measured log solubility in mols per litre'
    elif dataset_name == 'FreeSolv':
        smiles_name = 'smiles'
        target_name = 'expt'
    elif dataset_name == 'Tox21':
        smiles_name = 'smiles'
        # target_name = 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        target_name = 'SR-MMP'
    elif dataset_name == 'ClinTox':
        smiles_name = 'smiles'
        # target_name = 'CT_TOX'
        target_name = 'FDA_APPROVED'
    elif dataset_name == 'hERG':
        smiles_name = 'smiles'
        target_name = 'Value'
    elif dataset_name == 'BENZENE':
        smiles_name = 'smiles'
        target_name = 'label'
    elif dataset_name == 'BBBP':
        smiles_name = 'smiles'
        target_name = 'p_np'
    elif dataset_name == 'QM9':
        smiles_name = 'GDN-17'
        target_name = 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'
    elif dataset_name == 'ZINC':
        smiles_name = 'smiles'
        target_name = 'expt'
    elif dataset_name == 'HIV':
        smiles_name = 'smiles'
        target_name = 'HIV_active'
    elif dataset_name == 'ChEMBL':
        smiles_name = 'smiles'
        target_name = 'expt'
    elif dataset_name == 'SIDER':
        smiles_name = 'smiles'
        # target_name = 'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications'
        target_name = 'Injury, poisoning and procedural complications'
    elif dataset_name == 'MUV':
        smiles_name = 'smiles'
        target_name = 'MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692','MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846','MUV-852','MUV-858','MUV-859'


    return smiles_name, target_name

def get_dataset_root(dataset_name):
    if dataset_name == 'BACE':
        data_root = 'BACE/bace.csv'
    elif dataset_name == 'Mutagenesis':
        data_root = 'Mutagenesis/CAS_N6512.csv'
    elif dataset_name == 'ZINC':
        data_root = 'ZINC/ZINC-1.csv'
    elif dataset_name == 'ESOL':
        data_root = 'ESOL/delaney-processed.csv'
    elif dataset_name == 'Lipophilicity':
        data_root = 'Lipophilicity/Lipophilicity.csv'
    elif dataset_name == 'BBBP':
        data_root = 'BBBP/BBBP.csv'
    elif dataset_name == 'BENZENE':
        data_root = 'BENZENE/benzene_smiles.csv'
    elif dataset_name == 'ClinTox':
        data_root = 'ClinTox/clintox.csv'
    elif dataset_name == 'FreeSolv':
        data_root = 'FreeSolv/SAMPL.csv'
    elif dataset_name == 'hERG':
        data_root = 'hERG/hERG.csv'
    elif dataset_name == 'Tox21':
        data_root = 'Tox21/tox21.csv'
    elif dataset_name == 'HIV':
        data_root = 'HIV/HIV.csv'
    elif dataset_name == 'ChEMBL':
        data_root = 'ChEMBL/chembl.csv'
    elif dataset_name == 'MUV':
        data_root = 'MUV/muv.csv'
    elif dataset_name == 'SIDER':
        data_root = 'SIDER/sider.csv'

    return data_root

def filter_invalid_smiles(smiles_list):
    valid_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() > 0 and len(mol.GetAtoms()) > 5 and len(mol.GetAtoms()) <400:
            valid_smiles_list.append(smiles)

    return valid_smiles_list


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


