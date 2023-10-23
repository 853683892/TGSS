import numpy as np
import networkx as nx
import torch
from torch_geometric import data
from torch_geometric.utils import dense_to_sparse

from typing import List, Tuple, Union

from rdkit import Chem

from collections import defaultdict


class Graph(nx.Graph):
    def __init__(self, smiles, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.smiles = smiles
        self.laplacians = None
        self.v_plus = None
        self.max_num_nodes = 200

    def get_edge_index(self):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
            features.append(data)

        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_one_degree=False, use_one=False, use_node_label=False):
        features = []

        for node, node_attrs in self.nodes(data=True):
            data = []
            if use_node_label and node_attrs["label"] is not None:
                data.extend(node_attrs["label"])

            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_one_degree:
                data.extend([self.degree(node)])

            if use_one:
                data.extend([1])

            features.append(data)

        return torch.Tensor(features)

    def get_smiles(self):
        return self.smiles

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value -1] = 1
    return vec


def parse_to_data(temp_dir):
    indicator_path = temp_dir / 'graph_indicator.txt'
    edges_path = temp_dir / 'A.txt'
    smiles_path = temp_dir / 'SMILES.txt'
    node_labels_path = temp_dir / 'node_labels.txt'   # 存储原子序号
    edge_labels_path = temp_dir / 'edge_labels.txt'
    node_attrs_path = temp_dir / 'node_attrs.txt'
    edge_attrs_path = temp_dir / 'edge_attrs.txt'
    graph_labels_path = temp_dir / 'graph_labels.txt'

    unique_node_labels = set()   # 集合
    unique_edge_labels = set()


    indicator, edge_indicator = [-1], [(-1, -1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(),1):
            line = line.rstrip("\n")  # 删除字符串末尾的字符，默认为空格
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)

    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            edge_indicator.append(edge)
            graph_id = indicator[edge[0]]
            graph_edges[graph_id].append(edge)

    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                unique_node_labels.add(node_label)
                graph_id = indicator[i]
                node_labels[graph_id].append(node_label)

    if edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                edge_label = int(line)
                unique_edge_labels.add(edge_label)
                graph_id = indicator[edge_indicator[i][0]]
                edge_labels[graph_id].append(edge_label)


    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split (",")
                node_attr = np.array([float(n) for n in nums])
                graph_id = indicator[i]
                node_attrs[graph_id].append(node_attr)

    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                edge_attr = np.array([float(n) for n in nums])
                graph_id = indicator[edge_indicator[i][0]]
                edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            nums = line.split(",")
            targets = np.array([np.nan if n == 'None' else float(n) for n in nums])
            graph_labels.append(targets)


    # get SMILES
    smiles_all = []
    with open(smiles_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            smiles_all.append(line)

    num_node_labels = max(unique_node_labels) if unique_node_labels !=set() else 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:
        num_node_labels += 1

    num_edge_labels = max(unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0 :
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs":  node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs,
        "smiles": smiles_all
    }, num_node_labels, num_edge_labels


def create_graph_from_to_data(graph_data, target, num_node_labels, num_edge_labels, smiles=None):
    nodes = graph_data["graph_nodes"]
    edges = graph_data["graph_edges"]

    G = Graph(target=target, smiles=smiles)

    for i, node in enumerate(nodes):
        label, attrs = None, None

        if graph_data["node_labels"] != []:
            label = one_hot(graph_data["node_labels"][i], num_node_labels)

        if graph_data["node_attrs"] != []:
            attrs = graph_data["node_attrs"][i]

        G.add_node(node, label=label, attrs=attrs)

    for i, edge in enumerate(edges):
        n1, n2 = edge
        label, attrs = None, None

        if graph_data["edge_labels"] != []:
            label = one_hot(graph_data["edge_labels"][i], num_edge_labels)
        if graph_data["edge_attrs"] != []:
            attrs = graph_data["edge_attrs"][i]

        G.add_edge(n1, n2, label=label, attrs=attrs)

    return G


class Data(data.Data):
    def __init__(self, x=None, y=None, edge_index=None, edge_attr=None, laplicans=None, v_plus=None, smiles=None, max_num_nodes=200):
        super().__init__(x, edge_index, edge_attr, y)

    def set_targets(self, target):
        self.y = target


MAX_ATOMIC_NUM = 95
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

BOND_FDIM = 14


def one_encoding(value: int, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    features = one_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
            one_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            one_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            one_encoding(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag']) + \
            one_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
            one_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]

    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += one_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond







