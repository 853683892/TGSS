import os

import numpy as np
import torch
from pathlib import Path

from utils.sequence_utils import *
from utils.graph_utils import *
from utils.graph_utils import parse_to_data, create_graph_from_to_data, Data
from rdkit import Chem
from networkx import normalized_laplacian_matrix
import pickle
from torch.nn.utils.rnn import pad_sequence
from utils.util import get_dataset_smiles_target
from utils.util import get_dataset_smiles_target


class DataEmbedding():
    def __init__(self,data_df, features_path, data_root, dataset_name,
                 use_one_attrs=True, use_one_degree=False, use_one=False,
                 use_node_attrs=True, use_node_degree=False,
                 save_temp_data=True, precompute_kron_indices=True,
                 max_length=0, max_reductions=10):
        self.whole_data_df = data_df
        self.features_path = features_path
        self.data_root = data_root
        self.dataset_name = dataset_name

        self.use_one_degree = use_one_degree
        self.use_one_attrs = use_one_attrs
        self.use_one = use_one

        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs

        self.save_temp_data = save_temp_data
        self.precompute_kron_indices = precompute_kron_indices
        self.KRON_REDUCTIONS = max_reductions

        self.temp_dir = self.features_path.parent / 'raw'
        if not self.temp_dir.exists():
            os.makedirs(self.temp_dir)

        self.smiles_col, self.target_col = get_dataset_smiles_target(self.dataset_name)
        self.target = self.target_col
        self.target_col = []
        self.target_col.append(self.target)
        # vocab_filepath = './BACE/bace.csv/vocab.txt'
        # self.tokenizer = SmilesTokenizer(vocab_file=vocab_filepath)
        # self.max_length = max_length if max_length else self.get_max_length()




    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return self._dim_edge_features

    @property
    def max_num_nodes(self):
        return self._max_num_nodes


    def load_data_from_df(self):
        temp_dir = self.temp_dir

        if os.path.exists(temp_dir / 'SMILES.txt'):
            print("File exits.")
            return

        df = self.whole_data_df

        fp_smiles = open(temp_dir / 'SMILES.txt', "w")
        fp_edge_index = open(temp_dir / 'A.txt', "w")
        fp_graph_indicator = open(temp_dir / 'graph_indicator.txt', "w")
        fp_node_labels = open(temp_dir / 'node_labels.txt', "w")
        fp_node_attrs = open(temp_dir / 'node_attrs.txt', "w")
        fp_edge_attrs = open(temp_dir / 'edge_attrs.txt', "w")
        fp_graph_labels = open(temp_dir / 'graph_labels.txt', "w")


        'Data augmentation'




        cnt = 1
        for idx,row in df.iterrows():
            # if len(self.target_col[0]) == 1:
            #     smiles, g_labels = row[self.smiles_col], row[self.target_col].values
            # else:
            #     smiles = row[self.smiles_col]
            #     g_labels = []
            #     for i in range(len(self.target_col[0])):
            #         target = self.target_col[0]
                    # print(row[target[i]])
                    # g_labels.append(row[target[i]])
            # g_labels = np.array(g_labels)
            a = [0.0]
            smiles, g_labels = row[self.smiles_col], row[self.target_col].values
            g_labels = g_labels.tolist()
            if np.isnan(g_labels):
                g_labels = a
            g_labels = np.array(g_labels)
            print(idx)
            print(smiles)
            try:
                fp_smiles.writelines(smiles+"\n")
                mol = Chem.MolFromSmiles(smiles)
                num_nodes = len(mol.GetAtoms())
                node_dict = {}
                for i, atom in enumerate(mol.GetAtoms()):
                    node_dict[atom.GetIdx()] = cnt + i
                    '存储原子序号'
                    fp_node_labels.writelines(str(atom.GetAtomicNum()) + "\n")
                    'one-hot编码'
                    fp_node_attrs.writelines(str(atom_features(atom))[1:-1] + "\n")

                fp_graph_indicator.write(f"{idx + 1}\n" * num_nodes)
                fp_graph_labels.write(','.join(['None' if np.isnan(g_label) else str(g_label) for g_label in g_labels]) + "\n")

                for bond in mol.GetBonds():
                    node_1 = node_dict[bond.GetBeginAtomIdx()]
                    node_2 = node_dict[bond.GetEndAtomIdx()]
                    fp_edge_index.write(f"{node_1},{node_2}\n{node_2},{node_1}\n")
                    fp_edge_attrs.write(str([1 if i else 0 for i in bond_features(bond)])[1:-1] + "\n")
                    fp_edge_attrs.write(str([1 if i else 0 for i in bond_features(bond)])[1:-1] + "\n")
                cnt += num_nodes

            except ValueError as e:
                print("the SMILES ({}) can not be converted to a Chem.Molecule.\nREASON:{}".format(smiles, e))

        fp_smiles.close()
        fp_graph_indicator.close()
        fp_edge_attrs.close()
        fp_edge_index.close()
        fp_node_labels.close()
        fp_node_attrs.close()
        fp_graph_labels.close()



    def process(self):

        features_path = self.features_path
        if self.save_temp_data and os.path.exists(features_path):
            seq_data, masks, edge_index, x, voc, y_all = pickle.load(open(features_path,"rb"))
            # dataset = torch.load(features_path)

            self._dim_features = len(voc)
            # self._dim_edge_features = len(dataset["edge_attr"])
        else:
            self.load_data_from_df()

            graphs_data, num_node_labels, num_edge_labels = parse_to_data(self.temp_dir)
            print(num_node_labels)
            self.smiles_list = graphs_data.pop("smiles")
            self.target_list = graphs_data.pop("graph_labels")
            self._max_num_nodes = max([len(v) for (k,v) in graphs_data['graph_nodes'].items()])

            # self.smiles_list_1 = self.smiles_list[:230000]
            # self.target_list_1 = self.target_list[:230000]
            # self.smiles_list_2 = self.smiles_list[230000:]
            # self.target_list_2 = self.target_list[230000:]


            dataset = []
            for i, (target, smiles) in enumerate(zip(self.target_list, self.smiles_list), 1):
                graph_data = {k: v[i] for (k,v) in graphs_data.items()}
                G = create_graph_from_to_data(graph_data, target, num_node_labels, num_edge_labels, smiles=smiles)
                G.max_num_nodes = self._max_num_nodes
                data = self._to_data(G)
                dataset.append(data)


            copy_data = []
            GAT_data = []
            for i in range(len(dataset)):
                a = dataset[i].edge_index.view(-1)
                b = dataset[i].x.view(-1)
                copy_data.append(a)
                GAT_data.append(b)


            x_all, y_all, voc = self.load_data_from_smiles(self.smiles_list, self.target_list)
            seq_data, label, mask = self.make_variables(x_all, y_all, voc)


            edge_index = pad_sequence(copy_data,batch_first=True,padding_value=9999)
            x = pad_sequence(GAT_data,batch_first=True,padding_value=9999)

            self._dim_features = len(voc)

            pickle.dump((seq_data, mask, edge_index, x, voc, label), open(features_path, "wb"))


            # smiles = dataset[0]
            # x = dataset[1]
            # edge_index = dataset[2]
            # max_num_nodes = dataset[3]

            # GSSLdataset = {
            #     "smiles": smiles,
            #     "seq_data": seq_data,
            #     "mask": mask,
            #     "voc": voc,
            #     "x": x,
            #     "edge_index": edge_index,
            #     "max_num_nodes": max_num_nodes,
            # }


            # self._dim_edge_features = len(edge_attr)

            # torch.save(GSSLdataset, features_path)




    def load_data_from_smiles(self, x_smiles, labels):
        x_all = []
        y_all = []

        for smiles, target in zip(x_smiles, labels):
            try:
                x_all.append(smiles)
                y_all.append(target)
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a RDkit Mol .\nREASONM: {}'.format(smiles, e))

        # voc = self.tokenizer.vocab
        voc = self.construct_vocabulary(x_all)
        return x_all, np.array(y_all), voc


    def make_veriables(self, lines, all_letters):
        sequence_and_length = [self.line2voc_arr(line, all_letters) for line in lines]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])

        return self.pad_sequences(vectorized_seqs, seq_lengths)


    # def make_variables(self, smiles, all_letters):
    #     input_idx, mask = [], []
    #     for i, smi in enumerate(smiles):
    #         smi_input_idx, smi_mask = self.tokenize_smiles(smi)
    #         input_idx.append(smi_input_idx)
    #         mask.append(smi_mask)
    #
    #     input_idx = torch.stack(input_idx, dim=0)
    #     mask = torch.stack(mask, dim=0)
    #
    #     return input_idx, mask


    def construct_vocabulary(self, x_smiles):
        voc = set()
        for i, smiles in enumerate(x_smiles):
            smiles = smiles.split(" ")[0]
            regex = '(\[[^\[\]]{1,10}\])@+-#%'
            smiles = self.replace_halogen(smiles)
            char_list = re.split(regex, smiles)
            for char in char_list:
                chars = [unit for unit in char]
                [voc.add(unit) for unit in chars]
        return list(voc)

    def replace_halogen(self, string):

        br = re.compile('Br')
        cl = re.compile('Cl')
        Se = re.compile('Se')
        se = re.compile('se')
        si = re.compile('Si')
        cu = re.compile('Cu')
        hg = re.compile('Hg')
        fe = re.compile('Fe')
        As = re.compile('As')
        cr = re.compile('Cr')
        bi = re.compile('Bi')
        ti = re.compile('Ti')
        al = re.compile('Al')
        ni = re.compile('Ni')
        pd = re.compile('Pd')
        na = re.compile('Na')
        zn = re.compile('Zn')
        ge = re.compile('Ge')
        sb = re.compile('Sb')
        co = re.compile('Co')
        ru = re.compile('Ru')
        Sn = re.compile('Sn')
        Li = re.compile('Li')
        Ag = re.compile('Ag')
        Mn = re.compile('Mn')
        Ga = re.compile('Ga')
        Zr = re.compile('Zr')
        Mo = re.compile('Mo')
        Te = re.compile('Te')
        te = re.compile('te')
        Au = re.compile('Au')
        Ca = re.compile('Ca')
        Rh = re.compile('Rh')
        Pt = re.compile('Pt')

        string = si.sub('M', string)
        string = br.sub('R', string)
        string = cl.sub('L', string)
        string = Se.sub('Q', string)
        string = se.sub('Q', string)
        string = cu.sub('z', string)
        string = hg.sub('G', string)
        string = fe.sub('l', string)
        string = As.sub('A', string)
        string = cr.sub('t', string)
        string = bi.sub('W', string)
        string = ti.sub('T', string)
        string = al.sub('y', string)
        string = ni.sub('m', string)
        string = pd.sub('g', string)
        string = na.sub('a', string)
        string = zn.sub('Z', string)
        string = ge.sub('k', string)
        string = sb.sub('d', string)
        string = co.sub('x', string)
        string = ru.sub('D', string)
        string = Sn.sub('j', string)
        string = Li.sub('q', string)
        string = Ag.sub('?', string)
        string = Mn.sub('!', string)
        string = Ga.sub('<', string)
        string = Zr.sub('>', string)
        string = Mo.sub('`', string)
        string = Te.sub('$', string)
        string = te.sub('$', string)
        string = Au.sub('X', string)
        string = Ca.sub('w', string)
        string = Rh.sub('J', string)
        string = Pt.sub('*', string)

        return string

    def make_variables(self, lines, targets, all_letters):
        sequence_and_length = [self.line2voc_arr(line, all_letters) for line in lines]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])

        return self.pad_sequences(vectorized_seqs, seq_lengths, targets)

    def letterToIndex(self, letter, all_letters):
        return all_letters.index(letter)

    def line2voc_arr(self, line, all_letters):
        arr = []
        regex = '(\[[^\[\]]{1,10}\])@+-#'
        line = self.replace_halogen(line.strip(' '))
        char_list = re.split(regex, line)
        for li, char in enumerate(char_list):
            if char.startswith('[,[['):
                arr.append(self.letterToIndex(char, all_letters))
            else:
                chars = [unit for unit in char]

                for i, unit in enumerate(chars):
                    arr.append(self.letterToIndex(unit, all_letters))
        return arr, len(arr)

    def pad_sequences(self, vectorized_seqs, seq_lengths, targets):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        mask_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
            mask_tensor[idx, :] = torch.LongTensor(([1] * seq_len) + ([0] * (seq_lengths.max() - seq_len)))

        target = torch.LongTensor(targets)

        return seq_tensor, target, mask_tensor


    # def tokenize_smiles(self, smiles):
    #     encoded_inputs = self.tokenizer(smiles, max_length = self.max_length, )


    # def get_max_length(self):
    #     max_length = 0
    #     for i, smi in enumerate(self.smiles_list):
    #         token_list = self.tokenizer._tokenize(smi.strip(" "))
    #         if len(token_list) >= max_length:
    #             max_length = len(token_list)
    #     return max_length


    def _to_data(self, G):
        '创建字典'
        datadict = {}

        # smiles = G.get_smiles()
        # datadict.update(smiles=smiles)

        node_features = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        datadict.update(x=node_features)
        print(node_features.shape)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        datadict.update(max_num_nodes=G.max_num_nodes)

        # if G.laplacians is not None:
        #     datadict.update(laplicans=G.laplacians)
        #     datadict.update(v_plus=G.v_plus)
        #
        # if G.number_of_edges() and G.has_edge_attrs:
        #     edge_attr = G.get_edge_attr()
        #     datadict.update(edge_attr=edge_attr)
        # else:
        #     edge_attr = torch.Tensor([])
        #     datadict.update(edge_attr=edge_attr)




        data = Data(**datadict)
        return data

    # def _precompute_kron_indices(self, G):
    #     laplacians = []
    #     v_plus_list = [] # reduction matrices
    #
    #     x = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
    #     lap = torch.Tensor(normalized_laplacian_matrix(G).todense()) # 归一化  I - D^{-1/2}AD^{-1/2}
    #
    #     laplacians.append(lap)
    #
    #     for _ in range(self.KRON_REDUCTIONS):
    #         if lap.shape[0] == 1:
    #             v_plus, lap = torch.tensor([1]), torch.eye(1) # 生成对角线为1，其余为0的二维数组
    #         else:
    #             v_plus, lap = self._vertex_decimation(lap)
    #
    #         laplacians.append(lap.clone())
    #         v_plus_list.append(v_plus.clone().long())
    #
    #     return laplacians, v_plus_list
    #
    #
    # def _power_iteration(self, A, num_simulations=30):
    #     b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1
    #
    #     for _ in range(num_simulations):
    #         # 计算矩阵乘积
    #         b_k1 = torch.mm(A, b_k)
    #         # 计算范数
    #         b_k1_norm = torch.norm(b_k1)
    #         # 重新归一化向量
    #         b_k = b_k1 / b_k1_norm
    #
    #     return b_k
    #
    #
    # def _vertex_decimation(self, L):
    #
    #     max_eigenvec = self._power_iteration(L)
    #     v_plus, v_minus = (max_eigenvec >= 0).squeeze(), (max_eigenvec <0).squeeze()
    #
    #     # 对角矩阵
    #     if torch.sum(v_plus) == 0.:
    #         if torch.sum(v_minus) == 0.:
    #             assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
    #             return torch.ones(v_minus.shape), L
    #         else:
    #             return v_minus, L
    #
    #     L_plus_plus = L[v_plus][:, v_plus]
    #     L_plus_minus = L[v_plus][:, v_minus]
    #     L_minus_minus = L[v_minus][:, v_minus]
    #     L_minus_plus = L[v_minus][:,v_plus]
    #
    #     L_new = L_plus_plus - torch.mm(torch.mm(L_plus_minus, torch.inverse(L_minus_minus)), L_minus_plus)
    #
    #     return v_plus, L_new





