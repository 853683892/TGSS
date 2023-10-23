import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import args
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import GATConv, global_max_pool
from torch.nn.utils.rnn import pad_sequence

class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, hidden_size, batch_size, dim_features, heads, dropout, aggregation):
        super(GNN, self).__init__()
        'GAT'
        self.num_layer = num_layer
        self.dim_features = dim_features
        self.hidden_size = hidden_size
        self.heads = heads
        self.emb_dim = emb_dim
        self.fc_max = nn.Linear(self.emb_dim * self.heads,self.emb_dim * self.heads)
        self.batch_size = batch_size
        self.dropout = dropout
        self.aggregation = aggregation


        self.layers = nn.ModuleList([])
        for i in range(self.num_layer):
            dim_input = 128 if i == 0 else self.emb_dim * self.heads
            conv = GATConv(dim_input, self.emb_dim, heads=self.heads, dropout=self.dropout)
            conv.aggr = self.aggregation
            self.layers.append(conv)

        self.fc1 = nn.Linear(self.heads * self.num_layer * self.emb_dim * 2, self.heads * self.emb_dim * 2)
        self.fc2 = nn.Linear(self.heads * self.emb_dim * 2, self.emb_dim * 2)
        self.fc3 = nn.Linear(self.emb_dim * 2, self.emb_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.linear_first = torch.nn.Linear(self.emb_dim, self.emb_dim)




    def forward(self, data, BiLSTM_feature, Transformer_feature):
    # def forward(self, data):
        x = data[3]
        edge_index = data[2]

        tensor1 = []
        len_index = []
        for i in range(edge_index.size(0)):
            ten = edge_index[i, :]
            index1 = torch.nonzero(ten == 9999).squeeze()
            if min(index1.shape) == 0:
                ten = ten
            else:
                ten = ten[:index1[0]]

            len1 = torch.max(ten)
            len_index.append(len1)
            tensor1.append(ten)

        tensor2 = []
        tensor2.append(tensor1[0].reshape(2, -1))

        for i in range(1, len(tensor1)):
            sum = 0
            for d in range(i):
                sum = sum + len_index[d]
            tensor1[i] = tensor1[i] + (sum + i)
            tensor1[i] = tensor1[i].reshape(2, -1)
            tensor2.append(tensor1[i])
        ten1 = tensor2[0]
        for i in range(1, len(tensor2)):
            ten1 = torch.cat((ten1, tensor2[i]), 1)
        edge_index = ten1


        GAT_tensor = []
        for i in range(x.size(0)):
            GATten = x[i,:]
            index2 = torch.nonzero(GATten==9999).squeeze()
            if min(index2.shape)==0:
                GATten = GATten
            else:
                GATten = GATten[:index2[0]]

            GAT_tensor.append(GATten)
        gat_tensor1 = []
        for i in range(len(GAT_tensor)):
            a = GAT_tensor[i].reshape(-1,128)
            gat_tensor1.append(a)
        gat_tensor = gat_tensor1[0]
        for i in range(1,len(gat_tensor1)):

            gat_tensor = torch.cat((gat_tensor,gat_tensor1[i]),0)

        x = gat_tensor

        batchlist = []
        for i in range(len(gat_tensor1)):
            batchlen = gat_tensor1[i].size(0)
            fullbatch = torch.full((1, batchlen), i)
            batchlist.append(fullbatch)
        GNN_batch = batchlist[0].view(-1)
        for i in range(1, len(batchlist)):
            lsatbatch = batchlist[i].view(-1)
            GNN_batch = torch.cat((GNN_batch, lsatbatch), 0)

        batch = GNN_batch.cuda()

        seq_x = data[0].long()
        voc = data[4]
        newT = seq_x.view(-1)
        countVOC = []
        for i in range(voc.size(1)):
            countvoc = torch.nonzero(newT == voc[0, i])
            countvoc = countvoc.view(-1)
            countVOC.append(countvoc)
        vocindex = []
        for i in range(len(countVOC)):
            a = countVOC[i]
            if min(a.shape) != 0:
                vocindex.append(a)
        VOCindex = vocindex[0]
        for i in range(1, len(vocindex)):
            VOCindex = torch.cat((VOCindex, vocindex[i]), 0)
        VOCindex = VOCindex.reshape(-1, 1)

        VOCindex = VOCindex.view(VOCindex.size(0) * VOCindex.size(1))
        VOCindex = VOCindex.sort().values

        size = seq_x.size(0) * seq_x.size(1)
        outputslist = []
        for i in range(0, size):
            outputslist.append(i)

        VOCindex = VOCindex.tolist()
        VOCindex = list(set(outputslist).difference(set(VOCindex)))
        VOCindex.sort()
        VOCindex = torch.tensor(VOCindex)
        VOCindex = VOCindex.cuda()

        outputs_all = []
        self.conv_acts = []
        self.attention_weights = []
        x_all = []


        for i,layer in enumerate(self.layers):
            x, attention_w = layer(x,edge_index,return_attention_weights=True)
            self.conv_acts.append(torch.relu(x))
            self.attention_weights.append(attention_w)
            x_all.append(x)

            x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


        GNN_feature = x
        BiLSTM_feature = torch.reshape(BiLSTM_feature, (-1, 128))
        Transformer_feature = torch.reshape(Transformer_feature, (-1, 128))
        BiLSTM_feature = torch.index_select(BiLSTM_feature, 0, VOCindex)
        Transformer_feature = torch.index_select(Transformer_feature, 0, VOCindex)



        # x = torch.tanh(self.linear_first(x))

        # x = global_max_pool(x,batch)


        return BiLSTM_feature, Transformer_feature, GNN_feature, batch
        # return x

class GNN_predicted(nn.Module):
    def __init__(self, args, molecule_model):
        super(GNN_predicted, self).__init__()

        self.molecule_model = molecule_model
        # self.num_layer = args.num_layer
        # self.emb_dim = args.GNN_emb_dim
        # self.pool = global_max_pool


        # self.mult = 1



        #
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(args.BiLSTM_emb_dim * 1, 1)
        self.MLP = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        # self.fc_1 = nn.Linear(128,64)
        # self.relu1 = torch.nn.ReLU()
        # self.fc_2 = nn.Linear(64,64)
        # self.relu2 = torch.nn.ReLU()
        # self.fc_3 = nn.Linear(64,1)


        # self.linear_final = nn.Linear(args.GNN_emb_dim, args.GNN_emb_dim)

        self.fc_B = nn.Linear(128, 128)
        self.fc_G = nn.Linear(128, 128)
        self.fc_T = nn.Linear(128, 128)
        self.fc_seq = nn.Linear(128, 128)
        self.fc_128 = nn.Linear(256, 128)
        self.fc_128_1 = nn.Linear(256, 128)

        self.fc_Con = nn.Linear(256,128)
        self.fc_Con1 = nn.Linear(256,128)


        self.merge_atten_seq = atten(1)
        self.merge_atten_graph = atten(1)

    def Elem_feature_Fusion(self, pred_BiLSTM, pred_GNN, pred_Transformer):
        Con = self.fc_128(torch.cat(pred_BiLSTM,pred_GNN,pred_Transformer),dim=1)
        pred_B = self.relu(self.fc_B(pred_BiLSTM))
        pred_G = self.relu(self.fc_G(pred_GNN))
        pred_T = self.relu(self.fc_T(pred_Transformer))
        B = Con * pred_B + pred_BiLSTM
        G = Con * pred_G + pred_GNN
        T = Con * pred_T + pred_Transformer
        pred = B + G + T
        return pred



    def from_pretain_GNN(self, model_file):
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_GT_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_BG_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_1_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_200000_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_100000_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_50000_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)
        # self.molecule_model.load_state_dict(torch.load(model_file + '/ChEMBL_GNN_10000_model.pth', map_location={'cuda:0': 'cuda:1'}), strict=False)

        return


    def forward(self, data, pred_BiLSTM, pred_Transformer):

        '下游任务修改为：原子个数N*Embedding'
        BiLSTM_feature, Transformer_feature, GNN_feature, batch = self.molecule_model(data, pred_BiLSTM, pred_Transformer)

        heat_B = BiLSTM_feature
        heat_T = Transformer_feature
        heat_G = GNN_feature

        '首先进行序列特征融合，然后进行序列和图特征融合'
        '消融实验——B+T'
        # Con = torch.cat((cBiLSTM, cTransformer), 2)
        # Con = self.fc_128(Con)
        # Con = self.merge_atten_seq(Con)
        # pred_B = self.relu(self.fc_B(cBiLSTM))
        # pred_T = self.relu(self.fc_T(cTransformer))
        # B = Con * pred_B + cBiLSTM
        # ones_seq = torch.ones(Con.shape).cuda()
        # T = (ones_seq - Con) * pred_T + cTransformer
        # seq = B + T


        '下游任务修改为：原子个数N*Embedding'
        # Con = torch.cat((BiLSTM_feature, Transformer_feature), 1)
        # Con = self.fc_128(Con)
        # Con = self.merge_atten_seq(Con)
        # Con = self.fc_Con(Con)
        # pred_B = self.relu(self.fc_B(BiLSTM_feature))
        # pred_T = self.relu(self.fc_T(Transformer_feature))
        # B = Con * pred_B + BiLSTM_feature
        # ones_seq = torch.ones(Con.shape).cuda()
        # T = (ones_seq - Con) * pred_T + Transformer_feature
        # seq = B + T
        #
        # Con1 = torch.cat((seq, GNN_feature), 1)
        # Con1 = self.fc_128_1(Con1)
        # Con1 = self.merge_atten_graph(Con1)
        # Con1 = self.fc_Con1(Con1)
        # pred_seq = self.relu(self.fc_seq(seq))
        # pred_G = self.relu(self.fc_G(GNN_feature))
        # Seq = Con1 * pred_seq + seq
        # ones_graph = torch.ones(Con1.shape).cuda()
        # G = (ones_graph - Con1) * pred_G + GNN_feature
        # pred = Seq + G
        #
        # heat_TGSS = pred
        ''


        '消融实验——different model'
        'B+T'
        Con = torch.cat((BiLSTM_feature, Transformer_feature), 1)
        Con = self.fc_128(Con)
        Con = self.merge_atten_seq(Con)
        Con = self.fc_Con(Con)
        pred_B = self.relu(self.fc_B(BiLSTM_feature))
        pred_T = self.relu(self.fc_T(Transformer_feature))
        B = Con * pred_B + BiLSTM_feature
        ones_seq = torch.ones(Con.shape).cuda()
        T = (ones_seq - Con) * pred_T + Transformer_feature
        pred = B + T
        heat_TGSS = pred

        'B+G'
        # Con = torch.cat((BiLSTM_feature, GNN_feature), 1)
        # Con = self.fc_128(Con)
        # Con = self.merge_atten_seq(Con)
        # Con = self.fc_Con(Con)
        # pred_B = self.relu(self.fc_B(BiLSTM_feature))
        # pred_G = self.relu(self.fc_T(GNN_feature))
        # B = Con * pred_B + BiLSTM_feature
        # ones_seq = torch.ones(Con.shape).cuda()
        # G = (ones_seq - Con) * pred_G + GNN_feature
        # pred = B + G

        'T+G'
        # Con = torch.cat((Transformer_feature, GNN_feature), 1)
        # Con = self.fc_128(Con)
        # Con = self.merge_atten_seq(Con)
        # Con = self.fc_Con(Con)
        # pred_T = self.relu(self.fc_B(Transformer_feature))
        # pred_G = self.relu(self.fc_T(GNN_feature))
        # T = Con * pred_T + Transformer_feature
        # ones_seq = torch.ones(Con.shape).cuda()
        # G = (ones_seq - Con) * pred_G + GNN_feature
        # pred = T + G

        'B'
        # pred = BiLSTM_feature

        'T'
        # pred = Transformer_feature

        'G'
        # pred = GNN_feature

        '消融实验——特征融合'
        # xiaorong_nopinjie = torch.cat((BiLSTM_feature, Transformer_feature),1)
        # xiaorong_nopinjie = self.fc_128(xiaorong_nopinjie)
        # xiaorong_nopinjie = torch.cat((xiaorong_nopinjie, GNN_feature),1)
        # xiaorong_nopinjie = self.fc_128_1(xiaorong_nopinjie)
        # pred = xiaorong_nopinjie




        # seq = torch.reshape(seq, (-1,128))
        # seq = torch.index_select(seq, 0, VOCindex)

        '序列、图特征融合'
        # Con1 = torch.cat((seq, cGNN), 2)
        # Con1 = self.fc_128(Con1)
        # Con1 = self.merge_atten_graph(Con1)
        # pred_seq = self.relu(self.fc_seq(seq))
        # pred_G = self.relu(self.fc_G(cGNN))
        # Seq = Con1 * pred_seq + seq
        # ones_graph = torch.ones(Con1.shape).cuda()
        # G = (ones_graph - Con1) * pred_G + cGNN
        # pred = Seq + G

        '消融实验——B+G'
        # Con_BG = torch.cat((cBiLSTM,cGNN),2)
        # Con_BG = self.fc_128(Con_BG)
        # Con_BG = self.merge_atten_seq(Con_BG)
        # pred_B = self.relu(self.fc_B(cBiLSTM))
        # pred_G = self.relu(self.fc_G(cGNN))
        # B = Con_BG * pred_B + cBiLSTM
        # ones = torch.ones(Con_BG.shape).cuda()
        # G = (ones - Con_BG) * pred_G + cGNN
        # pred = B + G

        '消融实验——T+G'
        # Con_TG = torch.cat((cTransformer,cGNN),2)
        # Con_TG = self.fc_128(Con_TG)
        # Con_TG = self.merge_atten_seq(Con_TG)
        # pred_T = self.relu(self.fc_B(cTransformer))
        # pred_G = self.relu(self.fc_G(cGNN))
        # T = Con_TG * pred_T + cTransformer
        # ones = torch.ones(Con_TG.shape).cuda()
        # G = (ones - Con_TG) * pred_G + cGNN
        # pred = T + G






        # batchsize = pred.size(1)
        # newbatch = []
        # for i in range(64):
        #     for j in range(batchsize):
        #         newbatch.append(i)
        #
        # newbatch = torch.tensor(newbatch)
        # newbatch = newbatch.cuda()

        'output'




        pred = global_max_pool(pred, batch)
        sne = pred
        pred = self.MLP(pred)





        return pred, sne, heat_B, heat_T, heat_G, heat_TGSS



class atten(nn.Module):
    def __init__(self, padding=3):
        super(atten, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,bias=False)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xc = x.unsqueeze(1)
        xt = torch.cat((xc, self.dropout(xc)), dim=2)
        att = self.sigmoid(self.conv1(xt))
        return att.squeeze(1)











