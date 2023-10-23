import torch.nn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

class BiLSTM(torch.nn.Module):
    def __init__(self, dim_features, emb_dim, hidden_dim, lstm_hid_dim, batch_size):

        super(BiLSTM, self).__init__()
        self.dim_features = dim_features
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lstm_hid_dim = lstm_hid_dim
        self.batch_size = batch_size

        self.embeddings = nn.Embedding(self.dim_features, self.emb_dim)
        # self.embeddings = nn.Embedding(49, self.emb_dim)
        self.lstm = torch.nn.LSTM(self.emb_dim, self.lstm_hid_dim, 2, batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(2 * self.lstm_hid_dim, self.hidden_dim)
        self.linear_first.bias.data.fill_(0)
        self.fc1 = nn.Linear(self.emb_dim,1)
        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim).cuda()))

    def forward(self, data):
        x = data[0].long()

        embeddings = self.embeddings(x)
        self.hidden_state = self.init_hidden()

        outputs, self.hidden_state = self.lstm(embeddings, self.hidden_state)

        '下游任务修改为：原子个数N*Embedding'
        # outputs = outputs[:,-1,:]






        return outputs


class BiLSTM_predicted(nn.Module):
    def __init__(self, args, molecule_model, dim_features):
        super(BiLSTM_predicted, self).__init__()

        self.molecule_model = molecule_model
        # self.emb_dim = args.BiLSTM_emb_dim
        # self.hidden_dim = args.BiLSTM_hidden_dim
        # self.lstm_hid_dim = args.BiLSTM_lstm_hid_dim
        # self.dim_features = dim_features
        # self.embeddings = nn.Embedding(self.dim_features, self.emb_dim)

        # self.batch_size = args.batch_size
        # self.lstm = torch.nn.LSTM(self.emb_dim, self.lstm_hid_dim, 2, batch_first=True, bidirectional=True)
        # self.linear_first = torch.nn.Linear(2 * self.lstm_hid_dim, self.hidden_dim)
        # self.linear_final = torch.nn.Linear(self.emb_dim, self.emb_dim)
        #
        # self.sigmoid = nn.Sigmoid()


    def from_pretain_BiLSTM(self, model_file, voc, device):

        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_BG_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_BT_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_1_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_200000_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_100000_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_50000_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_BiLSTM_10000_model.pth', map_location={'cuda:0': 'cuda:1'})['embeddings.weight']


        'ChEMBL'
        pretain_voc = ['P', 'V', '5', '%', 'H', '[', 'F', '8', '0', 's', 'y', '$', 'O', '1', 'A', 'c', '4', 'o', 'M', 'n', '7', 'R', '2', 'b', '@', 'Z', '#', '-', 'C', 'd', '/', '=', 'Q', '(', 'I', ')', '3', 'a', 'B', 'N', '\\', 'S', 'L', '9', '6', '+', ']', '?', 'p']
        'ZINC'
        # pretain_voc = ['[', '1', '3', ']', '\\', 'S', '(', 'o', 'R', 'L', 'n', '#', 'P', ')', '2', '4', '+', '5', '6', '7', '@', '-', '/', 'N', 'C', '=', 'I', 'O', '8', 'H', 'F', 'c', 's']
        pipei = []
        for i in range(len(voc)):
            for j in range(len(pretain_voc)):
                if voc[i] == pretain_voc[j]:
                    pipei.append(j)
                else:
                    j = j + 1

        embedding_all = []
        for i in range(len(pipei)):

            embedding_all.append(pretained_dict[pipei[i],:])

        embedding_pinjie = []
        for i in range(len(embedding_all)):
            a = embedding_all[0]
            b = torch.unsqueeze(a,dim=0)
            embedding_pinjie.append(b)

        finetune_embedding = torch.cat(embedding_pinjie,dim=0)

        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_BG_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_BT_model.pth', map_location={'cuda:0': 'cuda:1'})
        state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_1_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_200000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_100000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_50000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_BiLSTM_10000_model.pth', map_location={'cuda:0': 'cuda:1'})

        chazhi = len(voc) - finetune_embedding.size(0)
        # chazhi = len(voc)
        chazhitensor = torch.zeros(chazhi, 128)
        chazhitensor = Parameter(nn.init.uniform_(chazhitensor,-0.1,0.1)).to(device)
        finetune_embedding = torch.cat((finetune_embedding,chazhitensor),dim=0)
        state_dict.update({'embeddings.weight': finetune_embedding})


        self.molecule_model.load_state_dict(state_dict, strict=False)
        # print(state_dict)
        return

    def forward(self, data):
        outputs = self.molecule_model(data)




        # outputs = F.relu(self.fc1(outputs))
        # outputs = torch.tanh(self.linear_first(outputs))


        '添加Relu'
        # outputs = F.relu(self.linear_final(outputs))
        return outputs