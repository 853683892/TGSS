import copy
import math
import collections
import numpy as np

import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import args
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import global_max_pool


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from molecule, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate, type_vocab_size=5):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, input_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if input_mask is not None:
            token_type_embeddings = self.token_type_embeddings(input_mask)
            words_embeddings = words_embeddings + token_type_embeddings

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        # self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)



        return hidden_states


class Transformer(nn.Sequential):
    def __init__(self, dim_features, emb_dim,  batch_size, max_num_nodes=1000):
        super(Transformer, self).__init__()

        self.vocab_size = dim_features
        self.emb_dim = emb_dim
        self.hidden_dim = args.Transformer_hidden_dim
        self.batch_size = batch_size
        self.max_num_nodes = max_num_nodes
        self.dropout_rate = args.dropout

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.emb = Embeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            max_position_size=1024,
            dropout_rate=self.dropout_rate
        )

        self.encoder = Encoder_MultipleLayers(
            n_layer=args.Transformer_n_layer,
            hidden_size=self.hidden_dim,
            intermediate_size=args.Transformer_intermediate_size,
            num_attention_heads=args.Transformer_num_attention_heads,
            attention_probs_dropout_prob=args.Transformer_attention_probs_dropout,
            hidden_dropout_prob=args.Transformer_hidden_dropout_rate
        )

        self.linear_final = nn.Linear(self.hidden_dim, self.hidden_dim)

    def featurize(self, data):
        e = data[0].long()
        e_mask = data[1].long()

        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())

        return encoded_layers[:, 0]

    def forward(self, data):
        e = data[0].long()
        e_mask = data[1].long()

        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())


        '下游任务修改为：原子个数N*Embedding'
        # encoded_layers = encoded_layers[:,-1,:]




        return encoded_layers




class Transformer_predicted(nn.Sequential):
    def __init__(self, args, molecule_model, dim_features, max_num_nodes=1000):
        super(Transformer_predicted, self).__init__()

        # self.vocab_size = dim_features
        # self.emb_dim = args.Transformer_emb_dim,
        # self.hidden_dim = args.Transformer_hidden_dim
        # self.batch_size = args.batch_size
        # self.max_num_nodes = max_num_nodes
        # self.dropout_rate = args.dropout
        self.molecule_model = molecule_model
        #
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()



        # self.linear_final = nn.Linear(self.hidden_dim, self.hidden_dim)


        # self.fc = nn.Linear(args.BiLSTM_emb_dim * 1, 1)

        # self.fc_p = nn.Linear(256,1)



    def from_pretain_Transformer(self, model_file, voc, device):

        # dict1 = torch.load(model_file + '/ChEMBL_Transformer_TG_model.pth', map_location={'cuda:0': 'cuda:1'})

        # position_dict = torch.load(model_file + '/ChEMBL_Transformer_1_model.pth')['emb.position_embeddings.weight']

        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_TG_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_TB_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_1_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_200000_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_100000_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_50000_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']
        # pretained_dict = torch.load(model_file + '/ChEMBL_Transformer_10000_model.pth', map_location={'cuda:0': 'cuda:1'})['emb.word_embeddings.weight']

        'ChEMBL'
        pretain_voc = ['P', 'V', '5', '%', 'H', '[', 'F', '8', '0', 's', 'y', '$', 'O', '1', 'A', 'c', '4', 'o', 'M', 'n', '7', 'R', '2', 'b', '@', 'Z', '#', '-', 'C', 'd', '/', '=', 'Q', '(', 'I', ')', '3', 'a', 'B', 'N', '\\', 'S', 'L', '9', '6', '+', ']', '?', 'p']

        pipei = []
        for i in range(len(voc)):
            for j in range(len(pretain_voc)):
                if voc[i] == pretain_voc[j]:
                    pipei.append(j)
                else:
                    j = j + 1

        embedding_all = []
        for i in range(len(pipei)):
            embedding_all.append(pretained_dict[pipei[i], :])

        embedding_pinjie = []
        for i in range(len(embedding_all)):
            a = embedding_all[0]
            b = torch.unsqueeze(a, dim=0)
            embedding_pinjie.append(b)

        finetune_embedding = torch.cat(embedding_pinjie, dim=0)

        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_TG_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_TB_model.pth', map_location={'cuda:0': 'cuda:1'})
        state_dict = torch.load(model_file + '/ChEMBL_Transformer_1_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_200000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_100000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_50000_model.pth', map_location={'cuda:0': 'cuda:1'})
        # state_dict = torch.load(model_file + '/ChEMBL_Transformer_10000_model.pth', map_location={'cuda:0': 'cuda:1'})
        chazhi = len(voc) - finetune_embedding.size(0)


        chazhitensor = torch.zeros(chazhi, 128)
        chazhitensor = Parameter(nn.init.uniform_(chazhitensor, -0.1, 0.1)).to(device)
        finetune_embedding = torch.cat((finetune_embedding, chazhitensor), dim=0)
        state_dict.update({'emb.word_embeddings.weight': finetune_embedding})

        self.molecule_model.load_state_dict(state_dict, strict=False)
        # print(state_dict)
        return

    def forward(self, data):
        encoded_layers = self.molecule_model(data)


        # output = self.linear_final(encoded_layers[:, 0])

        # encoded_layers = self.linear_final(encoded_layers)


        return encoded_layers


