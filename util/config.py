import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=42)  # 42,0,80
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--splits', type=str, default='random')

# dataset
parser.add_argument('--dataset_name', type=str, default='ChEMBL')
# parser.add_argument('--dataset_name', type=str, default='ESOL')
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--num_workers', type=int, default=8)

# paramater
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2', type=float, default=0.1)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# BiLSTM
parser.add_argument('--BiLSTM_emb_dim', type=int, default=128)
parser.add_argument('--BiLSTM_lr_scale', type=float, default=1)
parser.add_argument('--BiLSTM_hidden_dim', type=int, default=128)
parser.add_argument('--BiLSTM_lstm_hid_dim', type=int, default=64)


# GNN
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--GNN_emb_dim', type=int, default=128)
parser.add_argument('--graph_pooling', type=str, default='max')
parser.add_argument('--GNN_lr_scale', type=float, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--train_eps', type=str, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--GNN_dropout', type=float, default=0.3)

# Transformer
parser.add_argument('--Transformer_emb_dim', type=int, default=128)
parser.add_argument('--Transformer_hidden_dim', type=int, default=128)
parser.add_argument('--Transformer_lr_scale', type=float, default=1)
parser.add_argument('--Transformer_n_layer', type=int, default=2)
parser.add_argument('--Transformer_intermediate_size', type=int, default=128)
parser.add_argument('--Transformer_num_attention_heads', type=int, default=4)
parser.add_argument('--Transformer_attention_probs_dropout', type=float, default=0.1)
parser.add_argument('--Transformer_hidden_dropout_rate', type=float, default=0.1)


# VAE paramter
parser.add_argument('--VAE_emb_dim', type=int, default=128)
parser.add_argument('--AE_loss', type=str, default='l2')
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.set_defaults(detach_target=True)
parser.add_argument('--beta', type=float, default=1)

# Verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

# loading and saving
parser.add_argument('--input_model_file', type=str, default='./output_model')
parser.add_argument('--output_model_dir', type=str, default='/xulei/GSSL/output_model/')

# finetune
parser.add_argument('--predicted_datasets', type=str, default='HIV')
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.add_argument('--Fine_lr', type=float, default=0.00001)

args = parser.parse_args()