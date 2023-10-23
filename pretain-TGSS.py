from pretain import Construct
from utils.config import args
# from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
# from DataLoader import DataLoader
from model.SSL_BiLSTM import BiLSTM
from model.SSL_GNN import GNN
from model.SSL_Transformer import Transformer
from model.VAE import VariationalAutoEncoder
import torch.optim as optim
import time
from tqdm import tqdm
import torch
import numpy as np
from utils.util import get_dataset_root
from pretain.Construct_dataset import *



def train(args, molecule_model_1D, molecule_model_2D, molecule_model_Tr, device, loader, optimizer):
    start_time = time.time()

    molecule_model_1D.train()
    molecule_model_2D.train()
    molecule_model_Tr.train()

    AE_loss_accum, AE_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader

    for step, batch in enumerate(l):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device))


        BiLSTM_features = molecule_model_1D(batch)
        Transformer_features = molecule_model_Tr(batch)
        # BiLSTM_features, Transformer_features, GNN_features = molecule_model_2D(batch, BiLSTM_features_1, Transformer_features_1)
        GNN_features = molecule_model_2D(batch)

        AE_loss_1 = VAE_1D_2D_model(BiLSTM_features, GNN_features)
        AE_loss_2 = VAE_2D_1D_model(GNN_features, BiLSTM_features)
        AE_loss_3 = VAE_Tr_2D_model(Transformer_features, GNN_features)
        AE_loss_4 = VAE_2D_Tr_model(GNN_features, Transformer_features)
        AE_loss_5 = VAE_1D_Tr_model(BiLSTM_features, Transformer_features)
        AE_loss_6 = VAE_Tr_1D_model(Transformer_features, BiLSTM_features)

        AE_acc_1 = AE_acc_2 = AE_acc_3 = AE_acc_4 = AE_acc_5 = AE_acc_6 = 0

        AE_loss = (AE_loss_1 + AE_loss_2 + AE_loss_3 + AE_loss_4 + AE_loss_5 + AE_loss_6) / 6


        AE_loss_accum += AE_loss.detach().cpu().item()

        AE_acc_accum += (AE_acc_1 + AE_acc_2 + AE_acc_3 + AE_acc_4 + AE_acc_5 + AE_acc_6) / 6


        loss = 0
        loss += AE_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)

    temp_loss = AE_loss_accum
    if temp_loss < optimal_loss and temp_loss > 0:
    # if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print('AE Loss: {:.5f}\tAE ACC: {:.5f}\tTime: {:.5f}'.format(AE_loss_accum, AE_acc_accum, time.time() - start_time))
    return



def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(molecule_model_1D.state_dict(), args.output_model_dir + args.dataset_name + '_BiLSTM_200000_model.pth')  # save model
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + args.dataset_name + '_GNN_200000_model.pth')  # save model
            torch.save(molecule_model_Tr.state_dict(), args.output_model_dir + args.dataset_name + '_Transformer_200000_model.pth')  # save model
            saver_dict = {
                'model_1D': molecule_model_1D.state_dict(),
                'model_2D': molecule_model_2D.state_dict(),
                'model_Tr': molecule_model_Tr.state_dict(),
                'VAE_1D_2D_model': VAE_1D_2D_model.state_dict(),
                'VAE_2D_1D_model': VAE_2D_1D_model.state_dict(),
                'VAE_Tr_2D_model': VAE_Tr_2D_model.state_dict(),
                'VAE_2D_Tr_model': VAE_2D_Tr_model.state_dict(),
                'VAE_1D_Tr_model': VAE_1D_Tr_model.state_dict(),
                'VAE_Tr_1D_model': VAE_Tr_1D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + args.dataset_name + '_model_200000_complete.pth')
        else:
            torch.save(molecule_model_1D.state_dict(), args.output_model_dir + args.dataset_name + '_BiLSTM_200000_model_final.pth')  # save model
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + args.dataset_name + '_GNN_200000_model_final.pth')  # save model
            torch.save(molecule_model_Tr.state_dict(), args.output_model_dir + args.dataset_name + '_Transformer_200000_model.pth')  # save model
            saver_dict = {
                'model_1D': molecule_model_1D.state_dict(),
                'model_2D': molecule_model_2D.state_dict(),
                'model_Tr': molecule_model_Tr.state_dict(),
                'VAE_1D_2D_model': VAE_1D_2D_model.state_dict(),
                'VAE_2D_1D_model': VAE_2D_1D_model.state_dict(),
                'VAE_Tr_2D_model': VAE_Tr_2D_model.state_dict(),
                'VAE_2D_Tr_model': VAE_2D_Tr_model.state_dict(),
                'VAE_1D_Tr_model': VAE_1D_Tr_model.state_dict(),
                'VAE_Tr_1D_model': VAE_Tr_1D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + args.dataset_name + '_model_200000_complete_final.pth')

    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)



    # 读取数据集
    dataset_root = get_dataset_root(args.dataset_name)
    data_root = './datasets/{}'.format(dataset_root) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset_name)
    dataset = Construct.construct_dataset(data_root, args.dataset_name)
    dataset_getter = Construct.construct_dataloader(dataset)
    pretain_dataset = dataset_getter.get_dataset
    pretain_loader = dataset_getter.get_model_dataset(pretain_dataset, batch_size=args.batch_size)
    dim_features = pretain_dataset.dim_features


    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 设置模型  1D-BiLSTM   2D-GNN  Transformer
    molecule_model_1D = BiLSTM(dim_features=dim_features, emb_dim=args.BiLSTM_emb_dim, hidden_dim=args.BiLSTM_hidden_dim, lstm_hid_dim=args.BiLSTM_lstm_hid_dim, batch_size=args.batch_size).to(device)
    molecule_model_2D = GNN(num_layer=args.num_layer, emb_dim=args.GNN_emb_dim, hidden_size=args.hidden_size, batch_size=args.batch_size, dim_features=dim_features, heads=args.heads, dropout=args.GNN_dropout, aggregation=args.graph_pooling).to(device)
    molecule_model_Tr = Transformer(dim_features=dim_features, emb_dim=args.Transformer_emb_dim, batch_size=args.batch_size).to(device)

    'BiLSTM-GAT'
    VAE_1D_2D_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    VAE_2D_1D_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    'Transformer-GAT'
    VAE_Tr_2D_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    VAE_2D_Tr_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    'BiLSTM-Transformer'
    VAE_1D_Tr_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    VAE_Tr_1D_model = VariationalAutoEncoder(emb_dim=args.VAE_emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)


    model_param_group = []
    model_param_group.append({'params': molecule_model_1D.parameters(), 'lr': args.lr * args.BiLSTM_lr_scale})
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.GNN_lr_scale})
    model_param_group.append({'params': molecule_model_Tr.parameters(), 'lr': args.lr * args.Transformer_lr_scale})
    model_param_group.append({'params': VAE_1D_2D_model.parameters(), 'lr': args.lr * args.BiLSTM_lr_scale})
    model_param_group.append({'params': VAE_2D_1D_model.parameters(), 'lr': args.lr * args.GNN_lr_scale})
    model_param_group.append({'params': VAE_Tr_2D_model.parameters(), 'lr': args.lr * args.Transformer_lr_scale})
    model_param_group.append({'params': VAE_2D_Tr_model.parameters(), 'lr': args.lr * args.GNN_lr_scale})
    model_param_group.append({'params': VAE_1D_Tr_model.parameters(), 'lr': args.lr * args.BiLSTM_lr_scale})
    model_param_group.append({'params': VAE_Tr_1D_model.parameters(), 'lr': args.lr * args.Transformer_lr_scale})


    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    # train
    for epoch in range(1, args.num_epoch + 1):
        print('epoch:{}'.format(epoch))
        train(args, molecule_model_1D, molecule_model_2D, molecule_model_Tr, device, pretain_loader, optimizer)


    save_model(save_best=True)

