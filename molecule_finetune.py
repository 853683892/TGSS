import pickle

import numpy

from utils.config import args
from utils.util import get_num_task
from finetune import finetune_Construct
from finetune import finetune_DataWrapper
from finetune import finetune_Construct_dataset
import torch
import numpy as np
from model.SSL_GNN import GNN,GNN_predicted
from model.SSL_BiLSTM import BiLSTM,BiLSTM_predicted
from model.SSL_Transformer import Transformer,Transformer_predicted
import torch.optim as optim
import torch.nn as nn
from os.path import join
from sklearn.metrics import roc_auc_score, mean_squared_error
from utils.util import get_dataset_root
from math import sqrt
import csv
import pandas as pd




def train(model_GNN, model_BiLSTM, model_Transformer, device, loader, optimizer_GNN, optimizer_BiLSTM, optimizer_Transformer):
    model_GNN.train()
    model_BiLSTM.train()
    model_Transformer.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
        y = batch[5]


        pred_BiLSTM = model_BiLSTM(batch)
        pred_Transformer = model_Transformer(batch)
        pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)


        y = y.view(pred.shape).to(torch.float64)

        is_valid = y ** 2 > 0

        loss_mat = criterion(pred.double(), y)

        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        model_GNN.zero_grad()
        model_BiLSTM.zero_grad()
        model_Transformer.zero_grad()
        # optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        loss.backward()
        optimizer_GNN.step()
        optimizer_BiLSTM.step()
        optimizer_Transformer.step()
        # optimizer.step()
        total_loss += loss.detach().item()


    return total_loss / (step + 1)


def eval(model_GNN, model_BiLSTM, model_Transformer, device, loader, dataset_work):
    model_GNN.eval()
    model_BiLSTM.eval()
    model_Transformer.eval()

    y_true = []
    y_scores = []
    # snelist = []
    for step, batch in enumerate(loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
        with torch.no_grad():
            pred_BiLSTM = model_BiLSTM(batch)
            pred_Transformer = model_Transformer(batch)
            pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)

        true = batch[5].view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)

        # snelist.append(sne)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

    # snelist = torch.cat(snelist, dim=0).cpu().detach().numpy()


    if dataset_work == 0:
        scores = roc_auc_score(y_true, y_scores)
    elif dataset_work == 1:
        scores = sqrt(mean_squared_error(y_true, y_scores))

    return scores, y_true, y_scores

def test(model_GNN, model_BiLSTM, model_Transformer, device, loader, dataset_work):
        model_GNN.eval()
        model_BiLSTM.eval()
        model_Transformer.eval()

        y_true = []
        y_scores = []
        snelist = []
        for step, batch in enumerate(loader):
            batch = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device),
            batch[5].to(device))
            with torch.no_grad():
                pred_BiLSTM = model_BiLSTM(batch)
                pred_Transformer = model_Transformer(batch)
                pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)

            true = batch[5].view(pred.shape)

            y_true.append(true)
            y_scores.append(pred)

            snelist.append(sne)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()

        snelist = torch.cat(snelist, dim=0).cpu().detach().numpy()

        if dataset_work == 0:
            scores = roc_auc_score(y_true, y_scores)
        elif dataset_work == 1:
            scores = sqrt(mean_squared_error(y_true, y_scores))

        return scores, y_true, y_scores, snelist
    # return y_true, y_scores, snelist


def test_xulei_100(model_GNN, model_BiLSTM, model_Transformer, device, loader, dataset_work):
    model_GNN.eval()
    model_BiLSTM.eval()
    model_Transformer.eval()

    for step, batch in enumerate(loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
        with torch.no_grad():
            pred_BiLSTM = model_BiLSTM(batch)
            pred_Transformer = model_Transformer(batch)
            pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)

            torch.save(heat_B, "/xulei/GSSL/heat_B-100.pth")
            torch.save(heat_T, "/xulei/GSSL/heat_T-100.pth")
            torch.save(heat_G, "/xulei/GSSL/heat_G-100.pth")
            torch.save(heat_TGSS, "/xulei/GSSL/heat_TGSS-100.pth")


    return pred

def test_xulei_150(model_GNN, model_BiLSTM, model_Transformer, device, loader, dataset_work):
    model_GNN.eval()
    model_BiLSTM.eval()
    model_Transformer.eval()

    for step, batch in enumerate(loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
        with torch.no_grad():
            pred_BiLSTM = model_BiLSTM(batch)
            pred_Transformer = model_Transformer(batch)
            pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)

            torch.save(heat_B, "/xulei/GSSL/heat_B-150.pth")
            torch.save(heat_T, "/xulei/GSSL/heat_T-150.pth")
            torch.save(heat_G, "/xulei/GSSL/heat_G-150.pth")
            torch.save(heat_TGSS, "/xulei/GSSL/heat_TGSS-150.pth")


    return pred

def test_xulei_300(model_GNN, model_BiLSTM, model_Transformer, device, loader, dataset_work):
    model_GNN.eval()
    model_BiLSTM.eval()
    model_Transformer.eval()

    for step, batch in enumerate(loader):
        batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device))
        with torch.no_grad():
            pred_BiLSTM = model_BiLSTM(batch)
            pred_Transformer = model_Transformer(batch)
            pred, sne, heat_B, heat_T, heat_G, heat_TGSS = model_GNN(batch, pred_BiLSTM, pred_Transformer)

            torch.save(heat_B, "/xulei/GSSL/heat_B-300.pth")
            torch.save(heat_T, "/xulei/GSSL/heat_T-300.pth")
            torch.save(heat_G, "/xulei/GSSL/heat_G-300.pth")
            torch.save(heat_TGSS, "/xulei/GSSL/heat_TGSS-300.pth")


    return pred

if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device))
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    eval_metric = roc_auc_score

    'datasets'
    dataset_root = get_dataset_root(args.predicted_datasets)
    data_root = './datasets/{}'.format(dataset_root) if args.input_data_dir == '' else '{}/{}/'.format(
        args.input_data_dir, args.dataset_name)
    dataset = finetune_Construct.construct_dataset(data_root, args.predicted_datasets)
    dataset_getter = finetune_Construct.construct_dataloader(dataset)
    predicted_dataset = dataset_getter.get_dataset
    train_loader, valid_loader, test_loader, voc, test_xulei_loader = dataset_getter.get_model_dataset(predicted_dataset, batch_size=args.batch_size)
    dim_features = predicted_dataset.dim_features

    dataset_work = get_num_task(args.predicted_datasets)

    'model-2D-GNN'
    molecule_model_GNN = GNN(num_layer=args.num_layer, emb_dim=args.GNN_emb_dim, hidden_size=args.hidden_size, batch_size=args.batch_size, dim_features=dim_features, heads=args.heads, dropout=args.GNN_dropout, aggregation=args.graph_pooling)
    model_GNN = GNN_predicted(args=args, molecule_model=molecule_model_GNN)

    'model-1D-LSTM'
    molecule_model_BiLSTM = BiLSTM(dim_features=dim_features, emb_dim=args.BiLSTM_emb_dim, hidden_dim=args.BiLSTM_hidden_dim, lstm_hid_dim=args.BiLSTM_lstm_hid_dim, batch_size=args.batch_size)
    model_BiLSTM = BiLSTM_predicted(args=args, molecule_model=molecule_model_BiLSTM, dim_features=dim_features)

    'model-Transformer'
    molecule_model_Transformer = Transformer(dim_features=dim_features, emb_dim=args.Transformer_emb_dim, batch_size=args.batch_size)
    model_Transformer = Transformer_predicted(args=args, molecule_model=molecule_model_Transformer, dim_features=dim_features)



    if not args.input_model_file == '':
        model_GNN.from_pretain_GNN(args.input_model_file)
        model_BiLSTM.from_pretain_BiLSTM(args.input_model_file, voc, device)
        model_Transformer.from_pretain_Transformer(args.input_model_file, voc, device)


    model_GNN.to(device)
    model_BiLSTM.to(device)
    model_Transformer.to(device)
    print(model_GNN)
    print(model_BiLSTM)
    print(model_Transformer)

    'optimizer'
    model_param_group_GNN = [{'params': model_GNN.molecule_model.parameters(), 'lr': args.lr * args.lr_scale}]
    model_param_group_BiLSTM = [{'params': model_BiLSTM.molecule_model.parameters(), 'lr': args.lr * args.lr_scale}]
    model_param_group_Transformer = [{'params': model_Transformer.molecule_model.parameters(), 'lr': args.lr * args.lr_scale}]

    # model_param_group = [{'params': model_GNN.molecule_model.parameters(), 'lr': args.lr},
    #                      {'params': model_BiLSTM.molecule_model.parameters(), 'lr': args.lr},
    #                      {'params': model_Transformer.molecule_model.parameters(), 'lr': args.lr}]

    optimizer_GNN = optim.Adam(model_param_group_GNN, lr=args.lr, weight_decay=args.decay)
    optimizer_BiLSTM = optim.Adam(model_param_group_BiLSTM, lr=args.lr, weight_decay=args.decay)
    optimizer_Transformer = optim.Adam(model_param_group_Transformer, lr=args.lr, weight_decay=args.decay)

    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)


    if dataset_work == 1:
        criterion = nn.MSELoss()
    elif dataset_work == 0:
        criterion = nn.BCEWithLogitsLoss()


    train_auc_list, valid_auc_list, test_auc_list = [], [], []

    best_val_auc, best_val_idx = -1, 0

    for epoch in range(1, args.num_epoch + 1):
        loss = train(model_GNN,model_BiLSTM, model_Transformer, device, train_loader, optimizer_GNN, optimizer_BiLSTM, optimizer_Transformer)


        print('Epoch: {}\nLoss: {}'.format(epoch, loss))

        if args.eval_train:
            train_auc, train_target, train_pred = eval(model_GNN, model_BiLSTM, model_Transformer, device, train_loader, dataset_work)
        else:
            train_auc = 0

        val_auc, val_target, val_pred = eval(model_GNN, model_BiLSTM, model_Transformer, device, valid_loader, dataset_work)
        test_auc, test_target, test_pred, snelist = test(model_GNN, model_BiLSTM, model_Transformer, device, test_loader, dataset_work)
        # if epoch == 1:
        #     xulei = test_xulei_100(model_GNN, model_BiLSTM, model_Transformer, device, test_xulei_loader,
        #                                        dataset_work)
        # if epoch == 150:
        #     xulei = test_xulei_150(model_GNN, model_BiLSTM, model_Transformer, device, test_xulei_loader,
        #                                        dataset_work)
        # if epoch == 300:
        #     xulie = test_xulei_300(model_GNN, model_BiLSTM, model_Transformer, device, test_xulei_loader,
        #                                        dataset_work)


        # np.save("/xulei/GSSL/sne.npy",snelist)
        # with open('/xulei/GSSL/sne-table.pkl', 'wb') as f:
        #     pickle.dump(test_target, f)

        train_auc_list.append(train_auc)
        valid_auc_list.append(val_auc)
        test_auc_list.append(test_auc)
        if epoch % 10 == 0:
            print('Epoch:{}\ntrain:{:.6f}\tval:{:.6f}\ttest:{:.6f}'.format(epoch, train_auc, val_auc, test_auc))

        if test_auc > best_val_auc:
        # if test_auc < best_val_auc:
            best_val_auc = test_auc
            best_val_idx = epoch - 1
            if not args.output_model_dir == '':
                output_model_dir = join(args.output_model_dir, args.predicted_datasets + '_test_model_best_1.pth')
                saved_model_dict = {
                    'molecule_model_GNN': molecule_model_GNN.state_dict(),
                    'molecule_model_BiLSTM': molecule_model_BiLSTM.state_dict(),
                    'molecule_model_Transformer': molecule_model_Transformer.state_dict(),
                    'model_GNN': model_GNN.state_dict(),
                    'model_BiLSTM': model_BiLSTM.state_dict(),
                    'model_Transformer': model_Transformer.state_dict()
                }
                torch.save(saved_model_dict, output_model_dir)

                filename = join(args.output_model_dir, args.predicted_datasets + 'evaluation_best_1.pth')
                np.savez(filename, val_target=val_target, val_pred=val_pred,
                         test_target=test_target, test_pred=test_pred)

    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_auc_list[best_val_idx],
                                                                 valid_auc_list[best_val_idx],
                                                                 test_auc_list[best_val_idx]))
    'test'
    # model_GNN.from_pretain_GNN(args.input_model_file)
    # model_BiLSTM.from_pretain_BiLSTM(args.input_model_file, voc, device)
    # model_Transformer.from_pretain_Transformer(args.input_model_file, voc, device)
    # test_auc, test_target, test_pred, snelist = test(model_GNN, model_BiLSTM, model_Transformer, device, test_loader,
    #                                                  dataset_work)
    #
    # test_target, test_pred, snelist = test_xulei(model_GNN, model_BiLSTM, model_Transformer, device, test_xulei_loader,
    #                                        dataset_work)



    df = pd.DataFrame(data=test_auc_list)
    # df2 = pd.DataFrame(data=valid_auc_list)
    # df3 = pd.DataFrame(data=test_auc_list)
    df.to_csv("/xulei/GSSL/test output/model/B+T-HIV.csv", encoding='utf-8', index=False)
    # df1.to_csv("/xulei/GSSL/test output/feature fusion/No-ESOL.csv", encoding='utf-8', index=False)
    # df2.to_csv("/xulei/GSSL/test output/model/T+G-BACE-valid.csv", encoding='utf-8', index=False)
    # df3.to_csv("/xulei/GSSL/test output/feature fusion/Yes-Mutagenesis.csv", encoding='utf-8', index=False)



    if args.output_model_dir != '':
        output_model_path = join(args.output_model_dir, args.predicted_datasets + 'model_final_1.pth')
        saved_model_dict = {
            'molecule_model_GNN': molecule_model_GNN.state_dict(),
            'molecule_model_BiLSTM': molecule_model_BiLSTM.state_dict(),
            'molecule_model_Transformer': molecule_model_Transformer.state_dict(),
            'model_GNN': model_GNN.state_dict(),
            'model_BiLSTM': model_BiLSTM.state_dict(),
            'model_Transformer': model_Transformer.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)









