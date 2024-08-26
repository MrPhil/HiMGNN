# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import sys
from sklearn.metrics import f1_score
from utils_ import load_data,loadFeature_drug,loadFeature_virus
import torch.nn as nn
import sklearn
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import average_precision_score,precision_recall_curve, recall_score,precision_score,accuracy_score,confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
import dgl
from scipy import sparse
from model2_schnetSA import HiMGNN
from dataset import *
from torch_geometric.data import DataLoader
import csv
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class DDA_PU_loss(nn.Module):        
    def __init__(self):
        super(DDA_PU_loss,self).__init__()
        
    def forward(self,drug_virus_reconstruct,drug_virus,drug_virus_mask,pos_x_index,pos_y_index,neg_x_index,neg_y_index,alpha):
        alpha=alpha
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss_mat = loss_fn(drug_virus_reconstruct, drug_virus)
        #pos_x_index, pos_y_index=drug_virus.nozero       
        loss = (loss_mat[pos_x_index, pos_y_index].sum()*((1-alpha)/2) + loss_mat[neg_x_index, neg_y_index].sum()*(alpha/2))
        # lamda_u = 
        # lamda_v = 1    
        #reg = lamda_u * (torch.trace(torch.mm(x_m.t(), x_m))) + lamda_v * (torch.trace(torch.mm(x_d.t(), x_d)))
        #loss = loss + reg        
        return loss   

def evaluate(model,g,batch_data,features_d,features_p,noise,DDAtrain,DDAtest,DDAvalid):
    model.eval()
    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    pred_list = []
    ground_truth = []
    with torch.no_grad():
        d_x,p_x,logits = model(g, batch_data,features_d,features_p,noise)
        logits=logits.cpu().numpy()

    DDAvalid=DDAvalid.cpu().numpy()
    
    DDAtest=DDAtest.cpu().numpy()

    for ele in DDAvalid:
        pred_list.append(logits[ele[0],ele[1]])
        ground_truth.append(ele[2])

    valid_auc = roc_auc_score(ground_truth, pred_list)
            #print (valid_auc)
    valid_aupr = average_precision_score(ground_truth, pred_list)

    pred_list_pre = []
    for i in range(len(pred_list)):
        if pred_list[i] > 0.5:
            pred_list_pre.append(np.array([1.0]))
        else:
            pred_list_pre.append(np.array([0.0]))
    acc, precision, recall, F1 = accuracy_score(ground_truth, pred_list_pre),precision_score(ground_truth, pred_list_pre), recall_score(ground_truth, pred_list_pre,average='micro'),f1_score(ground_truth, pred_list_pre)
    print('acc:', acc, 'precision:', precision, 'recall:',recall,'F1:',F1)


    if valid_aupr >= best_valid_aupr:
        
        best_valid_aupr = valid_aupr
        best_valid_auc = valid_auc
        pred_list = []
        ground_truth = []
        for ele in DDAtest:
            pred_list.append(logits[ele[0],ele[1]])
            ground_truth.append(ele[2])
        test_auc = roc_auc_score(ground_truth, pred_list)
        fpr,tpr,thresholds=roc_curve(ground_truth, pred_list)
        test_aupr = average_precision_score(ground_truth, pred_list)
        pred_list_pre = []
        for i in range(len(pred_list)):
            if pred_list[i] > 0.5:
                pred_list_pre.append(np.array([1.0]))
            else:
                pred_list_pre.append(np.array([0.0]))

        acc= accuracy_score(ground_truth, pred_list_pre)

    print('acc:', acc)
    print ('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)

    return best_valid_auc, best_valid_aupr,test_auc, test_aupr,fpr,tpr,precision,recall,acc,F1,ground_truth,pred_list,pred_list_pre

def train_and_evaluate(DDAtrain, DDAvalid, DDAtest, graph, pos_x_index, pos_y_index, neg_x_index, neg_y_index,
                           drug_virus_train, train_mask, batch_data, features_d, features_p,noise, epochs, in_size, out_size,
                           loss_alpha,fold):

    pos_x_index=torch.tensor(pos_x_index,dtype=torch.long)
    pos_y_index=torch.tensor(pos_y_index,dtype=torch.long)
    neg_x_index=torch.tensor(neg_x_index,dtype=torch.long)
    neg_y_index=torch.tensor(neg_y_index,dtype=torch.long)
    DDAtrain=torch.from_numpy(DDAtrain).long()
    DDAvalid=torch.from_numpy(DDAvalid).long()
    DDAtest=torch.from_numpy(DDAtest).long()
    drug_virus_train=torch.from_numpy(drug_virus_train).float()
    train_mask=torch.from_numpy(train_mask).float()

    model = HiMGNN(
                all_meta_paths=[[['similarity'], ['dv', 'vd'], ['similarity', 'dv', 'vd']],
                                [['similarity'], ['vd', 'dv'], ['similarity', 'vd', 'dv']]],#3
                in_size=in_size,#features_d.shape[1],
                hidden_size=args['hidden_units'],
                out_size=args['out_size'],
                num_heads=args['num_heads'],
                dropout=args['dropout'],
                GAT_Layers=args['Gat_layers'],
                W_size=args['W_size'],
                Nei_number=args['Nei_number'],
                lamda=args['lamda'],
                cutoff=args['cutoff']).to(args['device'])
        
    loss_fcn=DDA_PU_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    
    DDAtrain = DDAtrain.to(args['device'])
    DDAvalid = DDAvalid.to(args['device'])
    DDAtest = DDAtest.to(args['device'])
    train_mask = train_mask.to(args['device'])
    drug_virus = drug_virus_train.to(args['device'])
    pos_x_index=pos_x_index.to(args['device'])
    pos_y_index=pos_y_index.to(args['device'])
    neg_x_index=neg_x_index.to(args['device'])
    neg_y_index=neg_y_index.to(args['device'])

    val_au=0
    save_model=0
    best_test=[]

    for epoch in range(epochs):
        model.train()

        d,p,logits = model(graph,batch_data,features_d,features_p,noise)
        loss = loss_fcn(logits,drug_virus,train_mask,pos_x_index,pos_y_index,neg_x_index,neg_y_index,loss_alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        best_valid_auc, best_valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall,acc,F1,ground_truth,pred_list,pred_list_pre = evaluate(model, graph,
                                                                                                     batch_data,
                                                                                                     features_d,
                                                                                                     features_p,noise,
                                                                                                     DDAtrain, DDAtest,
                                                                                                     DDAvalid)
        print('Epoch {:d} | Train Loss {:.4f} | best_valid_auc {:.4f} | best_valid_aupr {:.4f} |'
              'test_auc {:.4f} |test_aupr {:.4f}'.format(
            epoch + 1, loss.item(), best_valid_auc, best_valid_aupr,test_auc, test_aupr))

        if save_model < test_auc and val_au<best_valid_auc:
            save_model=test_auc
            val_au=best_valid_auc
            best_test=[test_auc, test_aupr,fpr,precision,recall,acc,F1]
            ground_pred=[]
            ground_pred_pre = []
            model_best=model
            epoch_best=epoch
            for ground_i in range(len(ground_truth)):
                ground_pred.append([ground_truth[ground_i],pred_list[ground_i]])
                ground_pred_pre.append([ground_truth[ground_i], pred_list_pre[ground_i][0]])


    torch.save(model_best, args['log_dir'] + '/model_{:d}_{:d}.pt'.format(fold, epoch_best))
    StorFile(ground_pred, args['log_dir'] + '/ground_pred{:d}.csv'.format(fold))
    StorFile(ground_pred_pre, args['log_dir'] + '/ground_pred_pre{:d}.csv'.format(fold))

    return best_test

    

def get_train(DDAtrain,num_drug,num_virus):   

    drug_virus = np.zeros((num_drug,num_virus))
    mask = np.zeros((num_drug,num_virus))
    #print (DDAtrain)
    pos_x_index=[]
    pos_y_index=[]
    neg_x_index=[]
    neg_y_index=[]
    for ele in DDAtrain:
        drug_virus[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
        if ele[2]==1:
            pos_x_index.append(ele[0])
            pos_y_index.append(ele[1])
        if ele[2]==0:
            neg_x_index.append(ele[0])
            neg_y_index.append(ele[1])    
    train_mask=mask
    return pos_x_index,pos_y_index,neg_x_index,neg_y_index,drug_virus,train_mask


import time
def main(args):

    dataset = Dataset(root='./dataset/'+args['data']+'/', path='drugs.csv',model_name='allenai/scibert_scivocab_uncased') ####
    dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, follow_batch=['pos1'])
    for i, batch_data in enumerate(tqdm(dataset_loader)):
        batch_data.pos = batch_data.pos1
        batch_data.z = batch_data.z1.type(dtype=torch.int32)
        batch_data.batch = batch_data.pos1_batch
        batch_data = batch_data.to(args['device'])
    torch.save(batch_data.batch, args['log_dir'] + '/'+args['data']+'_batch_data.pt')
    in_size=args['in_size']
    alpha=args['alpha']
    out_size=args['out_size']
    k_CV=args['k_cv']
    sample_times=args['sample_times']

    data_set,graph_old,num_drug,num_virus = load_data(args['dataset'],args['ratio'],args['network_path'],split=False,simi_xita=args['xita'],dir=args['log_dir'])

    hd = loadFeature_drug(args['network_path'])
    hp = loadFeature_virus(args['network_path'])
    noise_hp = torch.randn((num_virus, hp.shape[1]))
    hp = hp*(1-args['lamda'][1])+noise_hp*args['lamda'][1]
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])
    noise = noise_hp.to(args['device'])
    in_size = [in_size, features_p.shape[1]]

    test_auc_round = []
    test_aupr_round = []

    for r in range(sample_times):       
        print ('sample round',r+1)
        np.random.seed(int(time.time()))
        kf = StratifiedKFold(n_splits=k_CV, shuffle=True, random_state=666)
        test_auc_fold = []
        test_aupr_fold = []
        test_acc_fold=[]

        resuly=[]
        fold=1
        for train_index, test_index in kf.split(data_set[:,:2],data_set[:,2]):

            DDAtrain, DDAtest = data_set[train_index], data_set[test_index]
            DDAtrain, DDAvalid = train_test_split(DDAtrain, test_size=0.125, random_state=666)

            print ("#############%d fold"%fold+"#############")
            fold=fold+1

            drug_drug = np.loadtxt(args['network_path'] + 'd_d.txt')
            virus_virus = np.loadtxt(args['network_path'] + 'v_v.txt')
            drug_virus = np.zeros((drug_drug.shape[0],virus_virus.shape[0]))
            for DDAtrain_i in range(len(DDAtrain)):
                if DDAtrain[DDAtrain_i][2]==1:
                    drug_virus[DDAtrain[DDAtrain_i][0],DDAtrain[DDAtrain_i][1]]=1

            virus_drug = drug_virus.T
            drug_drug = np.where(drug_drug > args['xita'], 1, 0)
            virus_virus = np.where(virus_virus > args['xita'], 1, 0)

            d_d = dgl.graph(sparse.csr_matrix(drug_drug), ntype='drug',
                            etype='similarity')
            p_p = dgl.graph(sparse.csr_matrix(virus_virus), ntype='virus', etype='similarity')
            d_p = dgl.bipartite(sparse.csr_matrix(drug_virus), 'drug', 'dv', 'virus')
            p_d = dgl.bipartite(sparse.csr_matrix(virus_drug), 'virus', 'vd', 'drug')
            num_drug = d_d.number_of_nodes()
            num_virus = p_p.number_of_nodes()
            dg = dgl.hetero_from_relations([d_d, d_p, p_d])
            pg = dgl.hetero_from_relations([p_p, p_d, d_p])
            graph = [dg, pg]


            pos_x_index,pos_y_index,neg_x_index,neg_y_index,drug_virus_train,train_mask=get_train(DDAtrain,num_drug,num_virus)
            best_test = train_and_evaluate(
                DDAtrain, DDAvalid, DDAtest, graph, pos_x_index, pos_y_index, neg_x_index, neg_y_index,
                drug_virus_train, train_mask, batch_data,features_d, features_p,noise, args['num_epochs'], in_size, out_size, alpha,fold-1)   ######
            test_auc_fold.append(round(best_test[0],4))
            test_aupr_fold.append(round(best_test[1],4))
            test_acc_fold.append(round(best_test[5],4)) #5


        test_auc_fold.extend([round(np.mean(test_auc_fold[1:]),4),np.std(test_auc_fold[1:], ddof=1)])
        test_aupr_fold.extend([round(np.mean(test_aupr_fold[1:]),4),np.std(test_aupr_fold[1:], ddof=1)])
        test_acc_fold.extend([round(np.mean(test_acc_fold[1:]),4),np.std(test_acc_fold[1:], ddof=1)])

        resuly.extend([test_auc_fold,test_aupr_fold,test_acc_fold])

        test_auc_round.append(round(np.mean(test_auc_fold[1:6]),4))
        test_aupr_round.append(round(np.mean(test_auc_fold[1:6]),4))
        StorFile(resuly, args['log_dir'] + '/results.csv')



if __name__ == '__main__':
    import argparse
    from utils_ import setup

    parser = argparse.ArgumentParser('HiMGNN')
    parser.add_argument('-s', '--seed', type=int, default=666,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--pre_trained', action='store_true',
                        help='pre_trained hetero_data drug and virus feature ')
    parser.add_argument('-data', '--data', type=str, default='VDA2',
                        help='different dataset. eg.HDVD,VDA2')
    parser.add_argument('-r', '--ratio', type=str, default='one')
    parser.add_argument('-path', '--network_path', type=str, default='dataset/VDA2/',
                        help='different dataset path.eg.HDVD,VDA2')
    parser.add_argument('--device', type=str, default='cuda:0', help='Devices')
    parser.add_argument('--xita', type=float, default=0.1)
    parser.add_argument('--Nei_number', type=float, default=-1)
    parser.add_argument('--lamda', type=float, default=[0.3,0.4])

    args = parser.parse_args().__dict__
    args = setup(args)
    args['device']=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)
    main(args)


