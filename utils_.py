# -*- coding: utf-8 -*-

import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """

    log_dir = os.path.join(
        args['log_dir'],
        '{}'.format(args['dataset']))
    mkdir_p(log_dir)
    return log_dir


default_configure = {
    'lr': 0.001,
    'num_heads': [8],
    'hidden_units': 16,
    'dropout': 0.4,
    'weight_decay': 0.001,
    'num_epochs': 300,
    'k_cv': 5,
    'sample_times': 1,
    'in_size': 64,
    'out_size': 128,
    'W_size': 256,
    'Gat_layers': 2,
    'alpha': 0.5,
    'cutoff': 10.0
}

sampling_configure = {
    'batch_size': 32
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    if args['data'] == 'VDA2':
        args['dataset'] = 'VDA2'
    elif args['data'] == 'HDVD':
        args['dataset'] = 'HDVD'

    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


import time
def load_otherdata_test(network_path, r, simi_xita=0.05,dir='.'):

    drug_drug = np.loadtxt(network_path + 'd_d.txt')
    virus_virus = np.loadtxt(network_path + 'v_v.txt')
    drug_virus = np.loadtxt(network_path + 'd_v.txt')
    virus_drug = drug_virus.T
    dda_o = np.loadtxt(network_path + 'd_v.txt')

    drug_drug = np.where(drug_drug > simi_xita, 1, 0)
    virus_virus = np.where(virus_virus > simi_xita, 1, 0)

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
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dda_o)[0]):
        for j in range(np.shape(dda_o)[1]):
            if int(dda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dda_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    np.random.seed(276)
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    return data_set, graph, num_drug, num_virus

def load_otherdata_test2(network_path, r, simi_xita=0.05,dir='.'):

    drug_drug = np.loadtxt(network_path + 'd_d.txt')
    virus_virus = np.loadtxt(network_path + 'v_v.txt')
    drug_virus = np.loadtxt(network_path + 'd_v.txt')
    virus_drug = drug_virus.T
    dda_o = np.loadtxt(network_path + 'd_v.txt')

    drug_drug = np.where(drug_drug > simi_xita, 1, 0)
    virus_virus = np.where(virus_virus > simi_xita, 1, 0)

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
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dda_o)[0]):
        for j in range(np.shape(dda_o)[1]):
            if int(dda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dda_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    np.random.seed(892)
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    return data_set, graph, num_drug, num_virus


def load_data(dataset, r, network_path, split=True, simi_xita=0.05,dir='.'):
    if dataset == 'HDVD':
        if split != True:
            return load_otherdata_test(network_path, r, simi_xita=simi_xita,dir=dir)
    elif dataset == 'VDA2':
        return load_otherdata_test2(network_path, r, simi_xita=simi_xita, dir=dir)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

import csv

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def loadFeature_drug(network_path):
    index_drug = []
    ReadMyCsv1(index_drug, network_path + 'index_drug.csv')
    drugs = []
    ReadMyCsv1(drugs, network_path + 'drugs.csv')
    feature = np.load(network_path + 'embedding-ddi-1-49.npy')
    print(feature.shape)
    featureL = []
    for i in range(1, len(drugs)):
        for j in range(len(index_drug)):
            flag = 0
            if drugs[i][1] == index_drug[j][2]:
                featureL.append(feature[j])
                flag = 1
                break
        if flag == 0:
            featureL.append(np.random.randn(64))
    print('drug_pre_feature:', len(featureL), 64)

    return torch.tensor(featureL, dtype=torch.float32)

def loadFeature_virus(network_path):

    feature = np.load(network_path + 'virus_feature.npy')

    return torch.tensor(feature, dtype=torch.float32)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name, encoding='utf-8'))
    for row in csv_reader:
        save_list.append(row)
    return

