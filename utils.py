import json
import numpy as np
import scipy
import torch
import pickle
import sys
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg.eigen.arpack import eigsh

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize(adj):
    rowsum = torch.sum(adj, dim=1) + 1e-8
    d = torch.reciprocal(rowsum)
    d = torch.diag(d)
    return d.mm(adj)

def feature_standardize(f):
    mu = f.mean(dim=0)
    sigma = f.std(dim=0) + 1e-8
    f = (f - mu) / sigma
    return f

def feature_norm(f):
    rowsum = torch.sum(f**2, dim=1)
    rowsum = rowsum + 1e-8
    r_inv = torch.rsqrt(rowsum)
    r_mat_inv = torch.diag(r_inv).cuda()
    f = r_mat_inv.mm(f)
    return f

def cal_accuracy(output, y_val, val_mask):
    output = output[val_mask]
    y_val = y_val[val_mask]
    out_label = torch.argmax(output, dim=1)
    return sum([y_val[i][out_label[i]] for i in range(y_val.shape[0])])/float(y_val.shape[0])

def load_imdb(dataset=r'../../github/HAN/data/imdb/IMDB_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    node_label_list = ['label']
    MvsA = data['MvsA']
    MvsD = data['MvsD']
    M_features = data['MvsP'].tocsr()
    A_features = sp.csr_matrix((MvsA.shape[1], 1))
    D_features = sp.csr_matrix((MvsD.shape[1], 1))
    M_loop = sp.eye(MvsA.shape[0])
    A_loop = sp.eye(MvsA.shape[1])
    D_loop = sp.eye(MvsD.shape[1])
    adj_list = [M_loop, A_loop, D_loop, MvsA, MvsA.T, MvsD, MvsD.T]
    adj_type = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0)]
    features = [M_features, A_features, D_features]
    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]])
        else:
            print("type %s node have not label" %(t))
            exit(0)
    return adj_list, adj_type, features, label

def load_dblp(dataset=r'../../github/HAN/data/DBLP_four_area/DBLP_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    node_label_list = ['paper_label', 'author_label']
    PvsA = data['PvsA']
    PvsC = data['PvsC']
    P_features = data['PvsT'].tocsr()
    A_features = sp.csr_matrix((PvsA.shape[1], 1))
    C_features = sp.csr_matrix((PvsC.shape[1], 1))
    P_loop = sp.eye(PvsA.shape[0])
    A_loop = sp.eye(PvsA.shape[1])
    C_loop = sp.eye(PvsC.shape[1])
    adj_list = [P_loop, A_loop, C_loop, PvsA, PvsA.T, PvsC, PvsC.T]
    adj_type = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0)]
    features = [P_features, A_features, C_features]
    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]].toarray())
        else:
            print("type %s node have not label" %(t))
            exit(0)
    return adj_list, adj_type, features, label


def load_aminer(dataset=r'../../github/HAN/data/Aminer/Aminer_processed.mat', target_node=[0]):
    data = sio.loadmat(dataset)
    node_label_list = ['PvsC', 'AvsC']
    PvsA = data['PvsA']
    PvsP = data['PvsP']
    AvsA = data['AvsA']
    P_features = sp.csr_matrix(data['PvsF'])
    A_features = sp.csr_matrix(data['AvsF'])
    P_loop = sp.eye(PvsA.shape[0])
    A_loop = sp.eye(PvsA.shape[1])
    adj_list = [P_loop, A_loop, PvsA, PvsA.T, PvsP, AvsA]
    adj_type = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, 1)]
    features = [P_features, A_features]
    label = []
    for t in target_node:
        if t < len(node_label_list):
            label.append(data[node_label_list[t]].toarray())
        else:
            print("type %s node have not label" %(t))
            exit(0)
    return adj_list, adj_type, features, label


def load_heterogeneous_data(dataset_str,
                            train_rate=0.1,
                            val_rate=0.1,
                            test_rate=0.1,
                            target_node=[0]): # {'imdb', 'dblp', 'aminer'}
    """Load data."""
    dataset_dict = {'imdb': load_imdb, 'dblp': load_dblp, 'aminer': load_aminer}

    adj, adj_type, features, labels = dataset_dict[dataset_str](target_node=target_node)
    if len(labels) < 1:
        print("Error: nodes have not labels!")
        exit(0)
    sum_labels = [np.sum(l, axis=1) for l in labels]
    all_sample_with_label = [np.where(l>0)[0] for l in sum_labels]
    for i in range(len(labels)):
        print("we have %s samples (type %s) with label"%(len(all_sample_with_label[i]), target_node[i]))
    
    num_targets = [len(l) for l in all_sample_with_label]
    num_train = [int(num * train_rate) for num in num_targets]
    num_val = [int(num * val_rate) for num in num_targets]
    num_test = [int(num * test_rate) for num in num_targets]
    
    idx_random = all_sample_with_label#list(range(num_targets))
    for i in range(len(all_sample_with_label)):
        random.shuffle(idx_random[i])
    idx_train = [idx_random[i][0 : num_train[i]] for i in range(len(idx_random))]
    idx_val = [idx_random[i][num_train[i] : num_train[i] + num_val[i]] for i in range(len(idx_random))]
    idx_test = [idx_random[i][num_train[i] + num_val[i] : num_train[i] + num_val[i] + num_test[i]] for i in range(len(idx_random))]
    
    train_mask = [sample_mask(idx_train[i], labels[i].shape[0]) for i in range(len(labels))]
    val_mask = [sample_mask(idx_val[i], labels[i].shape[0]) for i in range(len(labels))]
    test_mask = [sample_mask(idx_test[i], labels[i].shape[0]) for i in range(len(labels))]

    y_train = [np.zeros(labels[i].shape) for i in range(len(labels))]
    y_val = [np.zeros(labels[i].shape) for i in range(len(labels))]
    y_test = [np.zeros(labels[i].shape) for i in range(len(labels))]
    for i in range(len(labels)):
        y_train[i][train_mask[i], :] = labels[i][train_mask[i], :]
        y_val[i][val_mask[i], :] = labels[i][val_mask[i], :]
        y_test[i][test_mask[i], :] = labels[i][test_mask[i], :]

    print("all adj shape:", [a.shape for a in adj])
    print("responding adj type:", [a for a in adj_type])
    print("all features of nodes shape:", [f.shape for f in features])
    print("all y_train num:", [len(y) for y in idx_train])
    print("all y_val num:", [len(y) for y in idx_val])
    print("all y_test num:", [len(y) for y in idx_test])
    return adj, adj_type, features, y_train, y_val, y_test, train_mask, val_mask, test_mask