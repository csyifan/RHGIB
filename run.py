import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('util/')

from model import basehgnn
from util.data import load_data
from util.pytorchtools import EarlyStopping
from args import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):

    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)

    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]

    g = g.to(device)
    
    node_seq = [[i] for i in range(features_list[0].shape[0])]
    node_seq = torch.tensor(node_seq).long()
    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    micro_f1 = torch.zeros(1)
    macro_f1 = torch.zeros(1)

    num_classes = dl.labels_train['num_classes']
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)

    
    net = basehgnn(num_classes, in_dims, args.hidden_dim, args.num_gnns, args.dropout,
                temper=args.temperature, num_type=len(node_cnt))

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # training loop
    net.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='base_{}.pt'.format(args.dataset))
    for epoch in range(args.epoch):
        t_start = time.time()
        # training
        net.train()

        logits, _ = net(g, features_list, train_seq, type_emb, node_type, args.l2norm)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, labels[train_idx])

        # autograd
        optimizer.zero_grad() 
        train_loss.backward()
        optimizer.step()

        t_end = time.time()

        # print training info
        print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, train_loss.item(), t_end-t_start))

        t_start = time.time()

        # validation
        net.eval()
        with torch.no_grad():
            logits, _ = net(g, features_list, val_seq, type_emb, node_type, args.l2norm)
            logp = F.log_softmax(logits, 1)
            val_loss = F.nll_loss(logp, labels[val_idx])
            pred = logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            pred = onehot[pred]
            print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))

        scheduler.step(val_loss)
        t_end = time.time()
        # print validation info
        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        # early stopping
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # testing with evaluate_results_nc
    net.load_state_dict(torch.load('base_{}.pt'.format(args.dataset)))
    net.eval()
    with torch.no_grad():
        logits, _ = net(g, features_list, test_seq, type_emb, node_type, args.l2norm)
        test_logits = logits

        pred = test_logits.cpu().numpy().argmax(axis=1)
        onehot = np.eye(num_classes, dtype=np.int32)
        pred = onehot[pred]
        result = dl.evaluate_valid(pred, dl.labels_test['data'][test_idx])
        print(result)
        micro_f1[0] = result['micro-f1']
        macro_f1[0] = result['macro-f1']

 
if __name__ == '__main__':
    run_model_DBLP(args)
