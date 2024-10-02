import argparse
import random
import sys
import dgl
import numpy as np
import torch
sys.path.append('util/')
from util.data import load_data
from data_process import *


ap = argparse.ArgumentParser(description='raic')
ap.add_argument('--feats-type', type=int, default=3,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2' +
                '4 - only term features (id vec for others);' +
                '5 - only term features (zero vec for others).')
ap.add_argument('--device', type=int, default=1)
ap.add_argument('--hidden-dim', type=int, default=256)
ap.add_argument('--dataset', type=str, default = 'DBLP')
ap.add_argument('--epoch', type=int, default=1000)
ap.add_argument('--patience', type=int, default=50)
ap.add_argument('--num-gnns', type=int, default=3)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=2023)
ap.add_argument('--dropout', type=float, default=0)
ap.add_argument('--weight-decay', type=float, default=0)
ap.add_argument('--l2norm', type=bool, default=True)
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--khop', type=int, default=2)
ap.add_argument('--mean', type=int, default=1)
ap.add_argument('--noise', type=float, default=0)

args = ap.parse_args()


feats_type = args.feats_type
features_list, adj, labels, train_val_test_idx, dl = load_data(args.dataset)
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
# device= torch.device('cpu')
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
g_ori = dgl.DGLGraph(adj+(adj.T))
g_ori = dgl.remove_self_loop(g_ori)

if args.noise == 0:
    g = g_ori
    adjM = adj
else:
    g, adjM = add_noise(g_ori, args.noise) # with noise

all_nodes = np.arange(features_list[0].shape[0])

node_seq = [[i] for i in range(features_list[0].shape[0])]
node_seq = torch.tensor(node_seq).long()

node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]

g = g.to(device)
train_seq = node_seq[train_idx]
val_seq = node_seq[val_idx]
test_seq = node_seq[test_idx]

num_classes = dl.labels_train['num_classes']
type_emb = torch.eye(len(node_cnt)).to(device)
node_type = torch.tensor(node_type).to(device)

if args.noise == 0:
    etype = get_etype(dl)
else:
    etype = get_etype_noise(dl, g) # with noise
    