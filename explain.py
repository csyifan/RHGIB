from args import *
from util.data_process import *
from util.data import load_data
from util.pytorchtools import EarlyStopping
from model import basehgnn
from Explainer import RAIC
import pandas as pd

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

net = basehgnn(num_classes, in_dims, args.hidden_dim, args.num_gnns, args.dropout, temper=args.temperature, num_type=len(node_cnt))
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

net.load_state_dict(torch.load('base_{}.pt'.format(args.dataset), map_location = 'cuda:0'))
num_etypes = len(dl.links['count'])*2+1

explainer = RAIC(num_etypes, g, net, features_list, in_dims, args.hidden_dim, args.khop, device)
explainer.vae_train(adjM, 100)
explainer.train(train_idx, val_idx, train_seq, type_emb, node_type, args.l2norm, num_etypes, in_dims, etype, adjM, args.mean, val_seq)
mae,mse,rmse = explainer.explain(test_idx, test_seq, type_emb, node_type, args.l2norm, etype, adjM, num_etypes)

