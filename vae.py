from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx): 
        sparse_mx = sparse_mx.tocoo() 
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(num_features))
        self.beta = Parameter(torch.Tensor(num_features))
        self.register_buffer("moving_avg", torch.zeros(num_features))
        self.register_buffer("moving_var", torch.ones(num_features))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("momentum", torch.tensor(momentum))
        self._reset()
    
    def _reset(self):
        self.gamma.data.fill_(1)
        self.beta.data.fill_(0)
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            self.moving_avg = self.moving_avg * self.momentum + mean * (1 - self.momentum)
            self.moving_var = self.moving_var * self.momentum + var * (1 - self.momentum)
        else:
            mean = self.moving_avg
            var = self.moving_var
            
        x_norm = (x - mean) / (torch.sqrt(var + self.eps))
        return x_norm * self.gamma + self.beta

class VAE(nn.Module):
    def __init__(self, input_dimensions, hidden_dim, out_dim, device):
        super().__init__()
        self.in_dims = input_dimensions
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        self.relu = ReLU()

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, self.hidden_dim) for in_dim in self.in_dims])

        self.first_gcn = GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu)
        self.second_gcn = GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu)
        
        self.mean_gcn = GraphConv(self.hidden_dim, self.out_dim, activation=F.relu)
        self.log_stddev_gcn = GraphConv(self.hidden_dim, self.out_dim, activation=F.relu)

    def weight_init_(self,mode = 'kaiming'):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                elif mode == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight)
    
    def encode(self, g, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        gh = torch.cat(h, 0)
        h = self.first_gcn(g, gh)
        h = self.second_gcn(g, h)

        mean = self.mean_gcn(g, h)
        log_stddev = self.log_stddev_gcn(g, h)

        gauss_noise = torch.randn(h.shape[0], self.out_dim, device = self.device)
        sampled_z = gauss_noise * torch.exp(log_stddev) + mean

        return sampled_z, mean, log_stddev
    
    def decode(self, sampled_z):
        # print('sampled_z:',sampled_z)
        adj_pred = torch.matmul(sampled_z,sampled_z.t())
        adj_pred = torch.sigmoid(adj_pred)
        return adj_pred
    
    def forward(self, g, features_list, adjM):
        sampled_z, mean, log_stddev = self.encode(g, features_list)
        adj_pred = self.decode(sampled_z)

        adj_pred = torch.triu(adj_pred,diagonal = 1)
        Loss = self.VAELoss(adj_pred, adjM, mean, log_stddev)
        return adj_pred, sampled_z, Loss
    
    def VAELoss(self, adj_pred, adjM, mean, log_stddev):
        total_loss = 0.0
        for k in range(5):

            adj_label = adjM + sp.eye(adjM.shape[0])  
            adj_label = sparse_to_tuple(adj_label)
            adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
            adj_label = adj_label.to(self.device)
            similar_loss = F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1))
            
            gauss_term = 1 + 2 * log_stddev - mean ** 2
            gauss_term = gauss_term - torch.exp(log_stddev) ** 2
            gauss_term = gauss_term.sum(1).mean()
            kl_loss = 0.5 / adj_pred.size(0) * gauss_term

            Loss = similar_loss + kl_loss
            total_loss += Loss
        
        total_loss = total_loss / 5
        return total_loss