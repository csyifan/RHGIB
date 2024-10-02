import numpy as np
import torch
import dgl
import scipy.sparse as sp
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import random
from scipy.sparse import csr_matrix

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

def get_sg_feature(sg, features_list):
    nodes = sg.nodes()
    ori_node_id = sg.ndata[dgl.NID][nodes.long()]
    
    feature = torch.vstack((features_list[0], features_list[1]))
    feature = torch.vstack((feature, features_list[2]))
    new_feature = feature[ori_node_id]
    return new_feature

def get_subgraph_pg(sg, features_list, node_cnt, seq, node_type):
    node_cnt_type = [sum(node_cnt[:i+1]) for i in range(len(node_cnt))]
    nodes = sg.nodes()
    ori_node_id = np.array(sg.ndata[dgl.NID][nodes.long()].cpu())
    new_feature_list = [[] for _ in range(len(node_cnt))]
    seq_nodes = []
    cnt = 0
    for nodes in ori_node_id:
        index = sum(1 for x in node_cnt_type if x <= nodes)
        if index == 0:
            new_feature = features_list[index][nodes]
            if nodes < seq.shape[0]:
                seq_nodes.append(cnt)
        else:
            new_feature = features_list[index][nodes-node_cnt_type[index-1]]
        new_feature_list[index].append(new_feature)
        cnt += 1
    a = new_feature_list
    tensor_dict = {}
    for sublist in a:
        for tensor in sublist:
            shape = tensor.size()
            if shape not in tensor_dict:
                tensor_dict[shape] = [tensor]
            else:
                tensor_dict[shape].append(tensor)

    new_feature_list = [torch.stack(tensor_list) for tensor_list in tensor_dict.values()]
    sg_node_type = node_type[ori_node_id]
    
    return new_feature_list, torch.tensor(seq_nodes).long().unsqueeze(1), sg_node_type

def get_subgraph(sg, features_list, node_cnt, seq, node_type):
    node_cnt_type = [sum(node_cnt[:i+1]) for i in range(len(node_cnt))]
    nodes = sg.nodes()
    ori_node_id = np.array(sg.ndata[dgl.NID][nodes.long()].cpu())
    new_feature_list = [[] for _ in range(len(node_cnt))]
    

    seq_nodes = []
    cnt = 0
    
    for node in ori_node_id:

        index = sum(1 for x in node_cnt_type if x <= node)
        
        if index == 0:

            new_feature = features_list[index][node]
            if node < seq.shape[0]:
                seq_nodes.append(cnt)
        else:

            new_feature = features_list[index][node - node_cnt_type[index-1]]
        

        new_feature_list[index].append(new_feature)
        cnt += 1
    

    for i in range(len(new_feature_list)):
        if new_feature_list[i]:
            new_feature_list[i] = torch.stack(new_feature_list[i])
        else:

            device = features_list[i][0].device
            new_feature_list[i] = torch.empty((0,) + features_list[i][0].size(), dtype=features_list[i][0].dtype, device=device)
    

    sg_node_type = node_type[ori_node_id]
    

    return new_feature_list, torch.tensor(seq_nodes).long().unsqueeze(1), sg_node_type


def find_index(a, b):
    for i, upper_bound in enumerate(b):
        if a < upper_bound:
            return i

def get_etype(dl):
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
                # edge2type[(v,u)] = k
    return edge2type

def get_etype_noise(dl, g):
    edge2type = {}
    
    node_counts = []
    cnt = 0
    for i in range(len(dl.nodes['count'])):
        cnt += dl.nodes['count'][i]
        node_counts.append(cnt)
        
    range_dict = {
    '0-1': 0,
    '1-2': 1,
    '1-3': 2,
    '0-2': 3,
    '0-3': 4,
    '2-3': 5,
    }  
    
    for u,v in zip(*g.edges()):
        u = int(u)
        v = int(v)
        if (u,v) not in edge2type:
            utype = find_index(u, node_counts)
            vtype = find_index(v, node_counts)
            if utype == vtype:
                edge2type[(v,u)] = len(dl.links['count'])
                edge2type[(u,v)] = len(dl.links['count'])
            else:
                key1 = f'{utype}-{vtype}'
                key2 = f'{vtype}-{utype}'
                kuv = range_dict.get(key1, None)
                kvu = range_dict.get(key2, None)
                if kuv != None :
                    a = kuv
                else:
                    a = kvu
                edge2type[(u,v)] = a
                edge2type[(v,u)] = a
                
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
                
    return edge2type


def get_efeat(edge2type, g, device):
    e_feat = []
    for u, v in zip(*g.edges()):
        u = g.ndata[dgl.NID][u].cpu().item()
        v = g.ndata[dgl.NID][v].cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    return e_feat

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):   
        sparse_mx = sparse_mx.tocoo()  
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_z_subgraph(sg, features_list, node_cnt, seq, node_type):
    node_cnt_type = [sum(node_cnt[:i+1]) for i in range(len(node_cnt))]
    nodes = sg.nodes()
    ori_node_id = np.array(sg.ndata[dgl.NID][nodes.long()].cpu())
    new_feature_list = []
    seq_nodes = []
    for nodes in ori_node_id:
        new_feature = features_list[nodes]
        new_feature_list.append(new_feature)
    feature = torch.cat([t.unsqueeze(0) for t in new_feature_list], dim=0)
    sg_node_type = node_type[ori_node_id]
    
    return feature, seq_nodes, sg_node_type

def add_noise(g_ori, ratio):
    noise_num = int(np.round(ratio * g_ori.num_edges()))
    noise_num = min(max(noise_num, 1),g_ori.num_edges())
    
    edge_index = [g_ori.edges()[0].tolist(),g_ori.edges()[1].tolist()]
    edge_index = torch.tensor(edge_index)
    
    adj = to_dense_adj(edge_index)
    adj = torch.squeeze(adj, dim = 0)
    
    row = random.choices(range(g_ori.num_nodes()), k=noise_num)
    col = random.choices(range(g_ori.num_nodes()), k=noise_num)
    for (i,j) in zip(row,col):
        adj[i,j] = 1 - adj[i,j]
    
    adjM = csr_matrix(adj)
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    return g, adjM