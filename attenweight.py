import torch
import torch.nn as nn
import dgl
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import edge_softmax
from dgl import function as fn

class AttentionWeight(nn.Module):
    def __init__(self, edge_feats, num_etypes, in_feats, out_feats, num_heads, negative_slope=0.2):
        super(AttentionWeight, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(out_feats)
        self._out_feats = out_feats
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        
        
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        self.leaky_relu = nn.ReLU()
        
        self.attn_mlp = nn.Linear(num_heads, 1, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_mlp.weight, gain=gain)

    def forward(self, graph, feat_list, e_feat):
        feat = feat_list
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = feat[0]
                h_dst = feat[1]
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = feat
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            
            graph.srcdata.update({'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            
            attention_weights = edge_softmax(graph, e).squeeze()
            
            
            return attention_weights