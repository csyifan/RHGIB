import torch
import torch_geometric as ptgeom
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import dgl
import sys
sys.path.append('..')
from data_process import *
from util.pytorchtools import EarlyStopping
import os
from attenweight import AttentionWeight
from vae import VAE
import pandas as pd

class RHGIB():
    def __init__(self, num_etypes, g, model_to_explain, feature_list, in_dims, hidden_dim, khop, device='cuda', epochs=100, lr=0.001, temp=(5.0, 2.0), reg_coefs=(1e-3, 1e-2),sample_bias=0):
        super().__init__()
        
        self.g = g
        self.model_to_explain = model_to_explain
        self.feature_list = feature_list
        self.in_dims = in_dims
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        
        vaeout = 32
        self.expl_embedding = 32
        
        self.khop = khop
        self.node_cnt = [features.shape[0] for features in self.feature_list]
        self.device = device
        self.VAE = VAE(self.in_dims, 64, vaeout, self.device).to(self.device)
        self.explainer_model = AttentionWeight(32, num_etypes, self.in_dims, self.expl_embedding, 1).to(self.device)

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001 
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(),device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    
    def _loss(self, masked_pred, original_pred, mask, reg_coefs, flag):

        entropy_reg = reg_coefs[1]
        
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if flag == 1:
            cce_loss = F.l1_loss(masked_pred, original_pred)
        else: 
            CELoss = nn.CrossEntropyLoss()
            original_pred = torch.argmax(original_pred, axis=1)
            cce_loss = CELoss(masked_pred, original_pred)
            
        all_loss = cce_loss + mask_ent_loss 

        return all_loss
    
    def _mask_graph_new(self, mask, rate):
        
        new_mask = torch.zeros(mask.shape[0]).to(self.device)
        _, idx = torch.sort(mask,descending=True)

        top_idx = idx[:int(rate*len(idx))]
        new_mask[top_idx]=1

        return new_mask

    def vae_train(self, adjM, vae_epoch):
        early_stopping = EarlyStopping(patience=30, verbose=True, save_path='testVAE.pt')
        optimizer = Adam(self.VAE.parameters(), lr=self.lr)
        for epoch in range(vae_epoch):
            self.VAE.train()
            _, _, Loss = self.VAE(self.g, self.feature_list, adjM)
            optimizer.zero_grad() 
            Loss.backward()
            optimizer.step()
            print('Epoch {:05d} | Train_Loss: {:.4f}'.format(epoch, Loss.item()))

            self.VAE.eval()
            with torch.no_grad():
                _, _, Loss = self.VAE(self.g, self.feature_list, adjM)

            early_stopping(Loss, self.VAE)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

    def train(self, train_idx, val_idx, train_seq, type_emb, node_type, l2norm, num_etypes, in_dims, etype, adjM, flag, val_seq):

        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam([{'params':self.VAE.parameters()}, {'params':self.explainer_model.parameters()}], self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        early_stopping = EarlyStopping(patience=20, verbose=True, save_path='exp.pt')
        
        with torch.no_grad():
            output, _ = self.model_to_explain(self.g, self.feature_list, train_seq, type_emb, node_type, l2norm)
            pred = torch.argmax(output, axis=1)

        for e in tqdm(range(0, self.epochs)):
            
            epoch_loss = 0

            _, sampled_z, vloss = self.VAE(self.g, self.feature_list, adjM)

            for n in range(len(train_idx)):
                
                self.explainer_model.train()
                self.VAE.train()
                optimizer.zero_grad()
                
                loss = torch.FloatTensor([0]).detach().to(self.device)
                vaeloss = 0

                sg, new_id = dgl.khop_in_subgraph(self.g, train_idx[n], k=self.khop)
                sg = dgl.add_self_loop(sg)
                sg_sampled_z, _, sg_node_type = get_z_subgraph(sg, sampled_z, self.node_cnt, train_seq, node_type)
                sg_feature_list, _, sg_node_type = get_subgraph(sg, self.feature_list, self.node_cnt, train_seq, node_type)
                t = temp_schedule(e)
                e_feat = get_efeat(etype, sg, self.device)

                sampling_weights = self.explainer_model(sg, sg_sampled_z, e_feat)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                masked_pred, _ = self.model_to_explain(sg, sg_feature_list, new_id.unsqueeze(1), type_emb, sg_node_type, l2norm, edge_weight = mask)

                id_loss = self._loss(masked_pred, output[n], mask, self.reg_coefs, flag)
                
                loss += id_loss
                epoch_loss += loss
            
            epoch_loss = torch.tensor(epoch_loss, requires_grad=True)
            epoch_loss = vloss + epoch_loss
            epoch_loss.backward()
            optimizer.step()
            
            self.explainer_model.eval()
            self.VAE.eval()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                val_loss = torch.FloatTensor([0]).detach().to(self.device)
                val_pred, _ = self.model_to_explain(self.g, self.feature_list, val_seq, type_emb, node_type, l2norm)
                _, sampled_z, vaeloss = self.VAE(self.g, self.feature_list, adjM)
                val_loss += vaeloss
                for n in range(len(val_idx)):
                    sg, new_id = dgl.khop_in_subgraph(self.g, train_idx[n], k=self.khop)
                    sg = dgl.add_self_loop(sg)

                    sg_sampled_z, _, sg_node_type = get_z_subgraph(sg, sampled_z, self.node_cnt, train_seq, node_type)
                    sg_feature_list, _, sg_node_type = get_subgraph(sg, self.feature_list, self.node_cnt, train_seq, node_type)
                    t = temp_schedule(e)
                    e_feat = get_efeat(etype,sg,self.device)
                    sampling_weights = self.explainer_model(sg, sg_sampled_z, e_feat)
                    mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                    
                    masked_pred, _ = self.model_to_explain(sg, sg_feature_list, new_id.unsqueeze(1), type_emb, sg_node_type, l2norm, edge_weight = mask)
                    
                    id_loss = self._loss(masked_pred, val_pred[n], mask, self.reg_coefs, flag)
                    
                    val_loss += id_loss
                    torch.cuda.empty_cache()
                    
                early_stopping(val_loss.item(), self.explainer_model)
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break
        

    def explain(self, idx, seq, type_emb, node_type, l2norm, etype, adjM, num_etypes):
        
        mseloss = nn.MSELoss()
        self.explainer_model.eval()
        self.VAE.eval()
        
        output, embeds = self.model_to_explain(self.g, self.feature_list, seq, type_emb, node_type, l2norm)
        
        mae = []
        rmse = []
        sgpred = []
        sgori = []
        mkpred = []
        
        _, sampled_z, _ = self.VAE(self.g, self.feature_list, adjM)
        
        for n in range(len(idx)):
            sg, new_id = dgl.khop_in_subgraph(self.g, idx[n], k=self.khop)
            
            sg_sampled_z, _, sg_node_type = get_z_subgraph(sg, sampled_z, self.node_cnt, seq, node_type)
            sg_feature_list, _, sg_node_type = get_subgraph(sg, self.feature_list, self.node_cnt, seq, node_type)
            
            e_feat = get_efeat(etype,sg,self.device)
            sampling_weights = self.explainer_model(sg, sg_sampled_z, e_feat)
            t = 0
            mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
            
            masked_pred, sg_ori_emb = self.model_to_explain(sg, sg_feature_list, new_id.unsqueeze(1), type_emb, sg_node_type, l2norm, edge_weight=mask)
            sgori.append(output[n].view(-1))
            sg_pred = torch.argmax(output[n], axis=0)
            mk_pred = torch.argmax(masked_pred, axis=1)
            sgpred.append(sg_pred.item())
            mkpred.append(mk_pred.item())

        sgpred = torch.tensor(sgpred).float()
        mkpred = torch.tensor(mkpred).float()
        mae = F.l1_loss(sgpred, mkpred)
        mse = mseloss(sgpred, mkpred)
        rmse = np.sqrt(mse)
        
        print("mae:",mae,"mse:",mse,"rmse:",rmse)
        
        return mae,mse,rmse
        
