import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903. This part of code refers to the implementation of https://github.com/Diego999/pyGAT.git

    """
    def __init__(self, device,in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.alpha = alpha  
        self.concat = concat 
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).to(device))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)).to(device))  

        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        self.leakyrelu = nn.LeakyReLU(self.alpha)  
        
        
        
    def forward(self, h, adj, k):

        Wh = torch.mm(h, self.W)

        adj = self._prepare_adj(adj, k)

        a_input = self._prepare_attentional_mechanism_input(Wh, adj)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze())  

        node_pairs = torch.nonzero(adj, as_tuple=True)
        
        adj_reconstructed = -9e15*torch.ones_like(adj)
        adj_reconstructed[node_pairs[0], node_pairs[1]] = e
        attention = F.softmax(adj_reconstructed, dim=1)

        h_prime = torch.matmul(attention, Wh)

        return self.leakyrelu(h_prime), adj_reconstructed
    
    def _prepare_adj(self, adj, k):
        
        sorted_values = torch.sort(adj.reshape(-1,), descending=True).values
        index = min(k * adj.shape[0] - 1, sorted_values.shape[0] - 1)
        parameter = sorted_values[index]
        adj = (adj >= parameter.data.cpu().numpy().item()).float()
        return adj
        

    def _prepare_attentional_mechanism_input(self, Wh, adj):
        N = Wh.size()[0] 

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0) 
        Wh_repeated_alternating = Wh.repeat(N, 1) 

        adj_mask = adj > 0 

        adj_mask = adj_mask.view(-1)

        Wh_repeated_in_chunks_filtered = Wh_repeated_in_chunks[adj_mask] 
        Wh_repeated_alternating_filtered = Wh_repeated_alternating[adj_mask] 

        all_combinations_matrix_filtered = torch.cat([Wh_repeated_in_chunks_filtered, Wh_repeated_alternating_filtered], dim=1)

        return all_combinations_matrix_filtered.view(-1, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class selfattention(nn.Module):
    def __init__(self, device,sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(sample_size, d_k,bias=False).to(device)
        self.key = nn.Linear(sample_size, d_k,bias=False).to(device)
        self.value = nn.Linear(sample_size, d_v,bias=False).to(device)
    
    def forward(self, x):
        x = x.mT
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)     
        att = torch.matmul(q, k.transpose(0,1)) / np.sqrt(self.d_k)
        att1 = torch.softmax(att, dim=1)
        output = torch.matmul(att1, v)
        return output.T,att
    
