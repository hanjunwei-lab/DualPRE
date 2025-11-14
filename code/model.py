# 模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, zscore
from layers import *


class DualPRE(nn.Module): 
    def __init__(self, device,Autodecoder_p,Autodecoder_c,npath, ncell, hidden , nOutputGAT_p, nOutputGAT_c, pgraph_k, cgraph_k, d_k, nheads, ntype,alpha,activation_function,dropout_rate): 
        """Dense version of GAT."""
        super(DualPRE, self).__init__() 
        
        nInputGAT_p = npath
        nInputGAT_c = ncell
        self.activation_function = activation_function
        self.ntype = ntype
        self.pgraph_k = pgraph_k
        self.cgraph_k = cgraph_k
        self.Autodecoder_p = Autodecoder_p
        self.Autodecoder_c = Autodecoder_c
   
        self.psample_LNlayer = nn.LayerNorm(hidden[0])

        self.psample_attentions1 = nn.ModuleList([GraphAttentionLayer(device,hidden[0], nOutputGAT_p, alpha=alpha, concat=True) for _ in range(nheads)])

        for i, attention in enumerate(self.psample_attentions1):  
            self.add_module('attentionps1_{}'.format(i), attention)

        self.psample_MultiHead1 = nn.ModuleList([selfattention(device,hidden[0], d_k, 1) for _ in range(nheads)])
        for i, attention in enumerate(self.psample_MultiHead1): 
            self.add_module('selfattentionps1_{}'.format(i), attention)

        self.psample_prolayer1 = nn.Linear(nOutputGAT_p*nheads, nOutputGAT_p, bias=False)
        self.psample_LNlayer1 = nn.LayerNorm(nOutputGAT_p)
        
        self.psample_prolayer_s = nn.Linear(1, hidden[0], bias=False) 

        self.psample_attentions2 = [GraphAttentionLayer(device,nOutputGAT_p*nheads, nOutputGAT_p, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.psample_attentions2):  
            self.add_module('attentionps1_{}'.format(i), attention)

        self.psample_MultiHead2 = [selfattention(device,hidden[0], d_k, 1) for _ in range(nheads)]
        for i, attention in enumerate(self.psample_MultiHead2):
            self.add_module('selfattentionps1_{}'.format(i), attention)

        self.psample_prolayer2 = nn.Linear(nOutputGAT_p*nheads, nOutputGAT_p, bias=False)
        self.psample_LNlayer2 = nn.LayerNorm(nOutputGAT_p) 
        self.csample_LNlayer = nn.LayerNorm(hidden[1])
        self.csample_attentions1 = nn.ModuleList([GraphAttentionLayer(device,hidden[1], nOutputGAT_c, alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.csample_attentions1): 
            self.add_module('attentioncs1_{}'.format(i), attention)
        

        self.csample_MultiHead1 = nn.ModuleList([selfattention(device,hidden[1], d_k, 1) for _ in range(nheads)])
        for i, attention in enumerate(self.csample_MultiHead1):  
            self.add_module('selfattentioncs1_{}'.format(i), attention)

        self.csample_prolayer1 = nn.Linear(nOutputGAT_c*nheads, nOutputGAT_c, bias=False)
        self.csample_LNlayer1 = nn.LayerNorm(nOutputGAT_c)
        self.csample_prolayer_s = nn.Linear(1, hidden[1], bias=False)
    
        self.csample_attentions2 = [GraphAttentionLayer(device,nOutputGAT_c*nheads, nOutputGAT_c, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.csample_attentions2): 
            self.add_module('attentioncs2_{}'.format(i), attention)
        

        self.csample_MultiHead2 = [selfattention(device,hidden[1], d_k, 1) for _ in range(nheads)]
        for i, attention in enumerate(self.csample_MultiHead2):
            self.add_module('selfattentioncs2_{}'.format(i), attention)
        

        self.csample_prolayer2 = nn.Linear(nOutputGAT_c*nheads, nOutputGAT_c, bias=False)
        self.csample_LNlayer2 = nn.LayerNorm(nOutputGAT_c)

        self.FClayer1 = nn.Linear(nOutputGAT_p + nOutputGAT_c, nOutputGAT_p + nOutputGAT_c)
        self.FClayer2 = nn.Linear(nOutputGAT_p + nOutputGAT_c, nOutputGAT_p + nOutputGAT_c) 
        self.FClayer3 = nn.Linear(nOutputGAT_p + nOutputGAT_c, self.ntype)
        self.dropout = nn.Dropout(p=dropout_rate)
      


    def forward(self,features, device):
        
        new_path_featmatrix, p_avg_GAT_matrix,pathway_reconstruction,correlation_matrix1_tensor,pathfe = self.AGAT(features[:,:343], device,channel="pathway")
        new_path_featmatrix, p_avg_SAT_matrix = self.TRAN(new_path_featmatrix, self.psample_prolayer_s, self.psample_MultiHead1,self.psample_LNlayer1)
        
        new_cell_featmatrix, c_avg_GAT_matrix,cell_reconstruction,correlation_matrix3_tensor,cellfe = self.AGAT(features[:,343:429], device,channel="cell")
        new_cell_featmatrix, c_avg_SAT_matrix = self.TRAN(new_cell_featmatrix, self.csample_prolayer_s, self.csample_MultiHead1,self.csample_LNlayer1)

        path_cell_x,path_cell_x1 = self.predictor(new_path_featmatrix,new_cell_featmatrix,device)

        
        output_dict = {
            "pathway_PSN": correlation_matrix1_tensor,
            "cell_PSN": correlation_matrix3_tensor,
            "pathway_GAT_weight": p_avg_GAT_matrix,
            "cell_GAT_weight": c_avg_GAT_matrix,
            "pathway_SAT_weight": p_avg_SAT_matrix,
            "cell_SAT_weight": c_avg_SAT_matrix,
            "new_features": path_cell_x1
        }

        
        if self.ntype != 1:
            path_cell_x_prob = torch.softmax(path_cell_x, dim=1)
            path_cell_x = path_cell_x.squeeze(-1) 
            path_cell_x_prob = path_cell_x_prob.squeeze(-1) 
            predicted_classes = torch.argmax(path_cell_x_prob, dim=1)
            return path_cell_x, path_cell_x_prob, predicted_classes,pathfe,pathway_reconstruction,cellfe,cell_reconstruction,output_dict
        else:
            return path_cell_x,pathfe,pathway_reconstruction,cellfe,cell_reconstruction,output_dict
        
        
        
    def AGAT(self, features, device, channel="pathway"):

        fe = features.detach().cpu().numpy()
        fe = torch.tensor(
            zscore(fe.astype(float), axis=0, ddof=1),
            dtype=torch.float32,
            device=device
        )

        if channel == "pathway":
            autodecoder = self.Autodecoder_p
            sample_attentions = self.psample_attentions1
            prolayer = self.psample_prolayer1
            graph_k = self.pgraph_k
        elif channel == "cell":
            autodecoder = self.Autodecoder_c
            sample_attentions = self.csample_attentions1
            prolayer = self.csample_prolayer1
            graph_k = self.cgraph_k
        else:
            raise ValueError(f"Unknown mode: {channel}")

        output, reconstruction = autodecoder(fe)

        correlation_matrix1_tensor = self.cosine_similarity_torch(output, output)

        new_featmatrix, avg_GAT_matrix = self.process_GAT_heads(
            sample_attentions,
            output,
            correlation_matrix1_tensor,
            graph_k
        )

        new_featmatrix = prolayer(new_featmatrix)

        return new_featmatrix, avg_GAT_matrix, reconstruction, correlation_matrix1_tensor, fe
        
        
    def TRAN(self,featmatrix, sample_prolayer_s, sample_MultiHead,sample_LNlayer):
        featmatrix1 = featmatrix 
        selflayeroutput = torch.zeros_like(featmatrix.mT)

        all_attention_weights = []
        for i, input_row in enumerate(featmatrix):

            input_row = input_row.unsqueeze(0) 
            input_row1 = sample_prolayer_s(input_row.mT)
            input_row1 = input_row1.mT

            temp_output = torch.zeros_like(input_row)

            sample_attention_weights = []
            for selfatt in sample_MultiHead:

                feat, attn_weight = selfatt(input_row1)
                temp_output += feat
                sample_attention_weights.append(attn_weight)

            temp_output /= len(sample_MultiHead)

            selflayeroutput[:, i] = temp_output.squeeze()

            avg_sample_attention = torch.mean(torch.stack(sample_attention_weights), dim=0)
            all_attention_weights.append(avg_sample_attention) 
            
        avg_attention_matrix = torch.mean(torch.stack(all_attention_weights), dim=0)
        selflayeroutput = selflayeroutput.mT
        
        selflayeroutput = selflayeroutput + featmatrix1
        selflayeroutput = sample_LNlayer(selflayeroutput)
        
        return selflayeroutput, avg_attention_matrix
    
    
        
    def cosine_similarity_torch(self,x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    

    def process_GAT_heads(self,attmodel,fea, graph,k):
        
        features = []
        attention_weights = []

        for att in attmodel:
            feat, attn_weight = att(fea, torch.abs(graph),k)
            features.append(feat)
            attention_weights.append(attn_weight)

        new_featmatrix = torch.cat(features, dim=1)

        avg_attention_matrix = torch.mean(torch.stack(attention_weights), dim=0)
        return new_featmatrix, avg_attention_matrix
        
    
    def predictor(self, path_feat, cell_feat, device):

        x1 = torch.cat((path_feat, cell_feat), dim=1)
        x1 = x1.to(device)

        x = self.FClayer1(x1)
        x = self.activation_function(x)
        x = self.dropout(x)

        x = self.FClayer2(x)
        x = self.activation_function(x)
        x = self.dropout(x)

        x = self.FClayer3(x)

        return x,x1
        
    
class Autodecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(Autodecoder, self).__init__()
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))

        
            
        for input_size, output_size in zip(hidden_dims, hidden_dims[1:]):
            
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
                
        modules.append(nn.Linear(hidden_dims[-1], code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        
        modules = []
        
        modules.append(nn.Linear(code_dim, hidden_dims[-1]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        
        for input_size, output_size in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dims[0], input_dim))
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x


