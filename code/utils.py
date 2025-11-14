import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split
# 读数据函数

def data_prepare_train(path, feature_matrix = "", 
                       sampleset = ""):  
    
    gene_features_tr = np.genfromtxt("{}{}.csv".format(path, feature_matrix), delimiter=',', skip_header=0, dtype=np.dtype(str))
    
    if gene_features_tr.shape[0] > 429:
       gene_features_tr = gene_features_tr[-429:, :]
    elif gene_features_tr.shape[0] < 429:
       sys.exit("Incomplete feature dimensions!") 
       
    sample_set_tr = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=1, dtype=np.dtype(int))
    gene_features_tr[gene_features_tr == 'NA'] = 0
    gene_features_tr = torch.tensor(gene_features_tr.astype(float), dtype=torch.float32)
    train_set, valid_set = train_test_split(np.arange(sample_set_tr.shape[0]), test_size=0.2, random_state=np.random.randint(0,1000)) 
    train_indices = train_set[:].astype(int) 
    valid_indices = valid_set[:].astype(int)  
    num_va = valid_indices.shape[0]
     
    feature_train, y_train = gene_features_tr[:, train_indices], sample_set_tr[train_indices, -1]
    feature_valid, y_valid = gene_features_tr[:, valid_indices], sample_set_tr[valid_indices, -1]

       

    if not isinstance(feature_train, torch.Tensor):
        feature_train = torch.tensor(feature_train, dtype=torch.float32, requires_grad=True)
    else:
        feature_train = feature_train.clone().detach().requires_grad_(True)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)
    else:
        y_train = y_train.clone().detach().requires_grad_(True)
    
    if not isinstance(feature_valid, torch.Tensor):
        feature_valid = torch.tensor(feature_valid, dtype=torch.float32, requires_grad=True)
    else:
        feature_valid = feature_valid.clone().detach().requires_grad_(True)

    if not isinstance(y_valid, torch.Tensor):
        y_valid = torch.tensor(y_valid, dtype=torch.float32, requires_grad=True)
    else:
        y_valid = y_valid.clone().detach().requires_grad_(True)

    
    return feature_train.mT, y_train, feature_valid.mT, y_valid


def data_prepare_external(path, feature_matrix = "", sampleset = "", internal = False):
    
    gene_features_ex = np.genfromtxt("{}{}.csv".format(path, feature_matrix), delimiter=',', skip_header=0, dtype=np.dtype(str),missing_values='NA', filling_values=0)
    
    if gene_features_ex.shape[0] > 429:
       gene_features_ex = gene_features_ex[-429:, :]
    elif gene_features_ex.shape[0] < 429:
       sys.exit("Incomplete feature dimensions!") 
    
    if internal == True:
       sample_set_ex = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=0, dtype=float,missing_values='NA', filling_values=0,ndmin=2)
    else:
       sample_set_ex = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=1, dtype=float,missing_values='NA', filling_values=0,ndmin=2)

    gene_features_ex[gene_features_ex == 'NA'] = 0
    feature_ex = torch.tensor(gene_features_ex.astype(float), dtype=torch.float32)
    
    y_ex = torch.tensor(sample_set_ex[:, -1])
    num_ex = gene_features_ex.shape[1]
    return feature_ex.T, y_ex, num_ex
    

def data_prepare_train_surv(path, feature_matrix = "", 
                       sampleset = ""):
    
    gene_features_tr = np.genfromtxt("{}{}.csv".format(path, feature_matrix), delimiter=',', skip_header=1, dtype=np.dtype(str))
    sample_set_tr = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=1, dtype=np.dtype(int))
    gene_features_tr[gene_features_tr == 'NA'] = 0
    gene_features_tr = torch.tensor(gene_features_tr.astype(float), dtype=torch.float32)
    train_set, valid_set = train_test_split(np.arange(sample_set_tr.shape[0]), test_size=0.2, random_state=np.random.randint(0,1000)) 
    train_indices = train_set[:].astype(int) 
    valid_indices = valid_set[:].astype(int)  
    num_va = valid_indices.shape[0]
    feature_train, y_train = gene_features_tr[:, train_indices], sample_set_tr[train_indices, -2:]
    feature_valid, y_valid = gene_features_tr[:, valid_indices], sample_set_tr[valid_indices, -2:]
    
    #feature_valid = np.concatenate((gene_features_tr[:, train_indices], gene_features_tr[:, valid_indices]), axis=1)
    #y_valid = np.concatenate((sample_set_tr[train_indices, 1:3], sample_set_tr[valid_indices, 1:3]), axis=0)
    feature_train, y_train = torch.tensor(feature_train), torch.tensor(y_train)
    feature_valid, y_valid = torch.tensor(feature_valid), torch.tensor(y_valid)
    return feature_train.mT, y_train, feature_valid.mT, y_valid
   

def data_prepare_external_surv(path, feature_matrix = "", sampleset = "",internal=False):
    
    gene_features_ex = np.genfromtxt("{}{}.csv".format(path, feature_matrix), delimiter=',', skip_header=0, dtype=np.dtype(str),missing_values='NA', filling_values=0)
    
    if gene_features_ex.shape[0] > 429:
       gene_features_ex = gene_features_ex[-429:, :]
    elif gene_features_ex.shape[0] < 429:
       sys.exit("Incomplete feature dimensions!") 
    
    if internal == True:
       sample_set_ex = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=0, dtype=float,missing_values='NA', filling_values=0,ndmin=2)
    else:
       sample_set_ex = np.genfromtxt("{}{}.csv".format(path, sampleset), delimiter=',', skip_header=1, dtype=float,missing_values='NA', filling_values=0,ndmin=2)

    
    gene_features_ex[gene_features_ex == 'NA'] = 0
    feature_ex = torch.tensor(gene_features_ex.astype(float), dtype=torch.float32)
    y_ex = torch.tensor(sample_set_ex[:, -2:])
    num_ex = gene_features_ex.shape[1]
    return feature_ex.T, y_ex, num_ex
    


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, pred, ytime, yevent):
        n_observed = yevent.sum(0)
        ytime_indicator = self.R_set(ytime)
        if torch.cuda.is_available():
            ytime_indicator = ytime_indicator.to(pred.device)
        
        risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
        diff = pred - torch.log(risk_set_sum)

        if diff.dim() == 1:
            diff = diff.unsqueeze(1) 

        if yevent.dim() == 1:
            yevent = yevent.unsqueeze(1)
        yevent = yevent.float() 
        
        sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
        cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
        
        return cost
    
    def R_set(self,x):
        n_sample = x.size(0)
        matrix_ones = torch.ones(n_sample, n_sample)
        indicator_matrix = torch.tril(matrix_ones)
        return(indicator_matrix)

def c_index(true_T, true_E, pred_risk, include_ties=True):

    order = np.argsort(-true_T.detach().cpu().numpy())
    true_T = true_T.detach().cpu().numpy()[order]
    true_E = true_E.detach().cpu().numpy()[order]
    pred_risk = pred_risk.detach().cpu().numpy()[order]
    result = concordance_index(true_T, -pred_risk, true_E)

    return result
    

def logrank_pvalue(true_T, true_E, pred_risk):

    true_T = np.asarray(true_T.detach().cpu().numpy()).flatten()
    true_E = np.asarray(true_E.detach().cpu().numpy()).flatten()
    pred_risk = np.asarray(pred_risk.detach().cpu().numpy()).flatten()

    median_risk = np.median(pred_risk)

    high_risk_group = (pred_risk > median_risk)
    low_risk_group = (pred_risk <= median_risk)

    if high_risk_group.sum() == 0 or low_risk_group.sum() == 0:
        print("Warning: One of the risk groups is empty!")
        return np.nan  
    
    result = logrank_test(
        true_T[high_risk_group], true_T[low_risk_group], 
        event_observed_A=true_E[high_risk_group], 
        event_observed_B=true_E[low_risk_group]
    )
    log10_pvalue = -np.log10(result.p_value)

    return log10_pvalue


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7  
        loss_y1 = -1 * self.alpha * \
            torch.pow((1 - preds), self.gamma) * \
            torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds,
                                                    self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)
    
