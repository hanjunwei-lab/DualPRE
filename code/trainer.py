import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score

from scipy.stats import pearsonr
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.utils.data as Dataset
from utils import *


# train function
def train(epoch, model, device, optimizer, loss_func, f_tra, y_tra, f_val, y_val, batch, task, ntype, cv, best_value,save,drop_last):
    t = time.time()
    
    tra_dataset = Dataset.TensorDataset(f_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=batch, shuffle=True,drop_last=drop_last)
    model.train()
    
    for batch_idx, (feature_tra, y_train) in enumerate(train_dataset):
        
        y_train = y_train.clone().detach().to(device).long()
        y_tpred, y_tpred_prob, y_tpred_classes,pathway_intput,pathway_reconstruction,cell_input,cell_reconstruction,output_dict = model(feature_tra.to(device), device)
        
        loss_train = loss_func(y_tpred, y_train)+nn.MSELoss()(pathway_intput,pathway_reconstruction)+nn.MSELoss()(cell_input,cell_reconstruction)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
    print(f"Epoch {epoch + 1} completed in {time.time() - t:.2f} seconds.")

    
    val_dataset = Dataset.TensorDataset(f_val, y_val)
    val_loader = Dataset.DataLoader(val_dataset, batch_size=batch, shuffle=False,drop_last=drop_last)

    model.eval()
    with torch.no_grad():
        
        pred_valid = []
        pred_valid_prob = []
        pred_valid_classes = []
        true_valid = []
        
        for batch_idx, (feature_val, y_valid) in enumerate(val_loader):
            y_valid = y_valid.to(device).long()
  
            y_vpred, y_vpred_prob, y_vpred_classes, pathway_input, pathway_reconstruction, cell_input, cell_reconstruction,output_dict = model(feature_val.to(device), device)

            pred_valid.extend(y_vpred.cpu().detach().numpy())
            pred_valid_prob.extend(y_vpred_prob.cpu().detach().numpy())
            pred_valid_classes.extend(y_vpred_classes.cpu().detach().numpy())
            true_valid.extend(y_valid.cpu().detach().numpy())
        
    true_valid = np.array(true_valid)

    pred_valid_tensor = torch.tensor(np.array(pred_valid), dtype=torch.float32).to(device)
    pred_valid_prob_tensor = torch.tensor(np.array(pred_valid_prob), dtype=torch.float32).to(device)
    true_valid_tensor = torch.tensor(true_valid.astype(int), dtype=torch.long).to(device)
    
    loss_valid = loss_func(pred_valid_tensor, true_valid_tensor) + \
        nn.MSELoss()(pathway_input, pathway_reconstruction) + \
        nn.MSELoss()(cell_input, cell_reconstruction)

    if len(np.unique(true_valid)) > 1:
        if ntype == 2: 
            AUROC_valid = roc_auc_score(true_valid, pred_valid_prob_tensor[:, 1].cpu().numpy())
            AUPRC_valid = average_precision_score(true_valid, pred_valid_prob_tensor[:, 1].cpu().numpy())
            accuracy_valid = accuracy_score(true_valid, pred_valid_classes)
            f1_valid = f1_score(true_valid, pred_valid_classes)

            print(f'Epoch: {epoch + 1:04d}, '
                  f'loss_valid: {loss_valid:.4f}, '
                  f'AUROC_valid: {AUROC_valid:.4f}, '
                  f'AUPRC_valid: {AUPRC_valid:.4f}, '
                  f'accuracy_valid: {accuracy_valid:.4f}, '
                  f'f1_valid: {f1_valid:.4f}')
          
        else: 
            accuracy_valid = accuracy_score(true_valid, pred_valid_classes)
            precision_valid = precision_score(true_valid, pred_valid_classes, average='weighted')
            recall_valid = recall_score(true_valid, pred_valid_classes, average='weighted')
            f1_valid = f1_score(true_valid, pred_valid_classes, average='weighted')

            print(f'Epoch: {epoch + 1:04d}, '
                  f'loss_valid: {loss_valid:.4f}, '
                  f'accuracy_valid: {accuracy_valid:.4f}, '
                  f'precision_valid: {precision_valid:.4f}, '
                  f'recall_valid: {recall_valid:.4f}, '
                  f'f1_valid: {f1_valid:.4f}')

    else:
        print("Skipping metrics calculation, as one or both sets contain only one class")

    os.makedirs("./model", exist_ok=True)

    if f1_valid >= best_value[0]:
        best_value[0] = f1_valid
        best_value[1] = epoch+1
        if save == True:
           torch.save(model,f"./model/best_model_{task}_cv{cv}_B{batch}.model")

        
    return best_value[1], loss_valid,best_value[0]

    

# test function
def compute_test(model, device, f_test, y_te, batch, ntype, drop_last):
    model.eval()
    loss_test, AUROC_test, accuracy_test, precision_test, f1_test = [], [], [], [], []
    pred_test, pred_test_prob, pred_test_classes, true_test = [], [], [], []
    
    dataset = Dataset.TensorDataset(f_test, y_te)
    test_dataset = Dataset.DataLoader(dataset, batch_size=batch, shuffle=False,drop_last=drop_last)
    
    for feature_test, y_test in test_dataset:
        
        y_test = y_test.to(device)
        y_pred, y_pred_prob, y_pred_classes,pathway_input,pathway_reconstruction,cell_input,cell_reconstruction,output_dict = model(feature_test.to(device), device)
        
        pred_test.extend(y_pred.cpu().detach().numpy())
        pred_test_prob.extend(y_pred_prob.cpu().detach().numpy())
        pred_test_classes.extend(y_pred_classes.cpu().detach().numpy())
        true_test.extend(y_test.cpu().detach().numpy())
    
    pred_test_tensor = torch.tensor(np.array(pred_test), dtype=torch.float32).to(device)
    pred_test_prob_tensor = torch.tensor(np.array(pred_test_prob), dtype=torch.float32).to(device)
    true_test_tensor = torch.tensor(true_test, dtype=torch.long).to(device) 

    if len(np.unique(true_test)) > 1:
        if ntype == 2:
            AUROC_test = roc_auc_score(true_test, pred_test_prob_tensor[:,1].cpu().detach().numpy())
            AUPRC_test = average_precision_score(true_test, pred_test_prob_tensor[:, 1].cpu().numpy())
            accuracy_test = accuracy_score(true_test, pred_test_classes)
            precision_test = precision_score(true_test, pred_test_classes)
            recall_test = recall_score(true_test, pred_test_classes)
            f1_test = f1_score(true_test, pred_test_classes)

            print("Test set results:",
                "AUROC_test= {:.4f}".format(AUROC_test),
                "AUPRC_test= {:.4f}".format(AUPRC_test),
                'accuracy_test= {:.4f}'.format(accuracy_test),
                "precision_test= {:.4f}".format(precision_test),
                "recall_test= {:.4f}".format(recall_test),
                "f1_test= {:.4f}\n".format(f1_test))
        else:
            accuracy_test = accuracy_score(true_test, pred_test_classes)
            precision_test = precision_score(true_test, pred_test_classes, average='weighted')
            recall_test = recall_score(true_test, pred_test_classes, average='weighted')
            f1_test = f1_score(true_test, pred_test_classes, average='weighted')
            
            print("Test set results:",
                #"loss_test = {:.4f}".format(loss_test),
                'accuracy_test= {:.4f}'.format(accuracy_test),
                #'AUPRC_test= {:.4f}'.format(AUPRC_test),
                "precision_test= {:.4f}".format(precision_test),
                "recall_test= {:.4f}".format(recall_test),
                "f1_test= {:.4f}\n".format(f1_test))

    else:
        print("Skipping metrics calculation, as one or both sets contain only one class")

    
    return pred_test_prob_tensor,output_dict



def train_surv(epoch, model, device,optimizer, loss_func, f_tra, y_tra, f_val, y_val, batch, task, cv, best_value, save,drop_last):
    t = time.time()
    tra_dataset = Dataset.TensorDataset(f_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=batch, shuffle=True, drop_last=drop_last)
    model.train()
    
    for batch_idx, (feature_tra, y_train) in enumerate(train_dataset):
        
        y_train = y_train.clone().detach().to(device).long()
        y_tpred, pathway_intput,pathway_reconstruction,cell_input,cell_reconstruction,output_dict = model(feature_tra.to(device), device)
        
        loss_train = loss_func(y_tpred, y_train[:,0],y_train[:,1])+nn.MSELoss()(pathway_intput,pathway_reconstruction)+nn.MSELoss()(cell_input,cell_reconstruction)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed in {time.time() - t:.2f} seconds.")

    
    val_dataset = Dataset.TensorDataset(f_val, y_val)
    val_loader = Dataset.DataLoader(val_dataset, batch_size=batch, shuffle=False,drop_last=drop_last)

    model.eval()
    with torch.no_grad():
        pred_valid = []
        true_valid = []
        cindex_valid = []
        
        for batch_idx, (feature_val, y_valid) in enumerate(val_loader):
            y_valid = y_valid.to(device).long()
            
            y_vpred, pathway_input, pathway_reconstruction, cell_input, cell_reconstruction, output_dict = model(feature_val.to(device), device)

            pred_valid.extend(y_vpred.cpu().detach().numpy())
            true_valid.extend(y_valid.cpu().detach().numpy())
        
    true_valid = np.array(true_valid)

    pred_valid_tensor = torch.tensor(np.array(pred_valid), dtype=torch.float32).to(device)
    true_valid_tensor = torch.tensor(true_valid.astype(int), dtype=torch.long).to(device)
    
    loss_valid = loss_func(pred_valid_tensor, true_valid_tensor[:,0], true_valid_tensor[:,1]) + \
        nn.MSELoss()(pathway_input, pathway_reconstruction) + \
        nn.MSELoss()(cell_input, cell_reconstruction)

    if torch.isnan(pred_valid_tensor).any():
       print("NaN detected in model output!")
       cindex_valid = torch.tensor(-1)
    else:   
       cindex_valid = c_index(true_valid_tensor[:,0], true_valid_tensor[:,1], pred_valid_tensor)
       logrank = logrank_pvalue(true_valid_tensor[:,0], true_valid_tensor[:,1], pred_valid_tensor)
    log_message = (
        f'Epoch: {epoch + 1:04d}, '
        f'loss_valid: {loss_valid.item():.4f}, '  
        f'cindex_valid: {cindex_valid.item():.4f}, ' 
        f'-log10pvalue: {logrank.item():.4f}, '
        f'time: {time.time() - t:.4f}s\n\n'
    )
    print(log_message)

    os.makedirs("./model", exist_ok=True)

    if cindex_valid >= best_value[0]:
        best_value[0] = cindex_valid
        best_value[1] = epoch+1
        if save == True:
           torch.save(model,f"./model/best_model_{task}_cv{cv}_B{batch}.model")
        
    return best_value[1], loss_valid,best_value[0]



def compute_test_surv(model,device,f_test, y_te,batch,drop_last):
    model.eval()
    loss_test, AUROC_test, accuracy_test, precision_test, f1_test = [], [], [], [], []
    pred_test, pred_test_prob, pred_test_classes, true_test = [], [], [], []

    
    dataset = Dataset.TensorDataset(f_test, y_te)
    test_dataset = Dataset.DataLoader(dataset, batch_size=batch, shuffle=False,drop_last=drop_last)
    
    for feature_test, y_test in test_dataset:
        
        y_test = y_test.to(device)
        y_pred, pathway_input,pathway_reconstruction,cell_input,cell_reconstruction,output_dict = model(feature_test.to(device), device)
        
        pred_test.extend(y_pred.cpu().detach().numpy())
        true_test.extend(y_test.cpu().detach().numpy())
    
    pred_test_tensor = torch.tensor(np.array(pred_test), dtype=torch.float32).to(device)
    true_test_tensor = torch.tensor(true_test, dtype=torch.long).to(device) 
    
    cindex_test = c_index(true_test_tensor[:,0], true_test_tensor[:,1], pred_test_tensor)
    logrank = logrank_pvalue(true_test_tensor[:,0], true_test_tensor[:,1], pred_test_tensor)

    
    log_message = (
        
        f'cindex_test: {cindex_test:.4f}, '
        f'-log10pvalue: {logrank:.4f}, '
    )
    print(log_message)
    
    return pred_test_tensor,output_dict