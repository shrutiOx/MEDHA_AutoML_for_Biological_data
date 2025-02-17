# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 02:16:06 2023

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 00:11:29 2023

@author: ADMIN
"""

import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam
import numpy as np

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
import logging
from MEDHA.AutoToolGraph.Selector   import Selector

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

_logger = logging.getLogger(__name__)


criterion = torch.nn.CrossEntropyLoss()

def cross_validation_with_val_set(dataset,
                                  folds,
                                  epochs,
                                  intepochs,
                                  space,
                                  input_channel,
                                  outchannel,
                                  max_nodes,
                                  percent_dec,
                                  batch_size,
                                  OptimizerDart,
                                  learning_rateDart,
                                  acc_thresold,
                                  R,
                                  weight_decay=0,
                                  logger=_logger):
    
    

    val_losses_ff, accsval_ff ,trainlossmean_ff = [], [], [] 
    for fold, (train_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        
        print('fold ',fold)
        print('len(train_dataset) CV ',len(train_dataset))
        print('len(val_dataset) CV ',len(val_dataset))
        
    

        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)


        myHpoObject = Selector(train_dataset,
                         intepochs,
                         learning_rateDart,
                         space,
                         input_channel,
                         outchannel,
                         max_nodes,
                         percent_dec,
                         OptimizerDart,
                         acc_thresold)
        model,createlist,space= myHpoObject.Calling_HPO_DART(max_evals=6,stoppage=3)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rateDart, weight_decay=weight_decay)



        val_losses_fold, accsval_fold ,trainlossmean_fold = [], [], [] 
        for epoch in range(1, epochs + 1):
            val_losses, accsval ,trainlosssum = [], [], [] ,0
            for i in range(0,R):
                '''R level metrics'''
                train_loss = train(model, optimizer, train_loader)
                val_losses.append(eval_loss(model, val_loader))
                accsval.append(eval_acc(model, val_loader))
                trainlosssum += train_loss

            '''mean of R for that epoch'''    
            accval_mean = np.mean(accsval)
            valoss_mean = np.mean(val_losses)
            train_loss_mean = trainlosssum/R
            
            eval_info = {
                ' (mean of R)'
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss_mean,
                'val_loss': valoss_mean[-1],
                'val_acc': accval_mean[-1]
            }
            logger.warning(eval_info)
            
            '''storing all epoch metrics'''
            accsval_fold.append(accval_mean)
            val_losses_fold.append(valoss_mean)
            trainlossmean_fold.append(train_loss_mean)
            
        '''mean of all epoch  per fold'''
            
        accval_fold_mean = np.mean(accsval_fold)
        valoss_fold_mean = np.mean(val_losses_fold)
        train_loss_fold_mean = np.mean(trainlossmean_fold) 
        
        '''storing  mean of all epochs from each fold'''
        val_losses_ff.append(valoss_fold_mean)
        accsval_ff.append(accval_fold_mean)
        trainlossmean_ff.append(train_loss_fold_mean)

            
    
    '''mean of all folds - final metrics'''
    loss,accsval,trainlossmean = tensor(val_losses_ff),  tensor(accsval_ff),tensor(trainlossmean_ff)

    loss_mean = loss.mean().item()

    accsval  = accsval.mean().item()
    acc_std = accsval.std().item()
    trainlossmeanf = trainlossmean.mean().item()

    print(f'Val Loss: {loss_mean:.4f} ',
          f'± {acc_std:.3f}' ,f' Validation_accuracy {accsval:.3f}', f' Training_loss_mean {trainlossmeanf:.3f}'  )

    return loss_mean,accsval, acc_std, model,trainlossmeanf,createlist,space




def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    #val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        #train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1)) 

    return train_indices, test_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.adj, data.mask)
        #out = torch.tensor(out,requires_grad=True)
        #loss = F.nll_loss(out, data.y.view(-1))
        #loss = F.nll_loss(out, data.y.view(-1))
        loss = criterion(out, data.y.view(-1))
        loss.backward()

        total_loss += loss.item() * num_graphs(data)

        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.adj, data.mask).max(1)[1]
        pred=pred.cpu()
        data.y = data.y.cpu()
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.adj, data.mask)
            #print('out ',out)
            #print('y ',data.y.view(-1))
        #loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        loss += criterion(out, data.y.view(-1)).item()

        #loss += F.nll_loss(out, data.y.view(-1)).item()* num_graphs(data)
        #loss += F.nll_loss(out, data.y.view(-1)).item()

    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)