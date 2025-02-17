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

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
import logging

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

_logger = logging.getLogger(__name__)


criterion = torch.nn.CrossEntropyLoss()

def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr=0.0001, 
                                  weight_decay=0,logger=_logger):

    val_losses, accs, accsval ,trainlossmean = [], [], [] ,[]
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)


        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)




        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            accsval.append(eval_acc(model, val_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
                'val_acc': accsval[-1]
            }
            logger.warning(eval_info)
            #if logger is not None:
            





    #loss, acc,accsval = tensor(val_losses), tensor(accs), tensor(accsval)
    #loss, acc,accsval = loss.view(folds, epochs), acc.view(folds, epochs),accsval.view(folds, epochs)
    #loss, argmin = loss.min(dim=1)
    #acc     = acc[torch.arange(folds, dtype=torch.long), argmin]
    #accsval = accsval[torch.arange(folds, dtype=torch.long), argmin]
    
    
    loss, acc,accsval,trainlossmean = tensor(val_losses), tensor(accs), tensor(accsval),tensor(train_loss)
    loss = loss.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    accsval  = accsval.mean().item()
    acc_std = acc.std().item()
    trainlossmeanf = trainlossmean.mean().item()

    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}' ,f' Validation_accuracy {accsval:.3f}', f' Training_loss_mean {trainlossmeanf:.3f}'  )

    return loss_mean, acc_mean, acc_std,accsval, model,trainlossmeanf


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


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