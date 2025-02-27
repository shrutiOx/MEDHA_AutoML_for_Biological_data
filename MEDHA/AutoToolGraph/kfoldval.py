# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:12:45 2023

@author: ADMIN
"""

import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam
import logging

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

_logger = logging.getLogger(__name__)


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr=0.0001, 
                                  weight_decay=0,logger=_logger):

    val_acc, accs= [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_acc.append(eval_acc(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_acc': val_acc[-1],
                'test_acc': accs[-1],

            }
            logger.warning(eval_info)

            

    #loss, acc = tensor(val_losses), tensor(accs)
    #loss = loss.view(folds, epochs)
    #loss, argmin = loss.min(dim=1)
    

    #loss_mean = loss.mean().item()
    #acc_mean = acc.mean().item()

    #acc_std = acc.std().item()

    loss, acc,accsval = tensor(train_loss), tensor(accs), tensor(val_acc)
    loss = loss.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    valacc_mean = accsval.mean().item()



    return loss_mean, acc_mean, acc_std,valacc_mean,model

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
        loss = F.nll_loss(out, data.y.view(-1))
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
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.adj, data.mask)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)