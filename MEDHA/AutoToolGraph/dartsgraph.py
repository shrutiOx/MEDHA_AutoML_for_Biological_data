
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:16:36 2023

@author: ADMIN
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# type: ignore

import copy
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.retiarii.oneshot import interface
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device
from torch_geometric.loader import DataLoader


_logger = logging.getLogger(__name__)


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]


class DartsTrainerGraph(interface.BaseOneShotTrainer):
    """
    DARTS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    learning_rate : float
        Learning rate to optimize the model.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """

    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, train_loader,test_loader, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        warnings.warn('DartsTrainer is deprecated. Please use strategy.DARTS instead.', DeprecationWarning)

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.model_optim = optimizer
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = 5.





    def _train_one_epoch(self, epoch):
        #print('got here')
        self.model.train()
        meters = AverageMeterGroup()
        #print('and here')
        #print(self.train_loader)
        #print(self.test_loader)
        
        for step,(( trn_data), (val_data)) in enumerate(zip(self.train_loader, self.test_loader)):
            trn_X = trn_data.x
            trn_y = trn_data.y 
            val_X = val_data.x
            val_y = val_data.y 
            trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
            val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

            train_adj=to_device(trn_data.adj,self.device)
            train_mask=to_device(trn_data.mask,self.device)


            val_adj=to_device(val_data.adj,self.device)
            val_mask=to_device(val_data.mask,self.device)
            

            #print('done X and y')

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                #print('going to unrolled')
                self._unrolled_backward(trn_X, trn_y, val_X, val_y,train_adj,train_mask,val_adj,val_mask)
            else:
                #print('going to backward')
                self._backward(val_X, val_y,val_adj,val_mask)
            self.ctrl_optim.step()
            
            #print('done phase 1')

            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y,train_adj,train_mask)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            #metrics = self.metrics(logits, trn_y)
            
            #print('metrics ',metrics)
            #logitsval, lossval = self._logits_and_lossval(val_X, val_y,val_adj,val_mask)
            logitsval, lossval = self._logits_and_loss(trn_X, trn_y,train_adj,train_mask)
            #metrics = self.metrics(logitsval, val_y)
            metrics = self.metrics(logitsval, trn_y)
            metrics['loss'] = lossval.item()
            #metrics['loss'] = loss.item()
            
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.warning('Epoch [%s/%s] Step [%s/%s]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.test_loader), meters)
                print('GPU memory consumption ',self._get_gpu_memory_summary())
                
            #print('done phase 2')
    def _get_gpu_memory_summary(self):
        """Returns a summary of GPU memory usage in MB."""
        allocated = round(torch.cuda.memory_allocated() / 1024**2, 1)
        reserved = round(torch.cuda.memory_reserved() / 1024**2, 1)
        return f"GPU Memory: Allocated: {allocated} MB, Reserved: {reserved} MB"
    
                
            #print('done phase 2')

    def _logits_and_loss(self, X, y,train_adj,train_mask):
        #print('inside logists and loss')
        logits = self.model(X,train_adj,train_mask)
        loss = self.loss(logits, y.view(-1))
        return logits, loss
    #my addition
    def _logits_and_lossval(self, X, y,val_adj,val_mask):
        #print('inside logists and loss')
        logitsval = self.model(X,val_adj,val_mask)
        lossval = self.loss(logitsval, y.view(-1))
        return logitsval,lossval

    def _backward(self, val_X, val_y,val_adj,val_mask):
        """
        Simple backward with gradient descent
        """
        #print('going to logits and loss from backward')
        _, loss = self._logits_and_loss(val_X, val_y,val_adj,val_mask)
        loss.backward()

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y,train_adj,train_mask,val_adj,val_mask):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.model_optim.param_groups[0]["lr"]
        momentum = self.model_optim.param_groups[0]["momentum"]
        weight_decay = self.model_optim.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay,train_adj,train_mask)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        #print('going to loss in unrolled')
        _, loss = self._logits_and_loss(val_X, val_y,val_adj,val_mask)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple([c.alpha for _, c in self.nas_modules])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        #print('going to hessian in unrolled')
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y,train_adj,train_mask)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay,adj,mask):
        """
        Compute unrolled weights w`
        """
        #print('inside compute virtual model unrolled')
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y,adj,mask)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.model_optim.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y,train_adj,train_mask):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            _logger.warning('In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d
                    
            _, loss = self._logits_and_loss(trn_X, trn_y,train_adj,train_mask)
            dalphas.append(torch.autograd.grad(loss, [c.alpha for _, c in self.nas_modules]))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
