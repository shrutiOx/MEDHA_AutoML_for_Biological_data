# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:14:09 2023

@author: ADMIN
"""


# import libraries
import torch
import torch.nn as nn
import numpy as np
import sys

import torch.nn.functional as F
import copy


import pandas as pd
import seaborn as sns 

# libraries for partitioning and batching the data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset


from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from torchvision import datasets, transforms

import scipy.stats as stats
import sklearn.metrics as skm







'''
Constructor takes in,

input_param,
nUnits,
nLayers,
out_param,
actfun,
bnormf,
test_loader
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

 
 
class CNNLSTM(nn.Module):
  def __init__(self,
               sample_data,
               out_param=0, 
               actfun= 'ReLU6',
               nLSTMlayers= 1,
               n_hiddenLSTM= 50,
               losstype=None):
    super().__init__()
   
    '''Architecture creation'''

    self.out_dimension     = out_param
    self.seq_len           = sample_data
    self.actfun            = actfun
    self.nLSTMlayers       = nLSTMlayers
    self.n_hiddenLSTM      = n_hiddenLSTM



    self.layers            = nn.ModuleDict()
    self.losstype = losstype

    '''LSTM Layer'''
    '''
    #when you have batch_first = True then put data in format - batch_size, sequence_length, input features.
    #input_features = number of rows for each data. As u are taking vector not matrix hence it is 1.
    #batch_size = batch size of dataloader
    #sequence length = length of the entire sequence = input in the fully-connected layer previously.
    '''
    self.layers['lstm']  = nn.LSTM(
            input_size   = 1, #as u are working with vector so it is 1
            hidden_size  = self.n_hiddenLSTM,
            num_layers   = self.nLSTMlayers,
            batch_first=True #now u put batch_size at first during putting input x.
            
            )
            
            
    self.layers['outlstm'] = nn.Linear(self.n_hiddenLSTM, self.out_dimension) 

  def reset_hidden_state(self):
        hidden      = (
            torch.zeros(self.nLSTMlayers, self.seq_len, self.n_hiddenLSTM).to(device),
            torch.zeros(self.nLSTMlayers, self.seq_len, self.n_hiddenLSTM).to(device) 
                           )                  

  
  def forward(self,x):
    
    actfun = getattr(torch.nn,self.actfun)
    actfun = actfun()
    x      = x.to(device)
    savethisx = x
    self.savethisx = savethisx
    batchsize      = len(x)
    
    '''getting the shape'''
    inlstm         = x.view(batchsize,self.seq_len,1)
    inlstm         = inlstm.to(device)
    
    listy_pred = []

    lstm_out, hidden =  self.layers['lstm'](x.view(batchsize,self.seq_len,1))
    #print( ' lstm_out ', lstm_out.shape)
    
    '''
    lstm_out  torch.Size([15, 12960, 20]) 15  = batch_size; 12960 = input sequence; 20 = number of hidden units in each layer
    last_time_step  torch.Size([15, 20])
    '''
    lstm_out = actfun(lstm_out)

    y_pred           = self.layers['outlstm'](lstm_out[:, -1, :]) #this will give 1 output per batch
    y_pred           = actfun(y_pred)
    if (self.losstype=='div'):
        y_pred = torch.nn.functional.log_softmax(y_pred,dim=1)

    
    return y_pred

  def givememyx(self,x):

        return self.savethisx.detach()
        #print(self.savethisx.shape)
    

    
  def Xavier_init(self,objectM):
    for p in objectM.named_parameters():
        if 'weight' in p[0]: 
            if len(p[1].shape) < 2:
                nn.init.xavier_normal_(p[1].unsqueeze(0))
            else:
                nn.init.xavier_normal_(p[1].data)
    return objectM





