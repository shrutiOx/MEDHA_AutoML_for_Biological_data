# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:24:40 2023

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

""" The below class is constructed to automate deep-neural-network (feed-forward) neural net

It has 5 functions within it including construvtor functionS.

The class takes in inputs as input parameters, number of units, number of layers,
output parameters (number), activation fucntion,  batch normalization flag (which accordingly enables or disables batch-norm) and test set loader in the contructor and returns the output. This class is used for object instantiation which will be used by the functions of this class, defined next.

The class Object can be used in class function Xavier_init to invoke Xavier weight initialization. This just overrides the default Kramming weight initialization (uniform) to Xavier (normal distribtuion)

The class Object can be used in the class's function - OptandLoss to output suitable optimizer and loss function. The loss function is set to default BCEWithLogitsLoss(), however based on the inputs such as 
learning_rate,momentum,L2 regularization parameter and optimizer set, the optimizer is customed.

Lastly it has a function called trainTheModel which trains the model into mini-batches and epochs. It outputs the trainaccuracy, testaccuracy, losses list over each epoch (each epoch has many mini-batches) and also outputs the "BestModel" which is obtained in one of the epoch. i.e within the code it finds which epoch outputs best test accuracy and accordingly save the model's parameters and hyperparameters.
"""

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

class DNNClassBCEFunc(nn.Module):
  def __init__(self,input_param,nUnits,nLayers,out_param,actfun,bnormf):
    super().__init__()
    '''Architecture creation'''
    # create dictionary to store the layers
    
    self.layers = nn.ModuleDict()                                               
    self.nLayers = nLayers                                                      #(depth)
  
    

    ### input layer
    
    self.layers['input'] = nn.Linear(input_param,nUnits)                        
    
    ### hidden layers
    
    for i in range(nLayers):                                                    
      self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)                      
      
      if bnormf == True:                                                        
          self.bnorm1 = nn.BatchNorm1d(nUnits)                                  #batch normalization                   

    ### output layer
    
    self.layers['output'] = nn.Linear(nUnits,out_param)                         

    # activation funcion to pass through
    
    self.actfun = actfun

  # forward pass

  def forward(self,x):
    
    # get activation function type
    # this code replaces torch.relu with torch.<self.actfun>
    
    actfun = getattr(torch.nn,self.actfun)
    actfun = actfun()
    
    #x = input
    
    x = actfun( self.layers['input'](x) )                                       

    # hidden layers
    
    for i in range(self.nLayers):
      x = actfun( self.layers[f'hidden{i}'](x) )                                
    
    # return output layer
    x =  self.layers['output'](x)


    return  x


  '''Function invokes Xavier weight initialization'''
    
  def Xavier_init(self,objectM):
    for p in objectM.named_parameters():
        if 'weight' in p[0]: 
            if len(p[1].shape) < 2:
                nn.init.xavier_normal_(p[1].unsqueeze(0))
            else:
                nn.init.xavier_normal_(p[1].data)
    return objectM
                
                
                
         
  '''#optimizer and lossfunction creation'''
  '''Function takes in
      objectM,
      learning_rate,
      moment,
      L2lambda,
      optimizerset
  '''

     





