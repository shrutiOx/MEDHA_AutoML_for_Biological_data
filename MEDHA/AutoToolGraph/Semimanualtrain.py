

# -*- coding: utf-8 -*-
"""Created on Thu Jun  1 18:00:23 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""


'''import libraries'''

import numpy as np
import pandas as pd


import time
import torch.nn.functional as F
import copy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms

from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble


import math

import scipy.stats as stats
import sklearn.metrics as skm

import nni

from nni.nas.fixed import fixed_arch


from sklearn.model_selection import KFold
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import nni.nas.fixed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
from torchvision import datasets,transforms
import torchvision.transforms as transforms
from torchmetrics.functional import kl_divergence

import nni

import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,DenseGCNConv,DenseGINConv,DenseGraphConv,DenseGATConv,dense_mincut_pool
from torch_geometric.nn import BatchNorm,GraphNorm,InstanceNorm,LayerNorm
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 
import scipy.stats as stats
import sklearn.metrics as skm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
#from nni.retiarii.oneshot.pytorch import dartsgraph
from MEDHA.AutoToolGraph import dartsgraph
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import torch_geometric.transforms as T 
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset

'''my modules'''


from MEDHA.AutoToolGraph.searchspace import GNN,Net


from MEDHA.AutoToolGraph.Dartstrainer_graph   import DartTrainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu" )


        
        
        
class SemiManualDart_train(nn.Module):
    def __init__(self):#a list of 2 inputs and each input should be a number
        super().__init__()

    def DartCaller(self,
                   input_channel=None,
                   hidden_channels=20,
                   outchannel=None,
                   attn_heads=10,
                   max_nodes=None,
                   droprate=0.25,
                   percent_dec=0.25,
                   num_epochs=10,
                   OptimizerDart = 'Adam',
                   batch_size=5, 
                   learning_rateDart     = 0.006,
                   L2lambdaDart          = 0.00002 ,
                   momentumDart          = 0.6,
                   train_loader = None,
                   test_loader = None):




        model_space = Net(input_channel,
                          hidden_channels,
                          outchannel,
                          attn_heads,
                          max_nodes,
                          droprate,
                          percent_dec)#bayesian
        
        #print('model_space ', model_space)
        


        '''Now you again need to retrain the DARTs-derived model on your training and test set as per nni tutorial-https://nni.readthedocs.io/en/latest/tutorials/darts.html'''


        DARTobj = DartTrainer()
        final_model,exported_arch,nas_modules,DARTacc = DARTobj.DARTTrain(modelspace=model_space,
                                                            train_loader=train_loader,
                                                            val_loader=test_loader,
                                                            learning_rate=learning_rateDart,
                                                            moment=momentumDart,
                                                            L2lambda=L2lambdaDart,
                                                            optimizerset=OptimizerDart,
                                                            epochs=num_epochs,
                                                            batches=batch_size)

        
        model = final_model
        model.to(device)
        
        createlist = {"input_channel : ":[],"hidden_channels : ":[],"outchannel : ":[],
                      "attn_heads : ":[],"max_nodes : ":[],"droprate : ":[],
                      "percent_dec : ":[],"num_epochs : ":[],
                      "OptimizerDart : ":[],"batch_size : ":[],"learning_rateDart : ":[],"L2lambdaDart : ":[],
                     "momentumDart : ":[]}
        

       
        createlist["input_channel : "].append(input_channel)
        createlist["hidden_channels : "].append(hidden_channels)
        createlist["outchannel : "].append(outchannel)
        createlist["attn_heads : "].append(attn_heads)
        createlist["max_nodes : "].append(max_nodes)
        createlist["droprate : "].append(droprate)
        createlist["percent_dec : "].append(percent_dec)
        createlist["OptimizerDart : "].append(OptimizerDart)
        createlist["batch_size : "].append(batch_size)
        createlist["learning_rateDart : "].append(learning_rateDart)
        createlist["L2lambdaDart : "].append(L2lambdaDart)
        createlist["momentumDart : "].append(momentumDart)


        
        return model,exported_arch,nas_modules,createlist,DARTacc

