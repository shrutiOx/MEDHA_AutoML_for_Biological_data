# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 02:13:42 2023

@author: ADMIN
"""



# -*- coding: utf-8 -*-
"""Created on Thu Jun  1 18:00:23 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""


'''import libraries'''

import numpy as np
import pandas as pd

from nni.retiarii.oneshot.pytorch import DartsTrainer,EnasTrainer
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

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import math

import scipy.stats as stats
import sklearn.metrics as skm
from nni.retiarii.nn.pytorch import Repeat
import nni

from nni.nas.fixed import fixed_arch
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

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
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
from hyperopt.early_stop import no_progress_loss
'''my modules'''


import torch
from MEDHA.AutoToolGraph.Semimanualtrain   import SemiManualDart_train
from MEDHA.AutoToolGraph.cross_validation_errica_graph_hpo   import cross_validation_with_val_set
from MEDHA.AutoToolGraph.Graphein_Caller   import Graphein_Caller

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu" )


class HPO_DART(nn.Module):
    def __init__(self,
                 input_channel=None,
                 outchannel=None,
                 max_nodes=100,
                 percent_dec=0.85,
                 batch_size=20,
                 space={ 
                         'hidden_channels': hp.choice('hidden_channels',[100,125]),
                         'attn_heads': hp.choice('attn_heads',[10,15,20]),
                         'droprate': hp.choice('droprate',[0.2,0.4,0.6]),
                         'num_epochs': hp.choice('num_epochs', [20,30,40])},
                 OptimizerDart='Adam',
                 learning_rateDart =  0.0001,
                 dataset=None,
                 train_loader=None,
                 test_loader = None,
                 acc_thresold=80,
                 epochs=50,
                 intepochs=10,
                 R=3):
        super().__init__()
        

        #print('space ',space)

            
        self.space = space

        self.input_channel = input_channel

        self.outchannel = outchannel
        self.max_nodes = max_nodes
        self.percent_dec = percent_dec
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset = dataset
        self.epochs=epochs
        self.intepochs=intepochs



        self.OptimizerDart = OptimizerDart
        self.learning_rateDart = learning_rateDart
        self.acc_thresold = acc_thresold
        self.R=R


        
        
        self.counter_not_considered=0
        
    def Call_Process(self):
            



            loss_mean,accsval, model,trainlossmeanf,createlist,space = cross_validation_with_val_set(
                                                                                                    dataset=self.dataset,
                                                                                                    folds=10,
                                                                                                    epochs=self.epochs,
                                                                                                    intepochs=self.intepochs,
                                                                                                    space= self.space,
                                                                                                    input_channel=self.input_channel,
                                                                                                    outchannel=self.outchannel,
                                                                                                    max_nodes=self.max_nodes,
                                                                                                    percent_dec=self.percent_dec,
                                                                                                    batch_size=self.batch_size,
                                                                                                    OptimizerDart=self.OptimizerDart,
                                                                                                    learning_rateDart = self.learning_rateDart,
                                                                                                    acc_thresold=self.acc_thresold,
                                                                                                    R=self.R)




            return loss_mean,accsval, model,trainlossmeanf,createlist,space
    


