# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 02:39:05 2023

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
from torch_geometric.loader import DenseDataLoader
'''my modules'''


import torch
from MEDHA.AutoToolGraph.Semimanualtrain   import SemiManualDart_train

from MEDHA.AutoToolGraph.Graphein_Caller   import Graphein_Caller
from MEDHA.AutoToolGraph.train_internal   import train_internal_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu" )


class Selector(nn.Module):
    def __init__(self,
                 train_dataset,
                 intepochs,
                 lr,
                 space,
                 input_channel,
                 outchannel,
                 max_nodes,
                 percent_dec,
                 OptimizerDart,
                 acc_thresold ):
        super().__init__()
        

        #print('space ',space)

            
        self.train_dataset = train_dataset

        self.intepochs = intepochs

        self.batch_size = 15
        self.lr = lr
        self.space = space
        self.input_channel = input_channel
        self.outchannel = outchannel
        self.max_nodes = max_nodes
        self.percent_dec = percent_dec
        self.OptimizerDart = OptimizerDart
        self.acc_thresold = acc_thresold        

        
        self.counter_not_considered=0
        
    def evaluate_model(self,myspace):
            
            print ('Params testing: ', myspace)
            
            print('starting here: counter_not_considered ',self.counter_not_considered)
            


            hidden_channels = int(myspace['hidden_channels'])      
            attn_heads      = int(myspace['attn_heads'])      
            droprate        = float(myspace['droprate'])
            num_epochs      = int(myspace['num_epochs'])  
            
            OptimizerDart         = self.OptimizerDart 
            learning_rateDart     = self.lr  
            


            
            datasetr = self.train_dataset.shuffle()
            print('len(datasetr) ',len(datasetr))
            n=len(datasetr)
            p=int(n * (90/100))
            print('p ',p)
            train_dataset_my = datasetr[:p]
            val_dataset = datasetr[p::]
            
            print('len(train_dataset) internal ',len(train_dataset_my))
            print('len(val_dataset) internal ',len(val_dataset))

            self.val_loader = DenseDataLoader(val_dataset, batch_size=self.batch_size)
            self.train_loader = DenseDataLoader(train_dataset_my, batch_size=self.batch_size)
            
            
            '''Calling DART trainer'''
            DartObject        =  SemiManualDart_train()

            modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(input_channel= self.input_channel,
                                                                                               hidden_channels=hidden_channels,
                                                                                               outchannel=self.outchannel,
                                                                                               attn_heads=attn_heads,
                                                                                               max_nodes=self.max_nodes,
                                                                                               droprate=droprate,
                                                                                               percent_dec=self.percent_dec,
                                                                                               num_epochs=num_epochs,
                                                                                               OptimizerDart = OptimizerDart,
                                                                                               batch_size=self.batch_size, 
                                                                                               learning_rateDart     = learning_rateDart,
                                                                                               train_loader = self.train_loader,
                                                                                               test_loader =  self.val_loader) 

            '''Now calling KFoldCrossValidator.'''
            #print('we are here')
            print('DARTacc ',DARTacc)
            #print('acc_thresold here ',self.acc_thresold)
            
            if DARTacc >=  self.acc_thresold:


                val_loss_mean, test_acc_mean, model,trainlossmeanf = train_internal_func( train_loader=self.train_loader,
                                                                                                    test_loader=self.val_loader,
                                                                                                    model=modelfinal,
                                                                                                    epochs=self.intepochs,
                                                                                                    batch_size=10,
                                                                                                    lr=learning_rateDart)

                print('internal val_loss_mean',val_loss_mean)
                print('internal test_acc_mean',test_acc_mean)
                
                return {'loss': val_loss_mean, 'status': STATUS_OK, 'model': model,'space': self.space,'ParameterList': ParameterList,'avg-train-loss': trainlossmeanf,'avg-test-accuracy':test_acc_mean}
            
            else:
                self.counter_not_considered +=1
                print('counter_not_considered :',self.counter_not_considered)
                return {'loss': (1000 - DARTacc), 'status': STATUS_OK, 'model': None,'space': self.space,'ParameterList': ParameterList,'avg-train-loss': None,'avg-test-accuracy':None}
    
    def Calling_HPO_DART(self,max_evals,stoppage):
        
        print('counter_not_considered :',self.counter_not_considered)



        
        myobj = Selector(self.train_dataset,
                         self.intepochs,
                         self.lr,
                         self.space,
                         self.input_channel,
                         self.outchannel,
                         self.max_nodes,
                         self.percent_dec,
                         self.OptimizerDart,
                         self.acc_thresold)

        trials = Trials()
        best   = fmin(fn          =  myobj.evaluate_model,
                    space         =  self.space,
                    algo          =  tpe.suggest,
                    max_evals     =  max_evals,
                    trials        =  trials,
                    early_stop_fn =  no_progress_loss(stoppage))
        

        
        modelfinal     = trials.best_trial['result']['model']
        space          = trials.best_trial['result']['space']
        createlist     = trials.best_trial['result']['ParameterList']
        
        return modelfinal,createlist,space
    

