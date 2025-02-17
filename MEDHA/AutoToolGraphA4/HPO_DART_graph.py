

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
from MEDHA.AutoToolGraph.crossvalidation   import cross_validation_with_val_set
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
                         'hidden_channels': hp.choice('hidden_channels',[50,100,125,150]),
                         'attn_heads': hp.choice('attn_heads',[10,15,20,30]),
                         'droprate': hp.choice('droprate',[0.2,0.4,0.6,0.8]),
                         'num_epochs': hp.choice('num_epochs', [10,15,7])},
                 OptimizerDart='Adam',
                 learning_rateDart =  0.00018,
                 dataset=None,
                 train_loader=None,
                 test_loader = None,
                 acc_thresold=80):
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
        



        self.OptimizerDart = OptimizerDart
        self.learning_rateDart = learning_rateDart
        self.acc_thresold = acc_thresold



        #print('acc_thresold ',acc_thresold)
        
        myspace = self.space
        
        self.counter_not_considered=0
        
    def evaluate_model(self,myspace):
            
            print ('Params testing: ', myspace)
            
            print('starting here: counter_not_considered ',self.counter_not_considered)
            


            hidden_channels = int(myspace['hidden_channels'])      
            attn_heads      = int(myspace['attn_heads'])      
            droprate        = float(myspace['droprate'])
            num_epochs      = int(myspace['num_epochs'])  
            
            OptimizerDart         = self.OptimizerDart 
            learning_rateDart     = self.learning_rateDart  
            

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
                                                                                               test_loader =  self.test_loader) 

            '''Now calling KFoldCrossValidator.'''
            #print('we are here')
            print('DARTacc ',DARTacc)
            #print('acc_thresold here ',self.acc_thresold)
            
            if DARTacc >=  self.acc_thresold:



                val_loss_mean, test_acc_mean, test_acc_std,val_acc, model,trainlossmeanf = cross_validation_with_val_set(
                                                                                                    dataset=self.dataset,
                                                                                                    model=modelfinal,
                                                                                                    folds=10,
                                                                                                    epochs=50,
                                                                                                    batch_size=15,
                                                                                                    lr=0.0001)


                return {'loss': val_loss_mean, 'status': STATUS_OK, 'model': model,'space': self.space,'ParameterList': ParameterList,'avg-train-loss': trainlossmeanf,'avg-test-accuracy':test_acc_mean,'avg-val-accuracy':val_acc,'test_acc_std':test_acc_std}
            
            else:
                self.counter_not_considered +=1
                print('counter_not_considered :',self.counter_not_considered)
                return {'loss': (1000 - DARTacc), 'status': STATUS_OK, 'model': model,'space': self.space,'ParameterList': ParameterList,'avg-train-loss': None,'avg-test-accuracy':None,'avg-val-accuracy':None,'test_acc_std':None}
    
    def Calling_HPO_DART(self,max_evals,stoppage):
        
        print('counter_not_considered :',self.counter_not_considered)



        
        myobj = HPO_DART(self.input_channel,
                         self.outchannel,
                         self.max_nodes,
                         self.percent_dec,
                         self.batch_size,
                         self.space,
                         self.OptimizerDart,
                         self.learning_rateDart,
                         self.dataset,
                         self.train_loader,
                         self.test_loader,
                         self.acc_thresold )

        trials = Trials()
        best   = fmin(fn          =  myobj.evaluate_model,
                    space         =  self.space,
                    algo          =  tpe.suggest,
                    max_evals     =  max_evals,
                    trials        =  trials,
                    early_stop_fn =  no_progress_loss(stoppage))
        
        avg_train_loss  = trials.best_trial['result']['avg-train-loss']
        loss            = trials.best_trial['result']['loss']
        avg_val_acc     = trials.best_trial['result']['avg-val-accuracy']
        avg_test_acc     = trials.best_trial['result']['avg-test-accuracy']
        avg_test_std    = trials.best_trial['result']['test_acc_std']  
        
        modelfinal     = trials.best_trial['result']['model']
        space          = trials.best_trial['result']['space']
        createlist     = trials.best_trial['result']['ParameterList']
        
        return avg_train_loss,loss,avg_val_acc,avg_test_acc,avg_test_std,modelfinal,space,createlist
    

