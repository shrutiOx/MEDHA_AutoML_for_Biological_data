
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


import MEDHA.AutoTool.Regression.Simple.Block_CNN_usableBN  

from MEDHA.AutoTool.Regression.Simple.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Regression.Simple.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Regression.Simple.DartTrainer   import DartTrainer

from MEDHA.AutoTool.Regression.Simple.SemiManualDART_train import SemiManualDart_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class HPO_DART(nn.Module):
    def __init__(self,
                 sample_data,
                 in_channel,
                 kernel,
                 outchannel,
                 dataSet,
                 lossfun,
                 batch_size,
                 acc_thresold=2,
                 pool_size = 1,
                 space= { 
                             'out_channel_input': hp.choice('out_channel_input',[25,50,75,100,125]),
                             'out_channel_f': hp.choice('out_channel_f',[25,50,60]),
                             'actfun': hp.choice('actfun',["ReLU6", "ReLU"]),
                             'drop': hp.uniform('drop', 0.1,0.3),
                             'UnitFCN_vars': hp.choice('UnitFCN_vars',[25,50,65]),
                             'nLayers_vars': hp.uniform('nLayers_vars', 1,3),
                             'loop': hp.uniform('loop', 1,2),
                             'num_epochDART': hp.uniform('num_epochDART',3,10)
                         },
                 optimizerset  =  'SGD',
                 learning_rate =  0.006,
                 L2lambdaDart  =  0.00002,
                 momentumDart  =  0.6):#a list of 2 inputs and each input should be a number
        super().__init__()
        
        self.space = space
        self.sample_data = sample_data
        self.in_channel = in_channel
        self.kernel = kernel
        self.outchannel = outchannel
        self.dataSet = dataSet
        self.lossfun = lossfun
        self.batch_size = batch_size
        
        self.acc_thresold  = acc_thresold
        self.optimizerset  = optimizerset
        self.learning_rate = learning_rate
        self.L2lambdaDart  = L2lambdaDart
        self.momentumDart  = momentumDart
        self.pool_size = pool_size
        
        myspace = self.space
        self.counter_not_considered=0
    
        
    def evaluate_model(self,myspace):
            
            print ('Params testing: ', myspace)
            
            pool_size = self.pool_size     

            out_channel_input = int(myspace['out_channel_input'])      
            out_channel_f = int(myspace['out_channel_f'])      
            actfun = (myspace['actfun'])
            
            optimizerset      = self.optimizerset 
            learning_rate     = self.learning_rate  
            L2lambdaDart      = self.L2lambdaDart  
            momentumDart      = self.momentumDart 
            
            drop =  float(myspace['drop']) 
            UnitFCN_vars =  int(myspace['UnitFCN_vars'])  
            nLayers_vars  = int(myspace['nLayers_vars'])  
            loop = int(myspace['loop']) 
            num_epochDART = int(myspace['num_epochDART']) 
            
            
            chooseblocks=['block1','block2','block3','block4','block5','block6','block7']
            
            '''Calling DART trainer'''
            DartObject =  SemiManualDart_train()

            modelfinal,exported_arch,nas_modules,createlist,DARTacc = DartObject.DartCaller(out_channel_input = out_channel_input,
                                                                                            out_channel_f     = out_channel_f,
                                                                                            drop              = drop,
                                                                                            UnitFCN_vars      = UnitFCN_vars,
                                                                                            nLayers_vars      = nLayers_vars,
                                                                                            loop              = loop,
                                                                                            pool_size         = pool_size,
                                                                                            actfun            = actfun,
                                                                                            num_epochs        = num_epochDART,
                                                                                            OptimizerDart     = optimizerset,
                                                                                            sample_data       = self.sample_data,
                                                                                            in_channel        = self.in_channel,
                                                                                            kernel            = [1,3,5],
                                                                                            batch_size        = self.batch_size, 
                                                                                            outchannel        = self.outchannel,
                                                                                            chooseblocks      = chooseblocks,
                                                                                            learning_rateDart     = learning_rate,
                                                                                            L2lambdaDart          = L2lambdaDart ,
                                                                                            momentumDart          = momentumDart,
                                                                                            dart_dataset          = self.dataSet,
                                                                                            lossfuntype           = self.lossfun) 
            

            
            '''Now calling KFoldCrossValidator.'''
            if DARTacc <=  self.acc_thresold:
                train_loss_all = []
                test_loss_all = []

                train_loss_all,test_loss_all, avg_train_loss,avg_test_loss,bestmodel,pearsoncorrArr,spearmancorrArr,R_squareArr = DartObject.KFoldCrossValidator(k                  = 5,
                                                                                                              crossvalidator_dataset = self.dataSet,
                                                                                                              model                  = modelfinal,
                                                                                                              lossfuntype            = self.lossfun,
                                                                                                              num_epochs             = 3,
                                                                                                              batch_size             = self.batch_size)


                return {'loss': avg_test_loss, 'status': STATUS_OK, 'model': bestmodel,'space': self.space,'createlist': createlist,'pearsoncorrArr':pearsoncorrArr,'spearmancorrArr':spearmancorrArr,'R_squareArr':R_squareArr}
            else:
                return {'loss': (1000 - DARTacc), 'status': STATUS_OK, 'model': None,'space': self.space,'createlist': createlist,'pearsoncorrArr':None,'spearmancorrArr':None,'R_squareArr':None}

    def Calling_HPO_DART(self,max_evals,stoppage):
        
        myobj = HPO_DART(self.sample_data,
                         self.in_channel,
                         self.kernel,
                         self.outchannel,
                         self.dataSet,
                         self.lossfun,
                         self.batch_size,
                         self.acc_thresold,
                         self.pool_size,
                         self.space,
                         self.optimizerset,
                         self.learning_rate,
                         self.L2lambdaDart,
                         self.momentumDart)

        trials = Trials()
        best = fmin(fn=myobj.evaluate_model,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    early_stop_fn =  no_progress_loss(stoppage))
        
        loss = trials.best_trial['result']['loss']
        modelfinal = trials.best_trial['result']['model']
        space = trials.best_trial['result']['space']
        createlist = trials.best_trial['result']['createlist']
        pearsoncorrArr = trials.best_trial['result']['pearsoncorrArr']
        spearmancorrArr = trials.best_trial['result']['spearmancorrArr']
        R_squareArr = trials.best_trial['result']['R_squareArr']
        
        return loss,modelfinal,space,createlist,pearsoncorrArr,spearmancorrArr,R_squareArr

