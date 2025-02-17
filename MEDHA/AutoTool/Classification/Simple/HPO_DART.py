
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


import MEDHA.AutoTool.Classification.Simple.Block_CNN_usableBN  

from   MEDHA.AutoTool.Classification.Simple.AutoDL_CNNspaceBN          import CNNModelSpace

from   MEDHA.AutoTool.Classification.Simple.TrainerandOptimizer        import TrainerandOptimizer

from   MEDHA.AutoTool.Classification.Simple.DartTrainer                import DartTrainer

from   MEDHA.AutoTool.Classification.Simple.SemiManualDART_train       import SemiManualDart_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu" )


class HPO_DART(nn.Module):
    def __init__(self,
                 sample_data=None,
                 in_channel=1,
                 kernel=[1,3,5],
                 outchannel=0,
                 dataSet=None,
                 lossfun=None,
                 batch_size=5,
                 acc_thresold=65,
                 pool_size = 1,
                 drop = 0,
                 space={ 
                         'out_channel_input': hp.choice('out_channel_input',[50,100,125]),
                         'out_channel_f': hp.choice('out_channel_f',[25,50]),
                         'actfun': hp.choice('actfun',["ReLU6", "ReLU",'LeakyReLU']),
                         'UnitFCN_vars': hp.choice('UnitFCN_vars',[25,50]),
                         'nLayers_vars': hp.choice('nLayers_vars', [ 1,2]),
                         'loop':  hp.choice('loop', [1,2]),
                         'num_epochDART': hp.choice('num_epochDART', [3,5,7])},
                 threshold =0 ,
                 predtype='binary',
                 optimizerset='Adam',
                 learning_rate =  0.00018,
                 L2lambdaDart =  0.03,
                 momentumDart =  0.0):
        super().__init__()
        

        #print('space ',space)

            
        self.space = space
        self.sample_data = sample_data
        self.in_channel = in_channel
        self.kernel = kernel
        self.outchannel = outchannel
        self.dataSet = dataSet
        self.lossfun = lossfun
        self.batch_size = batch_size
        self.threshold = threshold
        self.predtype = predtype
        self.acc_thresold = acc_thresold
        self.optimizerset = optimizerset
        self.learning_rate = learning_rate
        self.L2lambdaDart = L2lambdaDart
        self.momentumDart = momentumDart
        self.pool_size = pool_size
        self.drop = drop


        #print('acc_thresold ',acc_thresold)
        
        myspace = self.space
        
        self.counter_not_considered=0
        
    def evaluate_model(self,myspace):
            
            print ('Params testing: ', myspace)
            
            print('starting here: counter_not_considered ',self.counter_not_considered)
            
            pool_size         =  self.pool_size 

            out_channel_input = int(myspace['out_channel_input'])      
            out_channel_f     = int(myspace['out_channel_f'])      
            actfun            = (myspace['actfun'])
            
            optimizerset      = self.optimizerset 
            learning_rate     = self.learning_rate  
            L2lambdaDart      = self.L2lambdaDart  
            momentumDart      = self.momentumDart 
            

            
            dropt              = self.drop
            UnitFCN_vars      =  int(myspace['UnitFCN_vars'])  
            nLayers_vars      =  int(myspace['nLayers_vars'])  
            loop              =  int(myspace['loop']) 
            num_epochDART     =  int(myspace['num_epochDART']) 
            
            
            chooseblocks      =  ['block1','block2','block3','block4','block5','block6','block7']
            
            '''Calling DART trainer'''
            DartObject        =  SemiManualDart_train()

            
            modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input = out_channel_input,
                                                                                               out_channel_f     = out_channel_f,
                                                                                               drop              = dropt,
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
                                                                                               learning_rateDart = learning_rate,
                                                                                               L2lambdaDart      = L2lambdaDart ,
                                                                                               momentumDart      = momentumDart,
                                                                                               dart_dataset      = self.dataSet,
                                                                                               lossfuntype       = self.lossfun,
                                                                                               threshold         = self.threshold) 
            
            '''Now calling KFoldCrossValidator.'''
            #print('we are here')
            print('DARTacc ',DARTacc)
            #print('acc_thresold here ',self.acc_thresold)
            
            if DARTacc >=  self.acc_thresold:

                #train_loss       = []
                #train_acc        = []
                #validation_acc   = []
                


                train_loss,train_acc,validation_acc,validation_loss,avg_train_loss ,avg_train_acc,avg_val_acc,avg_val_loss,bestmodel= DartObject.KFoldCrossValidator(k=5,
                                                                                                                                          crossvalidator_dataset=self.dataSet,
                                                                                                                                          batch_size=self.batch_size,
                                                                                                                                          model=modelfinal,
                                                                                                                                          OptimizerKfold = 'Adam',
                                                                                                                                          lossfuntype=self.lossfun,
                                                                                                                                          num_epochs=5,
                                                                                                                                          predtype=self.predtype)


                return {'loss': avg_val_loss, 'status': STATUS_OK, 'model': bestmodel,'space': self.space,'ParameterList': ParameterList,'avg-train-accuracy': avg_train_acc,'avg-train-loss': avg_train_loss,'avg-validation-accuracy':avg_val_acc}
            
            else:
                self.counter_not_considered +=1
                print('counter_not_considered :',self.counter_not_considered)
                return {'loss': (1000 - DARTacc), 'status': STATUS_OK, 'model': None,'space': self.space,'ParameterList': ParameterList,'avg-train-accuracy': None,'avg-train-loss': None,'avg-validation-accuracy':None}
    
    def Calling_HPO_DART(self,max_evals,stoppage):
        
        print('counter_not_considered :',self.counter_not_considered)

        
        myobj = HPO_DART(self.sample_data,
                         self.in_channel,
                         self.kernel,
                         self.outchannel,
                         self.dataSet,
                         self.lossfun,
                         self.batch_size,
                         self.acc_thresold,
                         self.pool_size,
                         self.drop,
                         self.space,
                         self.threshold,
                         self.predtype,
                         self.optimizerset ,
                         self.learning_rate  ,
                         self.L2lambdaDart  ,
                         self.momentumDart )

        trials = Trials()
        best   = fmin(fn          =  myobj.evaluate_model,
                    space         =  self.space,
                    algo          =  tpe.suggest,
                    max_evals     =  max_evals,
                    trials        =  trials,
                    early_stop_fn =  no_progress_loss(stoppage))
        
        avg_train_loss = trials.best_trial['result']['avg-train-loss']
        avg_train_acc  = trials.best_trial['result']['avg-train-accuracy']
        loss           = trials.best_trial['result']['loss']
        avg_val_acc    = trials.best_trial['result']['avg-validation-accuracy']
        
        modelfinal     = trials.best_trial['result']['model']
        space          = trials.best_trial['result']['space']
        createlist     = trials.best_trial['result']['ParameterList']
        
        return loss,avg_val_acc,avg_train_acc,avg_train_loss,modelfinal,space,createlist
    

