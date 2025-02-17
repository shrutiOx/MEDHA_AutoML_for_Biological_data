
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


import MEDHA.AutoTool.Classification.Advanced.Block_CNN_usableBN  

from MEDHA.AutoTool.Classification.Advanced.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Classification.Advanced.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Classification.Advanced.DartTrainer   import DartTrainer

from MEDHA.AutoTool.Classification.Advanced.SemiManualDART_train import SemiManualDart_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class HPO_DART_advanced(nn.Module):
    def __init__(self,
                 sample_data,
                 in_channel,
                 kernel,
                 outchannel,
                 dataSet,
                 concatflag=False,
                 lossfun='bce',
                 batch_size=5,
                 pool_size = 1,
                 modeltypeflag = 'dnn',
                 UnitFCN_vars = 50,
                 nLayers_vars = 1,
                 loop = 1,
                 spacednn= {  
                             'out_channel_input': hp.choice('out_channel_input',[25,50,75,100,125]),
                             'out_channel_f': hp.choice('out_channel_f',[25,50,75,100,125]),
                             'actfun': hp.choice('actfun',["ReLU6", "ReLU",'LeakyReLU']),
                             'drop': hp.uniform('drop', 0.3,0.5),
                             'unitsdnn': hp.choice('UnitFCN_vars',[60,100,150,250,300]),
                             'layersdnn': hp.choice('nLayers_vars', [ 1,2]),
                             'num_epochDART': hp.choice('num_epochDART', [3,5,7])
                         },
                 spacelstm= {  
                             'out_channel_input': hp.choice('out_channel_input',[25,50,75,100,125]),
                             'out_channel_f': hp.choice('out_channel_f',[25,50,75,100,125]),
                             'actfun': hp.choice('actfun',["ReLU6", "ReLU",'LeakyReLU']),
                             'drop': hp.uniform('drop', 0.3,0.5),
                              'nLSTMlayers': hp.choice('nLSTMlayers',[1,2]),
                              'n_hiddenLSTM': hp.choice('n_hiddenLSTM', [10,50,100,150]),
                             'num_epochDART': hp.choice('num_epochDART', [3,5,7]),
                         },
                  threshold =0 ,
                  predtype='binary',
                  optimizerset='Adam',
                  learning_rate =  0.00018,
                  L2lambdaDart =  0.03,
                  momentumDart =  0.0,
                  condata=None,
                  acc_thresold=70):#a list of 2 inputs and each input should be a number
        super().__init__()
        
        self.spacednn  = spacednn
        self.spacelstm = spacelstm
        self.sample_data = sample_data
        self.in_channel = in_channel
        self.kernel = kernel
        self.outchannel = outchannel
        self.dataSet = dataSet
        self.lossfun = lossfun
        self.batch_size = batch_size
        self.concatflag = concatflag
        self.condata = condata
        
        self.optimizerset = optimizerset
        self.learning_rate = learning_rate
        self.L2lambdaDart = L2lambdaDart
        self.momentumDart = momentumDart
        self.threshold = threshold
        self.predtype = predtype
        self.acc_thresold = acc_thresold
        self.pool_size = pool_size
        self.modeltypeflag = modeltypeflag
        self.UnitFCN_vars = UnitFCN_vars
        self.nLayers_vars = nLayers_vars
        self.loop = loop
        
        if self.modeltypeflag == 'dnn':
            myspace = self.spacednn
        else:
            myspace = self.spacelstm
            
        self.counter_not_considered=0
    
        
    def evaluate_model(self,myspace):
            
            print ('Params testing: ', myspace)
            print('starting here: counter_not_considered ',self.counter_not_considered)
            
            pool_size              =  self.pool_size      

            out_channel_input      =  int(myspace['out_channel_input'])      
            out_channel_f          =  int(myspace['out_channel_f'])      
            actfun                 =  (myspace['actfun'])
            
            optimizerset           =   self.optimizerset
            learning_rate          =   self.learning_rate
            L2lambdaDart           =   self.L2lambdaDart
            momentumDart           =   self.momentumDart
            
            drop                   =  float(myspace['drop']) 
  

            
            #loop                   =  int(myspace['loop']) 
            num_epochDART          =  int(myspace['num_epochDART']) 

            chooseblocks           = ['block1','block2','block3','block4','block5','block6','block7']
            
            modeltypeflag          =  self.modeltypeflag

            
            if modeltypeflag == 'dnn':
                layersdnn              =  int(myspace['layersdnn'])  
                unitsdnn               =   int(myspace['unitsdnn'])
                nLSTMlayers            =  0
                n_hiddenLSTM           =  0
                
            if modeltypeflag == 'lstm':
                nLSTMlayers           =  int(myspace['nLSTMlayers'])  
                n_hiddenLSTM           = int(myspace['n_hiddenLSTM']) 
                layersdnn           =  0
                unitsdnn           =  0

            
           
            
            '''Calling DART trainer'''
            DartObject =  SemiManualDart_train()

            modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input = out_channel_input,
                                                                                               out_channel_f     = out_channel_f,
                                                                                               drop              = drop,
                                                                                               UnitFCN_vars      = self.UnitFCN_vars,
                                                                                               nLayers_vars      = self.nLayers_vars,
                                                                                               loop              = self.loop,
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
            
            #ParameterList["modeltypeflag : "].append(modeltypeflag)
            #ParameterList["layersdnn : "].append(layersdnn)
            #ParameterList["unitsdnn : "].append(unitsdnn)
            #ParameterList["nLSTMlayers : "].append(nLSTMlayers)
            #ParameterList["n_hiddenLSTM : "].append(n_hiddenLSTM)

            
            '''Now calling Concatenator.'''

            #print('we are here')
            print('DARTacc ',DARTacc)
            
            if DARTacc >=  self.acc_thresold:
            
                usenet,Concat_dataset = DartObject.Concater(whole_Dataset       = self.dataSet,
                                                        model                  = modelfinal,
                                                        concatflag             = self.concatflag,
                                                        condata                = self.condata, 
                                                        modeltypeflag          = modeltypeflag,
                                                        out_param              = self.outchannel,
                                                        nUnits                 = unitsdnn,
                                                        nLayers                = layersdnn,
                                                        nLSTMlayers            = nLSTMlayers,
                                                        n_hiddenLSTM           = n_hiddenLSTM,
                                                        createlist=ParameterList
                                                       )
            
                '''Now calling KFoldCrossValidator.'''

                train_loss_all = []
                test_loss_all = []


                train_acc = []
                validation_acc = []
                
                
                
               
                
                train_loss_all,train_acc,validation_acc,avg_train_loss,avg_train_acc,avg_val_acc,bestmodel,avg_val_loss = DartObject.KFoldCrossValidator(k=5,
                                                                                                                                          crossvalidator_dataset=Concat_dataset,
                                                                                                                                          batch_size=self.batch_size,
                                                                                                                                          model=usenet,
                                                                                                                                          learning_rate     = 0.006,
                                                                                                                                          L2lambda          = 0.00002 ,
                                                                                                                                          momentum          = 0.0,
                                                                                                                                          OptimizerKfold = 'Adam',
                                                                                                                                          lossfuntype=self.lossfun,
                                                                                                                                          num_epochs=5,
                                                                                                                                          predtype=self.predtype)
            



                return {'loss': avg_val_loss, 'status': STATUS_OK, 'modelDart': modelfinal,'modelkfold': usenet,'space': myspace,'ParameterList': ParameterList,'avg-train-accuracy': avg_train_acc,'avg-train-loss': avg_train_loss,'avg-validation-accuracy':avg_val_acc}
        
            else:
                self.counter_not_considered +=1
                print('counter_not_considered :',self.counter_not_considered)
                return {'loss': (1000 - DARTacc), 'status': STATUS_OK, 'modelDart': None,'modelkfold': None,'space': myspace,'ParameterList': ParameterList,'avg-train-accuracy': None,'avg-train-loss': None,'avg-validation-accuracy':None}
            
            


    def Calling_HPO_DART(self,max_evals,stoppage):
        
        myobj = HPO_DART_advanced(self.sample_data,
                         self.in_channel,
                         self.kernel,
                         self.outchannel,
                         self.dataSet,
                         self.concatflag,
                         self.lossfun,
                         self.batch_size,
                         self.pool_size,
                         self.modeltypeflag,
                         self.UnitFCN_vars ,
                         self.nLayers_vars ,
                         self.loop,
                         self.spacednn,
                         self.spacelstm,
                         self.threshold,
                         self.predtype,
                         self.optimizerset,
                         self.learning_rate,
                         self.L2lambdaDart,
                         self.momentumDart,
                         self.condata,
                         self.acc_thresold)

        trials = Trials()
        
        if self.modeltypeflag == 'dnn':
            myspacec = self.spacednn
        else:
            myspacec = self.spacelstm
            
        best = fmin(fn=myobj.evaluate_model,
                    space=myspacec,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials,
                    early_stop_fn=no_progress_loss(stoppage))
        
        avg_train_loss = trials.best_trial['result']['avg-train-loss']
        avg_train_acc  = trials.best_trial['result']['avg-train-accuracy']
        avg_val_loss  = (trials.best_trial['result']['loss'])
        avg_val_acc  =  (trials.best_trial['result']['avg-validation-accuracy'])
        

        createlist     = trials.best_trial['result']['ParameterList']
        modelDart      = trials.best_trial['result']['modelDart']
        modelkfold     = trials.best_trial['result']['modelkfold']
        space          = trials.best_trial['result']['space']

        
        return avg_val_acc,avg_val_loss,avg_train_acc,avg_train_loss,modelDart,modelkfold,space,createlist

