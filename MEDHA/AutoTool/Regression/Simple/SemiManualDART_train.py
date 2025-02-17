

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
'''my modules'''


import MEDHA.AutoTool.Regression.Simple.Block_CNN_usableBN  

from MEDHA.AutoTool.Regression.Simple.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Regression.Simple.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Regression.Simple.DartTrainer   import DartTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''get my model'''


'''fixed params'''
out_channel_i = 25
out_channel_i2 = 50
increment = 25
num_conv_layers = 2


        
        
        
class SemiManualDart_train(nn.Module):
    def __init__(self):#a list of 2 inputs and each input should be a number
        super().__init__()

    def DartCaller(self,
                   out_channel_input = 100,
                   out_channel_f = 25,
                   drop = 0.2,
                   UnitFCN_vars = 50,
                   nLayers_vars = 1,
                   loop = 1,
                   pool_size = 1,
                   actfun = 'ReLU',
                   num_epochs=10,
                   OptimizerDart = 'SGD',
                   sample_data = None,
                   in_channel  = 1,
                   kernel      = [1,3,5],
                   batch_size=5, 
                   outchannel = 0,
                   chooseblocks=['block1','block2','block3','block4','block5','block6','block7'],
                   learning_rateDart     = 0.006,
                   L2lambdaDart          = 0.00002 ,
                   momentumDart          = 0.6,
                   dart_dataset = None,
                   lossfuntype = 'div'):




        model_space = CNNModelSpace(
                              sample_data, #user only
                              in_channel ,  
                              pool_size,      
                              kernel,
                              out_channel_input,
                              out_channel_i,
                              out_channel_i2,
                              out_channel_f,
                              increment,
                              num_conv_layers,
                              actfun,
                              drop,
                              UnitFCN_vars,#bayesian
                              nLayers_vars,
                              loop,
                              chooseblocks,
                              outchannel,
                              losstype=lossfuntype)#bayesian
        
        #print('model_space ', model_space)
        


        '''Now you again need to retrain the DARTs-derived model on your training and test set as per nni tutorial-https://nni.readthedocs.io/en/latest/tutorials/darts.html'''


        DARTobj = DartTrainer()
        final_model,exported_arch,nas_modules,DARTacc = DARTobj.DARTTrain(modelspace=model_space,
                                                            dataloader=dart_dataset,
                                                            learning_rate=learning_rateDart,
                                                            moment=momentumDart,
                                                            L2lambda=L2lambdaDart,
                                                            optimizerset=OptimizerDart,
                                                            lossfun=lossfuntype,
                                                            epochs=num_epochs,
                                                            batches=batch_size)

        
        model = final_model
        model.to(device)
        
        createlist = {"out_channel_input : ":[],"out_channel_f : ":[],"drop : ":[],
                      "UnitFCN_vars : ":[],"nLayers_vars : ":[],"loop : ":[],
                      "pool_size : ":[],"actfun : ":[],"num_epochs : ":[],
                      "OptimizerDart : ":[],"in_channel : ":[],"kernel : ":[],
                     "outchannel : ":[],"batch_size : ":[],"learning_rateDart : ":[],"L2lambdaDart : ":[],
                     "momentumDart : ":[],"lossfuntype : ":[],"chooseblocks : ":[]}
        

        
        createlist["out_channel_input : "].append(out_channel_input)
        createlist["out_channel_f : "].append(out_channel_f)
        createlist["drop : "].append(drop)
        createlist["UnitFCN_vars : "].append(UnitFCN_vars)
        createlist["nLayers_vars : "].append(nLayers_vars)
        createlist["loop : "].append(loop)
        createlist["pool_size : "].append(pool_size)
        createlist["actfun : "].append(actfun)
        createlist["num_epochs : "].append(num_epochs)
        createlist["OptimizerDart : "].append(OptimizerDart)
        createlist["in_channel : "].append(in_channel)
        createlist["kernel : "].append(kernel)
        createlist["outchannel : "].append(outchannel)
        createlist["batch_size : "].append(batch_size)
        createlist["learning_rateDart : "].append(learning_rateDart)
        createlist["L2lambdaDart : "].append(L2lambdaDart)
        createlist["momentumDart : "].append(momentumDart)
        createlist["lossfuntype : "].append(lossfuntype)
        createlist["chooseblocks : "].append(chooseblocks)
        

        
        return model,exported_arch,nas_modules,createlist,DARTacc

    def KFoldCrossValidator(self,
                            k=5,
                            crossvalidator_dataset=None,
                            batch_size=10,
                            model=None,
                            learning_rate     = 0.006,
                            L2lambda          = 0.00002 ,
                            momentum          = 0.6,
                            OptimizerKfold = 'Adam',
                            lossfuntype='div',
                            num_epochs=5):
        
        splits=KFold(n_splits=k,shuffle=True)
        foldperf={}
        dftest = []
        
        '''creating an object of class TrainerandOptimizer'''
        trainparter = TrainerandOptimizer()

        
        '''K-FOLD implementation'''
        history = {'train_loss': [],'test_loss':[],'pearsoncorrArr':[],'spearmancorrArr':[],'R_squareArr':[]}

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(crossvalidator_dataset)))):

            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader_Kfold = DataLoader(crossvalidator_dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader_Kfold = DataLoader(crossvalidator_dataset, batch_size=batch_size, sampler=test_sampler)
            

            
            trainLoss = []
            testLoss  = []
            pearsoncorrArr = []
            spearmancorrArr = []
            R_squareArr = []


            lossfun,optimizer                = trainparter.OptandLoss(model,learning_rate,momentum,L2lambda,OptimizerKfold,lossfuntype)
            
            trainLoss,testLoss,theBestModel,pearsoncorrArr,spearmancorrArr,R_squareArr  = trainparter.trainTheModel(model,num_epochs,lossfun,optimizer,train_loader_Kfold,test_loader_Kfold,trainLoss,testLoss,pearsoncorrArr,spearmancorrArr,R_squareArr)
            
            print('trainLoss in this Fold ',trainLoss)
            print('testLoss in this Fold ',testLoss)
            print("Training Loss , Test Loss of last epoch ", trainLoss[-1],testLoss[-1])
            
            
            history['train_loss'].append(trainLoss)
            history['test_loss'].append(testLoss)
            history['pearsoncorrArr'].append(pearsoncorrArr)
            history['spearmancorrArr'].append(spearmancorrArr)
            history['R_squareArr'].append(R_squareArr)
      
          
                


        avg_train_loss = np.mean(history['train_loss'])
        avg_test_loss  = np.mean(history['test_loss'])


        print('Performance of {} fold cross validation'.format(k))
        print("Average Training Loss, Average Testing Loss  ",avg_train_loss,avg_test_loss) 
        
        return history['train_loss'],history['test_loss'], avg_train_loss,avg_test_loss,model,history['pearsoncorrArr'],history['spearmancorrArr'],history['R_squareArr']
















