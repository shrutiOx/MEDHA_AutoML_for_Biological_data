# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:26:33 2023

@author: ADMIN
"""

import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

from torchmetrics.functional import kl_divergence

import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 

import scipy.stats as stats
import sklearn.metrics as skm
# libraries for partitioning and batching the data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

from nni.retiarii.nn.pytorch import Repeat

import nni
import MEDHA.AutoTool.Classification.Advanced.Block_CNN_usableBN  


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class TrainerandOptimizer():
  def __init__(self):
    super().__init__()
  def OptandLoss(self,objectM,learning_rate,moment,L2lambda,optimizerset,typeopt):
  # loss function
      if typeopt.lower() == 'bce':
          lossfun  = nn.BCEWithLogitsLoss()
      else:
          lossfun = nn.CrossEntropyLoss() 
          #lossfun = nn.KLDivLoss(reduction="sum")  
          #lossfun = nn.KLDivLoss()  
  # optimizer
      if (optimizerset  == 'SGD') or (optimizerset  == 'RMSprop'):   
           optifun   = getattr(torch.optim,optimizerset ) 
           optimizer =  optifun(objectM.parameters(),lr=learning_rate,momentum=moment,weight_decay=L2lambda)
      else:
          optifun   = getattr(torch.optim,optimizerset )                    
          optimizer = optifun(objectM.parameters(),lr=learning_rate,weight_decay=L2lambda)
      return lossfun,optimizer 


  def trainTheModel(self,objectM,numepochs,lossfun,optimizer,train_loader,test_loader,trainAcc,skAccScore,losses,valloss,predtype='binary'):

  # New! initialize a dictionary for the best model
    theBestModel = {'Accuracy':0, 'net':None}
    
  # loop over epochs
    for epochi in range(numepochs):
        
    # switch on training mode
        objectM.train()

    # loop over training data batches
        batchAcc  = []
        batchLoss = []
        #print('lossfun ',lossfun)
        
        
        for X,y in train_loader: #mini batches train_loader
        
            X = X.to(device)
            y = y.to(device)

            yHat = objectM(X)
            loss = lossfun(yHat,y)
           # print('loss ',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            yHat = yHat.cpu()
            y = y.cpu()
            
            if predtype.lower()=='binary':
                #print('we are here binary')
                batchAcc.append(100*torch.mean(((yHat>0) == y).float()))
                batchLoss.append( loss.item() )
                #print('batchLoss ',batchLoss)
            else:
                batchAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )
                batchLoss.append( loss.item() )

          
    # end of batch loop...

    # per epoch
        trainAcc.append( np.mean(batchAcc) )
        
        losses.append( np.mean(batchLoss) )
        #print('losses ',losses)

    ### test accuracy

        skacc         = []

        objectM.eval()
        
        Xt,yt = next(iter(test_loader)) #using dev-set 
        Xt = Xt.to(device)
        yt = yt.to(device)
              
        with torch.no_grad():
                                                 
            predlabels =  objectM(Xt)
            predlabels = predlabels.cpu()
            yt = yt.cpu() 
            testloss = lossfun(predlabels,yt)    
            #print('testloss ',testloss)                                   
         
        #per epoch
        if predtype.lower()=='binary':
            #print('we are here too binary valid')
            skacc.append(100*skm.accuracy_score (yt,predlabels>0))
        else:
            skacc.append(100*torch.mean((torch.argmax(predlabels,axis=1)==yt).float()) )
            
        #print('valid-accuracy ',skacc)

        valloss.append(np.mean(testloss.item()))
        skAccScore.append(skacc[-1])

        print('epochi',epochi)
        print('mean-train-accuracy : ',trainAcc[-1])
        print('mean-test-accuracy  : ',skAccScore[-1])
        print('mean-train-loss  : ',losses[-1])
        print('mean-validation-loss  : ',valloss[-1])
        
        # New! Store this model if it's the best so far -- here is the updated code
        if skAccScore[-1]>theBestModel['Accuracy']:
      
        # new best accuracy
            theBestModel['Accuracy'] = skAccScore[-1].item()
      
        # model's internal state
            theBestModel['net'] = copy.deepcopy( objectM.state_dict() ) #copying model's state 
  
    # function output
    return trainAcc,skAccScore,losses,theBestModel,valloss

  def Xavier_init(self,objectM):
    for p in objectM.named_parameters():
        if 'weight' in p[0]: 
            if len(p[1].shape) < 2:
                nn.init.xavier_normal_(p[1].unsqueeze(0))
            else:
                nn.init.xavier_normal_(p[1].data)
    return objectM