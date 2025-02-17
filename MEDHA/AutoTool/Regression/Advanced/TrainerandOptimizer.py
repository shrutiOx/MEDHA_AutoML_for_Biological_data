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
import MEDHA.AutoTool.Regression.Simple.Block_CNN_usableBN  
from scipy.stats import spearmanr
from sklearn.metrics import r2_score 
from scipy.stats import pearsonr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class TrainerandOptimizer():
  def __init__(self):
    super().__init__()
  def OptandLoss(self,objectM,learning_rate,moment,L2lambda,optimizerset,typeopt):
  # loss function
      if typeopt == 'mse':
          lossfun = nn.MSELoss() 
      elif typeopt == 'div':
          lossfun = nn.KLDivLoss(reduction="batchmean")  
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


  def trainTheModel(self,objectM,numepochs,lossfun,optimizer,train_loader,test_loader,trainLoss,testLoss,pearsoncorrArr,spearmancorrArr,R_squareArr):

  # New! initialize a dictionary for the best model
    theBestModel = {'loss':0, 'net':None}
    

    
  # loop over epochs
    for epochi in range(numepochs):
        
    # switch on training mode
        objectM.train()

        losssum = []
        kldivsum = []
        for X,y in train_loader: #mini batches train_loader
        
            X = X.to(device)
            y = y.to(device)
            
            '''predicting yHat'''
            
            yHat = objectM(X)
            #print('yHat : ',yHat.shape)
            #print('y : ',y.shape)
            
            '''predicting loss from yHat and y'''
            
            loss = lossfun(yHat,y)
            
            '''applying optimizer'''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            yHat = yHat.cpu()
            y    = y.cpu()
            losssum.append(loss.item())
            
            #yget=(np.exp(yHat.detach().cpu()))
            #yget = yget.cpu()
            #kldiv = kl_divergence(y,yget)
            #print('kldiv_accuracy ',kldiv)
            #kldivsum.append(kldiv.item())

            
        
       # trainLoss[epochi] = np.mean(loss.item())
        
        #mean testing-loss of all batches 
        
        
        losssum = np.array(losssum)
        
        '''getting mean of the losses'''
        
        trainLoss.append(np.mean(losssum))
        
        
        #print('train loss', losssum)
       # print('trainLoss array ',trainLoss)
        
        #kldivsum = np.array(kldivsum)
        #kldivtrain.append(np.mean(kldivsum))
        #print('KL-DIV acc', kldivsum)

          
    # end of batch loop...

    ### test accuracy


        objectM.eval()
        
        Xt,yt = next(iter(test_loader)) #using dev-set 
        Xt = Xt.to(device)
        yt = yt.to(device)
              
        with torch.no_grad():
            
            '''predicting test labels'''                                     
            predlabels =  objectM(Xt)
            predlabels = predlabels.cpu()
            yt         = yt.cpu() 
            
            '''predicting losses'''
            
            losstest = lossfun(predlabels,yt) 
            #print('yt[1].shape ',yt[1].shape)
            if yt[1].shape == torch.Size([1]):
                #print('predlabels ',predlabels.detach().cpu())
                #print('yt ',yt)
                #print('Calculating Spearman and Pearson R for regression when target is 1 coloumn')
               # print((predlabels.detach().cpu()))
               # print((yt))
                #pearsoncorr  = pearsonr((predlabels.detach().cpu()), (yt))
                pearsoncorr  = pearsonr(torch.flatten(predlabels.detach().cpu()), torch.flatten(yt))
                spearmancorr = spearmanr(torch.flatten(predlabels.detach().cpu()), torch.flatten(yt))
                R_square = r2_score(predlabels, yt) 
                
                
                
               # print('Pearsoncorrelation for validation set is : ',pearsoncorr)
               # print('Spearmancorrelation  for validation set is : ',spearmancorr)
               # print('R2 value for validation set is : ',R_square)
                

        
        
        '''getting mean of the losses'''
        testLoss.append(np.mean(losstest.item()))
        if yt[1].shape == torch.Size([1]):
            pearsoncorrArr.append(pearsoncorr)
            spearmancorrArr.append(spearmancorr)
            R_squareArr.append(R_square)
        else:
            pearsoncorrArr=[]
            spearmancorrArr=[]
            R_squareArr=[]
            
        print('epochi',epochi)
        print('mean trainLoss ',trainLoss)
        print('mean validation-loss ',testLoss)
        

        
        # New! Store this model if it's the best so far -- here is the updated code    
        if testLoss[-1]<theBestModel['loss']:
      
        # new best accuracy (least loss)
            theBestModel['loss'] = testLoss[-1].item()
      
        # model's internal state
            theBestModel['net'] = copy.deepcopy( objectM.state_dict() ) #copying model's state 
        

    #mean testing-loss of all batches per epoch

    
    return trainLoss,testLoss,theBestModel,pearsoncorrArr,spearmancorrArr,R_squareArr

  def Xavier_init(self,objectM):
    for p in objectM.named_parameters():
        if 'weight' in p[0]: 
            if len(p[1].shape) < 2:
                nn.init.xavier_normal_(p[1].unsqueeze(0))
            else:
                nn.init.xavier_normal_(p[1].data)
    return objectM