# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:23:12 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""



# import libraries
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper



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
import MEDHA.AutoTool.Regression.Simple.Block_CNN_usableBN  as Block_CNN_usableBN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
 #self.conv1 = nn.Conv2d( 1,10,kernel_size=5,stride=1,padding=1)
 
 



def Block_Caller(
                 in_channel ,  
                 pool_size,      
                 kernels,
                 out_channel_input,
                 out_channel_i,
                 out_channel_i2,
                 out_channel_f,
                 increment,
                 num_conv_layers,
                 actfun,
                 drop
                 ):
    
    inpbblock = Block_CNN_usableBN.InputBlock(in_channel,kernels, out_channel_input,pool_size,actfun, drop)
    out_channel_inputlayer = inpbblock.outC()
    
    block1 = Block_CNN_usableBN.Block1(out_channel_inputlayer,
                    kernels, 
                    out_channel_i,
                    out_channel_i2,
                    out_channel_f,
                    pool_size,
                    actfun,drop)
    block2 = Block_CNN_usableBN.Block2(out_channel_inputlayer,kernels, out_channel_f,pool_size,actfun,drop)
    
    block3 = Block_CNN_usableBN.Block3(out_channel_inputlayer,kernels, out_channel_f,pool_size,actfun,drop)
    
    block4 = Block_CNN_usableBN.Block4(out_channel_inputlayer,kernels, out_channel_f,pool_size,actfun,drop)
    
    block5 = Block_CNN_usableBN.Block5(out_channel_inputlayer,kernels, out_channel_f,pool_size,actfun,drop)
    
    block6 = Block_CNN_usableBN.Block6(out_channel_inputlayer,increment,num_conv_layers,kernels, out_channel_f,pool_size,actfun,drop)
    
    block7 = Block_CNN_usableBN.Block7(out_channel_inputlayer, out_channel_f,actfun,drop,pool_size)
    
    #block8 = Block_CNN_usable.Block8(out_channel_inputlayer,out_channel_i,increment,num_conv_layers,kernels, out_channel_f,pool_size,actfun,drop)
    
    out_channel_final = block7.outC()
    


    
    return inpbblock,out_channel_inputlayer,block1,block2,block3,block4,block5,block6,block7,out_channel_final
          
    

    
@model_wrapper
class CNNModelSpace(nn.Module):
  def __init__(self,
               sample_data, #user only
                    in_channel ,  
                    pool_size,      
                    kernels,
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
                    chooseblocks=['block1','block2','block3','block4','block5','block6','block7'],
                    outparams=0,
                    losstype='mse'):#bayesian
      
    super().__init__()
    self.cnnlayers = nn.ModuleDict()
    self.actfun = actfun
    self.UnitFCN_vars = UnitFCN_vars
    self.nLayers_vars = nLayers_vars
    self.losstype = losstype
    
    inpbblock,out_channel_inputlayer,block1,block2,block3,block4,block5,block6,block7,out_channel_final = Block_Caller( 
                                                                              in_channel ,  
                                                                              pool_size,      
                                                                              kernels,
                                                                              out_channel_input,
                                                                              out_channel_i,
                                                                              out_channel_i2,
                                                                              out_channel_f,
                                                                              increment,
                                                                              num_conv_layers,
                                                                              actfun,
                                                                              drop
                                                                              )  
    
    
    self.loop = loop 
    self.out_channel_final = out_channel_final
    self.out_channel_inputlayer = out_channel_inputlayer
    blocklist = []
    for i in chooseblocks:
        if i == 'block1':
            blocklist.append(block1)
        if i == 'block2':
            blocklist.append(block2)
        if i == 'block3':
            blocklist.append(block3)
        if i == 'block4':
            blocklist.append(block4)
        if i == 'block5':
            blocklist.append(block5)
        if i == 'block6':
            blocklist.append(block6)
        if i == 'block7':
            blocklist.append(block7)
        
    
    self.cnnlayers['inputlayer']   = inpbblock
    for i in range(loop):
        
        #self.cnnlayers['MAIN']         =  nn.LayerChoice([block1,block2,block3,block4,block5,block6,block7])
        
        self.cnnlayers['MAIN']         =  nn.LayerChoice(blocklist)
        
        self.batchnorm_inp                   = nn.BatchNorm2d(self.out_channel_inputlayer)
        self.batchnorm_main                  = nn.BatchNorm2d(self.out_channel_final)
    
        self.remlist = []
    
        for i in range(0,3 ): 
          self.cnnlayers[f'LastLayer{i}']  = nn.Conv2d(out_channel_final,
                                                      out_channel_inputlayer,
                                                      kernels[i],
                                                      padding  =  ((kernels[i]-1)//2))
        self.remlist.append(self.cnnlayers[f'LastLayer{i}'])
          
        '''make layer choice from three kinds of kernels for Block1_1'''
        self.cnnlayers['RemLayer']           =  nn.LayerChoice(self.remlist)


    
    
    SS = sample_data
    SS.to(device)
    
    
    self.cnnlayers = self.cnnlayers.to(device)
    
    actfun = getattr(torch.nn,self.actfun)
    actfun = actfun()
    
    SS = actfun( self.cnnlayers['inputlayer'](SS))
    for i in range(self.loop):

        
        SS = actfun( self.cnnlayers['MAIN'](SS))
        SS = actfun( self.cnnlayers['RemLayer'](SS))

    SUnits = SS.shape.numel()/SS.shape[0]
    SS = SS.view(-1,int(SUnits))
    numAr = SS.shape
    
    size_of_input = numAr[1]*numAr[0]
    #print(size_of_input)
    '''This is where MLP starts'''
    
    self.cnnlayers['inputFCN']         = nn.Linear(size_of_input,self.UnitFCN_vars)
    for i in range(self.nLayers_vars):                                                    
      self.cnnlayers[f'hidden{i}']  = nn.Linear(self.UnitFCN_vars,self.UnitFCN_vars)  
    self.cnnlayers['outputFCN']        = nn.Linear(self.UnitFCN_vars,outparams)
    
  def forward(self,x,flag=False):
      
      x = x.to(device)
      actfun = getattr(torch.nn,self.actfun)
      actfun = actfun()
      
      x = actfun( self.cnnlayers['inputlayer'](x))
      x = (self.batchnorm_inp(x))
      
      for i in range(self.loop):
          #print('AM HERE')
          
          x = actfun( self.cnnlayers['MAIN'](x,True))
          x = (self.batchnorm_main(x))
          x = actfun( self.cnnlayers['RemLayer'](x))

      nUnits = x.shape.numel()/x.shape[0]
     # print('nUnits', nUnits)
      x = x.view(-1,int(nUnits))
      savethisx = x
      self.savethisx = savethisx

      '''' fully-connected layer'''


      x = actfun( self.cnnlayers['inputFCN'](x) ) 
      for i in range(self.nLayers_vars):
        x = actfun( self.cnnlayers[f'hidden{i}'](x) ) 
      x =  self.cnnlayers['outputFCN'](x)
      if (self.losstype=='div'):
          x = torch.nn.functional.log_softmax(x,dim=1)

 
      
      return x
  
  def givememyx(self,x):

        return self.savethisx.detach()
        #print(self.savethisx.shape)
  
  def Xavier_init(self,objectM):
            for p in objectM.named_parameters():
                if 'weight' in p[0]: 
                    if len(p[1].shape) < 2:
                        nn.init.xavier_normal_(p[1].unsqueeze(0))
                    else:
                        nn.init.xavier_normal_(p[1].data)
            return objectM
      
                                                   
 
                                                   
                                                   

    
  

    


