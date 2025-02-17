

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


import MEDHA.AutoTool.Classification.Advanced.Block_CNN_usableBN  

from MEDHA.AutoTool.Classification.Advanced.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Classification.Advanced.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Classification.Advanced.DartTrainer   import DartTrainer

from MEDHA.AutoTool.Classification.Advanced.DNNClassBCE_2 import DNNClassBCEFunc
from MEDHA.AutoTool.Classification.Advanced.CNNLSTM import CNNLSTM

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
                   lossfuntype = 'bce',
                   threshold=2):




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
                              outchannel)#bayesian
        
        #print('model_space ', model_space)
        


        '''Now you again need to retrain the DARTs-derived model on your training and test set as per nni tutorial-https://nni.readthedocs.io/en/latest/tutorials/darts.html'''


        DARTobj = DartTrainer()

        final_model,exported_arch,nas_modules,DARTacc = DARTobj.DARTTrain(modelspace = model_space,
                                                            dataloader          = dart_dataset,
                                                            learning_rate       = learning_rateDart,
                                                            moment              = momentumDart,
                                                            L2lambda            = L2lambdaDart,
                                                            optimizerset        = OptimizerDart,
                                                            lossfun             = lossfuntype,
                                                            epochs              = num_epochs,
                                                            batches             = batch_size,
                                                            threshold           = threshold)

        
        model = final_model
        model.to(device)
        
        createlist = {"out_channel_input : ":[],"out_channel_f : ":[],"drop : ":[],
                      "UnitFCN_vars : ":[],"nLayers_vars : ":[],"loop : ":[],
                      "pool_size : ":[],"actfun : ":[],"num_epochs : ":[],
                      "OptimizerDart : ":[],"in_channel : ":[],"kernel : ":[],
                     "outchannel : ":[],"batch_size : ":[],"learning_rateDart : ":[],"L2lambdaDart : ":[],
                     "momentumDart : ":[],"lossfuntype : ":[],"chooseblocks : ":[],"modeltypeflag : ":[],"UnitFCN_vars : ":[],
                     "nLayers_vars : ":[],"loop : ":[],"unitsdnn : ":[],"layersdnn : ":[],"n_hiddenLSTM : ":[],"nLSTMlayers : ":[],"input_param : ":[]}
        

        
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
    
    
    def Concater(self,whole_Dataset,model,concatflag,condata=None, modeltypeflag='dnn',
                 out_param=0,nUnits=50,nLayers=1,nLSTMlayers=1,n_hiddenLSTM=30,losstype=None,createlist=None):
        '''Now we get the trained vector derived from the DART trained model by predicting with the model'''
        
        whole_loader = DataLoader(whole_Dataset,batch_size=1, shuffle=False, drop_last=False)
        model = model.cpu()
        myconvl=[]
        myy=[]

        for Xtv,ytv in whole_loader:
            Xtv = Xtv.cpu()
            ytv = ytv.cpu()
            
            y_hat = model(Xtv,True)
            myConvV = model.givememyx(Xtv)
            myconvl.append(myConvV)
            myy.append(ytv)

        
        listdiv2=[]
        listdiv=[]

        
        print('undergoing concat phase part 1')
        for m in myconvl:
            for j in m:
                listdiv.append(j.tolist())
        for n in myy:
            for i in n:
                listdiv2.append(i.tolist())


        print('undergoing concat phase part 2')        
        myConvVector = torch.tensor(listdiv)
        myf = torch.tensor(listdiv2)   

        
        '''myConvVector is important->we will use this'''

        myConvVector = myConvVector.to(device)
        
        print(myConvVector.shape)
        print(condata.shape)

        if concatflag == True:
            ConcatenatedData = torch.cat((myConvVector,condata), dim=1)
        else:
            ConcatenatedData = myConvVector
    

        print('ConcatenatedData type ', ConcatenatedData.shape)
        print('labels type ', type(myf))
        print('myConvVector shape ', myConvVector.shape)
        print('labels shape ', myf.shape)

        """Now to put all into DNN/LSTM, so create torch dataset"""

        datatrainall       = ConcatenatedData.to(device)
        truelabeltrainall  =  myf.to(device)
        Concat_dataset = TensorDataset(datatrainall,truelabeltrainall)
        
        '''Now based on chosen model-type we will get the corresponding objects'''
        
        input_param = datatrainall.shape[1]
        out_param   = out_param
        
        if modeltypeflag.lower() == 'dnn':
            nUnits      = nUnits
            nLayers     = nLayers 
            '''object creation'''
            DNN_Net = DNNClassBCEFunc(input_param,
                                      nUnits,
                                      nLayers,
                                      out_param,
                                      'ReLU',
                                      True,
                                
                                      )    
            DNN_Net = DNN_Net.to(device)
            
            usenet = DNN_Net
            createlist["input_param : "].append(input_param)
            createlist["unitsdnn : "].append(nUnits)
            createlist["layersdnn : "].append(nLayers)
            createlist["modeltypeflag : "].append(modeltypeflag)

            #print('am here')
        elif modeltypeflag.lower() == 'lstm':
            nLSTMlayers        = nLSTMlayers 
            n_hiddenLSTM       = n_hiddenLSTM

            LSTMnet = CNNLSTM(
                         input_param,
                         out_param, 
                         'ReLU',
                         nLSTMlayers,
                         n_hiddenLSTM )
            LSTMnet = LSTMnet.to(device)
            
            usenet = LSTMnet
            createlist["input_param : "].append(input_param)
            createlist["nLSTMlayers : "].append(nLSTMlayers)
            createlist["n_hiddenLSTM : "].append(n_hiddenLSTM)
            createlist["modeltypeflag : "].append(modeltypeflag)

            #print('not here')
            
        else:
            print('Wrong model type passed')
        
        return usenet,Concat_dataset
        



    
    
    def KFoldCrossValidator(self,
                            k=5,
                            crossvalidator_dataset=None,
                            batch_size=10,
                            model=None,
                            learning_rate     = 0.006,
                            L2lambda          = 0.00002 ,
                            momentum          = 0.6,
                            OptimizerKfold = 'Adam',
                            lossfuntype='mse',
                            num_epochs=5,
                            predtype='binary'):
        
        splits=KFold(n_splits=k,shuffle=True)
        foldperf={}
        dftest = []
        
        '''creating an object of class TrainerandOptimizer'''
        trainparter = TrainerandOptimizer()

        
        '''K-FOLD implementation'''
        history = {'train_loss': [],'train_acc':[],'validation_acc':[],'validation_loss':[]}

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(crossvalidator_dataset)))):

            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader_Kfold = DataLoader(crossvalidator_dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader_Kfold = DataLoader(crossvalidator_dataset, batch_size=batch_size, sampler=test_sampler)
            

            

            trainAcc      = []
            SkAcc         = []
            losses        = []
            valloss       = []

            lossfun,optimizer = trainparter.OptandLoss(model,learning_rate,momentum,L2lambda,OptimizerKfold,lossfuntype)

            trainAcc,validacc,trainlosses, BestModel,valloss = trainparter.trainTheModel(model,num_epochs,lossfun,optimizer,train_loader_Kfold,test_loader_Kfold,trainAcc,SkAcc,losses,valloss,predtype)

            
            print("Training Loss,  Training Acc , Test Acc of last epoch of this fold ", trainlosses[-1],trainAcc[-1],validacc[-1])
            history['train_loss'].append(trainlosses)
            history['train_acc'].append(trainAcc)
            history['validation_acc'].append(validacc)
            history['validation_loss'].append(valloss)
            



        avg_train_loss = np.mean(history['train_loss'])
        avg_train_acc = np.mean(history['train_acc'])
        avg_test_acc = np.mean(history['validation_acc'])
        avg_val_loss = np.mean(history['validation_loss'])


        print('Performance of {} fold cross validation'.format(k))
        print("Average Training Loss, Average Training Acc, Average validation Acc ",avg_train_loss,avg_train_acc,avg_test_acc) 
        
        return history['train_loss'],history['train_acc'],history['validation_acc'], avg_train_loss,avg_train_acc,avg_test_acc,model,avg_val_loss
















