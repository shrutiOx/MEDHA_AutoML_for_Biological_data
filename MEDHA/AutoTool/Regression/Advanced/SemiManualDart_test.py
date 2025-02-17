

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
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import r2_score 

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
from torchvision import datasets,transforms
import torchvision.transforms as transforms
from torchmetrics.functional import kl_divergence
'''my modules'''


import MEDHA.AutoTool.Regression.Advanced.Block_CNN_usableBN  

from MEDHA.AutoTool.Regression.Advanced.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Regression.Advanced.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Regression.Advanced.DartTrainer   import DartTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''get my model'''


'''fixed params'''
out_channel_i = 25
out_channel_i2 = 50
increment = 25
num_conv_layers = 2


        
class SemiManualDart_test(nn.Module):
    def __init__(self):#a list of 2 inputs and each input should be a number
        super().__init__()
        
    def predict(self,
                rawdata,
                labelnames,
                test_loaderF,
                modeldart,#using DART derived model
                modelkfold,
                concatflagT=False,
                condataT=None,
                lossfun='div',
                needinexcel = 'no'):
        
        test_metrics  = [0]

        losses = []
        lossesKL = []
        prediction_df = pd.DataFrame()
        ypall = []


        input_data = list()
        input_data = rawdata
        input_data = list(input_data)
        input_data = ['Oligos'] + input_data
        labelnames = list(labelnames)
        prediction_df['input_data'] = (input_data)
        
        
        for col in labelnames:
            prediction_df[col] = ''

        storepredictions = []
        storepredictionsloss = []
        ypallloss = []
        storepredictions.append(labelnames)


        print('Now testing on independent test-set')

        #Xpre,ypre = next(iter(test_loaderF))
        #Xpre = Xpre.to(device)
        #ypre = ypre.to(device)
        
        '''Now we get the trained vector derived from the DART trained model by predicting with the model'''

        #y_hat_PRE = modeldart(Xpre)
        #myConvVector = modeldart.givememyx(Xpre)
        
        
        model = modeldart.cpu()
        myconvl=[]
        myy=[]
        #count1 = 0
        for Xtv,ytv in test_loaderF:
            Xtv = Xtv.cpu()
            ytv = ytv.cpu()
            
            y_hat = model(Xtv,True)
            myConvV = model.givememyx(Xtv)
            myconvl.append(myConvV)
            myy.append(ytv)
            #count1 += 1
            #print('in loop ',count1)
            
        #print(('myconvl ',myconvl))
        print('starting lists')
        
        listdiv2T=[]
        listdivT=[]
        
        #count2 = 0
        #count3 = 0
        
        print('Now testing on independent test-set PART 1')
        for m in myconvl:
            for j in m:
                listdivT.append(j.tolist())
                #count2 += 1
                #print('making pred ',count2)
        print('done')
        for n in myy:
            for i in n:
                listdiv2T.append(i.tolist())
                #count3 += 1
                #print('making y ',count3)
        
        #print('we r here now')
       # dfpred = pd.DataFrame()
       # dfpred['prediction'] = listdiv
        #dfpred['True value'] = listdiv2
                
        myConvVector = torch.tensor(listdivT)
        myf = torch.tensor(listdiv2T)   
        #myConvVector = torch.flip(myConvVector, [0, 1])
        #myf = torch.flip(myf, [1])
        
        #condata = torch.flip(condata, [0, 1])
        
        #print('outside loop')
        #print(condata)
        
        #Xtv,ytv = next(iter(whole_loader))
        #Xtv = Xtv.cpu()
        #ytv = ytv.cpu()
        #model = model.cpu()
        #y_hat = model(Xtv,True)
        #myConvVector = model.givememyx(Xtv)
        '''myConvVector is important->we will use this'''

        myConvVector = myConvVector.to(device)
        
        print(myConvVector.shape)
        print(condataT.shape) 
        
        '''myConvVector is important->we will use this'''

        if concatflagT == True:
            ConcatenatedData = torch.cat((myConvVector,condataT), dim=1)
        else:
            ConcatenatedData = myConvVector
        
        """Now to put all into DNN/LSTM, so create torch dataset"""

        datatrainall       = ConcatenatedData.to(device)
        #truelabeltrainall  =  ypre.to(device)
        truelabeltrainall  =  myf.to(device)
        Concat_dataset     = TensorDataset(datatrainall,truelabeltrainall)
        #test_loaderC       = DataLoader(Concat_dataset, batch_size=Concat_dataset.tensors[0].shape[0])
        test_loaderC       =DataLoader(Concat_dataset,batch_size=5, shuffle=False, drop_last=False)


        
        print('Now testing on independent test-set part-2')
        for Xp,yp in test_loaderC:
        #Xp,yp = next(iter(test_loaderF))

            Xp = Xp.to(device)
            yp = yp.to(device)
            predlabelsRL2 =  modelkfold(Xp)
            if yp[1].shape == torch.Size([1]):
                for i in torch.flatten(predlabelsRL2.detach().cpu()):
                    storepredictionsloss.append(i)
                    if lossfun == 'div':
                        storepredictions.append(np.exp(i.item()))
                    else:
                        storepredictions.append(i.item())
                for j in torch.flatten(yp):
                    ypall.append(j.cpu().item())
                    ypallloss.append(j.cpu())
            else:
                for i in predlabelsRL2.detach().cpu():
                      storepredictions.append(np.exp(i))
                      storepredictionsloss.append((i))


                for j in yp:
                    ypall.append(j.cpu())


        #yp = yp.cpu()
        listdiv= []
        listdiv2=[]
        listdiv3= []
        listdiv4=[]
        resultlist = {"Losses":[],"Pearsoncorrelation":[],"Spearmancorrelation":[],"R2-value":[]};
        
        
        dfpred = pd.DataFrame()
        
        if yp[1].shape > torch.Size([1]):
            for m in storepredictionsloss:
                for n in m:
                    ng = n.item()
                    listdiv.append(ng)
            
                listdiv2.append(listdiv)
                listdiv=[]
            
            for m1 in ypall:
                for n1 in m1:
                    n11 = n1.item()
                    listdiv3.append(n11)
            
                listdiv4.append(listdiv3)        
                listdiv3=[]
            
            
        if yp[1].shape == torch.Size([1]): 
            dfpred['prediction'] = storepredictions[1::]
            dfpred['True value'] = ypall
        
            dfpred['predictionloss'] = storepredictionsloss
            dfpred['True value loss'] = ypallloss

            ytorch = torch.tensor(dfpred['True value loss'])
            predtorch = torch.tensor(dfpred['predictionloss'])
            
        else:
            dfpred['predictionloss'] = listdiv2
            dfpred['True value loss'] = listdiv4
            predtorch =torch.tensor(dfpred['predictionloss'] )
            ytorch = torch.tensor(dfpred['True value loss'])
            
        #print(dfpred)
        print('shape of predicted tensor :', predtorch.shape)
        print('shape of true tensor :', ytorch.shape)
        
        if lossfun == 'div':
            #losses.append(  kl_divergence(dfpred['prediction'],dfpred['True value']))
            
            criterion = nn.KLDivLoss(reduction="batchmean")
            #yp = yp.to(device)
            lossesKL.append(  criterion(predtorch,ytorch))
            resultlist["Losses"].append(lossesKL)


            #print('KL DIV Test accuracy ',losses)
            print('KL DIV Test loss  ',lossesKL)
            if yp[1].shape == torch.Size([1]):
                print('Calculating Spearman and Pearson R for regression when target is 1 coloumn')

                #pearsoncorr  = pearsonr(torch.flatten(np.exp(predlabelsRL2.detach().cpu())), torch.flatten(yp.cpu()))
                pearsoncorr  = pearsonr(dfpred['prediction'],dfpred['True value'])
                spearmancorr = spearmanr(dfpred['prediction'],dfpred['True value']) 
                R_square = r2_score(dfpred['prediction'],dfpred['True value']) 
                
                print('Pearsoncorrelation  for test set is : ',pearsoncorr)
                print('Spearmancorrelation  for test set is : ',spearmancorr)
                print('R2 value for test set is : ',R_square)
                

                resultlist["Pearsoncorrelation"].append(pearsoncorr)
                resultlist["Spearmancorrelation"].append(spearmancorr)
                resultlist["R2-value"].append(R_square)
        else:
            criterion = nn.MSELoss() 
            yp = yp.to(device)
            lossesKL.append(  criterion(predtorch,ytorch))
            
            resultlist["Losses"].append(lossesKL)

            
            print('MSE  Test loss ',lossesKL)
            
            if yp[1].shape == torch.Size([1]):
                print('Calculating Spearman and Pearson R for regression when target is 1 coloumn')

                #pearsoncorr  = pearsonr(torch.flatten(predlabelsRL2.detach().cpu()), torch.flatten(yp.cpu()))
                #spearmancorr = spearmanr(torch.flatten(predlabelsRL2.detach().cpu()), torch.flatten(yp.cpu()))
                #R_square = r2_score(torch.flatten(predlabelsRL2.detach().cpu()), torch.flatten(yp.cpu())) 
                
                pearsoncorr  = pearsonr(dfpred['prediction'],dfpred['True value'])
                spearmancorr = spearmanr(dfpred['prediction'],dfpred['True value']) 
                R_square = r2_score(dfpred['prediction'],dfpred['True value']) 
                
                print('Pearsoncorrelation  for test set is : ',pearsoncorr)
                print('Spearmancorrelation  for test set is : ',spearmancorr)
                print('R2 value for test set is : ',R_square)
                
                resultlist["Pearsoncorrelation"].append(pearsoncorr)
                resultlist["Spearmancorrelation"].append(spearmancorr)
                resultlist["R2-value"].append(R_square)
                
                

        # open file in write mode
        
        
        with open(r'resultlist.txt', 'w') as fp:
            for key in resultlist:
                # write each item on a new line
                fp.write("%s\n" % resultlist[key])
            print(' We have saved resultlist.txt ')
            

            
        with open(r'Storepredictions.txt', 'w') as fp:
            for item in storepredictions:
                # write each item on a new line
                fp.write("%s\n" % item)
            print(' We have saved Storepredictions.txt ')

        
        if str(needinexcel).lower() == 'yes':
            with open(r'predictions.csv', 'w') as fp:
                for item in range(0,len(storepredictions)):
                    fp.write("%s," % input_data[item])
                    #fp.write("%s," % storepredictions[item][0])
                    if isinstance(storepredictions[item][0], str):
                        fp.write("%s," % storepredictions[item][0])
                    else:
                        fp.write("%s," % (storepredictions[item][0].item()))
                    for i in range(1,len(storepredictions[item])):
                        #print(storepredictions[item][i])
                        if isinstance(storepredictions[item][i], str):
                            fp.write("%s," % storepredictions[item][i])
                        else:
                            fp.write("%s," % (storepredictions[item][i].item()))
                        
                    fp.write("\n")
                print(' We have saved predictions.csv ')


        

        return lossesKL
    