

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
from sklearn.metrics import matthews_corrcoef
'''my modules'''


import MEDHA.AutoTool.Classification.Advanced.Block_CNN_usableBN  

from MEDHA.AutoTool.Classification.Advanced.AutoDL_CNNspaceBN   import CNNModelSpace

from MEDHA.AutoTool.Classification.Advanced.TrainerandOptimizer   import TrainerandOptimizer

from MEDHA.AutoTool.Classification.Advanced.DartTrainer   import DartTrainer

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
                test_loaderF,
                modeldart,
                modelkfold,
                concatflagT  = False,
                condataT     = None,
                indexT       = 0,
                labelsT      = 0,
                resulttype   = 'bceranking',
                indtruestart = 0,
                indtrueend   = 176):
        
        test_metrics      = [0]
        losses            = []
        lossesKL          = []
        prediction_df     = pd.DataFrame()
        testAccRL2        = []
        test_metrics      = [0,0,0,0,0,0]

        skacc   = []
        pracc   = []
        reacc   = []
        f1score = []

        skAccScoreT =         []
        preT        =         []
        reT         =         []
        f1scoreT    =         []
        MCCT        =  []
        storepredictions =    []

        print('Now testing on independent test-set')
        
        model = modeldart.cpu()
        myconvl=[]
        myy=[]

        
        for Xtv,ytv in test_loaderF:
            Xtv = Xtv.cpu()
            ytv = ytv.cpu()
            
            y_hat = model(Xtv,True)
            myConvV = model.givememyx(Xtv)
            myconvl.append(myConvV)
            myy.append(ytv)


        
        listdiv2T=[]
        listdivT=[]

        
        
        print('Now testing on independent test-set PART 1')
        for m in myconvl:
            for j in m:
                listdivT.append(j.tolist())
        for n in myy:
            for i in n:
                listdiv2T.append(i.tolist())

                

        myConvVector = torch.tensor(listdivT)
        myf = torch.tensor(listdiv2T)   

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
        truelabeltrainall  =  myf.to(device)
        Concat_dataset     = TensorDataset(datatrainall,truelabeltrainall)
        test_loaderC       =DataLoader(Concat_dataset,batch_size=5, shuffle=False, drop_last=False)

        
        print('Now testing on independent test-set-part-2')
        

        if resulttype == 'bceranking':
            for Xp,yp in test_loaderC: #using dev-set 
                  Xp = Xp.to(device)
                  yp = yp.to(device)
                  predlabelsRL2 =  modelkfold(Xp)
                  for i in predlabelsRL2.detach().cpu():
                      storepredictions.append(i.cpu())
            
            len(storepredictions)
            dfpred               =  pd.DataFrame()
            dfpred['index']      =  (indexT).tolist()
            dfpred['prediction'] =  (storepredictions)
            dfpred['truevalue']  =  (labelsT.cpu())
            dfpred               =  dfpred.sort_values(by=['prediction'],ascending=False)
            truepos              =  0
            trueneg              =  0
            
            
            for i in dfpred['truevalue'][indtruestart:indtrueend]:
                if  i == 1.0:
                    truepos += 1
                    
            for i in dfpred['truevalue'][indtrueend::]:
                if  i == 0.0:
                    trueneg += 1
                    
            accuracy = (truepos + trueneg)/len(dfpred['index'])
            print('accuracy ', accuracy)

            allpos = 0

            for i in dfpred['truevalue'][indtruestart:indtrueend]:
                    allpos += 1
                    
            precision = truepos/allpos
            print('precision ', precision)

            falseneg = 0
            falsepos = allpos-truepos
            for i in dfpred['truevalue'][indtrueend::]:
                if  i == 1.0:
                    falseneg += 1
                    
            recall = truepos/(truepos + falseneg)
            print('recall ', recall)

            numerator = precision*recall
            denominator = precision+recall

            f1_score = 2 * (numerator/denominator)
            print('f1_score ', f1_score)
            
            
            
            MCC = ((truepos*trueneg) - (falsepos*falseneg)) / math.sqrt((truepos+falsepos)*(truepos+falseneg)*(trueneg+falsepos)*(trueneg+falseneg))
            print('MCC score ', MCC)

            print('number of true positive : ', truepos,'number of true negative : ', trueneg)
            dfpred.to_csv('out.csv')  
            print('Saving complete for bce-ranking')
            
            return accuracy,precision,recall,f1_score,MCC
        
        elif resulttype == 'bcedefault':
            for Xp,yp in test_loaderF: #using dev-set 
                  Xp = Xp.to(device)
                  yp = yp.to(device)
                  predlabelsRL2 =  model(Xp)
                  for i in predlabelsRL2.detach().cpu():
                      storepredictions.append((i.cpu()))
                      matches=predlabelsRL2.cpu()>0
                      skacc.append(100*skm.accuracy_score(yp.cpu(),matches))
                      pracc.append(100*skm.precision_score(yp.cpu(),matches))
                      reacc.append(100*skm.recall_score(yp.cpu(),matches))
                      f1score.append(100*skm.f1_score (yp.cpu(),matches))
                      MCC.append(100*matthews_corrcoef (yp.cpu(),matches)) 
            
            
            skAccScoreT.append(np.mean(skacc))
            preT.append(np.mean(pracc))
            reT.append(np.mean(reacc))
            f1scoreT.append(np.mean(f1score))
            MCCT.append(np.mean(MCC))

            print('Accuracy : ',skAccScoreT)
            print('Precision : ',preT)
            print('Recall : ',reT)
            print('F1-Score : ',f1scoreT)
            print('MCCT : ',MCCT)

            with open(r'Predictions_binarylabel_out.txt', 'w') as fp:
                for item in storepredictions:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                print('Saving complete for bce-default')
                
                
            return skAccScoreT,preT,reT,f1scoreT,MCCT
            
            
        else:         
            for Xp,yp in test_loaderC: #using dev-set 
                  Xp = Xp.to(device)
                  yp = yp.to(device)
                  predlabelsRL2 =  modelkfold(Xp)
                  for i in predlabelsRL2.detach().cpu():
                      storepredictions.append(torch.argmax(i.cpu()))
                      matches=torch.argmax(predlabelsRL2.cpu(),axis=1)
                      skacc.append(100*skm.accuracy_score(yp.cpu(),matches))
                      pracc.append(100*skm.precision_score(yp.cpu(),matches,average='micro'))
                      reacc.append(100*skm.recall_score(yp.cpu(),matches,average='micro'))
                      f1score.append(100*skm.f1_score (yp.cpu(),matches,average='micro'))
            
            
            skAccScoreT.append(np.mean(skacc))
            preT.append(np.mean(pracc))
            reT.append(np.mean(reacc))
            f1scoreT.append(np.mean(f1score))

            print('Accuracy : ',skAccScoreT)
            print('Precision : ',preT)
            print('Recall : ',reT)
            print('F1-Score : ',f1scoreT)

            with open(r'Predictions_multilabel_out.txt', 'w') as fp:
                for item in storepredictions:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                print('Saving complete for multi-label')
                
            
            return skAccScoreT,preT,reT,f1scoreT