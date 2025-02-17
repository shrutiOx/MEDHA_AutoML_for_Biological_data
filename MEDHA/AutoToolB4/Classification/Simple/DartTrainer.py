# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 01:14:03 2023

@author: ADMIN
"""

'''import libraries'''

import numpy as np
import pandas as pd

from nni.retiarii.oneshot.pytorch import DartsTrainer,EnasTrainer
import time
import torch.nn.functional as F
import copy
import torch

from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms



import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import math



from nni.retiarii.nn.pytorch import Repeat
import nni

from nni.nas.fixed import fixed_arch
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from sklearn.model_selection import KFold
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import nni.nas.fixed




import random


import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from torchmetrics.functional import kl_divergence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" )

class DartTrainer():

  def __init__(self):
    super().__init__()
  def DARTTrain(   self,modelspace,
                   dataloader,
                   learning_rate=0.00018,
                   moment=0,
                   L2lambda=0.0001,
                   optimizerset='Adam',
                   lossfun='bce',
                   epochs=10,
                   batches=5,
                   threshold=0):


        self.saveaccuracy=[]
        def dartaccuracy(threshold,yHat, y):
                    

            
            if str.lower(lossfun) == 'bce':
                #print(yHat)
                trainval_accuracydarts =  {"accuracy BCE ": 100*(torch.mean(((yHat>threshold) == y).float()))}
                
                fromhere = (100*(torch.mean(((yHat>threshold) == y).float()))).item()
                self.saveaccuracy.append(fromhere)
                #print('Accuracy_per_batch_size BCE ', fromhere)

                        
            else:
                trainval_accuracydarts={"accuracy-cross-entropy ": 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) }

                fromhere = (100*torch.mean((torch.argmax(yHat,axis=1)==y).float())).item()
                self.saveaccuracy.append(fromhere)
                #print('Accuracy_per_batch_size CSE ', fromhere)
                #print(self.saveaccuracy)
            return trainval_accuracydarts


        '''criterion for DARTs'''     

        if str.lower(lossfun) == 'bce':

            criterion = nn.BCEWithLogitsLoss()

        else:

            criterion = nn.CrossEntropyLoss()

        '''optimizer for DARTs'''
        
        if (optimizerset  == 'SGD') or (optimizerset  == 'RMSprop'): 
                    optifun   = getattr(torch.optim,optimizerset ) 
                    optimizer = optifun(modelspace.parameters(),lr=learning_rate,momentum=moment,weight_decay=L2lambda)
        else:
                    optifun   = getattr(torch.optim,optimizerset ) 
                    optimizer = optifun(modelspace.parameters(),lr=learning_rate,weight_decay=L2lambda)

        print("Starting DARTS")


        trainer1 = DartsTrainer(
                        model=modelspace.to(device),
                        loss=criterion,
                        metrics=lambda yHat, y: dartaccuracy(threshold,yHat, y),
                        optimizer=optimizer,
                        num_epochs=epochs,
                        dataset=dataloader,
                        batch_size=batches,
                        log_frequency=10,
                        workers=0,
                        device=device
                        )



        exp = RetiariiExperiment(modelspace, trainer1)
        exp.run()
        mymodel = trainer1.model.to(device)
        
        exported_arch = exp.export_top_models()
        print('exported_arch ', exported_arch)

        final_model = mymodel
        #print(self.saveaccuracy)
        DARTacc=np.mean(self.saveaccuracy)
        print('Mean accuracy after all trials from DART-Tuner : ',DARTacc)
        #print('loss ?? ' ,DARTacc)  
        nas_modules = trainer1.nas_modules
        return    final_model ,exported_arch,nas_modules,DARTacc
    
    
