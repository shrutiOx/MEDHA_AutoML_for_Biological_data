# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:39:59 2023

@author: ADMIN
"""

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

import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,DenseGCNConv,DenseGINConv,DenseGraphConv,DenseGATConv,dense_mincut_pool
from torch_geometric.nn import BatchNorm,GraphNorm,InstanceNorm,LayerNorm
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 
import scipy.stats as stats
import sklearn.metrics as skm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
from MEDHA.AutoToolGraph import dartsgraph
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import torch_geometric.transforms as T 
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset
print('All imports completed')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#F.nll_loss(out, data.y.view(-1)).item()
#device = torch.device("cpu" )

class DartTrainer():

  def __init__(self):
    super().__init__()
  def DARTTrain(   self,modelspace,
                   train_loader,
                   val_loader,
                   learning_rate=0.00018,
                   moment=0,
                   L2lambda=0.0001,
                   optimizerset='Adam',
                   epochs=10,
                   batches=5):


        self.saveaccuracy=[]
        def dartaccuracy(yHat, y):
                    

            
                correct=0
                y=y.view(-1)

                batch_size = y.size(0)
                pred = yHat.max(dim=1)[1]  # Use the class with highest probability.

                correct += int((pred == y).sum())
                self.saveaccuracy.append(100*(correct/ batch_size))
                return {"acc1": 100*(correct/ batch_size)}


        '''criterion for DARTs'''     


        criterion = nn.CrossEntropyLoss()


        '''optimizer for DARTs'''
        
        if (optimizerset  == 'SGD') or (optimizerset  == 'RMSprop'): 
                    optifun   = getattr(torch.optim,optimizerset ) 
                    optimizer = optifun(modelspace.parameters(),lr=learning_rate,momentum=moment,weight_decay=L2lambda)
        else:
                    optifun   = getattr(torch.optim,optimizerset ) 
                    optimizer = optifun(modelspace.parameters(),lr=learning_rate,weight_decay=L2lambda)

        print("Starting DARTS")


        trainer1 = dartsgraph.DartsTrainerGraph(
                        model=modelspace.to(device),
                        loss=criterion,
                        metrics=lambda yHat, y: dartaccuracy(yHat, y),
                        optimizer=optimizer,
                        num_epochs=epochs,
                        train_loader=train_loader,
                        test_loader=val_loader,
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
    
    
