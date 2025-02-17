# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 01:15:07 2023

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:35:04 2023

@author: ADMIN
"""




'''import libraries'''

import numpy as np
import pandas as pd
import torch.nn as nn

import torch
import random
import pickle

from MEDHA.ModelReproduction.Regression.Simple.ModelReproductionSimpleReg import ModelReproduction
from torch.utils.data import DataLoader,TensorDataset
from MEDHA.AutoTool.Regression.Simple.SemiManualDart_test import SemiManualDart_test
from MEDHA.DataProcessing.Regression.Simple.DataProcessorTest  import DataPreprocessTest

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

reproobj = ModelReproduction(parameterlist='ParameterList.pkl',
seqtype = 'dna',
mainmodel_statedict='Best_Semi_Manual.pt',
numchannels=1,
max_length_of_trainseq=17)


modelfinal = reproobj.GetModel()


datapathtest = "hollerer_rbs_test.csv"



testobj = DataPreprocessTest(datacsv=datapathtest,
                             inslicestart = 1,
                             insliceend = 2,
                             outslicestart = 2,
                             outsliceend = 3,
                             indstart = 0,
                             indend = 1,
                             seqtype = 'dna',
                             numchannels = 1)




test_Dataset,test_loaderF,sampledataT,dataT,labelsT,indexofdata,labelnames = testobj.GetData()



test_loaderF=DataLoader(test_Dataset,batch_size=10, shuffle=False, drop_last=False)

testobject = SemiManualDart_test()
lossesKL = testobject.predict(rawdata=indexofdata,
                                       labelnames=labelnames,
                                       test_loaderF=test_loaderF,
                                       model=modelfinal,#using the model obtained from k-fold validation
                                       lossfun='mse',
                                       needinexcel='no'
                                       )
