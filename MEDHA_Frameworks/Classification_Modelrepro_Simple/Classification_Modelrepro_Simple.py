# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:15:44 2023

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

from MEDHA.ModelReproduction.Classification.Simple.ModelReproductionSimpleClass import ModelReproduction
from torch.utils.data import DataLoader,TensorDataset
from MEDHA.AutoTool.Classification.Simple.SemiManualDart_test import SemiManualDart_test
from MEDHA.DataProcessing.Classification.Simple.DataProcessorTest  import DataPreprocessTest

print('imports complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

reproobj = ModelReproduction(parameterlist='ParameterList_class_simple_01.pkl',
seqtype = 'proteinpadded',
mainmodel_statedict='Best_Semi_Manual_class_simple_01.pt',
numchannels=1,
max_length_of_trainseq=1370)


modelfinal = reproobj.GetModel()


datapathtest = "test_data_deepacr.csv"



testobj = DataPreprocessTest(datacsv=datapathtest,
                             inslicestart = 1,
                             insliceend = 2,
                             outslicestart = 2,
                             outsliceend = 3,
                             indstart = 0,
                             indend = 1,
                                 seqtype = 'proteinpadded',
                                 numchannels = 1)



test_Dataset,test_loaderW,sampledataT,dataT,labelsT,indexofdata = testobj.GetData()




test_loaderF=DataLoader(test_Dataset,batch_size=5, shuffle=False, drop_last=False)

testobject = SemiManualDart_test()



skAccScoreT,preT,reT,f1scoreT,MCCT= testobject.predict( test_loaderF,
 modelfinal,
 indexofdata,
 labelsT,
 resulttype='bceranking',
 indtruestart=0,
 indtrueend=177#using the model obtained from k-fold validation
)
