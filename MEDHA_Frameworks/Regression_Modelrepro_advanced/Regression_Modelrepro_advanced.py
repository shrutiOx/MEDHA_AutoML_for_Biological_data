# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:00:28 2023

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

from MEDHA.ModelReproduction.Regression.Advanced.ModelReproductionAdvancedReg import ModelReproduction
from torch.utils.data import DataLoader,TensorDataset
from MEDHA.AutoTool.Regression.Advanced.SemiManualDart_test import SemiManualDart_test
from MEDHA.DataProcessing.Regression.Advanced.DataProcessorTest  import DataPreprocessTest

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

reproobj = ModelReproduction(parameterlist='ParameterList_reg_adv_RBS_sm.pkl',
seqtype = 'dna',
numchannels=1,
max_length_of_trainseq=17,
statedict_modelDart='Best_adv_reg_RBS_sm_modeldart.pt',
statedict_modelFold='Best_adv_reg_RBS_sm_modelkfold.pt',)


modelfinal,use_Net = reproobj.GetModel()


datapathtest = 'hollerer_rbs_test_adv.csv'



testobj = DataPreprocessTest(datacsv          = datapathtest,
                             inslicestart     = 1,
                             insliceend       = 2,
                             concatslicestart = 3,
                             concatsliceend   = 7,
                             outslicestart    = 2,
                             outsliceend      = 3,
                             indstart         = 0,
                             indend           = 1,
                             numchannels      = 1,
                             seqtype          = 'dna')


test_Dataset,test_loaderF,sampledataT,dataT,labelsT,condataT,indexofdata,labelnames = testobj.GetDataConcat()





testobject = SemiManualDart_test()
lossesKL = testobject.predict(rawdata=indexofdata,
                                       labelnames=labelnames,
                                       test_loaderF=test_loaderF,
                                       modeldart=modelfinal,#using the model obtained from k-fold validation
                                       modelkfold=use_Net,
                                       concatflagT=True,
                                       condataT=condataT,
                                       lossfun='mse',
                                       needinexcel='no'
                                       )