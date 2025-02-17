# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 01:18:23 2023



@author: Shruti Sarika chakraborty
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
from  MEDHA.AutoTool.Regression.Advanced.SemiManualDart_test import SemiManualDart_test
from  MEDHA.DataProcessing.Regression.Advanced.DataProcessorTest  import DataPreprocessTest

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

reproobj = ModelReproduction(parameterlist='ParameterList_adv_Auto_01.pkl',
seqtype = 'dna',
numchannels=1,
max_length_of_trainseq=79,
statedict_modelDart='Best_modelDart_FORECAST_autoadv.pt',
statedict_modelFold='Best_modelfold_FORECAST_autoadv.pt',)


modelfinal,use_Net = reproobj.GetModel()


datapathtest = 'forecast_adv_test.csv'



testobj = DataPreprocessTest(datacsv          = datapathtest,
                             inslicestart     = 1,
                             insliceend       = 2,
                             concatslicestart = 2,
                             concatsliceend   = 3,
                             outslicestart    = 5,
                             outsliceend      = None,
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
                                       lossfun='div',
                                       needinexcel='yes'
                                       )