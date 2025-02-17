# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 03:07:58 2023

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

from MEDHA.ModelReproduction.Classification.Advanced.ModelReproductionAdvancedClass import ModelReproduction
from torch.utils.data import DataLoader,TensorDataset
from MEDHA.AutoTool.Classification.Advanced.SemiManualDart_test import SemiManualDart_test
from MEDHA.DataProcessing.Classification.Advanced.DataProcessorTest  import DataPreprocessTest

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

reproobj = ModelReproduction(parameterlist='ParameterList_class_adv_01.pkl',
seqtype = 'proteinpadded',
statedict_modelDart='modeldart_semi_class.pt',
statedict_modelFold='modelkfold_semi_class.pt',
numchannels=1,
max_length_of_trainseq=350)




modelfinal,usenet = reproobj.GetModel()


datapathtest = "concatenated_preacr_test_data.csv"




testobj = DataPreprocessTest(datacsv          = datapathtest,
                             inslicestart     = 1,
                             insliceend       = 2,
                             concatslicestart = 3,
                             concatsliceend   = 1603,
                             outslicestart    = 2,
                             outsliceend      = 3,
                             indstart         = 0,
                             indend           = 1,
                             numchannels      = 1,
                             seqtype          = 'proteinpadded')



test_Dataset,test_loaderF,sampledataT,dataT,labelsT,condataT,indexofdata = testobj.GetDataConcat()



testobject = SemiManualDart_test()



accuracy,precision,recall,f1_score,MCC = testobject.predict( test_loaderF,
modeldart    = modelfinal,
modelkfold   = usenet,
concatflagT  = True,
condataT     = condataT,
indexT       = indexofdata,
labelsT      = labelsT,
resulttype   = 'bceranking',
indtruestart = 0,
indtrueend   = 176
)

