
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 03:21:52 2023

@author:  SHRUTI SARIKA CHAKRABORTY
"""
import torch
from MEDHA.DataProcessing.Regression.Simple.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Regression.Simple.DataProcessorTest  import DataPreprocessTest
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
from MEDHA.AutoTool.Regression.Simple.HPO_DART import HPO_DART
from MEDHA.AutoTool.Regression.Simple.SemiManualDart_test import SemiManualDart_test
from torch.utils.data import DataLoader,TensorDataset
import pickle 


print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''Step - 1'''
'''get my train data'''

datapathtrain = 'hollerer_rbs_train.csv'

datapathtest = 'hollerer_rbs_test.csv'

dataobj =  DataPreprocessTrain(datacsv=datapathtrain,
                               inslicestart = 0,
                               insliceend = 1,
                               outslicestart = 1,
                               outsliceend = 2,  
                               #customalphabet = ['A',
                               #                  'G',
                               #                  'C'	,
                               #                  'T',
                               #                  '56',
                               #                  '49',
                               #                  '42'],
                               #customscheme = 7,
                               numchannels = 1,
                               seqtype = 'DNA')


whole_Dataset,whole_loader,sampledata,data,labels= dataobj.GetData()

'''Step - 2'''
'''Now calling HPO-DART-Tuner. Note that you do not need to pass space in parameter unless you want to configure it. In that case please follow the exact notations and structure'''



myHpoObject = HPO_DART(
sample_data=sampledata,
in_channel=1,
kernel=[1,3,5],
outchannel=1,
dataSet=whole_Dataset,
lossfun='mse',
batch_size=1000,
acc_thresold=1,
space= { 
            'out_channel_input': hp.choice('out_channel_input',[25,100,125]),
            'out_channel_f': hp.choice('out_channel_f',[25,50]),
            'actfun': hp.choice('actfun',["ReLU6", "ReLU"]),
            'drop': hp.uniform('drop', 0.1,0.3),
            'UnitFCN_vars': hp.choice('UnitFCN_vars',[25,50]),
            'nLayers_vars': hp.uniform('nLayers_vars', 1,2),
            'loop': hp.uniform('loop', 1,2),
            'num_epochDART': hp.uniform('num_epochDART',3,5)
        },)


loss,modelfinal,space,ParameterList,pearsoncorrArr,spearmancorrArr,R_squareArr = myHpoObject.Calling_HPO_DART(max_evals=4,stoppage=3)

print('pearsoncorrArr :',pearsoncorrArr)
print('spearmancorrArr :',spearmancorrArr)
print('R_squareArr :',R_squareArr)
print('BEST Loss :',loss)

'''Step - 3'''
'''save and load ParameterList as dictionary'''



with open('ParameterListHPODART.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterListHPODART.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    
'''Step - 4'''
'''getting test data'''

testobj = DataPreprocessTest(datacsv=datapathtest,
                             inslicestart = 1,
                             insliceend = 2,
                             outslicestart = 2,
                             outsliceend = 3,
                             indstart = 0,
                             indend = 1,
                             #customalphabet = ['A',
                             #                  'G',
                             #                  'C'	,
                             #                  'T',
                             #                  '56',
                             #                  '49',
                             #                  '42'],
                             #    customscheme = 7,
                                 seqtype = 'dna',
                                 numchannels = 1)



test_Dataset,test_loaderF,sampledataT,dataT,labelsT,indexofdata,labelnames = testobj.GetData()

'''Step - 5'''
'''Now to test the model performance on an independent test-set'''



test_loaderF=DataLoader(test_Dataset,batch_size=100, shuffle=False, drop_last=False)


testobject = SemiManualDart_test()



lossesKL = testobject.predict(rawdata=indexofdata,
                                       labelnames=labelnames,
                                       test_loaderF=test_loaderF,
                                       model=modelfinal,#using the model obtained from k-fold validation
                                       lossfun='mse',
                                       needinexcel='no'
                                       )






'''saving best model'''

torch.save(modelfinal.state_dict(),'Best_Auto_HPOdartRBS.pt')
print('saved the Best_Auto model')