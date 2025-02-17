

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 03:21:52 2023

@author:  SHRUTI SARIKA CHAKRABORTY
"""
import torch
from MEDHA.DataProcessing.Classification.Simple.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Classification.Simple.DataProcessorTest  import DataPreprocessTest
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
from MEDHA.AutoTool.Classification.Simple.HPO_DART import HPO_DART
import pickle 
from MEDHA.AutoTool.Classification.Simple.SemiManualDart_test import SemiManualDart_test
from torch.utils.data import DataLoader,TensorDataset


print('imports complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" )
#print('device here', device)
torch.cuda.empty_cache()
#torch.cuda.max_memory_allocated(device=device)





'''Step - 1'''
'''1.Get training data '''

datapathtrain = "train_data_deepacr.csv"



datapathtest = "test_data_deepacr.csv"



dataobj =  DataPreprocessTrain(datacsv=datapathtrain,
                               inslicestart = 1,
                               insliceend = 2,
                               outslicestart = 2,
                               outsliceend = 3,  
                               numchannels = 1,
                               seqtype = 'proteinpadded')


whole_Dataset,whole_loader,sampledata,data,labels= dataobj.GetData()




'''Step - 2'''
'''Now calling HPO-DART-Tuner. Note that you do not need to pass space in parameter unless you want to configure it. In that case please follow the exact notations and structure'''





myHpoObject = HPO_DART(
sample_data=sampledata,
in_channel=1,
kernel=[1,3,5],
outchannel=1,
dataSet=whole_Dataset,
lossfun='bce',
batch_size=5, 
acc_thresold=65,
space={ 
        'out_channel_input': hp.choice('out_channel_input',[50,100,125]),
        'out_channel_f': hp.choice('out_channel_f',[25,50]),
        'actfun': hp.choice('actfun',["ReLU6", "ReLU",'LeakyReLU']),
        'drop': hp.uniform('drop', 0.0,0.3),
        'UnitFCN_vars': hp.choice('UnitFCN_vars',[25,50]),
        'nLayers_vars': hp.choice('nLayers_vars', [ 1,2]),
        'loop':  hp.choice('loop', [1,2]),
        'num_epochDART': hp.choice('num_epochDART', [3,5])},
threshold=2,
predtype='binary',
L2lambdaDart=0.05)



avg_val_loss,avg_valid_acc,avg_train_acc,avg_train_loss,modelfinal,space,ParameterList = myHpoObject.Calling_HPO_DART(max_evals=5,stoppage=3)

'''Step - 3'''
'''save and load ParameterList (this contains Params used in DART-Tuner) as dictionary'''



with open('ParameterListHPODART_class_simple_01.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterListHPODART_class_simple_01.pkl', 'rb') as f:
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
                             seqtype = 'proteinpadded',
                             numchannels = 1)



test_Dataset,test_loaderG,sampledataT,dataT,labelsT,indexofdata = testobj.GetData()



'''Step - 5'''
'''Now to test the model performance on an independent test-set'''



test_loaderF=DataLoader(test_Dataset,batch_size=5, shuffle=False, drop_last=False)


testobject = SemiManualDart_test()



skAccScoreT,preT,reT,f1scoreT,MCCT= testobject.predict( test_loaderF,
 modelfinal,
 indexofdata,
 labelsT,
 resulttype='bceranking',
 indtruestart=0,
 indtrueend=176#using the model obtained from k-fold validation
)


'''saving best model'''

torch.save(modelfinal.state_dict(),'Best_Auto_HPOdart_simple_class_01.pt')
print('saved the Best_Auto model')





'''6. getting the results'''

print('Printing test accuracy, precision, recall, F1-Score')

print('Test-Accuracy : ',skAccScoreT)
print('Test-Precision : ',preT)
print('Test-Recall : ',reT)
print('Test-F1-Score : ',f1scoreT)
print(' Test-MCC : ',MCCT )


print('Printing autotuner obtained validation accuracy, train accuracy and train loss (u can see individual run"s obtained results from report')


avg_valid_acc,avg_train_acc,avg_train_loss
print(' avg_valid_acc : ',avg_valid_acc)
print('  avg_train_acc : ',avg_train_acc)
print('  avg_train_loss : ',avg_train_loss)


