
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 03:21:52 2023

@author:  SHRUTI SARIKA CHAKRABORTY
"""
import torch
from MEDHA.DataProcessing.Classification.Advanced.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Classification.Advanced.DataProcessorTest  import DataPreprocessTest
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
from MEDHA.AutoTool.Classification.Advanced.HPO_DART import HPO_DART_advanced 
import pickle 
from MEDHA.AutoTool.Classification.Advanced.SemiManualDart_test import SemiManualDart_test

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



'''Step - 1'''
'''1.Get training data for PreAcrs'''

datapathtrain = "concatenated_preacr_train_data.csv"

datapathtest = "concatenated_preacr_test_data.csv"

'''Step - 1'''
'''get my train data'''



                 
dataobj =  DataPreprocessTrain(datacsv            = datapathtrain,
                               inslicestart       = 1,
                               insliceend         = 2,
                               concatslicestart   = 3,
                               concatsliceend     = 1603,
                               outslicestart      = 2,
                               outsliceend        = 3,  
                               numchannels        = 1,
                               seqtype            = 'proteinpadded')



whole_Dataset,whole_loader,sampledata,data,labels,condata = dataobj.GetDataConcat()





'''Step - 2'''
'''Now calling HPO-DART-Tuner. Note that you do not need to pass space in parameter unless you want to configure it. In that case please follow the exact notations and structure'''






myHpoObject = HPO_DART_advanced(
                            sample_data = sampledata,
                            in_channel  = 1,
                            kernel      = [1,3,5],
                            outchannel  = 1,
                            dataSet     = whole_Dataset,
                            concatflag  = True,
                            lossfun     = 'bce',
                            batch_size  = 5,
			    modeltypeflag = 'dnn',
			    UnitFCN_vars = 50,
                	    nLayers_vars = 1,
                            threshold   = 2,
                            predtype    = 'binary',
                            condata     = condata,
                            acc_thresold= 65)

avg_val_acc,avg_val_loss,avg_train_acc,avg_train_loss,modelDart,modelkfold,space,ParameterList = myHpoObject.Calling_HPO_DART(max_evals=3,stoppage=2)

 


'''Step - 3'''
'''save and load ParameterList as dictionary.This dict stores the parameters used in DART training above'''



with open('ParameterList_adv_Auto_01.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterList_adv_Auto_01.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    
    
'''Step - 4'''
'''getting test data'''




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



'''Step - 5'''    
'''Now to test the model performance on an independent test-set'''



testobject = SemiManualDart_test()




accuracy,precision,recall,f1_score,MCC = testobject.predict( test_loaderF,
modeldart    = modelDart,
modelkfold   = modelkfold,
concatflagT  = True,
condataT     = condataT,
indexT       = indexofdata,
labelsT      = labelsT,
resulttype   = 'bceranking',
indtruestart = 0,
indtrueend   = 176
)


'''saving best model'''


torch.save(modelDart.state_dict(),'modeldart_autoadv_class.pt')
print('saved the modeldart model for classification advanced')

torch.save(modelkfold.state_dict(),'modelkfold_autoadv_class.pt')
print('saved the modelkfold model for classification advanced')


'''6. getting the results'''

print('Printing test accuracy, precision, recall, F1-Score')

print('Test-Accuracy : ',accuracy)
print('Test-Precision : ',precision)
print('Test-Recall : ',recall)
print('Test-F1-Score : ',f1_score)


print('Printing autotuner obtained validation accuracy, train accuracy and train loss (u can see individual run"s obtained results from report')




print(' avg_valid_acc : ',avg_val_acc)
print('  avg_train_acc : ',avg_train_acc)
print('  avg_train_loss : ',avg_train_loss)


