
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 03:21:52 2023

@author: ADMIN
"""
import torch
from MEDHA.DataProcessing.Regression.Simple.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Regression.Simple.DataProcessorTest  import DataPreprocessTest
from MEDHA.AutoTool.Regression.Simple.SemiManualDART_train import SemiManualDart_train
import pickle 
from MEDHA.AutoTool.Regression.Simple.SemiManualDart_test import SemiManualDart_test
from torch.utils.data import DataLoader,TensorDataset

print('import complete')


#import sys
#sys.stdout = open('E:/AT OXFORD FINALLY/THE PHD/TERM PAPER 1 plan and others/THE FINAL LIBRARY DESIGN/MEDHA/logout.txt', 'w')




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
                               seqtype = 'dna')



whole_Dataset,whole_loader,sampledata,data,labels= dataobj.GetData()




'''Step - 2'''
'''Now calling DART-Tuner'''



DartObject =  SemiManualDart_train()



''' put whole_Dataset in dart_dataset and sampledata in sample_data. If you do not wish that, then you can divide the whole data into training and testing set and put train_dataset in below for dart_dataset ''' 

modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input = 100,
                                                        out_channel_f = 25,
                                                        drop = 0.2,
                                                        UnitFCN_vars = 50,
                                                        nLayers_vars = 1,
                                                        loop = 1,
                                                        pool_size = 1,
                                                        actfun = 'ReLU',
                                                        num_epochs=5,
                                                        OptimizerDart = 'SGD',
                                                        sample_data = sampledata,
                                                        in_channel  = 1,
                                                        kernel      = [1,3,5],
                                                        batch_size=1000, 
                                                        outchannel = 1,
                                                        chooseblocks=['block1','block2','block3','block4','block5','block6','block7'],
                                                        learning_rateDart     = 0.006,
                                                        L2lambdaDart          = 0.00002 ,
                                                        momentumDart          = 0.6,
                                                        dart_dataset = whole_Dataset,
                                                        lossfuntype = 'mse') 
#'block1','block2','block3','block4','block5','block6','block7'


#print(' exported_arch ',exported_arch)
#print(' nas_modules ',nas_modules)
#print(' createlist :::: ',createlist)

'''Step - 3'''
'''save and load ParameterList as dictionary.This dict stores the parameters used in DART training above'''



with open('ParameterList.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterList.pkl', 'rb') as f:
    loaded_param = pickle.load(f)

'''Step - 4'''
'''Now doing K-FOLD cross validation.Put whole_Dataset in crossvalidator_dataset;modelfinal in  model'''

train_loss_all = []
test_loss_all = []





train_loss_all,test_loss_all, avg_train_loss,avg_test_loss,bestmodel,pearsoncorrArr,spearmancorrArr,R_squareArr = DartObject.KFoldCrossValidator(k=5,
                                                                                                  crossvalidator_dataset=whole_Dataset,
                                                                                                  model=modelfinal,
                                                                                                  lossfuntype='mse',
                                                                                                  num_epochs=3,
                                                                                                  batch_size=1000)



print(' train_loss_all ',train_loss_all)
print(' test_loss_all ',test_loss_all)
print(' avg_train_loss ',avg_train_loss)
print(' avg_test_loss ',avg_test_loss)
print('pearsoncorrArr :', pearsoncorrArr)
print('spearmancorrArr:', spearmancorrArr)
print('R_squareArr:', R_squareArr)

#print('bestmodel ',bestmodel)

'''Step - 5'''
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



    
'''Step - 6'''    
'''Now to test the model performance on an independent test-set'''



test_loaderF=DataLoader(test_Dataset,batch_size=100, shuffle=False, drop_last=False)


testobject = SemiManualDart_test()



lossesKL = testobject.predict(rawdata=indexofdata,
                                       labelnames=labelnames,
                                       test_loaderF=test_loaderF,
                                       model=bestmodel,#using the model obtained from k-fold validation
                                       lossfun='mse',
                                       needinexcel='no'
                                       )




''' Saving models and configurations '''

'''save and load exported_arch as dictionary'''



with open('exported_archfinal.pkl', 'wb') as f:
    pickle.dump(exported_arch, f)

print('saved the DART chosen confirguration')

with open('exported_archfinal.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    
    
'''saving best model'''

torch.save(bestmodel.state_dict(),'Best_Semi_Manual.pt')
print('saved the Best_Semi_Manual model')


'''saving nas-module'''

with open(r'nas-module.txt', 'w') as fp:
    fp.write("%s\n" % nas_modules)
    print('Saved the nas-module configuration, which shows how to infer DART chosen confirguration in exported_archfinal')
    
    
#sys.stdout.close()