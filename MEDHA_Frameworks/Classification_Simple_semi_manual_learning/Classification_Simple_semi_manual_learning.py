# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:46:05 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""

import torch
from MEDHA.DataProcessing.Classification.Simple.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Classification.Simple.DataProcessorTest  import DataPreprocessTest
from MEDHA.AutoTool.Classification.Simple.SemiManualDART_train import SemiManualDart_train
from MEDHA.AutoTool.Classification.Simple.SemiManualDart_test import SemiManualDart_test
from torch.utils.data import DataLoader,TensorDataset
import pickle

print('imports complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()




'''Step - 1'''
'''1.Get training data for PreAcrs'''


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
'''2.Call semi-manual DART for PreAcr training data'''




DartObject =  SemiManualDart_train()



''' put whole_Dataset in dart_dataset and sampledata in sample_data. If you do not wish that, then you can divide the whole data into training and testing set and put train_dataset in below for dart_dataset ''' 



modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input =  100,
                                                                           out_channel_f = 25,
                                                                           drop = 0,
                                                                           UnitFCN_vars = 25,
                                                                           nLayers_vars = 2,
                                                                           loop = 1,
                                                                           pool_size = 1,
                                                                           actfun = 'ReLU',
                                                                           num_epochs=3,
                                                                           OptimizerDart = 'Adam',
                                                                           sample_data = sampledata,
                                                                           in_channel  = 1,
                                                                           kernel      = [1,3,5],
                                                                           batch_size=5, 
                                                                           outchannel = 1,
                                                                           chooseblocks=['block1','block2','block3','block4','block5','block6','block7'],
                                                                           learning_rateDart     = 0.00018,
                                                                           L2lambdaDart          = 0.03 ,
                                                                           momentumDart          = 0.0,
                                                                           dart_dataset = whole_Dataset,
                                                                           lossfuntype = 'BCE',
                                                                           threshold=0) 
#'block1','block2','block3','block4','block5','block6','block7'

print('DARTacc ',DARTacc)


'''Step - 3'''
'''3.Do k-fold cross validation for training data'''





train_loss,train_acc,validation_acc,validation_loss,avg_train_loss ,avg_train_acc,avg_val_acc,avg_val_loss,bestmodel = DartObject.KFoldCrossValidator(k=5,
crossvalidator_dataset=whole_Dataset,
batch_size=10,
model=modelfinal,
OptimizerKfold = 'Adam',
lossfuntype='bce',
num_epochs=5,
predtype='binary')



print(' train_loss_all ',train_loss)
print(' avg_train_acc ',avg_train_acc)
print(' avg_validation_acc ',avg_val_acc)
print(' avg_validation_loss ',avg_val_loss)

'''Step - 4'''
'''4. Call test set'''

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




'''Step - 5'''
'''5. Predict test set'''



test_loaderF=DataLoader(test_Dataset,batch_size=5, shuffle=False, drop_last=False)

testobject = SemiManualDart_test()



skAccScoreT,preT,reT,f1scoreT,MCCT= testobject.predict( test_loaderF,
 bestmodel,
 indexofdata,
 labelsT,
 resulttype='bceranking',
 indtruestart=0,
 indtrueend=177#using the model obtained from k-fold validation
)

'''Step - 6'''
'''6. getting the results'''

print('Printing test accuracy, precision, recall, F1-Score')

print('Accuracy : ',skAccScoreT)
print('Precision : ',preT)
print('Recall : ',reT)
print('F1-Score : ',f1scoreT)
print(' MCC : ',MCCT )

print('Printing K-Fold accuracies and loss ')

print(' train_loss_all K-FOLD : ',train_loss)
print('  train accuracy K-FOLD : ',train_acc)
print('  validation accuracy K-FOLD : ',validation_acc)
print('  validation loss K-FOLD : ',validation_loss)

print(' average train loss K-FOLD : ',avg_train_loss)
print(' average train accuracy K-FOLD : ',avg_train_acc)
print(' average validation accuracy K-FOLD : ',avg_val_acc)
print(' average validation loss K-FOLD : ',avg_val_loss)






'''Step - 7'''
'''7. Save parameter list of DART and final model'''

'''save and load exported_arch as dictionary'''




with open('exported_archfinal_class_simple_01.pkl', 'wb') as f:
    pickle.dump(exported_arch, f)

print('saved the DART chosen confirguration')

with open('exported_archfinal_class_simple_01.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    
'''save and load ParameterList as dictionary'''

with open('ParameterList_class_simple_01.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterList_class_simple_01.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    
    
'''saving best model'''

torch.save(bestmodel.state_dict(),'Best_Semi_Manual_class_simple_01.pt')
print('saved the Best_Semi_Manual model for classification')


'''saving nas-module'''

with open(r'nas-module_class_simple_01.txt', 'w') as fp:
    fp.write("%s\n" % nas_modules)
    print('Saved the nas-module configuration, which shows how to infer DART chosen confirguration in exported_archfinal')