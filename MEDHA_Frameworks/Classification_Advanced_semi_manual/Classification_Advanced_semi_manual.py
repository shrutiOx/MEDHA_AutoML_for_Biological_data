

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:46:05 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""

import torch
from MEDHA.DataProcessing.Classification.Advanced.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Classification.Advanced.DataProcessorTest  import DataPreprocessTest
from MEDHA.AutoTool.Classification.Advanced.SemiManualDART_train import SemiManualDart_train
from MEDHA.AutoTool.Classification.Advanced.SemiManualDart_test import SemiManualDart_test
import pickle

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
'''2.Call semi-manual DART for PreAcr training data'''




DartObject =  SemiManualDart_train()



''' put whole_Dataset in dart_dataset and sampledata in sample_data. If you do not wish that, then you can divide the whole data into training and testing set and put train_dataset in below for dart_dataset ''' 


modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input = 50,
                                                                                   out_channel_f = 75,
                                                                                   drop = 0.4,
                                                                                   UnitFCN_vars = 65,
                                                                                   nLayers_vars = 1,
                                                                                   loop = 1,
                                                                                   pool_size = 1,
                                                                                   actfun = 'ReLU',
                                                                                   num_epochs=7,
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
                                                                                   lossfuntype = 'bce',
                                                                                   threshold=2) 

#'block1','block2','block3','block4','block5','block6','block7'

print('DARTacc ',DARTacc)


usenet,Concat_dataset = DartObject.Concater(whole_Dataset       = whole_Dataset,
                                        model                  = modelfinal,
                                        concatflag             = True,
                                        condata                = condata, 
                                        modeltypeflag          = 'dnn',
                                        out_param              = 1,
                                        nUnits=100,
                                        nLayers=1,
                                        createlist=ParameterList
                                       )




'''Step - 3'''
'''3.Do k-fold cross validation for training data'''





'''Now calling KFoldCrossValidator.'''

train_loss_all = []
test_loss_all = []

train_loss = []
train_acc = []
validation_acc = []

train_loss_all,train_acc,validation_acc,avg_train_loss,avg_train_acc,avg_val_acc,bestmodel,avg_val_loss = DartObject.KFoldCrossValidator(k=3,
                                                                                                                          crossvalidator_dataset=Concat_dataset,
                                                                                                                          batch_size=5,
                                                                                                                          model=usenet,
                                                                                                                          learning_rate     = 0.006,
                                                                                                                          L2lambda          = 0.00002 ,
                                                                                                                          momentum          = 0.0,
                                                                                                                          OptimizerKfold = 'Adam',
                                                                                                                          lossfuntype='bce',
                                                                                                                          num_epochs=5,
                                                                                                                          predtype='binary')




print(' train_loss_all ',train_loss_all)
print(' avg_train_acc ',avg_train_acc)
print(' avg_val_acc ',avg_val_acc)

'''Step - 4'''
'''4. Call test set'''

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
'''5. Predict test set'''




testobject = SemiManualDart_test()



accuracy,precision,recall,f1_score,MCC = testobject.predict( test_loaderF,
modeldart    = modelfinal,
modelkfold   = bestmodel,
concatflagT  = True,
condataT     = condataT,
indexT       = indexofdata,
labelsT      = labelsT,
resulttype   = 'bceranking',
indtruestart = 0,
indtrueend   = 176
)

'''Step - 6'''
'''6. getting the results'''

print('Printing test accuracy, precision, recall, F1-Score')

print('Accuracy : ',accuracy)
print('Precision : ',precision)
print('Recall : ',recall)
print('F1-Score : ',f1_score)


print('Printing K-Fold accuracies and loss ')

print(' train_loss_all K-FOLD : ',train_loss_all)
print('  train accuracy K-FOLD : ',train_acc)
print('  test accuracy K-FOLD : ',validation_acc)

print(' average train loss K-FOLD : ',avg_train_loss)
print(' average train accuracy K-FOLD : ',avg_train_acc)
print(' average test accuracy K-FOLD : ',avg_val_acc)
print(' average test loss K-FOLD : ',avg_val_loss)






'''Step - 7'''
'''7. Save parameter list of DART and final model'''

'''save and load exported_arch as dictionary'''

 


with open('exported_archfinal_class_adv_01.pkl', 'wb') as f:
    pickle.dump(exported_arch, f)

print('saved the DART chosen confirguration')

with open('exported_archfinal_class_adv_01.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    
'''save and load ParameterList as dictionary'''

with open('ParameterList_class_adv_01.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterList_class_adv_01.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    
    
'''saving best model'''


torch.save(modelfinal.state_dict(),'modeldart_semi_class.pt')
print('saved the modeldart model for classification')

torch.save(bestmodel.state_dict(),'modelkfold_semi_class.pt')
print('saved the modelkfold model for classification')


'''saving nas-module'''

with open(r'nas-module_class_adv_01.txt', 'w') as fp:
    fp.write("%s\n" % nas_modules)
    print('Saved the nas-module configuration, which shows how to infer DART chosen confirguration in exported_archfinal')