# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 20:45:42 2023

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:15:21 2023

@author: ADMIN
"""



# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:46:05 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""

import torch
from MEDHA.DataProcessing.Regression.Advanced.DataProcessorTrain import DataPreprocessTrain
from MEDHA.DataProcessing.Regression.Advanced.DataProcessorTest  import DataPreprocessTest
from MEDHA.AutoTool.Regression.Advanced.SemiManualDART_train import SemiManualDart_train
from MEDHA.AutoTool.Regression.Advanced.SemiManualDart_test import SemiManualDart_test
import pickle

print('import complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



'''Step - 1'''
'''1.Get training data for PreAcrs'''

datapathtrain = 'hollerer_rbs_train_adv.csv'

datapathtest  = 'hollerer_rbs_test_adv.csv'

'''Step - 1'''
'''get my train data'''



                 
dataobj =  DataPreprocessTrain(datacsv            = datapathtrain,
                               inslicestart       = 1,
                               insliceend         = 2,
                               concatslicestart   = 3,
                               concatsliceend     = 7,
                               outslicestart      = 2,
                               outsliceend        = 3,  
                               numchannels        = 1,
                               seqtype            = 'dna')



whole_Dataset,whole_loader,sampledata,data,labels,condata = dataobj.GetDataConcat()




'''Step - 2'''
'''2.Call semi-manual DART for PreAcr training data'''




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
                                                                                   batch_size=10, 
                                                                                   outchannel = 1,
                                                                                   chooseblocks=['block1','block2','block3','block4','block5','block6','block7'],
                                                                                   learning_rateDart     = 0.006,
                                                                                   L2lambdaDart          = 0.00002 ,
                                                                                   momentumDart          = 0.6,
                                                                                   dart_dataset = whole_Dataset,
                                                                                   lossfuntype = 'mse') 

#'block1','block2','block3','block4','block5','block6','block7'

print('DARTacc ',DARTacc)

'''Step - 3'''
'''3.Call concatenator module'''

usenet,Concat_dataset = DartObject.Concater(whole_Dataset       = whole_Dataset,
                                        model                  = modelfinal,
                                        concatflag             = True,
                                        condata                = condata, 
                                        modeltypeflag          = 'dnn',
                                        out_param              = 1,
                                        losstype='mse',
                                        createlist=ParameterList
                                       )




'''Step - 4'''
'''4.Do k-fold cross validation for training data'''





'''Now calling KFoldCrossValidator.'''

train_loss_all = []
test_loss_all = []



train_loss_all,test_loss_all, avg_train_loss,avg_test_loss,bestmodel,pearsoncorrArr,spearmancorrArr,R_squareArr = DartObject.KFoldCrossValidator(usenet=usenet,#usenet is the model
k=5,
crossvalidator_dataset=Concat_dataset,
batch_size=100,
lossfuntype='mse',
num_epochs=3)




print(' train_loss_all ',train_loss_all)
print(' test_loss_all ',test_loss_all)
print(' avg_train_loss ',avg_train_loss)
print(' avg_test_loss ',avg_test_loss)
print('pearsoncorrArr :', pearsoncorrArr)
print('spearmancorrArr:', spearmancorrArr)
print('R_squareArr:', R_squareArr)

'''Step - 4'''
'''4. Call test set'''

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




'''Step - 5'''
'''5. Predict test set'''



#from torch.utils.data import DataLoader,TensorDataset
#test_loaderF=DataLoader(test_Dataset,batch_size=10, shuffle=False, drop_last=False)



testobject = SemiManualDart_test()



lossesKL = testobject.predict(rawdata=indexofdata,
                                       labelnames=labelnames,
                                       test_loaderF=test_loaderF,
                                       modeldart=modelfinal,
                                       modelkfold=bestmodel,
                                       concatflagT=True,
                                       condataT=condataT,
                                       lossfun='mse',
                                       needinexcel='no'
                                       )

'''Step - 6'''
'''6. getting the results'''

print('Printing lossesKL')

print('lossesKL : ',lossesKL)





'''Step - 7'''
'''7. Save parameter list of DART and final model'''

'''save and load exported_arch as dictionary'''

 


with open('exported_archfinal_reg_adv_RBS_sm.pkl', 'wb') as f:
    pickle.dump(exported_arch, f)

print('saved the DART chosen confirguration')

with open('exported_archfinal_reg_adv_RBS_sm.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    
'''save and load ParameterList as dictionary'''

with open('ParameterList_reg_adv_RBS_sm.pkl', 'wb') as f:
    pickle.dump(ParameterList, f)

print('saved the DART chosen confirguration')

with open('ParameterList_reg_adv_RBS_sm.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    
    
'''saving best model'''

torch.save(bestmodel.state_dict(),'Best_adv_reg_RBS_sm_modelkfold.pt')
print('saved the Best_Semi_Manual model for regression bestmodel')


torch.save(modelfinal.state_dict(),'Best_adv_reg_RBS_sm_modeldart.pt')
print('saved the Best_Semi_Manual model for regression modelfinal')

'''saving nas-module'''

with open(r'nas-module_reg_adv_RBS_sm.txt', 'w') as fp:
    fp.write("%s\n" % nas_modules)
    print('Saved the nas-module configuration, which shows how to infer DART chosen confirguration in exported_archfinal')