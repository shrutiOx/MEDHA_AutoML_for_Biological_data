

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:46:05 2023

@author: SHRUTI SARIKA CHAKRABORTY 
"""

import torch
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
import pickle 
from torch.utils.data import DataLoader,TensorDataset


from MEDHA.AutoToolGraph.HPO_DART_ERRICA_GRAPH  import HPO_DART
from MEDHA.AutoToolGraph.Graphein_Caller   import Graphein_Caller
from MEDHA.AutoToolGraph.Graphein_Caller_Test   import Graphein_Caller_Test
from MEDHA.AutoToolGraph.GetProcessedata import ProcessedDataExtractor
from torch_geometric.datasets import TUDataset
import os.path as osp
import time
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch_geometric.loader import DenseDataLoader
print('imports complete')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



'''Step - 1'''
'''1.Get  data'''



'''Step - 1'''
'''1.Get training data '''

'''Either load the data'''
'''
Graphein_object = Graphein_Caller(datacsv='structural_rearrangement_data_TRAIN.csv',
name='protein_train',
batch_size=20)


dataset,train_loader,val_loader,max_nodes= Graphein_object.GetData()


print('len(dataset) out',len(dataset))
'''


'''OR you can get processed data if data was already processed'''

max_nodes=792

Graphein_object = ProcessedDataExtractor(data_loc='./processed/data_protein_train.pt', 
                                                  name='protein_train', 
                                                  numlabels=1,
                                                  max_nodes=max_nodes)


dataset = Graphein_object.GetProcessedData()





'''Step - 2'''
'''Now calling HPO-DART-Tuner. Note that you do not need to pass space in parameter unless you want to configure it. In that case please follow the exact notations and structure'''


myHpoObject = HPO_DART(
input_channel=dataset.num_node_features,
outchannel=dataset.num_classes,
max_nodes=max_nodes,
percent_dec=0.25,
batch_size=5,
OptimizerDart='Adam',
learning_rateDart =  0.0001,
dataset=dataset,
acc_thresold=50,
epochs=5,
intepochs=3,
R=1)



loss_mean,accsval,  model,trainlossmeanf,createlist,space  = myHpoObject.Call_Process()




'''Step - 3'''
'''save and load ParameterList (this contains Params used in DART-Tuner) as dictionary'''



with open('ParameterListHPODART_class_simple_01.pkl', 'wb') as f:
    pickle.dump(createlist, f)

print('saved the DART chosen confirguration')

with open('ParameterListHPODART_class_simple_01.pkl', 'rb') as f:
    loaded_param = pickle.load(f)
    



'''saving best model'''

torch.save(model.state_dict(),'Best_Auto_HPOdart_simple_class_01.pt')
print('saved the Best_Auto model')





'''6. getting the results'''

print('Printing test accuracy, precision, recall, F1-Score')


print('Printing autotuner obtained validation accuracy, train accuracy and train loss (u can see individual run"s obtained results from report')


print('  avg_train_loss Autotuner : ',trainlossmeanf)
print('  avg_val_loss Autotuner : ',loss_mean)
print('  avg_val_acc Autotuner : ',accsval)






def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.adj, data.mask).max(1)[1]
        pred=pred.cpu()
        data.y = data.y.cpu()
        correct += pred.eq(data.y.view(-1)).sum().item()
    return ((correct / len(loader.dataset)),pred)





'''Step - 1'''
'''1.Get testing data '''

Graphein_object = Graphein_Caller_Test(datacsv='structural_rearrangement_data_TEST.csv',
name='protein_test',
batch_size=20)


datasettest,test_loader,max_nodes= Graphein_object.GetData()


print('len(dataset)  test out',len(datasettest))

accs = []
predictions = []

accs.append(eval_acc(model, test_loader)[0])

predictions.append(eval_acc(model, test_loader)[1])

acc=torch.tensor(accs)

acc_mean = acc.mean().item()

print('Test accuracy ',acc_mean)
