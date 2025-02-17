


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:46:05 2023

@author: SHRUTI SARIKA CHAKRABORTY 
"""

import torch

import torch

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand
import pickle 
from torch.utils.data import DataLoader,TensorDataset


from MEDHA.AutoToolGraph.HPO_DART_ERRICA_GRAPH  import HPO_DART
from MEDHA.AutoToolGraph.Graphein_Caller   import Graphein_Caller
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




dataset2 = TUDataset(root='data/TUDataset', name='ENZYMES')

listi=[]



for i in dataset2:
    listi.append(i.num_nodes)
max_nodes = max(listi)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'ENZYMES_dense')
dataset = TUDataset(
    path,
    name='ENZYMES',
    transform=T.ToDense(max_nodes),
    pre_filter=lambda data: data.num_nodes <= max_nodes,
)

#dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
datasetr = dataset.shuffle()

print('len(datasetr) out',len(datasetr))



'''Step - 2'''
'''Now calling HPO-DART-Tuner. Note that you do not need to pass space in parameter unless you want to configure it. In that case please follow the exact notations and structure'''


myHpoObject = HPO_DART(
input_channel=dataset.num_node_features,
outchannel=dataset.num_classes,
max_nodes=max_nodes,
percent_dec=0.80,
batch_size=15,
OptimizerDart='Adam',
learning_rateDart =  0.0001,
dataset=datasetr,
acc_thresold=80,
epochs=50,
intepochs=10,
R=3)



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


print('  avg_test_std Autotuner : ',acc_std)
