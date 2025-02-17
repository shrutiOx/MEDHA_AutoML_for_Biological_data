# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:11:58 2023

@author: ADMIN
"""


# -*- coding: utf-8 -*-
"""Created on Thu Jun  1 18:00:23 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""


'''import libraries'''

import numpy as np
import pandas as pd
import torch.nn as nn

import torch
import random
import pickle

from MEDHA.DataProcessing.Regression.Simple.DataProcessorTrain import DataPreprocessTrain
from MEDHA.AutoTool.Regression.Advanced.SemiManualDART_train import SemiManualDart_train
from MEDHA.AutoTool.Regression.Advanced.SemiManualDART_train import SemiManualDart_train
from MEDHA.AutoTool.Regression.Advanced.DNNClassBCE_2 import DNNClassBCEFunc
from MEDHA.AutoTool.Regression.Advanced.CNNLSTM import CNNLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
#device = torch.device("cpu" )

'''

1. Create a function that takes in the following
a.parameterlist.pkl
b.encoding type
c.mainmodel_statedict
d.max_length_of_sequences in training
e.number of channels

2.This function then should unload the pkl and see how many outputs are asked
3.Now from the given encoding type, it will create 2 toy examples with number of predictions = number of out-channels specified in the pkl
4.It will call DataProcessing unit on that toy data, with the encoding scheme given by users.
5.It will next call Semi-Manual train filling all details from pkl and getting data rom processed-toy
6.It will run it one time.
7.Now it will wrap the best-model with this placeholder and create final model which it will output.
8.Next user calls DataProcessor test
9.Next runs Semi-manual test with the recieved model and get the results.
10.Put 'ModelReproduction' from Simple same to same for classification and regression. Same observes for 'Advanced' as well.

'''

class ModelReproduction(nn.Module):
    def __init__(self,
                 parameterlist=None,
                 customalphabet = [],
                 customscheme = 0,
                 seqtype = 'custom',
                 max_length_of_trainseq=20,
                 statedict_modelDart=None,
                 statedict_modelFold=None,
                 numchannels=1 ):
        super().__init__()
        

        #print('space ',space)

            
        self.parameterlist               = parameterlist
        self.customalphabet              = customalphabet
        self.customscheme                = customscheme
        self.seqtype                     = seqtype

        self.max_length_of_trainseq      = max_length_of_trainseq
        self.statedict_modelDart         = statedict_modelDart
        self.statedict_modelFold         = statedict_modelFold
        self.numchannels                 = numchannels


        with open(parameterlist, 'rb') as f:
            loaded_param = pickle.load(f)
            
        out_channel_inputL = 0
        out_channel_fL = 0
        dropL = 0
        UnitFCN_varsL = 0
        nLayers_varsL = 0
        loopL = 0
        pool_sizeL = 0
        actfunL=None
        num_epochsL = 0
        OptimizerDartL = 0
        in_channelL = 0
        kernelL = 0
        outchannelL = 0
        batch_sizeL = 0
        learning_rateDartL = 0
        L2lambdaDartL = 0
        momentumDartL = 0
        lossfuntypeL = 0
        chooseblocksL = 0
        modeltypeflagL = None
        unitsdnnL = 0
        layersdnnL = 0
        n_hiddenLSTML = 0
        nLSTMlayersL = 0
        input_paramL = 0
        
        for i in loaded_param:
            if('out_channel_input' in i):
                out_channel_inputL = loaded_param[i]
            if('out_channel_f' in i):
                out_channel_fL = loaded_param[i]
            if('drop' in i):
                dropL = loaded_param[i]
            if('UnitFCN_vars' in i):
                UnitFCN_varsL = loaded_param[i]        
            if('nLayers_vars' in i):
                nLayers_varsL = loaded_param[i]
            if('loop' in i):
                loopL = loaded_param[i]
            if('pool_size' in i):
                pool_sizeL = loaded_param[i]
            if('actfun' in i):
                actfunL = loaded_param[i]  
            if('num_epochs' in i):
                num_epochsL = loaded_param[i]
            if('OptimizerDart' in i):
                OptimizerDartL = loaded_param[i]
            if('in_channel' in i):
                in_channelL = loaded_param[i]
            if('kernel' in i):
                kernelL = loaded_param[i]        
            if('outchannel' in i):
                outchannelL = loaded_param[i]
            if('batch_size' in i):
                batch_sizeL = loaded_param[i]
            if('learning_rateDart' in i):
                learning_rateDartL = loaded_param[i]
            if('L2lambdaDart' in i):
                L2lambdaDartL = loaded_param[i]  
            if('momentumDart' in i):
                momentumDartL = loaded_param[i]
            if('lossfuntype' in i):
                lossfuntypeL = loaded_param[i]
            if('chooseblocks' in i):
                chooseblocksL = loaded_param[i]
            if('modeltypeflag' in i):
                modeltypeflagL = loaded_param[i]
            if('unitsdnn' in i):
                unitsdnnL = loaded_param[i]
            if('layersdnn' in i):
               layersdnnL = loaded_param[i]  
            if('n_hiddenLSTM' in i):
                n_hiddenLSTML = loaded_param[i]
            if('nLSTMlayers' in i):
                nLSTMlayersL = loaded_param[i]
            if('input_param' in i):
                input_paramL = loaded_param[i]
        
        if ('protein' in self.seqtype):
            typeofdata='protein'
        elif('rna' in self.seqtype):
            typeofdata='rna'
        else:
            typeofdata='dna'
        
        dftoydata = pd.DataFrame()
        
        if typeofdata=='dna':
                    val = [('T')*self.max_length_of_trainseq]
        if typeofdata=='rna':
                    val = [('U')*self.max_length_of_trainseq]
        if typeofdata=='protein':
                    val = [('M')*self.max_length_of_trainseq]
            
        dftoydata.insert(0, "Sequences", val)
            
        for i in range(0,int(outchannelL[0])):
                    dftoydata[f'label{i}'] = random.randint(0,9)
        
        
        dftoydata.to_csv('toydata.csv')
        
    
        dataobj =  DataPreprocessTrain(datacsv='toydata.csv',
                               inslicestart = 1,
                               insliceend = 2,
                               outslicestart = 2,
                               outsliceend = None,  
                               customalphabet = self.customalphabet,
                               customscheme = self.customscheme,
                               numchannels = self.numchannels,
                               seqtype = self.seqtype)


        whole_Dataset,whole_loader,sampledata,data,labels= dataobj.GetData()
        
        
        
        DartObject =  SemiManualDart_train()
        modelfinal,exported_arch,nas_modules,ParameterList,DARTacc = DartObject.DartCaller(out_channel_input =   int(out_channel_inputL[0]),
                                                                                   out_channel_f = int(out_channel_fL[0]),
                                                                                   drop = int(dropL[0]),
                                                                                   UnitFCN_vars = int(UnitFCN_varsL[0]),
                                                                                   nLayers_vars = int(nLayers_varsL[0]),
                                                                                   loop = int(loopL[0]),
                                                                                   pool_size = int(pool_sizeL[0]),
                                                                                   actfun = (actfunL[0]),
                                                                                   num_epochs=1,
                                                                                   OptimizerDart = (OptimizerDartL[0]),
                                                                                   sample_data = sampledata,
                                                                                   in_channel  = int(in_channelL[0]),
                                                                                   kernel      = kernelL[0],
                                                                                   batch_size=int(batch_sizeL[0]),
                                                                                   outchannel = int(outchannelL[0]),
                                                                                   chooseblocks=chooseblocksL[0],
                                                                                   learning_rateDart     = int(learning_rateDartL[0]),
                                                                                   L2lambdaDart          = int(L2lambdaDartL[0]),
                                                                                   momentumDart          = int(momentumDartL[0]),
                                                                                   dart_dataset = whole_Dataset,
                                                                                   lossfuntype = (lossfuntypeL[0])) 
        
        self.modelfinal = modelfinal
        
        if modeltypeflagL[0] == 'dnn':
            DNN_Net = DNNClassBCEFunc(input_paramL[0],
                                  unitsdnnL[0],
                                  layersdnnL[0],
                                  outchannelL[0],
                                  'ReLU',
                                  True,
                                  lossfuntypeL[0]
                                  )   
        
            self.use_Net = DNN_Net.to(device)
        else:
            LSTMnet = CNNLSTM(
                     input_paramL[0],
                     outchannelL[0], 
                     'ReLU',
                     nLSTMlayersL[0],
                     n_hiddenLSTML[0],
                     lossfuntypeL[0])
            self.use_Net = LSTMnet.to(device)
        
    def GetModel(self):
        
        self.modelfinal.load_state_dict(torch.load(self.statedict_modelDart))
        self.use_Net.load_state_dict(torch.load(self.statedict_modelFold))
        
        return self.modelfinal , self.use_Net