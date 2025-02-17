# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:43:23 2023

@author: SHRUTI SARIKA CHAKRABORTY
"""





import numpy as np
import pandas as pd

import time
import torch.nn.functional as F
import copy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms





import scipy.stats as stats
import sklearn.metrics as skm
import nni
import math








"""#importing TRAINING an VALIDATION data"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu" )
print(device)

class DataPreprocessTest():
    def __init__(self, 
                 datacsv='None',
                 inslicestart = 0,
                 insliceend = 0,
                 outslicestart = 0,
                 outsliceend = None,
                 indstart = 0,
                 indend = 0,
                 customalphabet = [],
                 customscheme = 0,
                 numchannels = 0,
                 seqtype = 'custom'):
        super().__init__()
        self.datacsv    = datacsv
        self.inslicestart = inslicestart
        self.insliceend = insliceend
        self.indstart = indstart
        self.indend = indend
        self.outslicestart = outslicestart
        self.outsliceend = outsliceend
        self.customalphabet = customalphabet
        self.customscheme = customscheme
        self.numchannels = numchannels
        self.seqtype = seqtype

    def GetData(self):
        df  = pd.read_csv(self.datacsv) # reads the excel file into a dataframe

        data        =  df[df.columns[self.inslicestart:self.insliceend]].values
        indexofdata =  df[df.columns[self.indstart:self.indend]].values 
        
        if self.outsliceend == None:
            labels = df[df.columns[self.outslicestart::]].values
            #labelnames = df.columns[self.outslicestart::]
        else:
            labels = df[df.columns[self.outslicestart:self.outsliceend]].values
       # labelnames = df.columns[self.outslicestart:self.outsliceend]
        
        

        
       
        

        '''to pad the sequences with (*) - it finds max length of all sequences and then for rest which is less than max-length, the rest of the positions are filled with *'''
        
        def pad_sequence(sequence, max_length):
            padded = sequence[:max_length] + '*' * (max_length - len(sequence))
            return padded
        
        '''finds max length from the coloumn=Sequences from the csv file'''
        
        max_length = df['Sequences'].apply(len).max()
        listi = []


        '''here we do padding for all sequences present in the column-'Sequences' of the csv (input) after we replace all spaces -> ' ' with ''. Then we put those processed-padded sequences in an empty list
        '''
        
        for sequence in df['Sequences']:
            processed_seq = sequence.replace(' ', '')
            padded_sequence = pad_sequence(processed_seq, max_length)
            listi.append(padded_sequence)


        '''this is for one-hot encoding of protein(amino-acids). For RNA/DNA the values in alphabet (list below) will change'''
        
        if str.lower(self.seqtype) == 'proteinpadded':
            self.alphabet = ['A',
                    'R',
                    'N'	,
                    'D'	,
                    'C'	,
                    'Q'	,
                    'E'	,
                    'G'	,
                    'H'	,
                    'I'	,
                    'L'	,
                    'K',
                    'M'	,
                    'F'	,
                    'P'	,
                    'S'	,
                    'T'	,
                    'W',
                    'Y'	,
                    'V'	,
                    'B'	,
                    'Z'	,
                    'J'	,
                    'U'	,
                    'O'	,
                    'X',
                    '*']
            self.scheme = 27
        elif str.lower(self.seqtype) == 'protein':
            self.alphabet = ['A',
                    'R',
                    'N'	,
                    'D'	,
                    'C'	,
                    'Q'	,
                    'E'	,
                    'G'	,
                    'H'	,
                    'I'	,
                    'L'	,
                    'K',
                    'M'	,
                    'F'	,
                    'P'	,
                    'S'	,
                    'T'	,
                    'W',
                    'Y'	,
                    'V'	,
                    'B'	,
                    'Z'	,
                    'J'	,
                    'U'	,
                    'O'	,
                    'X']
            self.scheme = 26
        elif (str.lower(self.seqtype) == 'rna') or (str.lower(self.seqtype) == 'dna'):
            self.alphabet = ['A',
                    'G',
                    'C'	,
                    'T'	
                    ]
            self.scheme = 4
        elif (str.lower(self.seqtype) == 'rnapadded') or (str.lower(self.seqtype) == 'dnapadded'):
             self.alphabet = ['A',
                     'G',
                     'C'	,
                     'T',
                     '*' ]
             self.scheme = 5
        elif('epigenetic' in str.lower(self.seqtype)):
            self.alphabet = ['A',
                    'G',
                    'C'	,
                    'T',
                    'D',
                    'CT',
                    'H3',
                    'RR',
                    '*']
            self.scheme = 9
        elif (str.lower(self.seqtype) == 'custom'):
            print(' Using custom encoding scheme in test ')
            self.alphabet = self.customalphabet
            self.scheme =  self.customscheme
        
        def OneHot_encode_seq(sequence):
            alphabet = self.alphabet
            char_to_int     = dict((c, i) for i, c in enumerate(alphabet))
            int_to_char     = dict((i, c) for i, c in enumerate(alphabet))
            # integer encode input data
  
            integer_encoded = [char_to_int[char] for char in sequence  ]
            #print(integer_encoded)
            onehot_encoded = list()

            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet))] #u can put here greater than 27 logic too
                
                #if value != 20:
                letter[value] = 1
                #else:
                #    letter[value] = 0
                onehot_encoded.append(letter)

            return np.array(onehot_encoded)

        list2 = []
        '''now putting the former-list of processed-padded sequences through one-hot encoding function to create a new list which we put into the main dataframe-df as a new column - padded'''
        
        for i in listi:
            list2.append(OneHot_encode_seq(i))
        df['padded'] = list2

        list3 = []
        numS = df[df.columns[-1]].values #this fetches padded column which has processed-padded-onehot encoded data
        
        '''after fetching the 'padded' column from 'df'  dataframe now we reshape the processed-padded-onehot encdoed data into '1 x () x 27'. Now remember 27 has to be there due to encoding scheme and () contains anything that's left'''
        
        
        for i in numS:
            i.reshape(1,max_length,self.scheme)
            list3.append(i)
    
        '''This reshaped data is put into a new column of df called ->torch_sequence'''
        df['torch_sequence'] = list3



        '''now we convert from pandas dataframe to tensor'''
        data = torch.tensor( df['torch_sequence'] ).float()

        #data = data.reshape(data.shape[0],1,max_length,21)
        '''you have to put reshape on this again to make it 4D-1st dimension represents batch'''
        #data = data.reshape(data.shape[0],self.numchannels,-1,self.scheme) #27 due to encoding scheme
        data = data.reshape(data.shape[0],self.numchannels,-1,self.scheme) #27 due to encoding scheme
        '''getting labels'''

        
        #data = list(zip(data,data2))
 
        
        labels=(np.vstack(labels).astype(np.float64))
        #print('test labels shape : ',labels.shape)
        labels = torch.from_numpy(labels).float()
        print('test labels shape : ', labels.shape)



        '''Transferring data and labels to GPU'''

        '''all data'''
        data   = data.to(device)
        labels = labels.to(device)
        
        '''now this is the whole dataset. For training (specially K-FOLD) we will put this whole into the model'''

        test_Dataset  = TensorDataset(data,labels)
        test_loaderF  = DataLoader(test_Dataset,batch_size=test_Dataset.tensors[0].shape[0])

        dataT         =  data.to(device)
        labelsT       =  labels.to(device)
        sampledataT   =  dataT[0]
        


        
        return test_Dataset,test_loaderF,sampledataT,dataT,labelsT,indexofdata
    




