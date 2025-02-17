# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:16:33 2025

@author: ADMIN
"""

import torch
from graphein.ml import ProteinGraphListDataset, GraphFormatConvertor
import torch_geometric.transforms as T
import nni.retiarii.nn.pytorch as nn


def GetDataProcessed(data_loc=None,
                       max_nodes=None,
                       name=None,
                       numlabels=None):
      """Process the processed graphs and make graph datasets:inner level func"""
      datas=torch.load(data_loc)
      max_nodes = max_nodes
      ds = ProteinGraphListDataset(root=".", data_list=datas, name=name,transform=T.ToDense(max_nodes))
      #ds = torch.load(datas)
      
      #valid_indices = [data.index for data in ds if data.y.shape[0] == numlabels]
      #valid_indices = [data.index for data in datas if data.y.shape[0] == numlabels]
      maindataset = ds
      #maindataset = datas[valid_indices]
      
      all_labels = [graph.y for graph in maindataset]
      labels_tensor = torch.stack(all_labels)
      print('Shape of the labels : ' ,labels_tensor.shape, flush=True)
      
      return maindataset
    
class ProcessedDataExtractor(nn.Module):
     """Get structures to make graphs from Processed data."""
     
     def __init__(self, data_loc, 
                  name=None, 
                  max_nodes=None,
                  numlabels=None):

         self.data_loc     = data_loc
         self.numlabels     = numlabels
         self.max_nodes     = max_nodes
         self.name          = name
         
    
         
     #creating the third abstract function
     def GetProcessedData(self):
         
         maindataset = GetDataProcessed(data_loc=self.data_loc,
                              max_nodes=self.max_nodes,
                              name=self.name,
                              numlabels=self.numlabels)
         
         return maindataset