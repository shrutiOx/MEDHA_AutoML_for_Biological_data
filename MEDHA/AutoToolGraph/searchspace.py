

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:01:43 2023

@author: ADMIN
"""

import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,DenseGCNConv,DenseGINConv,DenseGraphConv,DenseGATConv,dense_mincut_pool
from torch_geometric.nn import BatchNorm,GraphNorm,InstanceNorm,LayerNorm
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 
import scipy.stats as stats
import sklearn.metrics as skm
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
from MEDHA.AutoToolGraph import dartsgraph
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import torch_geometric.transforms as T 
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset
print('All imports completed')






@model_wrapper
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,attn_heads,droprate,lin=True):
        super().__init__()
        
        '''
        Layer 1
        
        '''
        
        '''DenseGCNConv''' 
        #self.conv1     = DenseGCNConv(in_channels, hidden_channels*attn_heads)
        
        '''DenseGATConv'''
        self.gat1      = DenseGATConv(in_channels, hidden_channels,dropout=droprate, heads=attn_heads)
        
        
        '''DenseGraphConv'''
        self.grconv1   = DenseGraphConv(in_channels, hidden_channels*attn_heads)
        
        '''DenseSAGEConv'''
        self.sageconv1 = DenseSAGEConv(in_channels, hidden_channels*attn_heads)
        
        '''DenseGINConv'''
        self.GINconv1 = DenseGINConv(
            Sequential(Linear(in_channels, hidden_channels),
                        ReLU(),
                       Linear(hidden_channels, hidden_channels*attn_heads), ReLU()))

        
        '''batchnorm'''
        self.bn1 = GraphNorm(hidden_channels*attn_heads)
        
        
        '''
        Layer 2
        
        '''
        
        '''DenseGCNConv''' 
        #self.conv2     = DenseGCNConv(hidden_channels*attn_heads, hidden_channels*(attn_heads//4))
        
        '''DenseGATConv'''
        self.gat2      = DenseGATConv(hidden_channels*attn_heads, hidden_channels,dropout=droprate, heads=(attn_heads//4))
        
        
        '''DenseGraphConv'''
        self.grconv2   = DenseGraphConv(hidden_channels*attn_heads, hidden_channels*(attn_heads//4))
        
        '''DenseSAGEConv'''
        self.sageconv2 = DenseSAGEConv(hidden_channels*attn_heads, hidden_channels*(attn_heads//4))
        
        '''DenseGINConv'''
        self.GINconv2 = DenseGINConv(
            Sequential(Linear(hidden_channels*attn_heads, hidden_channels*(attn_heads//4)),
                        ReLU(),
                       Linear(hidden_channels*(attn_heads//4), hidden_channels*(attn_heads//4)), ReLU()))

        
        '''batchnorm'''
        self.bn2 = BatchNorm(hidden_channels*(attn_heads//4))
        
        
        '''
        Layer 3
        
        '''
        
        '''DenseGCNConv''' 
        #self.conv3     = DenseGCNConv(hidden_channels*(attn_heads//4), out_channels)
        
        '''DenseGATConv'''
        self.gat3      = DenseGATConv(hidden_channels*(attn_heads//4), out_channels,dropout=droprate, heads=1)
        
        
        '''DenseGraphConv'''
        self.grconv3   = DenseGraphConv(hidden_channels*(attn_heads//4), out_channels)
        
        '''DenseSAGEConv'''
        self.sageconv3 = DenseSAGEConv(hidden_channels*(attn_heads//4), out_channels)
        
        '''DenseGINConv'''
        self.GINconv3 = DenseGINConv(
            Sequential(Linear(hidden_channels*(attn_heads//4), hidden_channels),
                       ReLU(),
                       Linear(hidden_channels, out_channels), ReLU()))

        
        '''batchnorm'''
        self.bn3 = GraphNorm(out_channels)       
        

        if lin is True:
            self.lin = torch.nn.Linear( (hidden_channels*attn_heads)+ (hidden_channels*(attn_heads//4))+ out_channels,
                                       out_channels)
        else:
            self.lin = None

        self.skipconnect1                      = nn.InputChoice(n_candidates=4)
        self.skipconnect2                      = nn.InputChoice(n_candidates=4)
        self.skipconnect3                      = nn.InputChoice(n_candidates=4)


    def bn(self, i, x):
        #print('x in bn b4 ', x.size())
        batch_size, num_nodes, num_channels = x.size()
        #print('x in bn ', x.size())
        
        x = x.view(-1, num_channels)
        #print('x in bn 2 ', x.size())
        
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        #print('x in bn 3', x.size())
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x.float()
        #print(x0)

        '''Level 1'''
        
        #print('x in beginning ', x.size())
        #print(x0)
        
        #xconv1         = self.conv1(x0, adj, mask)
        xgat1          = self.gat1(x0, adj, mask)
        xgrconv1       = self.grconv1(x0, adj, mask)
        xsageconv1     = self.sageconv1(x0, adj, mask)
        xGINconv1      = self.GINconv1(x0, adj, mask)
        
        xchoice1       = self.skipconnect1([xgrconv1,xgat1,xsageconv1,xGINconv1])
        #x1             = self.bn1(xchoice1).relu()
        x1             = xchoice1.relu()
        
        #print('x in layer 1 ', x1.size())
        
        '''Level 2'''
        
        #print('x in beginning of layer 2', x1.size())
        
        #xconv2         = self.bn(2, self.conv2(x1, adj, mask).relu())
        
        xgat2          = self.bn(2, self.gat2(x1, adj, mask).relu())
        xgrconv2       = self.bn(2, self.grconv2(x1, adj, mask).relu())

        xsageconv2     = self.bn(2, self.sageconv2(x1, adj, mask).relu())
        xGINconv2      = self.bn(2, self.GINconv2(x1, adj, mask).relu())
        
        
        xchoice2       = self.skipconnect2([xgrconv2,xgat2,xsageconv2,xGINconv2])

        #x1 = F.dropout(x1, p=0.6, training=self.training)
        x2=xchoice2.relu()
        #x2 = F.dropout(x2, p=0.6, training=self.training)
        #x2=xchoice2        
        #print('x in layer 2 ', x2.size())       

        
        '''Level 3'''
        
        #print('x in beginning of layer 2', x2.size())
        
        #xconv3         = self.conv3(x2, adj, mask)
        xgat3          = self.gat3(x2, adj, mask)
        xgrconv3       = self.grconv3(x2, adj, mask)
        xsageconv3     = self.sageconv3(x2, adj, mask)
        xGINconv3      = self.GINconv3(x2, adj, mask)
        
        xchoice3       = self.skipconnect3([xgrconv3,xgat3,xsageconv3,xGINconv3])
        #x3             = self.bn3(xchoice3).relu()
        x3=xchoice3.relu()
        
        
        
        #x3 = F.dropout(x3, p=0.6, training=self.training)
     
        #print('x in layer 3 ', x3.size())            

        

        x = torch.cat([x1, x2, x3], dim=-1)
        #print('final x ', x.size())

        if self.lin is not None:
            x = self.lin(x).relu()
            #print('x lin ', x.size())
            


        return x

@model_wrapper
class Net(torch.nn.Module):
    def __init__(self,input_channel,hidden_channels,outchannel,attn_heads,max_nodes,droprate,percent_dec=0.25):
        super().__init__()

        num_nodes = ceil(percent_dec * max_nodes)#decreasing nodes
        
        '''Main layer 1'''

        
        self.gnn1_pool = GNN(input_channel, hidden_channels, num_nodes,attn_heads,droprate)#decrease of nodes layer-provides number of cluster to form next layer
        self.gnn1_embed = GNN(input_channel, hidden_channels, hidden_channels,attn_heads,droprate,lin=False)#this is x;provides actual number of nodes in this layer
        
        '''Main layer 2'''

        num_nodes  =  ceil(percent_dec * num_nodes)#decreasing nodes further
        

        in_channel =  (hidden_channels*attn_heads)+ (hidden_channels*(attn_heads//4))+ hidden_channels
        
        self.gnn2_pool  = GNN(in_channel, hidden_channels, num_nodes,attn_heads,droprate)
        self.gnn2_embed = GNN(in_channel, hidden_channels, hidden_channels,attn_heads,droprate,lin=False)

        '''Main layer 3'''
        
        self.gnn3_embed = GNN(in_channel, hidden_channels, hidden_channels,attn_heads,droprate,lin=False)
        
        '''Linear layer'''

        self.lin11 = torch.nn.Linear(2*in_channel, hidden_channels)
        self.lin12 = torch.nn.Linear(3*in_channel, hidden_channels)
        self.lin13 = torch.nn.Linear(in_channel, hidden_channels)
        self.lin2  = torch.nn.Linear(hidden_channels, outchannel)
        
        self.skipconnect11                      = nn.InputChoice(n_candidates=4)


    def forward(self, x, adj1,mask=None):
        
        '''Let us create 2 heirarchies here. One heirarchy will have 2 Diff-Pool layers so 3 layers in total - 1st, 2nd, 3rd. The next heirarchy will have only 1 DiffPool layer so 2 layers in total - 1st and 3rd'''
        

        '''Heirarchy 1'''


        '''layer 1'''
        hs11 = self.gnn1_pool(x, adj1, mask) #s
        
        #print('gnn1_embed')
        hx11 = self.gnn1_embed(x, adj1, mask) #x
        
        '''doing mean 1 for heirarchy 1'''
        hs_xmean_1 = hx11.mean(dim=1)
        
        '''Heirarchical Pooling layer 1'''

        hsx01, adj, l1, e1 = dense_diff_pool(hx11, adj1, hs11, mask)
        
        '''layer 2'''
        

        hs12 = self.gnn2_pool(hsx01, adj)
        
        #print('gnn2_embed')
        hx12 = self.gnn2_embed(hsx01, adj)
        
        '''doing mean 2'''
        hs_xmean_2 = hx12.mean(dim=1)
        
        '''Heirarchical Pooling layer 2'''
        
        hsx11, adj, l2, e2 = dense_diff_pool(hx12, adj, hs12)
        
        '''layer 3'''
        
        #print('gnn3_embed')
        hsx21 = self.gnn3_embed(hsx11, adj)
        
        '''Taking mean layer 3'''

        hs_xmean_3 = hsx21.mean(dim=1)
        
        #print('xmean_1 ',xmean_1.size())
        #print('xmean_2 ',xmean_2.size())
        #print('xmean_3 ',xmean_3.size())
        
        
        x11 = torch.cat([hs_xmean_1, hs_xmean_2, hs_xmean_3], dim=-1)
        x1 = self.lin12(x11).relu()

        
        x21 = torch.cat([hs_xmean_1, hs_xmean_2], dim=-1)
        x2 = self.lin11(x21).relu()

        
        x31 = torch.cat([hs_xmean_2, hs_xmean_3], dim=-1)
        x3 = self.lin11(x31).relu()

        
        x41 = torch.cat([hs_xmean_1, hs_xmean_3], dim=-1)
        x4 = self.lin11(x41).relu()

        #x51 = hs_xmean_3
        #x5 = self.lin13(x51).relu()        
        
        x = self.skipconnect11([x1,x2,x3,x4])


        
        x = self.lin2(x)
        #print('x  ',  x.size())
        return F.log_softmax(x, dim=1)