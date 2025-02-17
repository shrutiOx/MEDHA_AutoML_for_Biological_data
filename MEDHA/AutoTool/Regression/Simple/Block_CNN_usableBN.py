# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:24:40 2023

@author: Shruti Sarika Chakraborty
"""

# import libraries
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper



import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 

import scipy.stats as stats
import sklearn.metrics as skm
# libraries for partitioning and batching the data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

import nni

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)

'''
Think-Tank

So let's create the range of blocks/cells to be included in the model space
'''
'''defining input block
conv = (inchanel-outchannel)
pooling type and size decided by DARTS
kernel type decided by DARTs.
Outputs processed input and out-channel length
'''


class InputBlock(nn.Module):
    def __init__(self, in_channel,
                       inputkernels,#list of 3 numbers and each input should be a number
                       out_channel,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.actfun = actfun
        self.layers    = nn.ModuleDict()
        self.listinput = []
        self.pool_size = pool_size
        self.out_channel_inputlayer = out_channel
        self.drop = drop
        for i in range(0,3 ): 
          self.layers[f'InputBlocks{i}']  = nn.Conv2d(in_channel,
                                                      out_channel,
                                                      inputkernels[i],
                                                      padding  =  ((inputkernels[i]-1)//2))
          self.listinput.append(self.layers[f'InputBlocks{i}'])
        '''make layer choice from three kinds of kernels'''
        self.layers['inputlayer']           =  nn.LayerChoice(self.listinput)

        self.skipconnect                     = nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)
       
        

    def forward(self, x,flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        '''input block'''
        if (x.shape[-1] > self.pool_size):
            x0  = actfun(F.max_pool2d( self.layers['inputlayer'](x),self.pool_size )) #can use F.avg_pool2d
        
        if (x.shape[-1] > self.pool_size):
            x01 = actfun(F.avg_pool2d( self.layers['inputlayer'](x),self.pool_size)) 

            x = self.skipconnect([x0, x01])
        
        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])
        
        return xf
    
    def outC(self):
        return self.out_channel_inputlayer
    











'''Testing input block-The model makes choices of the kernel size in input layer by LayerChoice and also the pool-size and pooling type (max/avg) by InputChoice. Input channel and Output channel is different'''





'''Now starting level 1'''

'''Block 1'''


class Block1(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       block1kernels,#list of 3 numbers and each input should be a number
                       out_channel_i,
                       out_channel_i2,
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.actfun = actfun
        self.layers     = nn.ModuleDict()
        self.listblock1 = []
        self.listblock2 = []
        self.listblock3 = []
        self.out_channel_i = out_channel_i
        self.out_channel_i2 = out_channel_i2
        self.out_channel_f = out_channel_f

        
        
        
        self.drop = drop 
        
        self.pool_size  = pool_size
        self.out_channel = out_channel_f
        
        for i in range(0,3 ): 
          self.layers[f'Blocks1_1{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      out_channel_i,
                                                      block1kernels[i],
                                                      padding  =  ((block1kernels[i]-1)//2))
          self.listblock1.append(self.layers[f'Blocks1_1{i}'])
          
        '''make layer choice from three kinds of kernels for Block1_1'''
        self.layers['Blocks1_1']           =  nn.LayerChoice(self.listblock1)
        self.batchnorm_Bl1_1                  = nn.BatchNorm2d(self.out_channel_i)
        
        for i in range(0,3 ): 
          self.layers[f'Blocks1_2{i}']  = nn.Conv2d(out_channel_i,
                                                      out_channel_i2,
                                                      block1kernels[i],
                                                      padding  =  ((block1kernels[i]-1)//2))
          self.listblock2.append(self.layers[f'Blocks1_2{i}'])
          self.batchnorm_Bl1_2                  = nn.BatchNorm2d(self.out_channel_i2)
          
        '''make layer choice from three kinds of kernels for Block1_2'''
        self.layers['Blocks1_2']           =  nn.LayerChoice(self.listblock2)
        
        for i in range(0,3 ): 
          self.layers[f'Blocks1_3{i}']  = nn.Conv2d(out_channel_i2,
                                                      out_channel_f,
                                                      block1kernels[i],
                                                      padding  =  ((block1kernels[i]-1)//2))
          self.listblock3.append(self.layers[f'Blocks1_3{i}'])
         
        '''make layer choice from three kinds of kernels for Blocks1_3'''
        self.layers['Blocks1_3']           =  nn.LayerChoice(self.listblock3)

        
        self.skipconnect                    =  nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)
       

    def forward(self, x, flag=True):

        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        '''input block'''
        if flag==True:
            #print('for SS')
            x = actfun( self.layers['Blocks1_1'](x))
            x = actfun( self.layers['Blocks1_2'](x))
            x = actfun( self.layers['Blocks1_3'](x)) 
        else:
            #print('for x')
            x = actfun( self.layers['Blocks1_1'](x))
            x = (self.batchnorm_Bl1_1(x))
            x = actfun( self.layers['Blocks1_2'](x))
            x = (self.batchnorm_Bl1_2(x))
            x = actfun( self.layers['Blocks1_3'](x)) 
  
            
        if (x.shape[-1] > self.pool_size):
            x00 = F.max_pool2d(x,self.pool_size)
            

        
        if (x.shape[-1] > self.pool_size):
            x10 = F.avg_pool2d(x,self.pool_size )
            x = self.skipconnect([x00, x10])
        
        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])
        
        return xf
    
    def outC(self):
        return self.out_channel
    

'''Testing Block 1- The model makes choices of the kernel size in each layer by LayerChoice ( kernel-type per layer may be different ) and also the pool-size and pooling type (max/avg) by InputChoice Three types of channels are used. Input channel = output of Input layer; outchanel_intermediate1;outchanel_intermediate2; out_channel_final = this will be same for the outputs of all Blocks and is a common variable'''





class Block2(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       block2kernels,#list of 3 numbers and each input should be a number
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.layers     = nn.ModuleDict()
        self.out_channel = out_channel_f
        self.listblock2A = []
        self.actfun = actfun
        
        self.drop = drop

        self.pool_size  = pool_size
        
        for i in range(0,3 ): 
          self.layers[f'Blocks2{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      out_channel_f,
                                                      block2kernels[i],
                                                      padding  =  ((block2kernels[i]-1)//2))
          self.listblock2A.append(self.layers[f'Blocks2{i}'])
          
          
        '''make layer choice from three kinds of kernels for Block1_1'''
        self.layers['Blocks2']              =  nn.LayerChoice(self.listblock2A)

        
        
        self.skipconnect                   =  nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)


    def forward(self, x, flag=True):

        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        '''input block'''
        
        x = actfun( self.layers['Blocks2'](x))
        
        if (x.shape[-1] > self.pool_size):
            x00 = F.max_pool2d(x,self.pool_size )

        if (x.shape[-1] > self.pool_size):
            x10 = F.avg_pool2d(x,self.pool_size )

            x = self.skipconnect([x00, x10])
        
        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])

       # if self.dropoutflag == True:
         #   xB1 = F.dropout(xB1,p=self.dropinit,training=self.training)
        return xf
    def outC(self):
        return self.out_channel
     
        
'''Testing Block 2- The model makes choices of the kernel size in each layer by LayerChoice  and also the pool-size and pooling type (max/avg) by InputChoice  Input channel = output of Input layer;  out_channel_final = this will be same for the outputs of all Blocks and is a common variable'''




        
class Block3(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       block3kernels,#list of 3 numbers and each input should be a number
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.layers     = nn.ModuleDict()
        self.out_channel = out_channel_f
        self.listblock3_0 = []
        self.listblock3_1 = []
        self.actfun = actfun
        self.out_channel_inputlayer = out_channel_inputlayer

        self.pool_size  = pool_size
        self.drop = drop
        
        for i in range(0,3 ): 
          self.layers[f'Blocks3_0{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      out_channel_inputlayer,
                                                      (block3kernels[i],1),
                                                      padding  =  ((block3kernels[i]-1)//2))
          
          self.listblock3_0.append(self.layers[f'Blocks3_0{i}'])
          self.batchnorm_Bl3_0                  = nn.BatchNorm2d(self.out_channel_inputlayer)
          
          self.layers[f'Blocks3_1{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      out_channel_f,
                                                      (1,block3kernels[i]),
                                                      padding  =  ((block3kernels[i]-1)//2))
          
          self.listblock3_1.append(self.layers[f'Blocks3_1{i}'])


        
        self.skipconnectBL                   =  nn.InputChoice(n_candidates=3)

        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)

       

    def forward(self, x, flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        ''' block choice'''
        
        
        x0 = actfun(  self.listblock3_0[0](x))
        x0 = actfun(  self.listblock3_1[0](x0))
        
        x1= actfun(  self.listblock3_0[0](x))
        x1 = actfun(  self.listblock3_1[0](x1))
        
        x2= actfun(  self.listblock3_0[0](x))
        x2 = actfun(  self.listblock3_1[0](x2))
        
        x_block = self.skipconnectBL([x0, x1,x2])
        if flag == False:
            x_block = (self.batchnorm_Bl3_0(x))
    
        if (x_block.shape[-1] > self.pool_size):
            x_block = F.avg_pool2d(x_block,self.pool_size )
            
        xB1 = F.dropout(x_block,p=self.drop)
        xB0 = x_block

        xf = self.skipconnectD([xB1, xB0])


        return xf
    def outC(self):
        return self.out_channel
    
    
'''Testing Block 3- The important difference is that the kernel size in first layer is (1,k) while in second is (k,1) .k = same for both layers. The pool-type is average while pool-size is InputChoice. The kernel-size in layers is chosen byInputChoice'''




  


class Block4(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       block4kernels,#list of 3 numbers and each input should be a number
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.layers     = nn.ModuleDict()
        self.listblock41 = []
        self.listblock42 = []
        self.listblock43 = []
        self.listblock44 = []
        self.actfun = actfun

        
        self.pool_size  = pool_size
        self.out_channel = out_channel_f
        self.drop = drop

        '''Block_4_1'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks4_1{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      64,
                                                      block4kernels[i],
                                                      padding  =  ((block4kernels[i]-1)//2))
          self.listblock41.append(self.layers[f'Blocks4_1{i}'])
          
          
        '''make layer choice from three kinds of kernels for Blocks4_1'''
        self.layers['Blocks4_1']           =  nn.LayerChoice(self.listblock41)
        self.batchnorm_B41                  = nn.BatchNorm2d(64)
        
        '''Block_4_2'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks4_2{i}']  = nn.Conv2d(64,
                                                     128 ,
                                                      block4kernels[i],
                                                      padding  =  ((block4kernels[i]-1)//2))
          self.listblock42.append(self.layers[f'Blocks4_2{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks4_2'''
        self.layers['Blocks4_2']           =  nn.LayerChoice(self.listblock42)
        self.batchnorm_B42                  = nn.BatchNorm2d(128)
        
        '''Block_4_3'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks4_3{i}']  = nn.Conv2d(  128,
                                                      256 ,
                                                      block4kernels[i],
                                                      padding  =  ((block4kernels[i]-1)//2))
          self.listblock43.append(self.layers[f'Blocks4_3{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks4_3'''
        self.layers['Blocks4_3']           =  nn.LayerChoice(self.listblock43)
        self.batchnorm_B43                  = nn.BatchNorm2d(256)
        
        '''Block_4_4'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks4_4{i}']  = nn.Conv2d(  256,
                                                      out_channel_f ,
                                                      block4kernels[i],
                                                      padding  =  ((block4kernels[i]-1)//2))
          self.listblock44.append(self.layers[f'Blocks4_4{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks4_4'''
        self.layers['Blocks4_4']           =  nn.LayerChoice(self.listblock44)



        
        self.skipconnect01                   =  nn.InputChoice(n_candidates=2)
        self.skipconnect02                   =  nn.InputChoice(n_candidates=2)
        self.skipconnect03                   =  nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)
       

    def forward(self, x, flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        '''intermediate block 1'''
        
        xb0 = actfun( self.layers['Blocks4_1'](x))
        if flag == False:
            xb0 = (self.batchnorm_B41(xb0))
        
        if (xb0.shape[-1] > self.pool_size):
            xb00 = F.max_pool2d(xb0,self.pool_size )
            

        
        if (xb0.shape[-1] > self.pool_size):
            xb02 = F.avg_pool2d(xb0,self.pool_size )

            xb0 = self.skipconnect01([xb00, xb02])
        
        '''intermediate block 2'''
        
        xb1 = actfun( self.layers['Blocks4_2'](xb0))
        if flag == False:
            xb1 = (self.batchnorm_B42(xb1))
        
        if (xb1.shape[-1] > self.pool_size):
            xb10 = F.max_pool2d(xb1,self.pool_size )

        if (xb1.shape[-1] > self.pool_size):
            xb12 = F.avg_pool2d(xb1,self.pool_size )

            xb1 = self.skipconnect02([xb10, xb12])
        
        '''last intermediate & final block'''
        
        xb3 = actfun( self.layers['Blocks4_3'](xb1)) 
        if flag == False:
            xb3 = (self.batchnorm_B43(xb3))
            
        x = actfun( self.layers['Blocks4_4'](xb3))
        
        if (x.shape[-1] > self.pool_size):
            x00 = F.max_pool2d(x,self.pool_size )
            

        
        if (x.shape[-1] > self.pool_size):
            x10 = F.avg_pool2d(x,self.pool_size )

            x = self.skipconnect03([x00,x10])
        
        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])

       # if self.dropoutflag == True:
         #   xB1 = F.dropout(xB1,p=self.dropinit,training=self.training)
        return xf
    
    def outC(self):
        return self.out_channel
    

'''Testing Block 4- A VGG-Net inspired 4 layered structure - conv(in,64)->kernel(layerchoice)->avg/max pool and pool size/type chosen by InputChoice -> conv(64,128)->kernel(layerchoice)->avg/max pool and pool size/type chosen by InputChoice-> conv(128,256)->kernel(layerchoice)->conv(256,final)->kernel(layerchoice)-> max/avg and pool-size determined by InputChoice'''






class Block5(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       block5kernels,#list of 3 numbers and each input should be a number
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.layers     = nn.ModuleDict()
        self.listblock51 = []
        self.listblock52 = []
        self.listblock53 = []
        self.listblock54 = []
        self.actfun = actfun

        self.drop = drop
        self.pool_size  = pool_size
        self.out_channel = out_channel_f

        '''Block_5_1'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks5_1{i}']  = nn.Conv2d(out_channel_inputlayer,
                                                      64,
                                                      block5kernels[i],
                                                      padding  =  ((block5kernels[i]-1)//2))
          self.listblock51.append(self.layers[f'Blocks5_1{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks5_1'''
        self.layers['Blocks5_1']           =  nn.LayerChoice(self.listblock51)
        self.batchnorm_B51                  = nn.BatchNorm2d(64)
        
        '''Block_5_2'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks5_2{i}']  = nn.Conv2d(64,
                                                     256 ,
                                                      block5kernels[i],
                                                      padding  =  ((block5kernels[i]-1)//2))
          self.listblock52.append(self.layers[f'Blocks5_2{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks5_2'''
        self.layers['Blocks5_2']           =  nn.LayerChoice(self.listblock52)
        self.batchnorm_B52                  = nn.BatchNorm2d(256)
        
        '''Block_5_3'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks5_3{i}']  = nn.Conv2d(  256,
                                                      64 ,
                                                      block5kernels[i],
                                                      padding  =  ((block5kernels[i]-1)//2))
          self.listblock53.append(self.layers[f'Blocks5_3{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks5_3'''
        self.layers['Blocks5_3']           =  nn.LayerChoice(self.listblock53)
        self.batchnorm_B53                  = nn.BatchNorm2d(64)
        
        '''Block_5_4'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks5_4{i}']  = nn.Conv2d(  64,
                                                      out_channel_f ,
                                                      block5kernels[i],
                                                      padding  =  ((block5kernels[i]-1)//2))
          self.listblock54.append(self.layers[f'Blocks5_4{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks5_4'''
        self.layers['Blocks5_4']           =  nn.LayerChoice(self.listblock54)


        
        self.skipconnect01                   =  nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)
       

    def forward(self, x, flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        '''intermediate block 1'''
        
        
        
        x = actfun( self.layers['Blocks5_1'](x))
        if flag == False:
            x = (self.batchnorm_B51(x))
        xclone = x.clone()
        x = actfun( self.layers['Blocks5_2'](x))
        if flag == False:
            x = (self.batchnorm_B52(x))
        x = actfun( self.layers['Blocks5_3'](x))
        if flag == False:
            x = (self.batchnorm_B53(x))
        x = x + xclone
        x = actfun( self.layers['Blocks5_4'](x))
        
        
        
        if (x.shape[-1] > self.pool_size):
            x00 = F.max_pool2d(x,self.pool_size )
            

        
        if (x.shape[-1] > self.pool_size):
            x10 = F.avg_pool2d(x,self.pool_size )
        
            x = self.skipconnect01([x00,x10])
        
        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])

       # if self.dropoutflag == True:
         #   xB1 = F.dropout(xB1,p=self.dropinit,training=self.training)
        return xf
    
    def outC(self):
        return self.out_channel
    

'''Testing Block 5- A Res-Net inspired 4 layered structure - conv(in,64)->kernel(layerchoice)--> conv(64,64)->kernel(layerchoice)--> conv(64,256)->kernel(layerchoice)->conv(256,final)->kernel(layerchoice)-> max/avg and pool-size determined by InputChoice'''






class Block6(nn.Module):
    def __init__(self, out_channel_inputlayer,
                       increment,
                       num_conv_layers,
                       block6kernels,#list of 3 numbers and each input should be a number
                       out_channel_f,
                       pool_size,
                       actfun,
                       drop):#a list of 2 inputs and each input should be a number
        super().__init__()
        self.layers     = nn.ModuleDict()
        self.listblock61 = []
        self.listblock62 = []
        self.listblock63 = []
        self.listblock64 = []
        self.num_conv_layers = num_conv_layers

        self.actfun = actfun
        self.drop = drop
        
        self.pool_size  = pool_size
        self.out_channel = out_channel_f
        self.out_channel_inputlayer = out_channel_inputlayer
        
       

        '''Other convolution layers:Block_6_2'''
        
        '''Doing with 3 types of kernel sizes each time'''
        
        for i in range(self.num_conv_layers):
            

          
          self.layers[f'hiddenconv0{i}'] = nn.Conv2d(self.out_channel_inputlayer,
                                                    self.out_channel_inputlayer+(increment),
                                                    block6kernels[0],
                                                    padding  =  ((block6kernels[0]-1)//2))
          


          self.out_channel_inputlayer=self.out_channel_inputlayer+(increment)
          self.layers[f'batchnorm0{i}'] = nn.BatchNorm2d(self.out_channel_inputlayer)
          
        
          
        out_chanel_ii = self.out_channel_inputlayer

          
        
        '''Block_6_3'''
        
        for i in range(0,3 ): 
          self.layers[f'Blocks6_3{i}']  = nn.Conv2d(out_chanel_ii,
                                                     out_channel_f ,
                                                      block6kernels[0],
                                                      padding  =  ((block6kernels[0]-1)//2))
          self.listblock63.append(self.layers[f'Blocks6_3{i}'])
          
        '''make layer choice from three kinds of kernels for Blocks5_2'''
        self.layers['Blocks6_3']             =  nn.LayerChoice(self.listblock63)

       
        self.skipconnect01                   =  nn.InputChoice(n_candidates=3)
        self.skipconnect02                   =  nn.InputChoice(n_candidates=2)
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)

    def forward(self, x, flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        

        
        '''doing for hiddenconv0'''
        for i in range(self.num_conv_layers):
            x = actfun(( self.layers[f'hiddenconv0{i}'](x) ))
            #x = F.dropout(x,p=0.4)
           # x = actfun(self.layers[f'batchnorm{i}'](x))
            if flag==False:
                x = (self.layers[f'batchnorm0{i}'](x))

            
        xin = x
        xin = actfun( self.layers['Blocks6_3'](xin))

        
        if (xin.shape[-1] > self.pool_size):
            x00 = F.max_pool2d(xin,self.pool_size )
        
        
        if (xin.shape[-1] > self.pool_size):
            x10 = F.avg_pool2d(xin,self.pool_size )

            xin = self.skipconnect02([x00, x10])
        
        xB1 = F.dropout(xin,p=self.drop)
        xB0 = xin

        xf = self.skipconnectD([xB1, xB0])
  
        
       # if self.dropoutflag == True:
         #   xB1 = F.dropout(xB1,p=self.dropinit,training=self.training)
        return xf
    
    def outC(self):
        return self.out_channel
    

'''Testing Block 6- Adaptive-CNN inspired network. conv(in,oC_in) ->hidden layers based on num_of_conv. Kernels decided by InputChoice. Oc_i{i} regulated by increments. Activation and Batchnorm applied to each hiddenlayers->conv(Oci_{i-last},Oc_final)-->max/avg pool and pool-type decided by InputChoice'''






class Block7(nn.Module):
    def __init__(self, out_channel_inputlayer,  out_channel_f,actfun,drop,pool_size):
        super().__init__()
        self.out_channel = out_channel_f
        self.actfun = actfun
        self.pool_size = pool_size
        
        self.depthwise = nn.Conv2d(out_channel_inputlayer, out_channel_inputlayer, kernel_size=3, groups=out_channel_inputlayer,padding  =  1)
        self.pointwise = nn.Conv2d(out_channel_inputlayer, out_channel_f, kernel_size=1,padding  =  0)
        
        self.skipconnect                   =  nn.InputChoice(n_candidates=2)
        
        self.skipconnectD                   =  nn.InputChoice(n_candidates=2)
        
        self.drop = drop


    def forward(self, x, flag=True):
        
        actfun = getattr(torch.nn,self.actfun)
        actfun = actfun()
        
        
        x= self.pointwise(self.depthwise(x))
        
        if (x.shape[-1] > self.pool_size):
            x00 = actfun(F.max_pool2d(x,self.pool_size ))
            

        
        if (x.shape[-1] > self.pool_size):
            x10 = actfun(F.avg_pool2d(x,self.pool_size ))

            x = self.skipconnect([x00,x10])

        xB1 = F.dropout(x,p=self.drop)
        xB0 = x

        xf = self.skipconnectD([xB1, xB0])        
        return xf
    
    def outC(self):
        return self.out_channel
    
'''Testing Block 7- DepthSepConvolution-Legacy--no-pooling'''




