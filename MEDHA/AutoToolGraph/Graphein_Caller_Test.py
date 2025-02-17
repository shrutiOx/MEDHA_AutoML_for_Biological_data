# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 05:35:55 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:21:30 2023

@author: ADMIN
"""



import torch

from MEDHA.AutoToolGraph.searchspace import GNN,Net


from MEDHA.AutoToolGraph.Dartstrainer_graph   import DartTrainer


from MEDHA.AutoToolGraph.Semimanualtrain   import SemiManualDart_train
from MEDHA.AutoToolGraph.crossvalidation   import cross_validation_with_val_set

import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,DenseGCNConv,DenseGINConv,DenseGraphConv,DenseGATConv
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

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import torch_geometric.transforms as T 
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelBinarizer
import pytorch_lightning as pl
#from tqdm.notebook import tqdm
from tqdm import tqdm

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_k_nn_edges
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot,meiler_embedding,expasy_protein_scale
from graphein.protein.features.sequence.embeddings import esm_sequence_embedding,biovec_sequence_embedding
from graphein.protein.features.sequence.sequence import molecular_weight
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial

from graphein.ml import ProteinGraphListDataset, GraphFormatConvertor,InMemoryProteinGraphDataset
import torch_geometric.transforms as T 
print('All imports completed')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()




'''
In "Free PDB" column of df u have the main IDs from which Graphein will get structures.
y is actually creating labels based on the type  in coloumn - "motion_type" in df dividing into a few classes
'''

'''Defining global functons necessary for edge-construction and node features'''
def spatial_graph_proteins(G: nx.Graph, distance_threshold: float = 5.0) -> nx.Graph:
    """
    Connect protein residues based on spatial distance threshold.

    Args:
        G (nx.Graph): Input graph with protein nodes
        distance_threshold (float): Maximum distance threshold for creating edges (Å)

    Returns:
        nx.Graph: Graph with added spatial edges between protein residues
    """
    # Standard amino acid residue names
    RESI_NAMES = [
        "5HP", "ABA", "ACE", "AIB", "ALA", "ARG", "ASN", "ASP", "ASH", "ASX",
        "BMT", "BOC", "CBX", "CEA", "CGU", "CME", "CRO", "CSD", "CSO", "CSS",
        "CSW", "CSX", "CXM", "CYS", "DAL", "DAR", "DCY", "DGL", "DGN", "DHI",
        "DIL", "DIV", "DLE", "DLY", "DPN", "DPR", "DSG", "DSN", "DSP", "DTH",
        "DTR", "DTY", "DVA", "FME", "FOR", "GLN", "GLU", "GLH", "GLX", "GLY",
        "HID", "HIE", "HIP", "HIS", "HYP", "ILE", "IVA", "KCX", "LEU", "LLP",
        "LYS", "LYN", "MET", "MLE", "MSE", "MVA", "NH2", "NLE", "NLW", "OCS",
        "ORN", "PCA", "PHE", "PRO", "PSW", "PTR", "PVL", "PYL", "SAR", "SEC",
        "SEP", "SER", "STY", "THR", "TPO", "TPQ", "TRP", "TYR", "TYS", "UNK",
        "VAL"
    ]

    # Store node coordinates and identify protein nodes
    node_coords = {}
    protein_nodes = []
    
    print("Identifying protein nodes and extracting coordinates...", flush=True)
    for node, data in G.nodes(data=True):
        try:
            # Extract coordinates
            if 'coords' in data:
                coords = data['coords']
            else:
                coords = np.array([
                    float(data.get('x_coord', 0)),
                    float(data.get('y_coord', 0)),
                    float(data.get('z_coord', 0))
                ])
            node_coords[node] = coords
            
            # Identify protein nodes by residue name
            residue_3letter = node.split(':')[1]
            if residue_3letter in RESI_NAMES:
                protein_nodes.append(node)
                
        except Exception as e:
            print(f"Warning: Could not process node {node}: {e}", flush=True)
            continue
    
    print(f"Found {len(protein_nodes)} protein nodes", flush=True)
    
    # Add edges based on distance threshold
    print("Computing distances and adding edges...", flush=True)
    edges_to_add = []
    
    # Iterate through protein nodes to create edges
    for i, node1 in enumerate(protein_nodes):
        coords1 = node_coords[node1]
        residue1 = G.nodes[node1].get("residue_name", "")
        
        # Only look at remaining nodes to avoid duplicate edges
        for node2 in protein_nodes[i+1:]:
            coords2 = node_coords[node2]
            residue2 = G.nodes[node2].get("residue_name", "")
            
            # Calculate distance
            dist = np.linalg.norm(coords1 - coords2)
            
            # Add edge if within threshold
            if dist <= distance_threshold:
                edges_to_add.append((
                    node1,
                    node2,
                    {
                        'kind': {'protein_prot_interaction'},
                        'distance': float(dist),
                        'residue1': residue1,
                        'residue2': residue2
                    }
                ))
    
    # Add all edges to graph at once
    G.add_edges_from(edges_to_add)
    print(f"Added {len(edges_to_add)} spatial edges", flush=True)
    
    return G





        
class Graphein_Caller_Test(nn.Module):
    def __init__(self, 
                 datacsv='None',
                 name='protein_1',
                 batch_size=5):
        super().__init__()
        self.datacsv    = datacsv
        self.name=name
        self.batch_size=batch_size

    def GetData(self):
        df  = pd.read_csv(self.datacsv) # reads the excel file into a dataframe

        pdbs = df["PDB_ID"]
        y    = torch.tensor(df["Labels"])
        
        constructors = {
            #"edge_construction_functions": [partial(add_k_nn_edges, k=3, long_interaction_threshold=0)],
            #"edge_construction_functions": [ add_peptide_bonds],
            "edge_construction_functions": [partial(spatial_graph_proteins, distance_threshold=5)],
            #"node_metadata_functions": [meiler_embedding],
            "node_metadata_functions": [amino_acid_one_hot],
            #"node_metadata_functions": [expasy_protein_scale],
            #"graph_metadata_functions": [molecular_weight]
            
        }

        config = ProteinGraphConfig(**constructors)
        print(config.dict())




        # Make graphs
        graph_list = []
        y_list = []

        for idx, pdb in enumerate(tqdm(pdbs)):
            try:
                graph_list.append(
                    construct_graph(pdb_code=pdb,
                                config=config
                               )
                    )
                y_list.append(y[idx])
            except:
                print(str(idx) + ' processing error...')
                pass



        '''
        Convert Nx graphs to PyTorch Geometric
        '''

        format_convertor = GraphFormatConvertor('nx', 'pyg',
                                                columns=['meiler','coords','edge_index','name','node_id','b_factor','amino_acid_one_hot','mask']
                                                )

        pyg_list = [format_convertor(graph) for graph in tqdm(graph_list)]

        listi=[]


        count=0


        for i in pyg_list:
            if i.coords.shape[0] == len(i.node_id):
                #print(len(i.node_id))
                pass
            else:
                pyg_list.remove(i)
                
        while(count<15):
            for i in pyg_list:
                if (len(i.node_id) > 800):

                    pyg_list.remove(i)

            count += 1
            
            

                
        for i in pyg_list:
            listi.append(i.num_nodes)
            
        max_nodes = np.max(listi)

        print('max_nodes ',max_nodes)


                

        for idx, g in enumerate(pyg_list):
            g.y = y_list[idx]
            #g.coords = torch.FloatTensor(g.coords[0].float())
            #print(g.meiler)
            #b_factor = np.asmatrix(g.b_factor)
            #g.x = np.concatenate((b_factor,g.amino_acid_one_hot),axis=1)
            g.x = g.amino_acid_one_hot
            g.amino_acid_one_hot=None
            g.node_id=None
            g.coords=None
            g.b_factor=None
            g.name=None
            g.num_nodes=None
           

        ds = ProteinGraphListDataset(root=".", data_list=pyg_list, name=self.name,transform=T.ToDense(max_nodes))
        print('ds ',ds)

        #print('ds[0] ',ds[0])

        dataset = ds.shuffle()


        test_loader = DenseDataLoader(dataset, batch_size=self.batch_size)




        
        return dataset,test_loader,max_nodes
