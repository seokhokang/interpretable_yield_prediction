import os, sys
import numpy as np
import torch
from dgl.convert import graph


class GraphDataset():

    def __init__(self, data_id, split_id):

        self._data_id = data_id
        self._split_id = split_id
        self.load()


    def load(self):
        
        if self._data_id in [1, 2]:
            data = np.load('./data/molgraph_dataset_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)
            rsmi = np.load('./data/split/dataset_%d_%d.npz'%(self._data_id, self._split_id))
        elif self._data_id == 3:
            data = np.load('./data/molgraph_test_%d.npz' %self._split_id, allow_pickle=True)
            rsmi = np.load('./data/split/test_%d.npz'%self._split_id)

        rmol_dict, pmol_dict, yld = data['rmol'], data['pmol'], data['yld']
        self.rsmi_list = rsmi['rxn']
        
        use_col = []
        for j in range(self.rsmi_list.shape[1]-1):
            if len(np.unique(self.rsmi_list[:,j])) > 1:
                use_col.append(j)
    
        self.rmols = rmol_dict[:,use_col]
        self.pmols = pmol_dict
        
        self.rsmi_list =  self.rsmi_list[:,use_col]
        self.yld = yld


    def to_graph(self,mol):
        
        g = graph((mol['src'], mol['dst']), num_nodes = mol['n_node'])
        g.ndata['node_feats'] = torch.FloatTensor(mol['ndata'])
        g.edata['edge_feats'] = torch.FloatTensor(mol['edata'])
    
        return g
    

    def __getitem__(self, idx):

        rs = [self.to_graph(x) for x in self.rmols[idx]]
        p = self.to_graph(self.pmols[idx])

        label = self.yld[idx]
        
        return *rs, p, label
        
        
    def __len__(self):

        return self.yld.shape[0]
