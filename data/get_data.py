import os, csv
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures


node_in_feats = 49
edge_in_feats = 10

atom_types = ['Li','B','C','N','O','F','Na','P','S','Cl','K','Fe','Br','Pd','I','Cs']

charge_types = [1, 2, -1]

degree_types = [1, 2, 3, 4]

hybridization_types = [Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3,
                       Chem.rdchem.HybridizationType.SP3D,
                       Chem.rdchem.HybridizationType.SP3D2]

numHs_types = [1, 2, 3]

valence_types = [1, 2, 3, 4, 5, 6]

chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

bond_types = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

bond_dir_types = [Chem.rdchem.BondDir.ENDUPRIGHT,
                  Chem.rdchem.BondDir.ENDDOWNRIGHT]

bond_stereo_types = [Chem.rdchem.BondStereo.STEREOE,
                     Chem.rdchem.BondStereo.STEREOZ]

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
                                   
def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
   
def smi_to_graph(smi):

    def one_hot_encoding(x, types, verbose = True):
        if verbose and x not in types:
            if x !=0 and x != Chem.rdchem.HybridizationType.S: print('missing', x, types)
            
        return list(map(lambda s: x == s, types))

    def atom_featurizer(j, a):
        fea = (one_hot_encoding(a.GetSymbol(), atom_types)
               + one_hot_encoding(a.GetFormalCharge(), charge_types)
               + one_hot_encoding(a.GetDegree(), degree_types)
               + one_hot_encoding(a.GetHybridization(), hybridization_types)
               + one_hot_encoding(a.GetTotalNumHs(), numHs_types)
               + one_hot_encoding(a.GetTotalValence(), valence_types)
               + one_hot_encoding(a.GetChiralTag(), chiral_types)
               + [(j in D_list), (j in A_list)]
               + [a.GetIsAromatic(), a.IsInRing()]
               + [a.IsInRingSize(s) for s in [3, 4, 5, 6, 7, 8]]
              )
    
        return fea

    def bond_featurizer(b):
        fea = (one_hot_encoding(b.GetBondType(), bond_types)
               + one_hot_encoding(b.GetBondDir(), bond_dir_types)
               + one_hot_encoding(b.GetStereo(), bond_stereo_types)
               + [b.IsInRing(), b.GetIsConjugated()]
              )
        
        return fea

    def _DA(mol):
        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list
    
    if smi == '':
        n_node, ndata, edata, src, dst = 0, np.empty((0, node_in_feats), dtype = bool), np.empty((0, edge_in_feats), dtype = bool), [], []
        
    else:
        mol = Chem.MolFromSmiles(smi)
        
        D_list, A_list = _DA(mol)

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
            
        ndata = np.array([atom_featurizer(j, a) for j, a in enumerate(mol.GetAtoms())], dtype = bool)
            
        if n_edge > 0:
            edata = np.array([bond_featurizer(b) for b in mol.GetBonds()], dtype = bool)
            edata = np.vstack([edata, edata])
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
            src, dst = np.hstack([bond_loc[:,0], bond_loc[:,1]]), np.hstack([bond_loc[:,1], bond_loc[:,0]])
        else:
            edata, src, dst = np.empty((0, edge_in_feats), dtype = bool), [], []

    mol_dict = {'n_node': n_node,
                'ndata': ndata,
                'edata': edata,
                'src': src,
                'dst': dst}

    return mol_dict
    
    
def get_graph_data(rsmi_list, yld_list, filename):

    rmol_dict = [[smi_to_graph(x) for x in reactants] for reactants in rsmi_list[:,:-1]]
    pmol_dict = [smi_to_graph(x) for x in rsmi_list[:,-1]]

    np.savez_compressed(filename, rmol = rmol_dict, pmol = pmol_dict, yld = yld_list) 

    
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    for data_id in [1, 2]:
        for split_id in range(10):
            load_dict = np.load(os.path.join(BASE_DIR, 'split', 'dataset_%d_%d.npz'%(data_id, split_id)))
            
            rsmi_list = load_dict['rxn']
            yld_list = load_dict['yld']
            
            filename = os.path.join(BASE_DIR, 'molgraph_dataset_%d_%d.npz'%(data_id, split_id))
            print(filename)            
            get_graph_data(rsmi_list, yld_list, filename)

    for test_id in [1, 2, 3, 4]:
        load_dict = np.load(os.path.join(BASE_DIR, 'split', 'test_%d.npz'%test_id))

        rsmi_list = load_dict['rxn']
        yld_list = load_dict['yld']

        filename = os.path.join(BASE_DIR, './molgraph_test_%d.npz'%test_id)
        print(filename)
        get_graph_data(rsmi_list, yld_list, filename)
