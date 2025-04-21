import torch
import dgl

                        
def collate_reaction_graphs(batch):

    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(batchdata[-1])
    
    return *gs, labels


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass



list_smiles_1 = {
	'P2Et': 'CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC',
	'BTMG': 'CN(C)C(N(C)C)=NC(C)(C)C',
	'MTBD': 'CN1CCCN2CCCN=C12',
	'1-chloro-4-(trifluoromethyl)benzene': 'FC(F)(F)c1ccc(Cl)cc1',
	'1-bromo-4-(trifluoromethyl)benzene': 'FC(F)(F)c1ccc(Br)cc1',
	'1-iodo-4-(trifluoromethyl)benzene': 'FC(F)(F)c1ccc(I)cc1',
	'1-chloro-4-methoxybenzene': 'COc1ccc(Cl)cc1',
	'1-bromo-4-methoxybenzene': 'COc1ccc(Br)cc1',
	'1-iodo-4-methoxybenzene': 'COc1ccc(I)cc1',
	'1-chloro-4-ethylbenzene': 'CCc1ccc(Cl)cc1',
	'1-bromo-4-ethylbenzene': 'CCc1ccc(Br)cc1',
	'1-ethyl-4-iodobenzene': 'CCc1ccc(I)cc1',
	'2-chloropyridine': 'Clc1ccccn1',
	'2-bromopyridine': 'Brc1ccccn1',
	'2-iodopyridine': 'Ic1ccccn1',
	'3-chloropyridine': 'Clc1cccnc1',
	'3-bromopyridine': 'Brc1cccnc1',
	'3-iodopyridine': 'Ic1cccnc1',
	'4-phenylisoxazole': 'o1cc(cn1)c2ccccc2',
	'5-phenylisoxazole': 'o1nccc1c2ccccc2',
	'3-phenylisoxazole': 'o1ccc(n1)c2ccccc2',
	'ethyl-3-methylisoxazole-5-carboxylate': 'CCOC(=O)c1onc(C)c1',
	'3-methylisoxazole': 'Cc1ccon1',
	'ethyl-5-methylisoxazole-3-carboxylate': 'CCOC(=O)c1cc(C)on1',
	'5-phenyl-1,2,4-oxadiazole': 'c1ccc(-c2ncno2)cc1',
	'5-methylisoxazole': 'Cc1oncc1',
	'ethyl-isoxazole-3-carboxylate': 'CCOC(=O)c1ccon1',
	'benzo[c]isoxazole': 'o1cc2ccccc2n1',
	'ethyl-5-methylisoxazole-4-carboxylate': 'CCOC(=O)c1cnoc1C',
	'3,5-dimethylisoxazole': 'Cc1onc(C)c1',
	'ethyl-isoxazole-4-carboxylate': 'CCOC(=O)c1conc1',
	'methyl-isoxazole-5-carboxylate': 'COC(=O)c1oncc1',
	'benzo[d]isoxazole': 'o1ncc2ccccc12',
	'5-(2,6-difluorophenyl)isoxazole': 'Fc1cccc(F)c1c2oncc2',
	'3-methyl-5-phenylisoxazole': 'Cc1cc(on1)c2ccccc2',
	'N,N-dibenzylisoxazol-5-amine': 'C(N(Cc1ccccc1)c2oncc2)c3ccccc3',
	'N,N-dibenzylisoxazol-3-amine': 'C(N(Cc1ccccc1)c2ccon2)c3ccccc3',
	'5-methyl-3-(1H-pyrrol-1-yl)isoxazole': 'Cc1onc(c1)n2cccc2',
	'methyl-5-(furan-2-yl)isoxazole-3-carboxylate': 'COC(=O)c1cc(on1)c2occc2',
	'methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate': 'COC(=O)c1cc(on1)c2sccc2',
	'ethyl-3-methoxyisoxazole-5-carboxylate': 'CCOC(=O)c1onc(OC)c1',
	'XPhos': 'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2',
	't-BuXPhos': 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C',
	't-BuBrettPhos': 'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC',
	'AdBrettPhos': 'CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C',
}

list_smiles_2 = {
	'6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
	'6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
	'6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
	'6-iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
	'6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
	'potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
	'6-quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O',
	'2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
	'2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
	'2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
	'2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br',
	'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]',
	'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
	'P(Ph)3': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
	'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
	'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
	'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
	'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
	'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
	'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]', 
	'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
	'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]', 
	'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
	'NaOH': '[OH-].[Na+]', 
	'NaHCO3': '[Na+].OC([O-])=O', 
	'CsF': '[F-].[Cs+]', 
	'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 
	'KOH': '[K+].[OH-]', 
	'LiOtBu': '[Li+].[O-]C(C)(C)C', 
	'Et3N': 'CCN(CC)CC', 
	'MeCN': 'CC#N.O', 
	'THF': 'C1CCOC1.O', 
	'DMF': 'CN(C)C=O.O', 
	'MeOH': 'CO.O', 
	'MeOH/H2O_V2 9:1': 'CO.O', 
	'THF_V2': 'C1CCOC1.O',
}
