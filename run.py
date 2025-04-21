import numpy as np
import sys, csv, os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from model import training, inference

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import reactionMPNN

from util import MC_dropout

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_id', '-d', type=int) #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
parser.add_argument('--split_id', '-s', type=int) #data_id 1 & 2: 0-9, data_id 3: 1-4 
parser.add_argument('--train_size', '-t', type=int) #data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055]

args = parser.parse_args()


data_id = args.data_id
split_id = args.split_id
train_size = args.train_size
batch_size = 128
use_saved = False
model_path = './model/model_%d_%d_%d.pt' %(data_id, split_id, train_size)
if not os.path.exists('./model/'): os.makedirs('./model/')
        
data = GraphDataset(data_id, split_id)
frac_split = (train_size + 1e-5)/len(data)
train_set, test_set = split_dataset(data, [frac_split, 1 - frac_split], shuffle=False)
assert len(train_set) == train_size

train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

print('-- CONFIGURATIONS')
print('--- data_type:', data_id, split_id)
print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
print('--- max no. reactants: %d' %len(data.rmols[0]))
print('--- use_saved:', use_saved)
print('--- model_path:', model_path)

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)

# training 
train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

node_dim = data.rmols[0,0]['ndata'].shape[1]
edge_dim = data.rmols[0,0]['edata'].shape[1]
n_cat = data.rmols.shape[1]
net = reactionMPNN(node_dim, edge_dim, n_cat).cuda()

if use_saved == False:
    print('-- TRAINING')
    if len(test_set) > 0:
        net = training(net, cuda, train_loader, test_loader, train_y_mean, train_y_std)
    else:
        net = training(net, cuda, train_loader, None, train_y_mean, train_y_std)
    
    checkpoint = {
      'net': net.state_dict(),
      'y_mean': train_y_mean,
      'y_std': train_y_std
    }
    
    torch.save(checkpoint, model_path)
    
else:
    print('-- LOAD SAVED MODEL')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    train_y_mean = checkpoint['y_mean']
    train_y_std = checkpoint['y_std']


# inference
test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]

test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, cuda, test_loader, train_y_mean, train_y_std)
test_y_pred = np.clip(test_y_pred, 0, 100)

result = [mean_absolute_error(test_y, test_y_pred),
          mean_squared_error(test_y, test_y_pred) ** 0.5,
          r2_score(test_y, test_y_pred),
          stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
          
print('-- RESULT')
print('--- test size: %d' %(len(test_y)))
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))
np.savetxt("./result/summary_%d_%d_%d.csv"%(data_id, split_id, train_size), result, delimiter=",")

# interpretation on the entire dataset
data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

data_y = data.yld
data_y_component = []

net.eval()
MC_dropout(net)
with torch.no_grad():
    for batchidx, batchdata in enumerate(data_loader):
    
        inputs_rmol = [b.to(cuda) for b in batchdata[:-2]]
        inputs_pmol = batchdata[-2].to(cuda)
        
        inputs_rind = torch.eye(len(inputs_rmol)).to(cuda)
        data_y_component.append(np.mean([net(inputs_rmol, inputs_rind, inputs_pmol)[:,:,0].cpu().numpy() for _ in range(30)], 0))

data_y_component = np.vstack(data_y_component) * train_y_std
data_y_pred = np.sum(data_y_component, 1) + train_y_mean

out = np.hstack([data.rsmi_list, np.ones((len(data_y), 1)) * train_y_mean, data_y_component, data_y_pred.reshape(-1,1), data_y.reshape(-1,1)])

np.savetxt("./result/output_%d_%d_%d.csv"%(data_id, split_id, train_size), out, delimiter=",", fmt='%s')