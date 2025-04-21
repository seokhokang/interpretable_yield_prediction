import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

from util import MC_dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['node_feats']
        edge_feats = g.edata['edge_feats']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 n_cat, readout_feats = 1024,
                 predict_hidden_feats = 512, prob_dropout = 0.1):
        
        super(reactionMPNN, self).__init__()

        self.feats = readout_feats
        
        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats + n_cat, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )
        
        self.embed = nn.Sequential(
            nn.Linear(readout_feats * 2, readout_feats), nn.PReLU()
        )
        
    def forward(self, rmols, rinds, pmol):

        r_mol_embeds = torch.stack([torch.cat([self.mpnn(mol), torch.repeat_interleave(ind.unsqueeze(0), mol.batch_size, dim=0)], 1) for mol, ind in zip(rmols, rinds)])
        
        r_graph_feats = torch.mean(r_mol_embeds, 0)[:,:self.feats]
        p_graph_feats = self.mpnn(pmol)
        
        reaction_embed = self.embed(torch.cat([r_graph_feats, p_graph_feats], -1))
        reaction_embed = torch.repeat_interleave(reaction_embed.unsqueeze(0), len(r_mol_embeds), dim=0)
        
        out_r = self.predict(torch.cat([r_mol_embeds[:,:,:self.feats] - reaction_embed, r_mol_embeds[:,:,self.feats:]], 2)).permute(1,0,2)

        return out_r
        
        
def training(net, cuda, train_loader, val_loader, train_y_mean, train_y_std, n_epochs = 500, warmup_epochs = 10, val_monitor_epoch = 10, n_forward_pass = 5):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    loss_fn = nn.MSELoss(reduction = 'none')

    optimizer = AdamW(net.parameters(), lr = 1e-3, weight_decay = 5e-2)
    
    lr_scheduler_1 = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs * (train_size // batch_size))
    lr_scheduler_2 = CosineAnnealingLR(optimizer, T_max=(n_epochs - warmup_epochs) * (train_size // batch_size), eta_min=1e-5)
    lr_scheduler = SequentialLR(optimizer, [lr_scheduler_1, lr_scheduler_2], [warmup_epochs * (train_size // batch_size)])

    for epoch in range(n_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs_rmol = [b.to(cuda) for b in batchdata[:-2]]
            inputs_pmol = batchdata[-2].to(cuda)
            
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)
            
            inputs_rind = torch.eye(len(inputs_rmol)).to(cuda)
            
            out = net(inputs_rmol, inputs_rind, inputs_pmol)
            pred = out[:,:,0].sum(1)
            var = torch.exp(out[:,:,1]).sum(1)
            logvar = torch.log(var + 1e-10)

            loss = loss_fn(pred, labels)
            loss = (1 - 0.1) * loss.mean() + 0.1 * ( loss * torch.exp(-logvar) + logvar ).mean()
            
            loss_a = torch.stack([torch.mean(x).square() for x in out[:,:,0].permute(1,0)]).sum() #column mean close to zero
            loss_b = torch.triu(torch.corrcoef(out[:,:,0].permute(1,0)), diagonal = 1).square().sum() #decorrelation
            loss_c = torch.stack([torch.square(y) * torch.square(x).sum()/torch.var(x, correction=0) for x, y in zip(out[:,:,0], labels)]).mean()#maximization of row variance
            
            loss += loss_a + 0.01 * loss_b + 0.01 * loss_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            
            lr_scheduler.step()
            
            train_loss = loss.detach().item()
        
        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60))

        # validation
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            val_y = val_loader.dataset.dataset.yld[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(net, cuda, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(val_y, val_y_pred),
                      mean_squared_error(val_y, val_y_pred) ** 0.5,
                      r2_score(val_y, val_y_pred)]
                      
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))

    print('training terminated at epoch %d' %epoch)
    
    return net
    

def inference(net, cuda, test_loader, train_y_mean, train_y_std, n_forward_pass = 30):

    net.eval()
    MC_dropout(net)
    
    test_y_mean = []
    test_y_var = []

    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs_rmol = [b.to(cuda) for b in batchdata[:-2]]
            inputs_pmol = batchdata[-2].to(cuda)

            inputs_rind = torch.eye(len(inputs_rmol)).to(cuda)
            
            mean_list = []
            var_list = []
            for _ in range(n_forward_pass):
                out = net(inputs_rmol, inputs_rind, inputs_pmol)
                mean = out[:,:,0].sum(1)
                var = torch.exp(out[:,:,1]).sum(1)

                mean_list.append(mean.cpu().numpy())
                var_list.append(var.cpu().numpy())

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)

    return test_y_pred, test_y_epistemic, test_y_aleatoric