import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.utils import weight_norm
import math
from nilearn.connectome import ConnectivityMeasure
import copy
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from einops import rearrange







class ModuleTransformer(nn.Module):
    def __init__(self, n_region, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_region, num_heads)  #(in_features=128, out_features=128, bias=True)
        self.layer_norm1 = nn.LayerNorm(n_region)
        self.layer_norm2 = nn.LayerNorm(n_region)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_region))

    def forward(self, q, k, v):
        x_attend, attn_matrix = self.multihead_attn(q, k, v) #torch.Size([34, 2, 128]), torch.Size([2, 34, 150])
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix




class GMPool_temporal_attention_FC(nn.Module):
    def __init__(self, n_region, hidden_dim):
        super(GMPool_temporal_attention_FC, self).__init__()

        self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=1, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())


    def forward(self, X):

        minibatch_size, num_timepoints, num_nodes = X.shape[:3]
        X_enc = rearrange(X, 'b t c -> (b t) c')  # torch.Size([300, 116])
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=num_timepoints, b=minibatch_size) #torch.Size([176, 8, 128])
        X_enc, (hn, cn) = self.PE(X_enc)
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc)  # t b c
        X_enc = rearrange(X_enc, 't b c -> b t c', t=num_timepoints, b=minibatch_size)


        diff = X_enc.unsqueeze(2) - X_enc.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        distance = rearrange(distance, 'b n1 n2 c -> (b n1 n2) c')
        clf_output = self.mlp(distance)
        clf_output = rearrange(clf_output, '(b n1 n2) c -> b n1 n2 c', b=minibatch_size, n1=num_timepoints, n2=num_timepoints).squeeze(-1)
        mask = torch.eye(num_timepoints).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy
        # grouping_M = grouping_M + 1e-7




        values, vectors = torch.linalg.eigh(grouping_M)
        # values, vectors = torch.linalg.eig(grouping_M)
        # values = values.real
        # vectors = vectors.real


        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(grouping_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]

        new_X_list = []
        for k in range(grouping_M.size(0)):
            flatten_fc_list = []
            for i in range(15):
                selected_eigenvectors = vectors_copy[k, :, i]
                selected_eigenvectors = torch.abs(selected_eigenvectors)
                new_X = X[k] * selected_eigenvectors.unsqueeze(1)
                # fc = self.corrcoef(new_X.T)
                # flatten_fc = self.flatten_upper_triang_values(fc)
                flatten_fc_list.append(new_X)
            aa = torch.stack(flatten_fc_list) #torch.Size([5, 6670]) or torch.Size([10, 176, 116])
            new_X_list.append(aa)
        new_FC = torch.stack(new_X_list)


        return new_FC

    def flatten_upper_triang_values(self, X):  ##torch.Size([116, 116])


        mask = torch.triu(torch.ones_like(X), diagonal=1)
        A_triu_1d = X[mask == 1]

        return A_triu_1d


    def corrcoef(self, x):
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        return c






''' transformer '''
class GMNetwork(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, roi, hidden_dim, spa_hidden_dim, time, num_classes, dropout=0.2):
        super(GMNetwork, self).__init__()



        self.gm_temporal_model = GMPool_temporal_attention_FC(roi, hidden_dim)

        self.temporal_conv1 = nn.Conv2d(15, 32, kernel_size=(1, 116), stride=(1,))
        self.spatial_conv1 = nn.Conv2d(15, 32, kernel_size=(1, 176), stride=(1,))
        self.temporal_conv2 = nn.Conv1d(176, 1, kernel_size=1, stride=1)
        self.spatial_conv2 = nn.Conv1d(116, 1, kernel_size=1, stride=1)


        self.last_layer = nn.Sequential(nn.Linear(32, hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))


    def forward(self, inputs):
        fc_attention = {'temporal-result': []}


        temporal_result = self.gm_temporal_model(inputs)  ## model -> torch.Size([8, 10, 176, 116])


        tem_out = self.temporal_conv1(temporal_result).squeeze(-1)  # model -> torch.Size([8, 32, 176, 1]) -> torch.Size([8, 32, 176])



        for s in range(tem_out.shape[0]):
            b = (tem_out[s].cpu().detach().numpy() - np.min(tem_out[s].cpu().detach().numpy())) / (np.max(tem_out[s].cpu().detach().numpy()) - np.min(tem_out[s].cpu().detach().numpy()))
            fc_attention['temporal-result'].append(b)



        tem_out = rearrange(tem_out, 'b f c -> b c f')
        tem_out = self.temporal_conv2(tem_out).squeeze(1) # model -> torch.Size([8, 1, 32]) -> torch.Size([8, 32])


        logit = self.last_layer(tem_out)



        return logit, fc_attention


    def flatten_upper_triang_values(self, X):  ##torch.Size([116, 116])


        mask = torch.triu(torch.ones_like(X), diagonal=1)
        A_triu_1d = X[mask == 1]

        return A_triu_1d


    def corrcoef(self, x):
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        return c


