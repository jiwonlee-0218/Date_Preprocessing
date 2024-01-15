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
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat





class GMPool(nn.Module):
    def __init__(self, hidden):
        super(GMPool, self).__init__()

        self.MLP = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.MLP_sigmoid = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, X):

        time, hidden = X.shape[1:]
        X1 = X.unsqueeze(1)
        X2 = X.unsqueeze(2)
        # cos_sim = self.cos(X1, X2)

        diff = X2 - X1
        distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        clf_output = self.MLP_sigmoid(distance.reshape(-1, hidden)).reshape(-1, time, time)
        mask = torch.eye(time).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy



        values, vectors = torch.linalg.eigh(grouping_M)


        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(grouping_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]

        values[values <= 0] = 1e-7
        d = torch.sqrt(values)
        S = vectors_copy * d.unsqueeze(1)


        return S





########################################################################################################################


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout) #batch first X -> batch_first=False라면, (Time_step, Batch_size, Input_feature_dimension) 순서이다.


    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]] #torch.Size([149, 2, 116]) -> torch.Size([149, 2, 128]) -> torch.Size([34, 2, 128])





class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.MLP = nn.Sequential(nn.Linear(hidden_dim*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.MLP2 = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())




    def forward(self, x, e_s, node_axis=1):
        # assumes shape [... x node x ... x feature]  x.shape == torch.Size([34, 2, 116, 128])

        batchsize, timepoints = x.shape[0], x.shape[2]
        fc_sequence_list = []
        x = rearrange(x, 'b t n1 n2 -> (b t) n1 n2') #torch.Size([4, 150, 150, 128]) -> torch.Size([600, 150, 128])


        for timeseries in x:
            fc = self.corrcoef(timeseries.T)
            fc_sequence_list.append(fc)
        fc_sequence = torch.stack(fc_sequence_list) #torch.Size([600, 128, 128])
        fc_sequence = self.MLP(rearrange(fc_sequence, 'b n v -> b (n v)')) #torch.Size([600, 128])
        fc_sequence = rearrange(fc_sequence, '(b t) v -> b t v', b=batchsize, t=timepoints) #torch.Size([4, 150, 128])

        e_s = repeat(e_s, 'n1 n2 -> n1 t n2', t=timepoints)

        H = torch.cat((fc_sequence, e_s), dim=-1) #torch.Size([4, 150, 256])
        H =self.MLP2(rearrange(H, 'b t v -> (b t) v'))  # torch.Size([600, 256])
        H = rearrange(H, '(b t) v -> b t v', b=batchsize, t=timepoints)  # torch.Size([4, 150, 256])

        x_q = self.embed_query(H.mean(node_axis, keepdims=True)) #torch.Size([4, 1, 128])
        x_k = self.embed_key(H) #torch.Size([4, 150, 128])
        x_eigenvectorattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 'b t c -> b c t'))/np.sqrt(x_q.shape[-1])).squeeze(1)  #torch.Size([34, 2, 1, 116]) -> torch.Size([34, 2, 116])
        return (H * self.dropout(x_eigenvectorattention.unsqueeze(-1)))

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

class ModuleTransformer_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, q, k, v):
        x_attend, attn_matrix = self.multihead_attn(q, k, v)  #self.multihead_attn(query, key, value)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


class ModuleTransformer_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads) #(in_features=128, out_features=128, bias=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)  #self.multihead_attn(query, key, value)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix




class EigenvectorAttentionModel(nn.Module):
    def __init__(self, time_length, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum'):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        self.dropout = nn.Dropout(dropout)


        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(time_length+input_dim, hidden_dim) #Linear(in_features=266, out_features=128, bias=True)

        # defined from me
        self.MLP = nn.Sequential(nn.Linear(time_length+input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer_1(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1)
        self.gmpool = GMPool(hidden_dim)
        self.SE = nn.Sequential(nn.Linear(input_dim*input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.garo = ModuleGARO(hidden_dim) #128, 150
        self.tatt2 = ModuleTransformer_2(time_length, hidden_dim, num_heads=num_heads, dropout=0.1)
        self.linear_layers = nn.Linear(time_length, num_classes)








    def forward(self, v, t, fc):
        # assumes shape [minibatch x time x time] for v --> torch.Size([4, 150, 150])
        # assumes shape [time x minibatch x node] for t --> torch.Size([4, 150, 116])
        # assumes shape [minibatch x node x node] for fc  --> torch.Size([4, 116, 116])
        if v.shape[0] != t.shape[0]:
            v = v[:t.shape[0]]

        logit = 0.0
        attention = {'time-attention_1': [], 'time-attention_2': []}


        e_s = self.SE(rearrange(fc, 'b n v -> b (n v)'))

        batchsize, timepoints = t.shape[:2]
        h = torch.cat([v, t], dim=2) #torch.Size([4, 150, 266])
        X_enc = rearrange(h, 'b t c -> (b t) c')
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=timepoints, b=batchsize) #torch.Size([150, 4, 128])
        X_enc, (hn, cn) = self.PE(X_enc)
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc)
        X_enc = rearrange(X_enc, 't b c -> b t c')


        S = self.gmpool(X_enc)
        S_reshaped = S.transpose(1, 2).unsqueeze(3)
        X_enc_reshaped = X_enc.unsqueeze(1)
        result = S_reshaped * X_enc_reshaped
        eigenvector_attn = self.garo(result, e_s, node_axis=1)


        X_enc_result = torch.matmul(X_enc, torch.transpose(eigenvector_attn, 1,2))
        h_attend, time_attn = self.tatt2(rearrange(X_enc_result, 'b t v -> t b v')) #torch.Size([150, 4, 150]) 150dim을 가진 벡터 150개(length)가 들어간다

        h_dyn = self.cls_token(h_attend)
        logit = self.dropout(self.linear_layers(h_dyn))

        attention['time-attention_1'].append(_)
        attention['time-attention_2'].append(time_attn)





        attention['time-attention_1'] = torch.stack(attention['time-attention_1'], dim=1).detach().cpu()  #torch.Size([4, 1, 150, 150])
        attention['time-attention_2'] = torch.stack(attention['time-attention_2'], dim=1).detach().cpu()  #torch.Size([4, 1, 150, 150])


        return logit, attention