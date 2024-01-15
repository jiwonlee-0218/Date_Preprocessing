# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch.nn.utils import weight_norm
# import math
# from nilearn.connectome import ConnectivityMeasure
# import copy
# from typing import Optional, Tuple
# from torch.nn.utils.rnn import pad_sequence
# import torch
# import numpy as np
# import torch.nn as nn
# from einops import rearrange, repeat
#
#
#
#
#
# class GMPool(nn.Module):
#     def __init__(self, hidden):
#         super(GMPool, self).__init__()
#
#         self.MLP = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
#         self.MLP_sigmoid = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())
#         self.cos = nn.CosineSimilarity(dim=-1)
#
#     def forward(self, X):
#
#         time, hidden = X.shape[1:]
#         X1 = X.unsqueeze(1)
#         X2 = X.unsqueeze(2)
#         # cos_sim = self.cos(X1, X2)
#
#         diff = X2 - X1
#         distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
#         clf_output = self.MLP_sigmoid(distance.reshape(-1, hidden)).reshape(-1, time, time)
#         mask = torch.eye(time).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
#         clf_output_copy = clf_output.clone()
#         clf_output_copy[mask] = 1.
#         grouping_M = clf_output_copy
#
#
#
#         values, vectors = torch.linalg.eigh(grouping_M)
#
#
#         values, values_ind = torch.sort(values, dim=1, descending=True)
#
#         vectors_copy = vectors.clone()
#         for j in range(grouping_M.size(0)):
#             vectors_copy[j] = vectors_copy[j, :, values_ind[j]]
#
#         values[values <= 0] = 1e-7
#         d = torch.sqrt(values)
#         S = vectors_copy * d.unsqueeze(1)
#
#
#         return S
#
#
#
#
#
# ########################################################################################################################
#
#
# class ModuleTimestamping(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout) #batch first X -> batch_first=False라면, (Time_step, Batch_size, Input_feature_dimension) 순서이다.
#
#
#     def forward(self, t, sampling_endpoints):
#         return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]] #torch.Size([149, 2, 116]) -> torch.Size([149, 2, 128]) -> torch.Size([34, 2, 128])
#
#
#
#
#
# class ModuleMeanReadout(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#
#     def forward(self, x, node_axis=1):
#         return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)
#
#
# class ModuleGARO(nn.Module):
#     def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
#         super().__init__()
#         self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
#         self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
#         self.dropout = nn.Dropout(dropout)
#         self.MLP = nn.Sequential(nn.Linear(hidden_dim*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
#         self.MLP2 = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
#
#
#
#
#     def forward(self, x, e_s, node_axis=1):
#         # assumes shape [... x node x ... x feature]  x.shape == torch.Size([34, 2, 116, 128])
#
#         batchsize, timepoints = x.shape[0], x.shape[2]
#         fc_sequence_list = []
#         x = rearrange(x, 'b t n1 n2 -> (b t) n1 n2') #torch.Size([4, 150, 150, 128]) -> torch.Size([600, 150, 128])
#
#
#         for timeseries in x:
#             fc = self.corrcoef(timeseries.T)
#             fc_sequence_list.append(fc)
#         fc_sequence = torch.stack(fc_sequence_list) #torch.Size([600, 128, 128])
#         fc_sequence = self.MLP(rearrange(fc_sequence, 'b n v -> b (n v)')) #torch.Size([600, 128])
#         fc_sequence = rearrange(fc_sequence, '(b t) v -> b t v', b=batchsize, t=timepoints) #torch.Size([4, 150, 128])
#
#         e_s = repeat(e_s, 'n1 n2 -> n1 t n2', t=timepoints)
#
#         H = torch.cat((fc_sequence, e_s), dim=-1) #torch.Size([4, 150, 256])
#         H =self.MLP2(rearrange(H, 'b t v -> (b t) v'))  # torch.Size([600, 256])
#         H = rearrange(H, '(b t) v -> b t v', b=batchsize, t=timepoints)  # torch.Size([4, 150, 256])
#
#         x_q = self.embed_query(H.mean(node_axis, keepdims=True)) #torch.Size([4, 1, 128])
#         x_k = self.embed_key(H) #torch.Size([4, 150, 128])
#         x_eigenvectorattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 'b t c -> b c t'))/np.sqrt(x_q.shape[-1])).squeeze(1)  #torch.Size([34, 2, 1, 116]) -> torch.Size([34, 2, 116])
#         return (H * self.dropout(x_eigenvectorattention.unsqueeze(-1)))
#
#     def corrcoef(self, x):
#         mean_x = torch.mean(x, 1, keepdim=True)
#         xm = x.sub(mean_x.expand_as(x))
#         c = xm.mm(xm.t())
#         c = c / (x.size(1) - 1)
#         d = torch.diag(c)
#         stddev = torch.pow(d, 0.5)
#         c = c.div(stddev.expand_as(c))
#         c = c.div(stddev.expand_as(c).t())
#         c = torch.clamp(c, -1.0, 1.0)
#         return c
#
# class ModuleTransformer_1(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.layer_norm2 = nn.LayerNorm(input_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
#
#
#     def forward(self, q, k, v):
#         x_attend, attn_matrix = self.multihead_attn(q, k, v)  #self.multihead_attn(query, key, value)
#         x_attend = self.dropout1(x_attend) # no skip connection
#         x_attend = self.layer_norm1(x_attend)
#         x_attend2 = self.mlp(x_attend)
#         x_attend = x_attend + self.dropout2(x_attend2)
#         x_attend = self.layer_norm2(x_attend)
#         return x_attend, attn_matrix
#
#
# class ModuleTransformer_2(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads) #(in_features=128, out_features=128, bias=True)
#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.layer_norm2 = nn.LayerNorm(input_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
#
#
#     def forward(self, x):
#         x_attend, attn_matrix = self.multihead_attn(x, x, x)  #self.multihead_attn(query, key, value)
#         x_attend = self.dropout1(x_attend) # no skip connection
#         x_attend = self.layer_norm1(x_attend)
#         x_attend2 = self.mlp(x_attend)
#         x_attend = x_attend + self.dropout2(x_attend2)
#         x_attend = self.layer_norm2(x_attend)
#         return x_attend, attn_matrix
#
#
#
#
# class EigenvectorAttentionModel(nn.Module):
#     def __init__(self, time_length, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum'):
#         super().__init__()
#         assert cls_token in ['sum', 'mean', 'param']
#         if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
#         elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
#         elif cls_token=='param': self.cls_token = lambda x: x[-1]
#         else: raise
#         self.dropout = nn.Dropout(dropout)
#
#
#         self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None
#
#         self.num_classes = num_classes
#         self.sparsity = sparsity
#
#         # define modules
#         self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
#         self.initial_linear = nn.Linear(time_length+input_dim, hidden_dim) #Linear(in_features=266, out_features=128, bias=True)
#
#         # defined from me
#         self.MLP = nn.Sequential(nn.Linear(time_length+input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
#         self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
#         self.tatt = ModuleTransformer_1(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1)
#         self.gmpool = GMPool(hidden_dim)
#         self.SE = nn.Sequential(nn.Linear(input_dim*input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
#         self.garo = ModuleGARO(hidden_dim) #128, 150
#         self.tatt2 = ModuleTransformer_2(time_length, hidden_dim, num_heads=num_heads, dropout=0.1)
#         self.linear_layers = nn.Linear(time_length, num_classes)
#
#
#
#
#
#
#
#
#     def forward(self, v, t, fc):
#         # assumes shape [minibatch x time x time] for v --> torch.Size([4, 150, 150])
#         # assumes shape [time x minibatch x node] for t --> torch.Size([4, 150, 116])
#         # assumes shape [minibatch x node x node] for fc  --> torch.Size([4, 116, 116])
#         if v.shape[0] != t.shape[0]:
#             v = v[:t.shape[0]]
#
#         logit = 0.0
#         attention = {'time-attention_1': [], 'time-attention_2': []}
#
#
#         e_s = self.SE(rearrange(fc, 'b n v -> b (n v)'))
#
#         batchsize, timepoints = t.shape[:2]
#         h = torch.cat([v, t], dim=2) #torch.Size([4, 150, 266])
#         X_enc = rearrange(h, 'b t c -> (b t) c')
#         X_enc = self.MLP(X_enc)
#         X_enc = rearrange(X_enc, '(b t) c -> t b c', t=timepoints, b=batchsize) #torch.Size([150, 4, 128])
#         X_enc, (hn, cn) = self.PE(X_enc)
#         X_enc, _ = self.tatt(X_enc, X_enc, X_enc)
#         X_enc = rearrange(X_enc, 't b c -> b t c')
#
#
#         S = self.gmpool(X_enc)
#         S_reshaped = S.transpose(1, 2).unsqueeze(3)
#         X_enc_reshaped = X_enc.unsqueeze(1)
#         result = S_reshaped * X_enc_reshaped
#         eigenvector_attn = self.garo(result, e_s, node_axis=1)
#
#
#         X_enc_result = torch.matmul(X_enc, torch.transpose(eigenvector_attn, 1,2))
#         h_attend, time_attn = self.tatt2(rearrange(X_enc_result, 'b t v -> t b v')) #torch.Size([150, 4, 150]) 150dim을 가진 벡터 150개(length)가 들어간다
#
#         h_dyn = self.cls_token(h_attend)
#         logit = self.dropout(self.linear_layers(h_dyn))
#
#         attention['time-attention_1'].append(_)
#         attention['time-attention_2'].append(time_attn)
#
#
#
#
#
#         attention['time-attention_1'] = torch.stack(attention['time-attention_1'], dim=1).detach().cpu()  #torch.Size([4, 1, 150, 150])
#         attention['time-attention_2'] = torch.stack(attention['time-attention_2'], dim=1).detach().cpu()  #torch.Size([4, 1, 150, 150])
#
#
#         return logit, attention





#########################################################################################################################
#########################################################################################################################





# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch.nn.utils import weight_norm
# import math
# from nilearn.connectome import ConnectivityMeasure
# import copy
# from typing import Optional, Tuple
# from torch.nn.utils.rnn import pad_sequence
# import matplotlib.pyplot as plt
# from einops import rearrange
#
#
#
#
# class ModuleTransformer(nn.Module):
#     def __init__(self, n_region, hidden_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(n_region, num_heads)  #(in_features=128, out_features=128, bias=True)
#         self.layer_norm1 = nn.LayerNorm(n_region)
#         self.layer_norm2 = nn.LayerNorm(n_region)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_region))
#
#     def forward(self, q, k, v):
#         x_attend, attn_matrix = self.multihead_attn(q, k, v) #torch.Size([34, 2, 128]), torch.Size([2, 34, 150])
#         x_attend = self.dropout1(x_attend) # no skip connection
#         x_attend = self.layer_norm1(x_attend)
#         x_attend2 = self.mlp(x_attend)
#         x_attend = x_attend + self.dropout2(x_attend2)
#         x_attend = self.layer_norm2(x_attend)
#         return x_attend, attn_matrix
#
#
#
# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features):
#         super(MLP, self).__init__()
#
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, 1)
#         self.sigmoid = nn.Sigmoid()
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#
#     def forward(self, X):
#
#         X = F.relu(self.fc1(X))
#         X = self.fc2(X)
#         Y = self.sigmoid(X)
#
#         return Y
#
#
#
#
#
# class GMPool_attention_RT(nn.Module):
#     def __init__(self, roi, hidden_dim):
#         super(GMPool_attention_RT, self).__init__()
#
#         # self.mlp = MLP(roi, 200)
#         self.mlp = nn.Sequential(nn.Linear(roi, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
#                                  nn.Linear(hidden_dim, 1), nn.Sigmoid())
#
#
#
#     def forward(self, X):
#
#         minibatch_size, num_timepoints, num_nodes = X.shape[:3]
#
#         diff = X.unsqueeze(2) - X.unsqueeze(1)
#         distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
#         distance = rearrange(distance, 'b n1 n2 c -> (b n1 n2) c')
#         clf_output = self.mlp(distance)
#         clf_output = rearrange(clf_output, '(b n1 n2) c -> b n1 n2 c', b=minibatch_size, n1=num_timepoints, n2=num_timepoints).squeeze(-1)
#         mask = torch.eye(num_timepoints).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
#         clf_output_copy = clf_output.clone()
#         clf_output_copy[mask] = 1.
#         grouping_M = clf_output_copy
#         # grouping_M = grouping_M + 1e-7
#
#
#
#
#         values, vectors = torch.linalg.eigh(grouping_M)
#         # values, vectors = torch.linalg.eig(grouping_M)
#         # values = values.real
#         # vectors = vectors.real
#
#
#         values, values_ind = torch.sort(values, dim=1, descending=True)
#
#         vectors_copy = vectors.clone()
#         for j in range(grouping_M.size(0)):
#             vectors_copy[j] = vectors_copy[j, :, values_ind[j]]
#
#
#
#         selected_eigenvectors = vectors_copy[:, :, 0] #8개 batch의 첫번째 eigenvector
#         selected_eigenvectors = torch.abs(selected_eigenvectors)
#         new_X = X * selected_eigenvectors.unsqueeze(2) #torch.Size([8, 176, 116])
#
#
#
#         return new_X
#
#
#
#
#
#
#
# ''' transformer '''
# class GMNetwork(nn.Module):
#     """
#     class for the defintion of the DTC model
#     path_ae : path to load autoencoder
#     centr_size : size of the centroids = size of the hidden features
#     alpha : parameter alpha for the t-student distribution.
#     """
#
#     def __init__(self, roi, hidden_dim, num_classes, dropout=0.2):
#         super(GMNetwork, self).__init__()
#
#
#
#         self.gm_model = GMPool_attention_RT(roi, hidden_dim)
#         self.PE = nn.LSTM(roi, hidden_dim, num_layers=1, batch_first=True)
#         self.transformer_modules = ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=1, dropout=0.1)
#         self.cls_token = lambda x: x.sum(0)
#         self.linear_layers = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, inputs):
#         fc_attention = {'fc': [], 'time-attention': []}
#
#         result = self.gm_model(inputs)  ## model
#         X_enc, (hn, cn) = self.PE(result)
#         X_enc = rearrange(X_enc, 'b t c -> t b c') #torch.Size([8, 176, 128]) -> torch.Size([176, 8, 128])
#         out, time_attn = self.transformer_modules(X_enc, X_enc, X_enc) ## model
#         h_dyn = self.cls_token(out)
#         logit = self.dropout(self.linear_layers(h_dyn))
#
#
#         for timeseries in result:
#             fc = self.corrcoef(timeseries.T)
#             fc_attention['fc'].append(fc)
#
#         fc_attention['time-attention'].append(time_attn)
#
#         fc_attention['fc'] = torch.stack(fc_attention['fc']).detach().cpu()  #torch.Size([8, 116, 116])
#         fc_attention['time-attention'] = torch.stack(fc_attention['time-attention']).squeeze().detach().cpu()
#
#         return logit, fc_attention, result
#
#
#     def corrcoef(self, x):
#         mean_x = torch.mean(x, 1, keepdim=True)
#         xm = x.sub(mean_x.expand_as(x))
#         c = xm.mm(xm.t())
#         c = c / (x.size(1) - 1)
#         d = torch.diag(c)
#         stddev = torch.pow(d, 0.5)
#         c = c.div(stddev.expand_as(c))
#         c = c.div(stddev.expand_as(c).t())
#         c = torch.clamp(c, -1.0, 1.0)
#         return c



############################################################################################
############################################################################################
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

class GMPool_attention_FC(nn.Module):
    def __init__(self, roi, hidden_dim):
        super(GMPool_attention_FC, self).__init__()

        # self.mlp = MLP(roi, 200)
        self.mlp = nn.Sequential(nn.Linear(roi, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), nn.Sigmoid())



    def forward(self, X):

        minibatch_size, num_timepoints, num_nodes = X.shape[:3]

        diff = X.unsqueeze(2) - X.unsqueeze(1)
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
            for i in range(10):
                selected_eigenvectors = vectors_copy[k, :, i]
                selected_eigenvectors = torch.abs(selected_eigenvectors)
                new_X = X[k] * selected_eigenvectors.unsqueeze(1)
                fc = self.corrcoef(new_X.T)
                flatten_fc = self.flatten_upper_triang_values(fc)
                flatten_fc_list.append(flatten_fc)
            aa = torch.stack(flatten_fc_list) #torch.Size([5, 6670])
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

    def __init__(self, roi, hidden_dim, num_classes, dropout=0.2):
        super(GMNetwork, self).__init__()



        self.gm_model = GMPool_attention_FC(roi, hidden_dim)
        self.trans_model = Transformer2(
            n_classes=7,
            vocab_size=6670,

            d_model=128,
            word_pad_len=10,

            hidden_size=256,
            n_heads=4,
            n_encoders=1,
            dropout=0.0
        )
        self.cls_token = lambda x: x.sum(1)
        self.linear_layers = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, fc):

        flattened_fc_list = self.flatten_upper_triang_fc(fc)
        flattened_fc_list = flattened_fc_list.unsqueeze(1) #torch.Size([8, 1, 6670])
        result = self.gm_model(inputs)  ## model -> result: torch.Size([8, 10, 6670])
        sum_result_fc = flattened_fc_list + result
        out = self.trans_model(sum_result_fc)  ## out: torch.Size([8, 10, 128])


        h_dyn = self.cls_token(out)
        logit = self.dropout(self.linear_layers(h_dyn))




        return logit


    def flatten_upper_triang_fc(self, X):  ##torch.Size([116, 116])

        flattened_fc_list = []
        for i in range(X.shape[0]):
            mask = torch.triu(torch.ones_like(X[i]), diagonal=1)
            A_triu_1d = X[i][mask == 1]
            flattened_fc_list.append(A_triu_1d)

        values_arr = torch.stack(flattened_fc_list)
        return values_arr










############################################################################################
############################################################################################
class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Parameters
    ----------
    d_model : int
        Size of word embeddings
    word_pad_len : int
        Length of the padded sentence
    dropout : float
        Dropout
    """
    def __init__(self, d_model: int, word_pad_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()

        self.pe = torch.tensor([
            [pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)]
            for pos in range(word_pad_len)
        ])  # (batch_size, word_pad_len, emb_size)

        # PE(pos, 2i) = sin(pos / 10000^{2i / d_model})
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        # PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word embeddings
        Returns
        -------
        position encoded embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word Embeddings + Positional Encoding
        """
        # word embeddings + positional encoding
        embeddings = embeddings + nn.Parameter(self.pe, requires_grad=True).to('cuda')
        embeddings = self.dropout(embeddings)
        return embeddings




class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network
    Parameters
    ----------
    d_model : int
        Size of word embeddings
    hidden_size : int
        Size of position-wise feed forward network
    dropout : float
        Dropout
    """
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.5) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network
        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of position-wise feed-forward network
        """
        # eq.2: FFN = max(0, x W_1 + b_1) W_2 + b_2
        out = self.W_2(self.relu(self.W_1(x)))  # (batch_size, word_pad_len, d_model)
        out = self.dropout(out)

        out += x  # residual connection
        out = self.layer_norm(out)  # LayerNorm

        return out




class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Parameters
    ----------
    scale : float
        Scale factor (sqrt(d_k))
    dropout : float
        Dropout
    """
    def __init__(self, scale: float, dropout: float = 0.5) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Query
        K : torch.Tensor
            Key
        V : torch.Tensor
            Value
        mask : torch.Tensor (batch_size, 1, 1, word_pad_len)
            Padding mask metrix, None if it is not needed
        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Context vector
        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        # Q·K^T / sqrt(d_k)
        att = torch.matmul(Q / self.scale, K.transpose(2, 3))  # (batch_size, n_heads, word_pad_len, word_pad_len)

        # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)

        # eq.1: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k))·V
        att = self.dropout(self.softmax(att))  # (batch_size, n_heads, word_pad_len, word_pad_len)
        context = torch.matmul(att, V)  # (batch_size, n_heads, word_pad_len, d_k)  ## 얘가 찐 결과물 1번째 x

        return context, att


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    Parameters
    ----------
    d_model : int
        Size of word embeddings
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.5) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        # we assume d_v always equals d_k
        self.d_k = d_model // n_heads ##head_dim
        self.n_heads = n_heads

        # linear projections
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)  ## == (d_model, d_model)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k)

        # scaled dot-product attention
        scale = self.d_k ** 0.5  # scale factor
        self.attention = ScaledDotProductAttention(scale=scale)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * self.d_k, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data
        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed
        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network
        att: torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        batch_size = x.size(0)

        Q = self.W_Q(x)  # (batch_size, word_pad_len, n_heads * d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)  # (batch_size, word_pad_len, n_heads, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # (batch_size, n_heads, word_pad_len, d_k)

        # for n_heads axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, d_k)

        context, att = self.attention(Q, K, V, mask=mask)  # (batch_size, n_heads, word_pad_len, d_k)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)  # (batch_size, word_pad_len, n_heads * d_k)

        out = self.dropout(self.fc(context))  # (batch_size, word_pad_len, d_model)

        out = out + x  # residual connection
        out = self.layer_norm(out)  # LayerNorm  ## 얘가 찐 결과물 3번째 x

        return out, att


class EncoderLayer(nn.Module):
    """
    An encoder layer.
    Parameters
    ----------
    d_model : int
        Size of word embeddings
    n_heads : int
        Number of attention heads
    hidden_size : int
        Size of position-wise feed forward network
    dropout : float
        Dropout
    """
    def __init__(self, d_model: int, n_heads: int, hidden_size: int, dropout: float = 0.5) -> None:
        super(EncoderLayer, self).__init__()

        # an encoder layer has two sub-layers:
        #   - multi-head self-attention
        #   - positon-wise fully connected feed-forward network
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_size, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Input data
        mask : torch.Tensor (batch_size, 1, word_pad_len)
            Padding mask metrix, None if it is not needed
        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of the current encoder layer
        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att_out, att = self.attention(x, mask=mask)  # (batch_size, word_pad_len, d_model), (batch_size, n_heads, word_pad_len, word_pad_len)
        out = self.feed_forward(att_out)  # (batch_size, word_pad_len, d_model)
        return out, att


def get_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Mask tokens that are pads (not pad: 1, pad: 0)
    Parameters
    ----------
    seq : torch.Tensor (batch_size, word_pad_len)
        The sequence which needs masking
    pad_idx: index of '<pad>' (default is 0)
    Returns
    -------
    mask : torch.Tensor (batch_size, 1, word_pad_len)
        A padding mask metrix
    """
    seq = torch.sum(seq, dim=2)
    mask = (seq != pad_idx).unsqueeze(1).to('cuda')  # (batch_size, 1, word_pad_len)
    return mask

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=1, in_chans=116, embed_dim=90):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)



    def forward(self, x):
        B, H, W = x.shape # (64, 284, 116)
        x = x.transpose(1,2)  # (16, 116, 284)
        x = self.proj(x).transpose(1, 2)  # (64, 80, 284) -> (64, 284, 80)
        return x  # (64, 284, 80) patch sequence = 284 patch 1개당 80 dimension


class Transformer2(nn.Module):
    """
    Implementation of Transformer proposed in paper [1]. Only the encoder part
    is used here.
    `Here <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py>`_
    is the official TensorFlow implementation of Transformer.
    Parameters
    ----------
    n_classes : int
        Number of classes
    vocab_size : int
        Number of words in the vocabulary
    embeddings : torch.Tensor
        Word embedding weights
    d_model : int
        Size of word embeddings
    word_pad_len : int
        Length of the padded sequence
    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)
    hidden_size : int
        Size of position-wise feed forward network
    n_heads : int
        Number of attention heads
    n_encoders : int
        Number of encoder layers
    dropout : float
        Dropout
    References
    ----------
    1. "`Attention Is All You Need. <https://arxiv.org/abs/1706.03762>`_" \
        Ashish Vaswani, et al. NIPS 2017.
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,

        d_model: torch.Tensor,
        word_pad_len: int,

        hidden_size: int,
        n_heads: int,
        n_encoders: int,
        dropout: float = 0.5
    ) -> None:
        super(Transformer2, self).__init__()

        # embedding layer
        self.patch_embed = PatchEmbed(in_chans=vocab_size, embed_dim=d_model)
        self.embeddings = nn.Embedding(vocab_size, d_model)
        # self.set_embeddings(embeddings, fine_tune)
        # postional coding layer
        self.postional_encoding = PositionalEncoding(d_model, word_pad_len, dropout)

        # an encoder layer
        self.encoder = EncoderLayer(d_model, n_heads, hidden_size, dropout)
        # encoder is composed of a stack of n_encoders identical encoder layers
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(n_encoders)
        ])




    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = True) -> None:
        """
        Set weights for embedding layer
        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings
        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad = fine_tune)

    # def forward(self, text: torch.Tensor, words_per_sentence: torch.Tensor) -> torch.Tensor:
    def forward(self, text: torch.Tensor) -> torch.Tensor:

        """
        Parameters
        ----------
        text : torch.Tensor (batch_size, word_pad_len)
            Input data
        words_per_sentence : torch.Tensor (batch_size)
            Sentence lengths
        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores
        """
        # get padding mask
        # mask = get_padding_mask(text)

        # word embedding
        # text = text.type('torch.LongTensor')
        embeddings = self.patch_embed(text) # (batch_size, word_pad_len, emb_size)
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings  ####################################################################
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask = None)  # (batch_size, word_pad_len, d_model)


        return encoder_out #, att