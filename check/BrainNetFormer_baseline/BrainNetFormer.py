from matplotlib.pyplot import axis
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


class LayerGIN(nn.Module):
    def __init__(self, n_region, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        # print(a.shape, v.shape)
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v  # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleTransformer(nn.Module):
    def __init__(self, n_region, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_region, num_heads)  # (in_features=128, out_features=128, bias=True)
        self.layer_norm1 = nn.LayerNorm(n_region)
        self.layer_norm2 = nn.LayerNorm(n_region)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, n_region))

    def forward(self, q, k, v):
        x_attend, attn_matrix = self.multihead_attn(q, k, v)  # torch.Size([34, 2, 128]), torch.Size([2, 34, 150])
        x_attend = self.dropout1(x_attend)  # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


class BrainEncoder(nn.Module):
    def __init__(self, n_region, hidden_dim):
        super().__init__()
        # @ For X_enc
        self.MLP = nn.Sequential(nn.Linear(n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.PE = nn.LSTM(hidden_dim, hidden_dim, 1)
        self.tatt = ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=1, dropout=0.1)
        # @ For e_s
        self.SE = nn.Sequential(nn.Linear(n_region * n_region, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

    def forward(self, t, a):
        timepoints, batchsize = t.shape[:2]  # t.shape = torch.Size([150, 2, 116])
        X_enc = rearrange(t, 't b c -> (b t) c')  # torch.Size([300, 116])
        X_enc = self.MLP(X_enc)
        X_enc = rearrange(X_enc, '(b t) c -> t b c', t=timepoints, b=batchsize)
        X_enc, (hn, cn) = self.PE(X_enc)
        X_enc, _ = self.tatt(X_enc, X_enc, X_enc)  # t b c

        e_s = self.SE(rearrange(a, 'b n v -> b (n v)'))  # b c
        return X_enc, e_s


class BrainDecoder(nn.Module):
    def __init__(self, n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size, dropout=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.sparsity = sparsity


        self.cls_token = lambda x: x.sum(0)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.initial_linear = nn.Linear(window_size, hidden_dim)  # D+N -> D
        self.sigmoid = nn.Sigmoid()

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.transformer_modules.append(
                ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparse=True):
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100 - self.sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v,
                                        (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))

    def forward(self, dyn_t, dyn_a, e_s, X_enc, sampling_endpoints):
        logit = 0.0

        # temporal attention
        attention_list = []
        h_dyn_list = []
        minibatch_size, num_timepoints, num_nodes = dyn_a.shape[:3]

        h = rearrange(dyn_t, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)

        dyn_a_ = self._collate_adjacency(dyn_a)  # a: b t v v
        for layer, (G, T, L) in enumerate(zip(self.gnn_layers, self.transformer_modules, self.linear_layers)):
            # graph convolution
            h = G(h, dyn_a_)
            h_bridge = rearrange(h, '(b t n) c -> b t n c', t=num_timepoints, b=minibatch_size,
                                 n=num_nodes)  # b t n c  torch.Size([2, 34, 116, 128])

            # spatial attention readout
            q_s = repeat(e_s, 'b c -> t b c', t=num_timepoints) + X_enc[[p - 1 for p in sampling_endpoints]]  # torch.Size([34, 2, 128])
            q_s = repeat(q_s, "t b c -> b t c l", l=1)  # b t c 1   torch.Size([2, 34, 128, 1])
            SA_t = torch.softmax(torch.matmul(h_bridge, q_s), axis=2) + 1
            # SA_t = 1.0

            X_dec = h_bridge * SA_t  # b t n c
            X_dec = X_dec.sum(axis=2)  # b t c
            X_dec = rearrange(X_dec, 'b t c -> t b c')

            # time attention
            h_attend, time_attn = T(X_dec, X_enc, X_enc)  # h_attend: t b c
            h_dyn = self.cls_token(h_attend)  # h_dyn: b c
            logit += self.dropout(L(h_dyn))

            attention_list.append(time_attn)

        # logit = self.sigmoid(logit)
        tatt = torch.stack(attention_list, dim=1)
        time_attention = tatt.detach().cpu()  # b 4 t t

        return logit, time_attention


class BrainNetFormer(nn.Module):
    def __init__(self, n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity, window_size, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.sparsity = sparsity


        self.encoder = BrainEncoder(n_region, hidden_dim)
        self.decoder = BrainDecoder(n_region, hidden_dim, num_classes, num_heads, num_layers, sparsity,
                                    window_size, dropout)



    def forward(self, dyn_t, dyn_a, t, a, sampling_endpoints):
        X_enc, e_s = self.encoder(t, a)
        logit, time_attention = self.decoder(dyn_t, dyn_a, e_s, X_enc, sampling_endpoints)


        return logit, time_attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
