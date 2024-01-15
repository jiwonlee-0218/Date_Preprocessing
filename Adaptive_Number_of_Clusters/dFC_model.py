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

class LSTMClassification(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_features2, n_classes):
        super(LSTMClassification, self).__init__()

        self.lstm = nn.LSTM(in_features, hidden_features, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_features, hidden_features2, batch_first=True)
        self.fc = nn.Linear(hidden_features2, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):

        out_lstm, _ = self.lstm(X)
        out_lstm2, _ = self.lstm2(out_lstm)
        out_fc = self.fc(out_lstm2[:,-1])
        scores = self.sigmoid(out_fc)

        return scores

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, X):

        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        Y = self.sigmoid(X)

        return Y



## 시간 상관없이 같은 그룹에 속하는 모든 R-dimension의 vector들을 시간에 상관없이 가지고와서 fc를 만든다.
# -> 때문에 시간 상관없이 그냥 같은 그룹에 속한 모든 vector를 가지고 와서 fc를 만든다 하지만 collasp에 빠져버림
#vocab_size = 6670, d_model = 512, word_pad_len = 176, hidden_size = 128, n_heads = 4, n_encoders = 3
class GMPool_1(nn.Module):
    def __init__(self, args, batch, time, roi):
        super(GMPool_1, self).__init__()

        self.batch = batch
        self.time = time
        self.roi = roi
        self.device = args.device
        self.mlp = MLP(self.roi, 200)



    def forward(self, X):


        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff**2) + 1e-9)
        clf_output = self.mlp(distance.view(-1, self.roi)).view(self.batch, self.time, self.time, -1).squeeze()
        mask = torch.eye(self.time).unsqueeze(0).repeat(self.batch, 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy   #torch.allclose(grouping_M[0], grouping_M[0].transpose(0, 1)) -> True : symmetric한 행렬이다.


        grouping_M = grouping_M + 1e-7
        values, vectors = torch.linalg.eigh(grouping_M)
        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(self.batch):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]
        # vectors = vectors[:,:,values_ind[0]] #vectors[0] == torch.flip(pp[0], [1]) -> True

        values[values <= 0] = 1e-7
        d = torch.sqrt(values)
        d = torch.diag_embed(d)
        S = torch.matmul(vectors, d) #torch.Size([4, 176, 176])



        # Group = torch.argmax(S, dim=2) #torch.Size([4, 176])
        # Group = F.softmax(S, dim=2) #torch.Size([4, 176, 176])
        Group = F.gumbel_softmax(S, tau=0.00001, dim=2) #torch.Size([4, 176, 176])
        # Group = F.gumbel_softmax(S * 1e+7, tau=0.0001, dim=2)  안된다


        # k = torch.argmax(Group, dim=2)
        # print(len(torch.unique(k[0])))
        Group_T = Group.permute(0, 2, 1)

        group_weight = Group_T.unsqueeze(2)
        X = X.permute(0, 2, 1).unsqueeze(1)
        result = torch.mul(group_weight, X)

        FC_input = []
        for i in range(self.batch):
            is_zero = (result[i] == 0).all(dim=1).all(dim=1)
            non_zero_tensors = result[i][~is_zero]
            output_tensor = make_functional_connectivity(non_zero_tensors)
            upper_tri_vectors = flatten_upper_triang_values(output_tensor)
            FC_input.append(upper_tri_vectors)



        # FC_output = torch.stack(FC_input)
        padding_FC_output = padding_sequences(FC_input, 176)  #max padding sequence
        return padding_FC_output, grouping_M, Group



## 같은 그룹에 속하는 R-dimension의 vector들을 가지고와서 평균내서 R-dimension의 vector를 만든다.
# -> 이들은 시간을 고려해서 Group을 만들며 그렇기 때문에 궁극적으로 따른 Group개수만큼 R-dimension의 vector가 생긴다.
class GMPool_2(nn.Module):
    def __init__(self, args, batch, time, roi):
        super(GMPool_2, self).__init__()

        self.batch = batch
        self.time = time
        self.roi = roi
        self.device = args.device
        self.mlp = MLP(self.roi, 200)



    def forward(self, X):


        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff**2) + 1e-9)
        clf_output = self.mlp(distance.view(-1, self.roi)).view(self.batch, self.time, self.time, -1).squeeze()
        mask = torch.eye(self.time).unsqueeze(0).repeat(self.batch, 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy   #torch.allclose(grouping_M[0], grouping_M[0].transpose(0, 1)) -> True : symmetric한 행렬이다.


        grouping_M = grouping_M + 1e-7


        values, vectors = torch.linalg.eigh(grouping_M)
        # values, vectors = torch.linalg.eig(grouping_M)
        # values = values.real
        # vectors = vectors.real


        values, values_ind = torch.sort(values, dim=1, descending=True)

        vectors_copy = vectors.clone()
        for j in range(self.batch):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]


        values[values <= 0] = 1e-7
        d = torch.sqrt(values)
        d = torch.diag_embed(d)
        S = torch.matmul(vectors, d) #torch.Size([4, 176, 176])
        S = S*1e+3



        Group = F.gumbel_softmax(S, tau=0.0001, dim=2) #torch.Size([4, 176, 176])
        soft_indices = torch.sum(Group * torch.linspace(0, Group.size(-1) - 1, steps=Group.size(-1)).to(self.device), dim=-1)


        X = X.permute(0, 2, 1)
        subejct_mean_vectors = []

        for i in range(self.batch):
            group_indices = soft_indices[i]
            MASK = (group_indices + 1) * self.toggle(group_indices)
            final = (MASK.sum(dim=0) ** (-1)).detach() * MASK

            result = X[i].unsqueeze(0) * final.unsqueeze(1)

            result_matrix = torch.zeros((self.roi, len(result))).to(self.device)

            for j in range(len(result)):
                selected_x = result[j]
                nonzero_columns = (selected_x != 0).any(dim=0)
                mean_nonzero_columns = selected_x[:, nonzero_columns].mean(dim=1)
                result_matrix[:, j] = mean_nonzero_columns


            result_matrix = result_matrix.permute(1, 0)
            subejct_mean_vectors.append(result_matrix)


        # FC_output = torch.stack(subejct_mean_vectors)
        padding_FC_output = padding_sequences(subejct_mean_vectors, 176)  # max padding sequence
        return padding_FC_output, grouping_M, soft_indices

    def toggle(self, tensor):
        mask = tensor[:-1] != tensor[1:]
        mask = torch.cat((torch.tensor([True]).to(self.device), mask))
        group_counts = torch.cumsum(mask, dim=0)[mask]
        num_groups = len(group_counts)
        # print(num_groups)
        group_assignments = torch.zeros((num_groups, len(tensor))).to(self.device)
        current_group = 0
        for i in range(len(tensor)):
            if i == 0 or tensor[i] != tensor[i - 1]:
                current_group += 1
            group_assignments[current_group - 1, i] = 1
        return group_assignments


    def Toggle(self, x):
        tmp = torch.zeros_like(x)
        for i in range(len(x) - 1):
            if x[i + 1] != x[i]:
                tmp[i + 1] = 1
        bracket = torch.zeros([int(tmp.sum()) + 1, len(x)]).to(self.device)
        j = 0
        for i in range(len(x)):
            if tmp[i] == 1:
                j += 1
            bracket[j, i] = 1
        return bracket




class GMPool_3(nn.Module):
    def __init__(self, args, batch, time, roi):
        super(GMPool_3, self).__init__()

        self.batch = batch
        self.time = time
        self.roi = roi
        self.device = args.device
        self.mlp = MLP(self.roi, 200)



    def forward(self, X):


        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff**2))
        clf_output = self.mlp(distance.view(-1, self.roi)).view(-1, self.time, self.time).squeeze()
        mask = torch.eye(self.time).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy   #torch.allclose(grouping_M[0], grouping_M[0].transpose(0, 1)) -> True : symmetric한 행렬이다.


        grouping_M = grouping_M + 1e-7


        values, vectors = torch.linalg.eigh(grouping_M)
        # values, vectors = torch.linalg.eig(grouping_M)
        # values = values.real
        # vectors = vectors.real


        values, values_ind = torch.sort(values, dim=1, descending=True)


        vectors_copy = vectors.clone()
        for j in range(grouping_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]


        values[values <= 0] = 1e-7
        d = torch.sqrt(values)
        d = torch.diag_embed(d)
        S = torch.matmul(vectors, d) #torch.Size([4, 176, 176])



        S = S * 1e+3
        Group = F.gumbel_softmax(S, tau=0.0001, dim=2) #torch.Size([4, 176, 176])
        soft_indices = torch.sum(Group * torch.linspace(0, Group.size(-1) - 1, steps=Group.size(-1)).to(self.device), dim=-1)



        X = X.permute(0, 2, 1)
        subejct_fc_vectors = []

        for i in range(grouping_M.size(0)):
            group_indices = soft_indices[i]
            MASK = (group_indices + 1) * self.toggle(group_indices)
            final = (MASK.sum(dim=0) ** (-1)).detach() * MASK

            result = X[i].unsqueeze(0) * final.unsqueeze(1)

            fc_vectors = []
            for j in range(len(result)):
                selected_x = result[j]
                corr1 = torch.corrcoef(selected_x)
                fc_vectors.append(corr1)

            FC_output = torch.stack(fc_vectors)


            flattened_FC_output = flatten_upper_triang_values(FC_output)
            subejct_fc_vectors.append(flattened_FC_output)


        padding_FC_output = padding_sequences(subejct_fc_vectors, 176)
        return padding_FC_output, grouping_M, soft_indices

    def toggle(self, tensor):
        mask = tensor[:-1] != tensor[1:]
        mask = torch.cat((torch.tensor([True]).to(self.device), mask))
        group_counts = torch.cumsum(mask, dim=0)[mask]
        num_groups = len(group_counts)
        # print(num_groups)
        group_assignments = torch.zeros((num_groups, len(tensor))).to(self.device)
        current_group = 0
        for i in range(len(tensor)):
            if i == 0 or tensor[i] != tensor[i - 1]:
                current_group += 1
            group_assignments[current_group - 1, i] = 1
        return group_assignments


    def Toggle(self, x):
        tmp = torch.zeros_like(x)
        for i in range(len(x) - 1):
            if x[i + 1] != x[i]:
                tmp[i + 1] = 1
        bracket = torch.zeros([int(tmp.sum()) + 1, len(x)]).to(self.device)
        j = 0
        for i in range(len(x)):
            if tmp[i] == 1:
                j += 1
            bracket[j, i] = 1
        return bracket




class GMPool_4(nn.Module):
    def __init__(self, args, batch, time, roi):
        super(GMPool_4, self).__init__()

        self.batch = batch
        self.time = time
        self.roi = roi
        self.device = args.device
        self.mlp = MLP(64, 200)


        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=116,
                out_channels=64,
                kernel_size=1,
                stride=1
            ),

        )


    def forward(self, X):


        # diff = X.unsqueeze(2) - X.unsqueeze(1)
        # distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        # distance_mean = torch.mean(distance, 3) ########### change
        # clf_output = self.mlp(distance_mean.view(-1, self.roi)).view(-1, self.time, self.time).squeeze()
        # mask = torch.eye(self.time).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        # clf_output_copy = clf_output.clone()
        # clf_output_copy[mask] = 1.
        # grouping_M = clf_output_copy   #torch.allclose(grouping_M[0], grouping_M[0].transpose(0, 1)) -> True : symmetric한 행렬이다.
        # grouping_M = grouping_M + 1e-7

        x = X.transpose(1, 2)
        y = self.conv_layer_1(x)
        out_cnn = y.permute(0, 2, 1)
        diff = out_cnn.unsqueeze(2) - out_cnn.unsqueeze(1)
        distance = torch.sqrt(torch.relu(diff ** 2) + 1e-9)
        clf_output = self.mlp(distance.reshape(-1, 64)).reshape(-1, self.time, self.time)
        mask = torch.eye(self.time).unsqueeze(0).repeat(clf_output.size(0), 1, 1).bool()
        clf_output_copy = clf_output.clone()
        clf_output_copy[mask] = 1.
        grouping_M = clf_output_copy
        # grouping_M = grouping_M + 1e-7



        # x = X.transpose(1, 2)
        # y = self.conv_layer_1(x)
        # out_cnn = y.permute(0, 2, 1)
        # cos_sim = torch.nn.CosineSimilarity(dim=3)
        # cos_distance = cos_sim(out_cnn.unsqueeze(2), out_cnn.unsqueeze(1))
        # cos_distance = cos_distance + 1e-7


        values, vectors = torch.linalg.eigh(grouping_M)
        # values, vectors = torch.linalg.eig(cos_distance)
        # values = values.real
        # vectors = vectors.real


        values, values_ind = torch.sort(values, dim=1, descending=True)


        vectors_copy = vectors.clone()
        for j in range(grouping_M.size(0)):
            vectors_copy[j] = vectors_copy[j, :, values_ind[j]]


        values[values <= 0] = 1e-7
        d = torch.sqrt(values)
        d = torch.diag_embed(d)
        S = torch.matmul(vectors, d) #torch.Size([4, 176, 176])



        S = S * 1e+3
        Group = F.gumbel_softmax(S, tau=0.0001, dim=2) #torch.Size([4, 176, 176])
        soft_indices = torch.sum(Group * torch.linspace(0, Group.size(-1) - 1, steps=Group.size(-1)).to(self.device), dim=-1)



        X = X.permute(0, 2, 1)
        subejct_fc_vectors = []

        for i in range(grouping_M.size(0)):
            group_indices = soft_indices[i]
            MASK = (group_indices + 1) * self.toggle(group_indices)
            final = (MASK.sum(dim=0) ** (-1)).detach() * MASK

            result = X[i].unsqueeze(0) * final.unsqueeze(1)

            fc_vectors = []
            for j in range(len(result)):
                selected_x = result[j]
                corr1 = torch.corrcoef(selected_x)
                fc_vectors.append(corr1)

            FC_output = torch.stack(fc_vectors)


            flattened_FC_output = flatten_upper_triang_values(FC_output)
            subejct_fc_vectors.append(flattened_FC_output)


        padding_FC_output = padding_sequences(subejct_fc_vectors, 176)
        return padding_FC_output, grouping_M, soft_indices

    def toggle(self, tensor):
        mask = tensor[:-1] != tensor[1:]
        mask = torch.cat((torch.tensor([True]).to(self.device), mask))
        group_counts = torch.cumsum(mask, dim=0)[mask]
        num_groups = len(group_counts)
        # print(num_groups)
        group_assignments = torch.zeros((num_groups, len(tensor))).to(self.device)
        current_group = 0
        for i in range(len(tensor)):
            if i == 0 or tensor[i] != tensor[i - 1]:
                current_group += 1
            group_assignments[current_group - 1, i] = 1
        return group_assignments


    def Toggle(self, x):
        tmp = torch.zeros_like(x)
        for i in range(len(x) - 1):
            if x[i + 1] != x[i]:
                tmp[i + 1] = 1
        bracket = torch.zeros([int(tmp.sum()) + 1, len(x)]).to(self.device)
        j = 0
        for i in range(len(x)):
            if tmp[i] == 1:
                j += 1
            bracket[j, i] = 1
        return bracket




''' transformer '''
class GMNetwork(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self,  args, batch, time, roi):
        super(GMNetwork, self).__init__()

        self.batch_size = args.batch_size
        self.device = args.device

        self.gm_model = GMPool_4(args, batch, time, roi)
        # self.lstm_layer = LSTMClassification(in_features=6670, hidden_features=512, hidden_features2=128, n_classes=2)
        self.trans_model = Transformer2(
                                    n_classes = args.n_labels,
                                    vocab_size = 6670,

                                    d_model = 512,
                                    word_pad_len = 176,

                                    hidden_size = 1024,
                                    n_heads = 4,
                                    n_encoders = 3,
                                    dropout =0.0
                                )

    def forward(self, inputs):

        result, M, S = self.gm_model(inputs) ## model
        out = self.trans_model(result) ## model
        # out = self.lstm_layer(result)  ## model

        return out, M, S


################################################################################################
def make_functional_connectivity(input_tensor):
    b, n, _ = input_tensor.size()
    input_tensor = input_tensor - input_tensor.mean(dim=2, keepdim=True)
    input_tensor = input_tensor / (input_tensor.std(dim=2, keepdim=True) + 1e-7)
    output_tensor = torch.bmm(input_tensor, input_tensor.transpose(1, 2)) / (input_tensor.size(2) - 1)
    return output_tensor

# def padding_sequences(sequences, max_len):
#     max_len = max_len  # 50
#
#     # out = F.pad(a,(0,0,0,max_len-a.shape[0]),'constant',10)  (3, 6) --> (10, 6)
#     sequences[0] = F.pad(sequences[0], (0, 0, 0, max_len - sequences[0].shape[0]), 'constant', 0)
#
#     seqs = pad_sequence(sequences, batch_first=True)
#
#     return seqs

def padding_sequences(sequences, max_len):
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_sequences = F.pad(padded_sequences, (0, 0, 0, max_len - padded_sequences.size(1)), "constant", 0)

    return padded_sequences



def flatten_upper_triang_values(X):  ##torch.Size([8, 116, 116])

    values_list = []
    for i in range(X.shape[0]):
        mask = torch.triu(torch.ones_like(X[i]), diagonal=1)
        A_triu_1d = X[i][mask == 1]
        values_list.append(A_triu_1d)

        # mask = torch.triu(torch.ones_like(X[i], dtype=torch.bool), diagonal=1)
        # A_triu_1d = torch.masked_select(X[i], mask)
        # values_list.append(A_triu_1d)

    values_arr = torch.stack(values_list)
    return values_arr


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
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

        # classifier
        self.fc = nn.Linear(word_pad_len * d_model, n_classes)



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
        mask = get_padding_mask(text)

        # word embedding
        # text = text.type('torch.LongTensor')
        embeddings = self.patch_embed(text) # (batch_size, word_pad_len, emb_size)
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings  ####################################################################
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask = mask)  # (batch_size, word_pad_len, d_model)

        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # (batch_size, word_pad_len * d_model)
        scores = self.fc(encoder_out)  # (batch_size, n_classes)

        return scores #, att