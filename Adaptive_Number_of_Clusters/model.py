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



# class LSTMClassification(nn.Module):
#     def __init__(self, in_features, hidden_features, n_classes):
#         super(LSTMClassification, self).__init__()
#
#         self.lstm = nn.LSTM(in_features, hidden_features, batch_first=True)
#         self.fc = nn.Linear(hidden_features, n_classes)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, X):
#
#         out_lstm, _ = self.lstm(X)
#         out_fc = self.fc(out_lstm[:,-1])
#         scores = self.sigmoid(out_fc)
#
#         return scores



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



###################### 논문과 동일하게 pooling S와 ROI timeseries data를 곱해서 새로운 ROI timeseries data를 만든다
class GMPool(nn.Module):
    def __init__(self, args, batch, time, roi):
        super(GMPool, self).__init__()

        self.batch = batch
        self.time = time
        self.roi = roi
        self.device = args.device
        self.mlp = MLP(self.roi, 200)




    def forward(self, X):
        try:

            diff = X.unsqueeze(2) - X.unsqueeze(1)
            distance = torch.sqrt(torch.relu(diff**2) + 1e-9)
            clf_output = self.mlp(distance.view(-1, self.roi)).view(self.batch, self.time, self.time, -1).squeeze()
            mask = torch.eye(self.time).unsqueeze(0).repeat(self.batch, 1, 1).bool()
            # clf_output[mask]=1.
            # grouping_M = clf_output
            clf_output_copy = clf_output.clone()
            clf_output_copy[mask] = 1.
            grouping_M = clf_output_copy   #torch.allclose(grouping_M[0], grouping_M[0].transpose(0, 1)) -> True : symmetric한 행렬이다.

            # make the matrix symmetric
            # grouping_M = (grouping_M + grouping_M.transpose(1, 2)) / 2
            grouping_M = grouping_M + 1e-6

            values, vectors = torch.linalg.eigh(grouping_M)
            # values, vectors = torch.real(values), torch.real(vectors)
            values, values_ind = torch.sort(values, dim=1, descending=True)

            vectors_copy = vectors.clone()
            for j in range(self.batch):
                vectors_copy[j] = vectors_copy[j, :, values_ind[j]]
            # vectors = vectors[:,:,values_ind[0]] #vectors[0] == torch.flip(pp[0], [1]) -> True

            values[values <= 0] = 1e-7
            d = torch.sqrt(values)
            # d = torch.nan_to_num(d)
            d = torch.diag_embed(d)
            S = torch.matmul(vectors, d)





            # Group = torch.argmax(S, dim=2)
            # FC_input = []
            # for i in range(self.batch):# 0
            #     FC = make_functional_connectivity(X[i], Group[i])
            #     FC_input.append(FC)
            #
            # padding_FC_output = padding_sequences(FC_input, 88)
            # return padding_FC_output






            SX = torch.matmul(S.transpose(1, 2), X)
            return SX, grouping_M, S

        except:
            print('wrong')




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

        self.gm_model = GMPool(args, batch, time, roi)
        # self.lstm_layer = LSTMClassification(in_features=116, hidden_features=64, n_classes=2)
        self.trans_model = Transformer2(
                                    n_classes = 2,
                                    vocab_size = 116,

                                    d_model = 64,
                                    word_pad_len = 176,

                                    hidden_size = 32,
                                    n_heads = 2,
                                    n_encoders = 1,
                                    dropout =0.0
                                )

    def forward(self, inputs):

        SX, M, S = self.gm_model(inputs) ########################### model

        out = self.trans_model(SX) ############################# model
        # out = self.lstm_layer(SX)

        return out, M, S


################################################################################################
def make_functional_connectivity(inputs, Group):




    fc_list = []

    tmp = Group[0].item()
    change = 0
    index = 0

    for i in range(Group.size(0)):
        change += 1

        if tmp != Group[i].item():
            x = inputs[index:change - 1, :]
            x = torch.mean(x, 0, keepdim=True)
            # sub = np.reshape(x.detach().cpu().numpy(), (1,) + x.shape)
            # fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
            fc_list.append(x)
            tmp = Group[i].item()
            index = i

        if i == Group.size(0) - 1:
            x = inputs[index:change, :]  ## == aa[index:, :]
            x = torch.mean(x, 0, keepdim=True)
            # sub = np.reshape(x.detach().cpu().numpy(), (1,) + x.shape)
            # fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
            fc_list.append(x)

    return torch.cat(fc_list, 0)  ## torch.stack(fc_list, 0) == (1, 1, 116)



def padding_sequences(sequences, max_len):
    max_len = max_len  # 50

    # out = F.pad(a,(0,0,0,max_len-a.shape[0]),'constant',10)  (3, 6) --> (10, 6)
    sequences[0] = F.pad(sequences[0], (0, 0, 0, max_len - sequences[0].shape[0]), 'constant', 0)

    seqs = pad_sequence(sequences, batch_first=True)

    return seqs

def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt

def flatten_upper_triang_values(X):
    upper_tri = torch.triu(X, diagonal=1)  ##torch.Size([8, 116, 116])

    values_list = []
    for i in range(X.shape[0]):
        arr = upper_tri[i]
        values = arr[arr != 0]
        values_list.append(values)
        arr = 0

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
        embeddings = embeddings + nn.Parameter(self.pe, requires_grad=False).to('cuda')
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

        out = out + x  # residual connection
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
        # mask = get_padding_mask(text)

        # word embedding
        # text = text.type('torch.LongTensor')
        embeddings = self.patch_embed(text) # (batch_size, word_pad_len, emb_size)
        embeddings = self.postional_encoding(embeddings)

        encoder_out = embeddings  ####################################################################
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, mask = None)  # (batch_size, word_pad_len, d_model)

        encoder_out = encoder_out.view(encoder_out.size(0), -1)  # (batch_size, word_pad_len * d_model)
        scores = self.fc(encoder_out)  # (batch_size, n_classes)

        return scores #, att