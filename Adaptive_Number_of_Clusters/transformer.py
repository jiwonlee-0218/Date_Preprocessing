import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import copy
import os
import json
from typing import Dict, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm





import argparse
import yaml

# yahoo_answers_classes = [
#     'Society & Culture',
#     'Science & Mathematics',
#     'Health',
#     'Education & Reference',
#     'Computers & Internet',
#     'Sports',
#     'Business & Finance',
#     'Entertainment & Music',
#     'Family & Relationships',
#     'Politics & Government'
# ]
#
# dbpedia_classes = [
#     'Company',
#     'EducationalInstitution',
#     'Artist',
#     'Athlete',
#     'OfficeHolder',
#     'MeanOfTransportation',
#     'Building',
#     'NaturalPlace',
#     'Village',
#     'Animal',
#     'Plant',
#     'Album',
#     'Film',
#     'WrittenWork'
# ]
#
# ag_news_classes = [
#     'World',
#     'Sports',
#     'Business',
#     'Sci / Tech'
# ]
#
# class Config:
#     """Convert a ``dict`` into a ``Class``"""
#     def __init__(self, entries: dict = {}):
#         for k, v in entries.items():
#             if isinstance(v, dict):
#                 self.__dict__[k] = Config(v)
#             else:
#                 self.__dict__[k] = v
#
# def load_config(file_path: str) -> dict:
#     """
#     Load configuration from a YAML file
#     Parameters
#     ----------
#     file_path : str
#         Path to the config file (in YAML format)
#     Returns
#     -------
#     config : dict
#         Configuration settings
#     """
#     f = open(file_path, 'r', encoding = 'utf-8')
#     config = yaml.load(f.read(), Loader = yaml.FullLoader)
#     return config
#
# def parse_opt() -> Config:
#     parser = argparse.ArgumentParser()
#     # config file
#     parser.add_argument(
#         '--config',
#         type = str,
#         default = 'ag_news/transformer.yaml',
#         help = 'path to the configuration file (yaml)'
#     )
#     args = parser.parse_args()
#     config_dict = load_config(args.config)
#     config = Config(config_dict)
#
#     return config
#
#
#
# ###############################################################################################################
# def init_embeddings(embeddings: torch.Tensor) -> None:
#     """
#     Fill embedding tensor with values from the uniform distribution.
#     Parameters
#     ----------
#     embeddings : torch.Tensor
#         Word embedding tensor
#     """
#     bias = np.sqrt(3.0 / embeddings.size(1))
#     torch.nn.init.uniform_(embeddings, -bias, bias)
#
#
# def load_embeddings(
#     emb_file: str,
#     word_map: Dict[str, int],
#     output_folder: str
# ) -> Tuple[torch.Tensor, int]:
#     """
#     Create an embedding tensor for the specified word map, for loading into the model.
#     Parameters
#     ----------
#     emb_file : str
#         File containing embeddings (stored in GloVe format)
#     word_map : Dict[str, int]
#         Word2id map
#     output_folder : str
#         Path to the folder to store output files
#     Returns
#     -------
#     embeddings : torch.Tensor
#         Embeddings in the same order as the words in the word map
#     embed_dim : int
#         Dimension of the embeddings
#     """
#     emb_basename = os.path.basename(emb_file)
#     cache_path = os.path.join(output_folder, emb_basename + '.pth.tar')
#
#     # no cache, load embeddings from .txt file
#     if not os.path.isfile(cache_path):
#         # find embedding dimension
#         with open(emb_file, 'r') as f:
#             embed_dim = len(f.readline().split(' ')) - 1
#             num_lines = len(f.readlines())
#
#         vocab = set(word_map.keys())
#
#         # create tensor to hold embeddings, initialize
#         embeddings = torch.FloatTensor(len(vocab), embed_dim)
#         init_embeddings(embeddings)
#
#         # read embedding file
#         for line in tqdm(open(emb_file, 'r'), total = num_lines, desc = 'Loading embeddings'):
#             line = line.split(' ')
#
#             emb_word = line[0]
#             embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
#
#             # ignore word if not in train_vocab
#             if emb_word not in vocab:
#                 continue
#
#             embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
#
#         # create cache file so we can load it quicker the next time
#         print('Saving vectors to {}'.format(cache_path))
#         torch.save((embeddings, embed_dim), cache_path)
#
#     # load embeddings from cache
#     else:
#         print('Loading embeddings from {}'.format(cache_path))
#         embeddings, embed_dim = torch.load(cache_path)
#
#     return embeddings, embed_dim
#
#
#
# def get_label_map(dataset: str) -> Tuple[Dict[str, int], Dict[int, str]]:
#     if dataset == 'ag_news':
#         classes = ag_news_classes
#     elif dataset == 'dbpedia':
#         classes = dbpedia_classes
#     elif dataset == 'yahoo_answers':
#         classes = yahoo_answers_classes
#     else:
#         raise Exception("Dataset not supported: ", dataset)
#
#     label_map = {k: v for v, k in enumerate(classes)}
#     rev_label_map = {v: k for k, v in label_map.items()}
#
#     return label_map, rev_label_map
#
# class DocDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
#     (for document classification).
#     Parameters
#     ----------
#     data_folder : str
#         Path to folder where data files are stored
#     split : str
#         Split, one of 'TRAIN' or 'TEST'
#     """
#     def __init__(self, data_folder: str, split: str) -> None:
#         split = split.upper()
#         assert split in {'TRAIN', 'TEST'}
#         self.split = split
#
#         # load data
#         self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))
#
#     def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
#         return torch.LongTensor(self.data['docs'][i]), \
#                torch.LongTensor([self.data['sentences_per_document'][i]]), \
#                torch.LongTensor(self.data['words_per_sentence'][i]), \
#                torch.LongTensor([self.data['labels'][i]])
#
#     def __len__(self) -> int:
#         return len(self.data['labels'])
#
#
# class SentDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches
#     (for sentence classification).
#     Parameters
#     ----------
#     data_folder : str
#         Path to folder where data files are stored
#     split : str
#         Split, one of 'TRAIN' or 'TEST'
#     """
#     def __init__(self, data_folder: str, split: str) -> None:
#         split = split.upper()
#         assert split in {'TRAIN', 'TEST'}
#         self.split = split
#
#         # load data
#         self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))
#
#     def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
#         return torch.LongTensor(self.data['sents'][i]), \
#                torch.LongTensor([self.data['words_per_sentence'][i]]), \
#                torch.LongTensor([self.data['labels'][i]])
#
#     def __len__(self) -> int:
#         return len(self.data['labels'])
#
#
#
# def load_data(
#     config: Config, split: str, build_vocab: bool = True
# ) -> Union[DataLoader, Tuple[DataLoader, torch.Tensor, int, Dict[str, int], int, int]]:
#     """
#     Load data from files output by ``prepocess.py``.
#     Parameters
#     ----------
#     config : Config
#         Configuration settings
#     split : str
#         'trian' / 'test'
#     build_vocab : bool
#         Build vocabulary or not. Only makes sense when split = 'train'.
#     Returns
#     -------
#     split = 'test':
#         test_loader : DataLoader
#             Dataloader for test data
#     split = 'train':
#         build_vocab = Flase:
#             train_loader : DataLoader
#                 Dataloader for train data
#         build_vocab = True:
#             train_loader : DataLoader
#                 Dataloader for train data
#             embeddings : torch.Tensor
#                 Pre-trained word embeddings (None if config.emb_pretrain = False)
#             emb_size : int
#                 Embedding size (config.emb_size if config.emb_pretrain = False)
#             word_map : Dict[str, int]
#                 Word2ix map
#             n_classes : int
#                 Number of classes
#             vocab_size : int
#                 Size of vocabulary
#     """
#     split = split.lower()
#     assert split in {'train', 'test'}
#
#
#
#     # train
#     train_loader = DataLoader(
#         DocDataset(config.output_path, 'train') if config.model_name in ['han'] else SentDataset(config.output_path, 'train'),
#         batch_size = config.batch_size,
#         shuffle = True,
#         num_workers = config.workers,
#         pin_memory = True
#     )
#
#     if build_vocab == False:
#         return train_loader
#
#     else:
#         # load word2ix map
#         with open(os.path.join(config.output_path, 'word_map.json'), 'r') as j:
#             word_map = json.load(j)
#         # size of vocabulary
#         vocab_size = len(word_map)
#
#         # number of classes
#         label_map, _ = get_label_map(config.dataset)
#         n_classes = len(label_map)
#
#         # word embeddings
#         if config.emb_pretrain == True:
#             # load Glove as pre-trained word embeddings for words in the word map
#             emb_path = os.path.join(config.emb_folder, config.emb_filename)
#             embeddings, emb_size = load_embeddings(
#                 emb_file = os.path.join(config.emb_folder, config.emb_filename),
#                 word_map = word_map,
#                 output_folder = config.output_path
#             )
#         # or initialize embedding weights randomly
#         else:
#             embeddings = None
#             emb_size = config.emb_size
#
#         return train_loader, embeddings, emb_size, word_map, n_classes, vocab_size

#######################################################################################################################################################################################################

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
        self.patch_embed = PatchEmbed(in_chans=6670, embed_dim=1000)
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



#########################################################################################################################


# config = parse_opt()
# train_loader, embeddings, emb_size, word_map, n_classes, vocab_size = load_data(config, 'train', True)
#
# # model = Transformer2(
# #             n_classes = n_classes,
# #             vocab_size = vocab_size,
# #             embeddings = embeddings,
# #             d_model = emb_size,
# #             word_pad_len = config.word_limit,
# #             fine_tune = config.fine_tune_word_embeddings,
# #             hidden_size = config.hidden_size,
# #             n_heads = config.n_heads,
# #             n_encoders = config.n_encoders,
# #             dropout = config.dropout
# #         )
#
#
# for i, batch in enumerate(train_loader):
#     sentences, words_per_sentence, labels = batch
#
#     sentences = sentences.to(device)  # (batch_size, word_limit)
#     words_per_sentence = words_per_sentence.squeeze(1).to(device)  # (batch_size)
#     labels = labels.squeeze(1).to(device)  # (batch_size)
#
#     scores = model(sentences, words_per_sentence)