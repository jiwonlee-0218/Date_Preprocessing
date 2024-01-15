import torch.nn as nn
import torch
from utils import compute_similarity
from sklearn.cluster import AgglomerativeClustering
import gc
from torchinfo import summary

# Only LSTM ---------------------------------------------------------------------------------------------------------------------------------------------
class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_lstm=[50, 32]):  # 내 모델에 사용될 구성품을 정의 및 초기화하는 메서드
        super(TAE, self).__init__()

        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(filter_lstm = self.filter_lstm)
        self.tae_decoder = TAE_decoder(filter_lstm = self.filter_lstm)


    def forward(self, x):        #  init에서 정의된 구성품들을 연결하는 메서드

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features, out_deconv   # features는 clustering을 위해 encoder의 output을 사용


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_lstm):
        super(TAE_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 284 , 50)
        self.lstm_1 = nn.LSTM(
            input_size=116,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
        )
        self.act_lstm_1 = nn.Tanh()

        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
        )
        self.act_lstm_2 = nn.Tanh()



    def forward(self, x):   # x : (1, 284, 116)

        ## encoder
        out_lstm1, _ = self.lstm_1(x)
        out_lstm1_act = self.act_lstm_1(out_lstm1)

        out_lstm2, _ = self.lstm_2(out_lstm1_act)
        out_lstm2_act = self.act_lstm_2(out_lstm2)

        return out_lstm2_act


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, filter_lstm):
        super(TAE_decoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]


        # upsample
        self.lstm_1 = nn.LSTM(
            input_size=self.hidden_lstm_2,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
        )
        self.act_lstm_1 = nn.Tanh()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=116,
            batch_first=True,
        )


    def forward(self, features):


        out_deconv1, _ = self.lstm_1(features)  # (1, 284, 64)
        out_deconv1 = self.act_lstm_1(out_deconv1)  # (1, 284, 64)
        out_deconv2, _ = self.lstm_2(out_deconv1)    # (1, 284, 116)

        return out_deconv2



# Only CNN ---------------------------------------------------------------------------------------------------------------------------------------------
class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_1=80):  # 내 모델에 사용될 구성품을 정의 및 초기화하는 메서드
        super(TAE, self).__init__()
        self.filter_1 = filter_1

        self.tae_encoder = TAE_encoder(filter_1=self.filter_1)
        self.tae_decoder = TAE_decoder(filter_1=self.filter_1)


    def forward(self, x):        #  init에서 정의된 구성품들을 연결하는 메서드

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features, out_deconv   # features는 clustering을 위해 encoder의 output을 사용


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1):
        super(TAE_encoder, self).__init__()

        self.filter_1 = filter_1

        ## CNN PART
        ### output shape (batch_size, 7 , n_hidden = 284)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=116,out_channels=self.filter_1,kernel_size=1,stride=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):   # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2) # (1, 116, 284)
        out_cnn = self.conv_layer(x) # (1, 80, 284)
        out_cnn = out_cnn.permute(0, 2, 1)  # (1, 284, 80)


        return out_cnn


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, filter_1):
        super(TAE_decoder, self).__init__()

        self.filter_1 = filter_1


        # upsample
        self.deconv_layer = nn.ConvTranspose1d(in_channels=self.filter_1,out_channels=116,kernel_size=1,stride=1)

    def forward(self, features):

        features = features.transpose(1, 2)
        out_deconv = self.deconv_layer(features)  # (1, 116, 284)
        out_deconv = out_deconv.permute(0,2,1)  # (1, 284, 116)
        # out_deconv_nnn = out_deconv.view(out_deconv.shape[0], self.n_hidden, -1) #########################################
        return out_deconv
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# DW
##############################################################################################################
#
# class TAE(nn.Module):
#
#     def __init__(self, args, filter_1=80, filter_lstm=[50, 10]):
#         super(TAE, self).__init__()
#
#         ## CNN PART
#         ### output shape (batch_size, 7 , n_hidden = 284)
#         self.conv1 = nn.Conv1d(in_channels=116, out_channels=80, kernel_size=1, stride=1)
#         # self.conv2 = nn.Conv1d(in_channels=80, out_channels=64, kernel_size=1, stride=1)
#         # self.dconv2 = nn.ConvTranspose1d(in_channels=64,out_channels=80, kernel_size=1, stride=1,)
#         self.dconv1 = nn.ConvTranspose1d(in_channels=80,out_channels=116, kernel_size=1, stride=1,)
#
#         self.act = nn.LeakyReLU()
#
#
#     def forward(self, x):
#         # x : (batch, 284, 116) (batch, time, rois)
#         ## encoder
#         x = x.transpose(1,2) # (1, 116, 284) (batch, rois, time)
#         feat = self.act(self.conv1(x)) # (batch, feat, time)
#         # feat = self.act(self.conv2(feat))
#         # out = self.act(self.dconv2(feat)) #  (batch, rois, time)
#         out = self.dconv1(feat)
#
#         out = out.contiguous().permute(0, 2, 1)
#         return feat, out

