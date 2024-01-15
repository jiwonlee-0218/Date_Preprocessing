import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from functools import partial
import math



class DRNN_decoder2(nn.Module):
    def __init__(self, n_input, filters):
        super(DRNN_decoder2, self).__init__()

        self.n_input = n_input #10
        self.recon = filters #116


        self.rnn = nn.GRU(self.n_input, self.recon, num_layers=1, batch_first=True)



    def forward(self, features):


        ## decoder
        output, fcn_latent = self.rnn(features)

        return output #(16, 150, 116)




class DRNN_decoder(nn.Module):
    def __init__(self, n_input, filters, recon):
        super(DRNN_decoder, self).__init__()

        self.n_input = n_input #10

        self.hidden_lstm_1 = filters[1][0] #50
        self.hidden_lstm_2 = filters[0] #64

        self.recon = recon #116


        self.rnn_1 = nn.GRU(self.n_input, self.hidden_lstm_1, num_layers=1, batch_first=True)
        self.rnn_2 = nn.GRU(self.hidden_lstm_1, self.hidden_lstm_2, num_layers=1, batch_first=True)

        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.hidden_lstm_2,
            out_channels=self.recon,
            kernel_size=1,
            stride=1,
        )



    def forward(self, features):


        ## decoder
        output_1, fcn_latent_1 = self.rnn_1(features)
        output_2, fcn_latent_2 = self.rnn_2(output_1)


        features = output_2.transpose(1, 2)
        out_deconv = self.deconv_layer(features)
        out_deconv = out_deconv.permute(0, 2, 1)


        return out_deconv #(16, 150, 116)










class LSTM_decoder(nn.Module):
    def __init__(self, n_input, filters, recon):
        super(LSTM_decoder, self).__init__()

        self.hidden_lstm_1 = filters[0] #100
        self.hidden_lstm_2 = filters[1] #50
        self.hidden_lstm_3 = filters[2] #10
        self.n_input = n_input
        self.recon = recon #116


        # self.rnn = nn.GRU(self.n_input, self.backto_input, num_layers=1, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=self.n_input, hidden_size=self.hidden_lstm_2, batch_first=True)
        self.act_lstm_1 = nn.ReLU()
        self.lstm_2 = nn.LSTM(input_size=self.hidden_lstm_2, hidden_size=self.hidden_lstm_1, batch_first=True)
        self.act_lstm_2 = nn.ReLU()
        self.lstm_3 = nn.LSTM(input_size=self.hidden_lstm_1, hidden_size=self.recon, batch_first=True)
        self.act_lstm_3 = nn.Tanh()

    def forward(self, features):

        ## decoder
        out_lstm1, _ = self.lstm_1(features)
        out_lstm1_act = self.act_lstm_1(out_lstm1)

        out_lstm2, _ = self.lstm_2(out_lstm1_act)
        out_lstm2_act = self.act_lstm_2(out_lstm2)

        out_lstm3, _ = self.lstm_3(out_lstm2_act)
        out_lstm3_act = self.act_lstm_2(out_lstm3)

        return out_lstm3_act








class CNN_decoder(nn.Module):
    def __init__(self, n_input, filters):
        super(CNN_decoder, self).__init__()

        self.n_input = n_input
        self.backto_input = filters

        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.n_input,
            out_channels=self.backto_input,
            kernel_size=1,
            stride=1,
        )



    def forward(self, features):


        ## decoder
        features = features.transpose(1, 2)
        out_deconv = self.deconv_layer(features)
        out_deconv = out_deconv.permute(0, 2, 1)


        return out_deconv #(16, 150, 116)







class CNN_BiLSTM_decoder(nn.Module):
    def __init__(self, n_input, filters, recon):
        super(CNN_BiLSTM_decoder, self).__init__()

        self.hidden_lstm_1 = filters[1][0]
        self.hidden_lstm_2 = filters[0]

        self.n_input = n_input
        self.recon = recon #116


        # self.rnn = nn.GRU(self.n_input, self.backto_input, num_layers=1, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=self.n_input, hidden_size=self.hidden_lstm_1, batch_first=True)
        self.act_lstm_1 = nn.ReLU()
        self.lstm_2 = nn.LSTM(input_size=self.hidden_lstm_1, hidden_size=self.hidden_lstm_2, batch_first=True)
        self.act_lstm_2 = nn.ReLU()

        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.hidden_lstm_2,
            out_channels=self.recon,
            kernel_size=1,
            stride=1,
        )



    def forward(self, features):

        ## decoder
        out_lstm1, _ = self.lstm_1(features)
        out_lstm1_act = self.act_lstm_1(out_lstm1)

        out_lstm2, _ = self.lstm_2(out_lstm1_act)
        out_lstm2_act = self.act_lstm_2(out_lstm2)


        ## decoder
        out_lstm2_act = out_lstm2_act.transpose(1, 2)
        out_deconv = self.deconv_layer(out_lstm2_act)
        out_deconv = out_deconv.permute(0, 2, 1)


        return out_deconv





class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(DRNN, self).__init__()

        self.dilations = [1, 4, 8, 16]
        # self.dilations = [1, 16]
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        cell = nn.GRU

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden[i], dropout=dropout)
            else:
                c = cell(n_hidden[i-1], n_hidden[i], dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-1])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)  #(N, L, H)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))

            zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        hidden = hidden.cuda()

        return hidden




class LSTM_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, n_input, filter_lstm):
        super(LSTM_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.hidden_lstm_3 = filter_lstm[2]
        self.n_input = n_input

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 284 , 50)
        self.lstm_1 = nn.LSTM(
            input_size= self.n_input,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True

        )
        self.act_lstm_1 = nn.ReLU()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True

        )
        self.act_lstm_2 = nn.ReLU()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_3 = nn.LSTM(
            input_size=self.hidden_lstm_2,
            hidden_size=self.hidden_lstm_3,
            batch_first=True,
            bidirectional=True

        )





    def forward(self, x):  # x : (1, 284, 116)

        out_lstm1, _ = self.lstm_1(x)
        out_lstm1 = torch.sum(out_lstm1.view(out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1), dim=2, )
        out_lstm1_act = self.act_lstm_1(out_lstm1)


        out_lstm2, _ = self.lstm_2(out_lstm1_act) # (1, 6, 64)
        out_lstm2 = torch.sum(out_lstm2.view(out_lstm2.shape[0], out_lstm2.shape[1], 2, self.hidden_lstm_2), dim=2, )
        out_lstm2_act = self.act_lstm_2(out_lstm2)

        out_lstm3, _ = self.lstm_3(out_lstm2_act)  # (1, 6, 64)
        out_lstm3 = torch.sum(out_lstm3.view(out_lstm3.shape[0], out_lstm3.shape[1], 2, self.hidden_lstm_3), dim=2, )


        return out_lstm3




class CNN_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, n_input, filters):
        super(CNN_encoder, self).__init__()

        self.filter_1 = filters
        # self.filter_2 = filters[1]
        # self.filter_3 = filters[2]
        self.n_input = n_input

        ## CNN PART
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_input,
                out_channels=self.filter_1,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm1d(self.filter_1),
            nn.ReLU()
        )

        # self.conv_layer_2 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.filter_1,
        #         out_channels=self.filter_2,
        #         kernel_size=1,
        #         stride=1
        #     ),
        #     nn.BatchNorm1d(self.filter_2),
        #     nn.ReLU(),
        # )
        #
        # self.conv_layer_3 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.filter_2,
        #         out_channels=self.filter_3,
        #         kernel_size=1,
        #         stride=1
        #     ),
        #     nn.BatchNorm1d(self.filter_3),
        #     nn.ReLU(),
        # )


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2)
        x = self.conv_layer_1(x)
        # x = self.conv_layer_2(x)
        # x = self.conv_layer_3(x)
        out_cnn = x.permute(0, 2, 1) #(16, 150, 32)

        return out_cnn



class CNN_BiLSTM_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, n_input, filters, filter_lstm):
        super(CNN_BiLSTM_encoder, self).__init__()

        self.filters = filters
        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.n_input = n_input

        ## CNN PART
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_input,
                out_channels=self.filters,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU()
        )

        ## LSTM PART
        self.lstm_1 = nn.LSTM(
            input_size=self.filters,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True

        )
        self.act_lstm_1 = nn.Tanh()

        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True

        )
        self.act_lstm_2 = nn.Tanh()


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2)
        x = self.conv_layer_1(x)
        out_cnn = x.permute(0, 2, 1) #(16, 150, 32)

        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(out_lstm1.view(out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1), dim=2, )
        out_lstm1_act = self.act_lstm_1(out_lstm1)

        out_lstm2, _ = self.lstm_2(out_lstm1_act)  # (1, 6, 64)
        out_lstm2 = torch.sum(out_lstm2.view(out_lstm2.shape[0], out_lstm2.shape[1], 2, self.hidden_lstm_2), dim=2, )
        out_lstm2_act = self.act_lstm_2(out_lstm2)

        return out_lstm2_act




class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Encoder, self).__init__()


        self.drnn = DRNN(n_input=n_input, n_hidden=n_hidden, n_layers=2, cell_type='GRU')
        # self.cnn_encoder = CNN_encoder(n_input= n_hidden[-1], filters=filters)
        # self.lstm_encoder = LSTM_encoder(n_input=n_input, filter_lstm=filters)
        # self.cnn_lstm_encoder = CNN_BiLSTM_encoder(n_input=n_input,  filters=filters, filter_lstm=n_hidden)


    def forward(self, inputs):
        ## encoder
        # outputs_fw, states_fw = self.drnn(inputs)
        # cnn_output = self.cnn_encoder(outputs_fw)
        # return cnn_output




        # ''' dilated RNN '''
        outputs_fw, states_fw = self.drnn(inputs)
        # ''' backward '''
        # # inputs_bw = torch.flip(inputs, dims=[1])
        # # outputs_bw, states_bw = self.drnn(inputs_bw)
        states_fw = torch.cat(states_fw, 1)
        # # states_bw = torch.cat(states_bw, 1)
        # # final_states = torch.cat((states_fw, states_bw), 1)
        # # return outputs_fw, final_states
        return outputs_fw, states_fw





class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        # self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'), gain=1)
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'), gain=1)


    def forward(self, h_real):
        W = h_real.transpose(0,1)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.transpose(0,1), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans









class Seq2Seq(nn.Module):
    def __init__(self, args, n_input, n_hidden):  #116, 64, [100, 50]
        super(Seq2Seq, self).__init__()



        # self.f_enc = Encoder(n_input=n_input, filter_1=filter_1, n_hidden=n_hidden)
        # self.f_enc = Encoder(n_input=n_input, filters=filters, n_hidden=n_hidden)
        self.f_enc = Encoder(n_input=n_input, n_hidden=n_hidden)

        # self.f_dec = CNN_BiLSTM_decoder(n_input=n_hidden[-1], filters=[filters, n_hidden], recon=n_input)
        # self.f_dec = CNN_decoder(n_input=n_hidden[-1], filters = n_input)
        self.f_dec = DRNN_decoder2(n_input=n_hidden[-1], filters=n_input)

        self.kmeans = kmeans(args)


    def forward(self, inputs):


        features, _ = self.f_enc(inputs)  #torch.Size([16, 150, 10]), (16, 200)
        aa = features.reshape(-1, 10)  #torch.Size([2400, 10])

        kmeans_loss = self.kmeans(aa)

        input_recons = self.f_dec(features)


        return aa, input_recons, kmeans_loss











''' TCN Model '''

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
                                nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                stride=stride, padding=0, dilation=dilation)
                                )

        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
                                nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                stride=stride, padding=0, dilation=dilation)
                                )
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)
        features = features.permute(0, 2, 1)
        return features



class TCN_decoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_decoder, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in reversed(range(num_levels)):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)
        output = features.permute(0, 2, 1)
        return output



# TCNModel(input_size= 116, output_size=10, num_channels=[20] * 2, kernel_size=3, dropout=0.25)
class TCNModel(nn.Module):
    def __init__(self, args, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = TCN_decoder(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.kmeans = kmeans(args)

    def forward(self, inputs):
        emb = self.tcn(inputs)

        aa = emb.reshape(-1, 10)  # torch.Size([2400, 10])
        kmeans_loss = self.kmeans(aa)

        y = self.decoder(emb)

        return aa, y, kmeans_loss









''' TCM + DRNN '''

class TCM_DRNN_decoder(nn.Module):
    def __init__(self, n_input, n_hidden, recon):
        super(TCM_DRNN_decoder, self).__init__()

        self.n_input = n_input #10

        self.hidden_lstm_1 = n_hidden[0] #64
        self.hidden_lstm_2 = n_hidden[1] #50
        self.hidden_lstm_3 = n_hidden[2] #10

        self.recon = recon #116


        self.rnn_1 = nn.GRU(self.n_input, self.hidden_lstm_2, num_layers=1, batch_first=True)
        self.rnn_2 = nn.GRU(self.hidden_lstm_2, self.hidden_lstm_1, num_layers=1, batch_first=True)
        self.rnn_3 = nn.GRU(self.hidden_lstm_1, self.recon, num_layers=1, batch_first=True)




    def forward(self, features):


        ## decoder
        output_1, fcn_latent_1 = self.rnn_1(features)
        output_2, fcn_latent_2 = self.rnn_2(output_1)
        output_3, fcn_latent_3 = self.rnn_3(output_2)


        return output_3 #(16, 150, 116)



class TCM_DRNN_encoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, n_hidden=[64, 50, 10]):
        super(TCM_DRNN_encoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.drnn = DRNN(n_input=num_channels[-1], n_hidden=n_hidden, n_layers=4, cell_type='GRU')
        ############ DRNN LAYER !!!!!!


    def forward(self, x):
        x = x.permute(0, 2, 1) #(16, 116, 150)
        features = self.network(x) #(16, 64, 150)
        features = features.permute(0, 2, 1) #(16, 150, 64)
        outputs_fw, states_fw = self.drnn(features)  #torch.Size([16, 150, 10])
        return outputs_fw



class TCM_DRNN(nn.Module):
    def __init__(self, args, input_size, num_channels, kernel_size=2, dropout=0.2, n_hidden=[64, 50, 10]):
        super(TCM_DRNN, self).__init__()

        self.tcn = TCM_DRNN_encoder(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, n_hidden=n_hidden)
        self.f_dec = TCM_DRNN_decoder(n_input=n_hidden[-1], n_hidden=n_hidden, recon=input_size)
        self.kmeans = kmeans(args)




    def forward(self, x):

        emb = self.tcn(x) #emb=(16, 150, 10)

        aa = emb.reshape(-1, 10)  #torch.Size([2400, 10])
        kmeans_loss = self.kmeans(aa)

        input_recons = self.f_dec(emb) #(16, 150, 116)


        return aa, input_recons, kmeans_loss









''' only DRNN 3 layer '''

class DRNN_decoder(nn.Module):
    def __init__(self, n_input, n_hidden, recon):
        super(DRNN_decoder, self).__init__()

        self.n_input = n_input #10

        self.hidden_lstm_1 = n_hidden[0] #64
        self.hidden_lstm_2 = n_hidden[1] #50
        self.hidden_lstm_3 = n_hidden[2] #10

        self.recon = recon #116


        self.rnn_1 = nn.GRU(self.n_input, self.hidden_lstm_2, num_layers=1, batch_first=True)
        self.rnn_2 = nn.GRU(self.hidden_lstm_2, self.hidden_lstm_1, num_layers=1, batch_first=True)
        self.rnn_3 = nn.GRU(self.hidden_lstm_1, self.recon, num_layers=1, batch_first=True)




    def forward(self, features):


        ## decoder
        output_1, fcn_latent_1 = self.rnn_1(features)
        output_2, fcn_latent_2 = self.rnn_2(output_1)
        output_3, fcn_latent_3 = self.rnn_3(output_2)


        return output_3 #(16, 150, 116)





class DRNN_Model(nn.Module):
    def __init__(self, args, input_size, n_hidden):
        super(DRNN_Model, self).__init__()

        self.drnn = DRNN(n_input=input_size, n_hidden=n_hidden, n_layers=3, cell_type='GRU')
        self.drnn_dec = TCM_DRNN_decoder(n_input=n_hidden[-1], n_hidden=n_hidden, recon=input_size)
        self.kmeans = kmeans(args)




    def forward(self, x):

        outputs_fw, states_fw = self.drnn(x) #emb=(16, 150, 10)
        aa = outputs_fw.reshape(-1, 10)  #torch.Size([2400, 10])
        kmeans_loss = self.kmeans(aa)
        input_recons = self.drnn_dec(outputs_fw) #(16, 150, 116)


        return aa, input_recons, kmeans_loss






























































''' Transformer '''


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 150):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        self.pos_embedding = torch.zeros((maxlen, emb_size)).cuda()
        self.pos_embedding[:, ::2] = torch.sin(pos * den)
        self.pos_embedding[:, 1::2] = torch.cos(pos * den)


    def forward(self, token_embedding):
        seq_len = token_embedding.size(1)
        y = self.pos_embedding[:seq_len, :]
        return y





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super(Mlp, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) #torch.Size([8, 150, 128])
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) #torch.Size([8, 150, 10])
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale #(1)


        # diagnol filled +1
        # a = torch.zeros(286, 286).to('cuda')
        # a = a.fill_diagonal_(1)
        # for k in range(attn.size(0)):
        #     for i in range(attn.size(1)):
        #         attn[k][i] = attn[k][i] + a




        attn = attn.softmax(dim=-1) #(2)
        attn = self.attn_drop(attn) #(3)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, out_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = self.norm3(x)  ### modified  partial(nn.LayerNorm, eps=1e-6)


        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,in_chans, embed_dim):
        super(PatchEmbed, self).__init__()

        self.proj = nn.Conv1d(in_chans, 86, kernel_size=1, stride=1)


        self.proj2 = nn.Conv1d(86, embed_dim, kernel_size=1, stride=1)
        self.ru1 = nn.ReLU()
        self.ru2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.ru1(x)
        x = self.proj2(x)
        x = self.ru2(x)
        x = x.transpose(1, 2)
        return x  # (16, 150, 64) patch sequence = 284 patch 1개당 64 dimension


# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#
#     def __init__(self, in_chans, embed_dim):
#         super(PatchEmbed, self).__init__()
#
#         self.tcn = TemporalConvNet(in_chans, [embed_dim]*5, kernel_size=1, dropout=1)
#
#     def forward(self, x):
#         y = self.tcn(x)
#         return y






############# ENCODER ##############
class VisionTransformer(nn.Module):   # img_size [284]
    """ Vision Transformer """

    def __init__(self, in_chans=116, embed_dim=90,
                 num_heads=2, mlp_hidden_dim=8, out_dim=10, qkv_bias=False, norm_layer=nn.LayerNorm,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., depth=12,
                 ):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)


        self.positional_encoding = PositionalEncoding(embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim), requires_grad=False)  # torch.Size([1, 286, 116])
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, out_dim=out_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,  norm_layer=norm_layer
            ) for i in range(depth)
        ])

        self.device  = torch.device('cuda')




    def prepare_tokens(self, x):
        x_ = self.patch_embed(x)  # patch linear embedding  (16, 284, 64)
        # x_ = x_ + self.positional_encoding(x_)   # add positional encoding to each token (16, 284, 64) + (1, 284, 64)

        return self.pos_drop(x_)

    def forward(self, x):    # encoding layer


        x_ = self.prepare_tokens(x)   # torch.Size([16, 286, 64])  embed 1단계


        for i, blk in enumerate(self.blocks):  # block 1  transformer encoder
            x_ = blk(x_)                         # torch.Size([64, 18, 768])




        return x_





########### DECODER ##########
class RECHead(nn.Module):
    def __init__(self, embed_dim, in_chans, num_heads, mlp_hidden_dim, out_dim, qkv_bias=False, norm_layer=nn.LayerNorm, depth=1):
        super(RECHead, self).__init__()


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, out_dim=out_dim, qkv_bias=qkv_bias, qk_scale=None,
                drop=0., attn_drop=0., norm_layer=norm_layer
            ) for i in range(depth)
        ])


        self.convTrans = nn.ConvTranspose1d(embed_dim, in_chans, kernel_size=1, stride=1)


    def forward(self, x):



        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):  # block 1  transformer encoder
            x = blk(x)  # (16, 284, 64)

        x_rec = x.transpose(1, 2)  # (8, 64, 150)
        x_rec = self.convTrans(x_rec)  # (16, 116, 284)
        x_rec = x_rec.permute(0, 2, 1)  # (16, 284, 116)

        return x_rec





''' transformer with kmeans '''
class TransformerKmeans(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()




        # create full model
        #### ENCODER ####
        self.backbone =  VisionTransformer(
                                            in_chans=116, embed_dim=64,  depth=1, num_heads=2, mlp_hidden_dim=128, out_dim=10,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                           )


        #### DECODER ####
        self.rec_head = RECHead(
                                 embed_dim=64, in_chans=116, num_heads=2, mlp_hidden_dim=128, out_dim=10,
                                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 depth=1
                               )

        self.kmeans = kmeans(args)



    def forward(self, im):


        #### ENCODER ####
        encoded_out = self.backbone(im) # torch.Size([8, 150, 116]) -> torch.Size([8, 150, 64])

        aa = encoded_out.reshape(-1, 64)  # torch.Size([2400, 10])
        kmeans_loss = self.kmeans(aa)

        #### DECODER ####
        recons_imgs = self.rec_head(encoded_out)  # torch.Size([16, 286, 64]) -> torch.Size([16, 286, 116])




        return aa, recons_imgs, kmeans_loss


















''' transformer '''
class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()




        # create full model
        #### ENCODER ####
        self.backbone =  VisionTransformer(
                                            in_chans=116, embed_dim=64, num_heads=8, mlp_hidden_dim=128, out_dim=10,
                                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depth=1
                                           )


        #### DECODER ####
        self.rec_head = RECHead(
                                 embed_dim=64, in_chans=116, num_heads=8, mlp_hidden_dim=128, out_dim=10,
                                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 depth=1
                               )




    def forward(self, im):


        #### ENCODER ####
        encoded_out = self.backbone(im) # torch.Size([8, 150, 116]) -> torch.Size([8, 150, 64])


        #### DECODER ####
        recons_imgs = self.rec_head(encoded_out)  # torch.Size([16, 286, 64]) -> torch.Size([16, 286, 116])


        return encoded_out, recons_imgs
