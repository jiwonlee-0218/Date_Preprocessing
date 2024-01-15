import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm




class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(DRNN, self).__init__()

        self.dilations = [1, 4, 16]
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



class TemporalBlock_2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super(TemporalBlock_2, self).__init__()

        self.relu = nn.ReLU()


        padding1 = (kernel_size-1) * dilation[1]
        self.pad1 = torch.nn.ZeroPad2d((padding1, 0, 0, 0))
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation[1])
        self.net1 = nn.Sequential(self.pad1, self.conv1, self.relu)


        padding2 = (kernel_size - 1) * dilation[2]
        self.pad2 = torch.nn.ZeroPad2d((padding2, 0, 0, 0))
        self.conv2 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation[2])
        self.net2 = nn.Sequential(self.pad2, self.conv2, self.relu)



    def forward(self, x):

        y1 = self.net1(x)
        y2 = self.net2(x)
        SUM = y1 + y2

        return SUM




class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()

        self.relu = nn.ReLU()


        padding1 = (kernel_size-1) * dilation[0]
        self.pad1 = torch.nn.ZeroPad2d((padding1, 0, 0, 0))
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation[0])
        self.net1 = nn.Sequential(self.pad1, self.conv1, self.relu)


        padding2 = (kernel_size - 1) * dilation[1]
        self.pad2 = torch.nn.ZeroPad2d((padding2, 0, 0, 0))
        self.conv2 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation[1])
        self.net2 = nn.Sequential(self.pad2, self.conv2, self.relu)



    def forward(self, x):

        y1 = self.net1(x)
        y2 = self.net2(x)
        SUM = y1 + y2

        return SUM



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dilation, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        Residual_1 = []
        for i in range(3):
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            Residual_1 += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)]

        self.network = nn.Sequential(*Residual_1)
        self.downsample = nn.Conv1d(num_inputs, num_channels, 1)


        Residual_2 = []
        for i in range(4):
            in_channels = num_channels
            out_channels = num_channels
            Residual_2 += [TemporalBlock_2(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)]

        self.network_2 = nn.Sequential(*Residual_2)
        self.downsample_2 = nn.Conv1d(num_channels, num_channels, 1)


        Residual_3 = []
        for i in range(3):
            in_channels = num_channels
            out_channels = num_channels
            Residual_3 += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)]

        self.network_3 = nn.Sequential(*Residual_3)



    def forward(self, x):
        L1 = []
        L2 = []
        x = x.permute(0, 2, 1)
        for i in range(x.shape[1]):
            features = self.network(x[:,i].unsqueeze(1))
            res = self.downsample(x[:,i].unsqueeze(1))
            out = features + res

            y1 = self.network_2(out)
            z1 = self.downsample_2(out)
            out2 = y1 + z1

            y2 = self.network_2(out2)
            z2 = self.downsample_2(out2)
            out3 = y2 + z2

            y3 = self.network_3(out3)
            z3 = self.downsample_2(out3)
            out4 = y3 + z3

            L1.append(out4[:,0])
            L2.append(out4[:,1])

        L1 = torch.stack(L1, 1)
        L2 = torch.stack(L2, 1)

        return L1, L2



class M_TCNetwork_encoder(nn.Module):
    def __init__(self, input_size, num_channels, dilation, kernel_size=2, dropout=0.2, n_hidden=[64, 50, 10]):
        super(M_TCNetwork_encoder, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, dilation, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(116, 128)
        self.fc2 = nn.Linear(128, 32)
        self.drnn = DRNN(n_input=64, n_hidden=n_hidden, n_layers=3, cell_type='GRU')

    def forward(self, x):

        A1 = []
        A2 = []

        L1, L2 = self.tcn(x) #(16, 64, 150)

        for i in range(x.shape[1]):
            L1fc_out1 = self.fc1(L1[:,:,i])
            L1fc_out2 = self.fc2(L1fc_out1)
            A1.append(L1fc_out2)

        for i in range(x.shape[1]):
            L2fc_out1 = self.fc1(L2[:,:,i])
            L2fc_out2 = self.fc2(L2fc_out1)
            A2.append(L2fc_out2)

        A1 = torch.stack(A1, 1)
        A2 = torch.stack(A2, 1)

        out = torch.cat((A1, A2), dim=-1)
        outputs_fw, states_fw = self.drnn(out)
        return outputs_fw



# class M_TCNetwork_decoder(nn.Module):
#     def __init__(self, n_input, filters):
#         super(M_TCNetwork_decoder, self).__init__()
#
#         self.n_input = n_input
#         self.backto_input = filters
#
#         self.deconv_layer = nn.ConvTranspose1d(
#             in_channels=self.n_input,
#             out_channels=self.backto_input,
#             kernel_size=1,
#             stride=1,
#         )
#
#
#
#     def forward(self, features):
#
#
#         ## decoder
#         features = features.transpose(1, 2)
#         out_deconv = self.deconv_layer(features)
#         out_deconv = out_deconv.permute(0, 2, 1)
#
#
#         return out_deconv #(16, 150, 116)


class M_TCNetwork_decoder(nn.Module):
    def __init__(self, n_input, n_hidden, recon):
        super(M_TCNetwork_decoder, self).__init__()

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




class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        # self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'), gain=1)
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'))


    def forward(self, h_real):
        W = h_real.transpose(0,1)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.transpose(0,1), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans


class M_TCNetwork(nn.Module):
    def __init__(self, args, input_size, num_channels, dilation, kernel_size=2, dropout=0.2, n_hidden=[64, 50, 10]):
        super(M_TCNetwork, self).__init__()


        self.f_enc = M_TCNetwork_encoder(input_size, num_channels, dilation, kernel_size=kernel_size, dropout=dropout, n_hidden=n_hidden)
        self.f_dec = M_TCNetwork_decoder(n_input=n_hidden[-1], n_hidden=n_hidden, recon=116)
        self.kmeans = kmeans(args)

    def forward(self, x):

        emb = self.f_enc(x) #emb=torch.Size([16, 150, 10])
        aa = emb.reshape(-1, 10)  #torch.Size([2400, 10])
        kmeans_loss = self.kmeans(aa)

        input_recons = self.f_dec(emb)


        return aa, input_recons, kmeans_loss