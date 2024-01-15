import numpy as np
import torch
import torch.nn as nn






class Decoder(nn.Module):
    def __init__(self, n_input, n_hidden, device):
        super(Decoder, self).__init__()

        self.device = device


        self.rnn = nn.GRU(n_input, n_hidden, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(n_hidden, n_input)



    def forward(self, inputs, hidden):

        hidden = hidden.unsqueeze(0)

        # Initializing hidden state for first input with zeros
        hidden_state_collector = torch.empty((inputs.size(0), inputs.size(1), hidden.size(-1)), device=self.device)
        output, fcn_latent = self.rnn(inputs, hidden)
        hidden_state_collector = output
        reconstruction = self.output_layer(hidden_state_collector)


        return reconstruction, fcn_latent




class DRNN(nn.Module):

    def __init__(self, n_input, fw_n_hidden, bw_n_hidden, dropout=0, cell_type='GRU', device=None, batch_first=True):
        super(DRNN, self).__init__()

        # self.fw_n_hidden = fw_n_hidden #[100, 50, 50]
        # self.bw_n_hidden = bw_n_hidden #[50, 30, 30]
        n_layers = len(fw_n_hidden)
        self.device = device
        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError


        for i in range(n_layers):
            c = cell(n_input, fw_n_hidden[i], dropout=dropout)
            n_input = fw_n_hidden[i]

            layers.append(c)
        self.cells = nn.Sequential(*layers)


    def forward(self, inputs, hidden=None):
        inputs = inputs.to(self.device)
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-1]) ## layer의 마지막 hidden state들

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        if self.cell_type == "GRU":
            encoder_state_fw = torch.concat(outputs, dim=1)  #list:3 (56, 3) -> (56, 96)

        return inputs, encoder_state_fw  ## last layer(3) output, all layer last hidden states

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate, )  #inputs(286, 8, 1), _(286), -> _(143)
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
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        cell = cell.to(self.device)
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
            if self.device:
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
        if self.device:
            hidden = hidden.to(self.device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if self.device:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden








class Encoder_Decoder(nn.Module):
    def __init__(self, n_input, fw_n_hidden, bw_n_hidden, device):
        super(Encoder_Decoder, self).__init__()



        self.f_enc = DRNN(n_input=n_input, fw_n_hidden=fw_n_hidden, bw_n_hidden=bw_n_hidden, cell_type='GRU', device = device)
        self.f_dec = Decoder(n_input=n_input, n_hidden = np.sum(fw_n_hidden), device=device)

    def forward(self, inputs):
        f_outputs, f_final_hidden_state = self.f_enc(inputs)  #torch.Size([8, 286, 32]), (8, 96)

        recons_input, fcn_latent = self.f_dec(inputs, f_final_hidden_state)
        #(8, 286, 1), (1, 8, 96)


        return recons_input, fcn_latent, f_final_hidden_state


class classifier(nn.Module):
    def __init__(self, in_features, hidden_units = [128, 2]):
        super(classifier, self).__init__()


        self.hidden_units_0 = hidden_units[0]
        self.hidden_units_1 = hidden_units[1]
        self.fc1 = nn.Linear(in_features=in_features, out_features=self.hidden_units_0, bias=False)
        self.fc2 = nn.Linear(in_features=self.hidden_units_0, out_features=self.hidden_units_1, bias=False)


    def forward(self, inputs):

        x = self.fc1(inputs)
        out = self.fc2(x)

        return out



class kmeans(nn.Module):
    def __init__(self):
        super(kmeans, self).__init__()


        self.batch_size = 4
        self.hidden_dim = 32
        self.training_samples_num = 112
        self.n_clusters = 2
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)


    def forward(self, h_real):
        W = h_real.permute(1,0)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.permute(1,0), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans

