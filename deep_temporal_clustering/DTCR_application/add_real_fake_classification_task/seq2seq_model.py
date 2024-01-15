import numpy as np
import torch
import torch.nn as nn






class Decoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Decoder, self).__init__()



        self.rnn = nn.GRU(n_input, n_hidden, num_layers=1, batch_first=True)



    def forward(self, inputs):


        output, fcn_latent = self.rnn(inputs)



        return output #(16, 150, 116)




class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(DRNN, self).__init__()

        self.dilations = [1, 4, 16]
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
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

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



class Classifier(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, in_features, hidden_units):
        super(Classifier, self).__init__()

        self.hidden_units_1 = hidden_units[0]
        self.hidden_units_2 = hidden_units[1]
        self.fc1 = nn.Linear(in_features=in_features, out_features=self.hidden_units_1, bias=False)
        self.fc2 = nn.Linear(in_features=self.hidden_units_1, out_features=self.hidden_units_2, bias=False)


    def forward(self, inputs):

        #inputs: torch.Size([16, 400])
        x = self.fc1(inputs)
        out = self.fc2(x)

        return out

class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Encoder, self).__init__()



        self.drnn = DRNN(n_input=n_input, n_hidden=n_hidden, n_layers=3, cell_type='GRU')



    def forward(self, inputs):

        outputs_fw, states_fw = self.drnn(inputs)

        inputs_bw = torch.flip(inputs, dims=[1])
        outputs_bw, states_bw = self.drnn(inputs_bw)

        states_fw = torch.cat(states_fw, 1)
        states_bw = torch.cat(states_bw, 1)
        final_states = torch.cat((states_fw, states_bw), 1)
        return outputs_fw, final_states




class Seq2Seq(nn.Module):
    def __init__(self, args, n_input, n_hidden, classification_unit):
        super(Seq2Seq, self).__init__()

        self.f_enc = Encoder(n_input=n_input, n_hidden=n_hidden)
        self.f_dec = Decoder(n_input=n_hidden[-1], n_hidden = n_input)
        self.f_classifier = Classifier(in_features=np.sum(n_hidden)*2, hidden_units=classification_unit)
        self.kmeans = kmeans(args)

    def forward(self, inputs):


        f_outputs, f_final_hidden_state = self.f_enc(inputs)  #torch.Size([16, 150, 50]), (16, 400)

        # real_features = f_outputs[label[:,0]==1]
        # real_hidden_state = f_final_hidden_state[label[:,0]==1]

        # loss_kmeans = self.kmeans(real_hidden_state)
        #
        classifier_output = self.f_classifier(f_final_hidden_state)
        recons_input = self.f_dec(f_outputs)
        # real_recons = recons_input[label[:,0]==1]


        return f_outputs, f_final_hidden_state, classifier_output




class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)


    def forward(self, h_real):
        W = h_real.permute(1,0)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.permute(1,0), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans



