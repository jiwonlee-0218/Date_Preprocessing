import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

## TCN: Dilated Convolutions + Residual Connections + Weight Norm. + Dropout
''' contrastive with dilated convolution '''

def transpose(x):
    return x.transpose(-2, -1)


class Bidirection_DRNN(nn.Module):

    def __init__(self, args, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
        super(Bidirection_DRNN, self).__init__()

        self.dilations = [1, 4, 16]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.n_hidden = n_hidden
        self.batch_size = args.batch_size
        self.timeseries = args.timeseries

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
            bw_inputs = torch.flip(inputs, dims=[1]).transpose(0, 1)
            fw_inputs = inputs.transpose(0, 1)

        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                fw_outputs, _ = self.drnn_layer(cell, fw_inputs, dilation)
                bw_outputs, bw_ = self.drnn_layer(cell, bw_inputs, dilation)
                total_inputs = torch.sum(torch.concat((fw_outputs, bw_outputs), 2).view(fw_inputs.shape[1], self.timeseries, 2, self.n_hidden[i]), dim=2)
                fw_inputs = total_inputs.transpose(0, 1)
                bw_inputs = torch.flip(total_inputs, dims=[1]).transpose(0, 1)
            else:
                fw_inputs, hidden[i] = self.drnn_layer(cell, fw_inputs, dilation, hidden[i])


            outputs.append(fw_inputs[-1])

        # if self.batch_first:
        #     total_inputs = total_inputs.transpose(0, 1)  #(N, L, H)
        return total_inputs, outputs

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


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class CausalConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(CausalConvolutionBlock, self).__init__()


        # First causal convolution
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                                    in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)
                                    )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.LeakyReLU()


        # Second causal convolution
        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                                    out_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)
                                )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(in_channels, out_channels, 1)

        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, num_channels, kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        depth = len(num_channels)

        for i in range(depth):
            dilation_size = 2 ** i
            in_channels_block = in_channels if i == 0 else num_channels[i-1]
            out_channels_block = num_channels[i]
            layers += [CausalConvolutionBlock(in_channels_block, out_channels_block, kernel_size,
                                              stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size)]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(nn.Module):
    def __init__(self, args, input_size, num_channels, kernel_size=2, n_hidden=[96, 64, 32]):
        super(CausalCNNEncoder, self).__init__()

        causal_cnn = CausalCNN(input_size, num_channels, kernel_size)
        squeeze = SqueezeChannels()

        self.network = torch.nn.Sequential(causal_cnn, squeeze)
        self.drnn = DRNN(n_input=num_channels[-1], n_hidden=n_hidden, n_layers=3, cell_type='GRU')
        # self.bi_drnn = Bidirection_DRNN(args, n_input=num_channels[-1], n_hidden=n_hidden, n_layers=3, cell_type='GRU')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)
        features = features.permute(0, 2, 1)
        outputs_fw, states_fw = self.drnn(features)
        return outputs_fw



class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        # self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'), gain=1)
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size * args.timeseries, self.n_clusters, device='cuda'))
        self.F_aug = torch.nn.init.orthogonal_(torch.randn(self.batch_size * args.timeseries, self.n_clusters, device='cuda'))


    def forward(self, h_real, h_aug):
        W = h_real.transpose(0,1)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.transpose(0,1), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss


        W_aug = h_aug.transpose(0,1)
        WTW_aug = torch.matmul(h_aug, W_aug)
        FTWTWF_aug = torch.matmul(torch.matmul(self.F_aug.transpose(0,1), WTW_aug), self.F_aug)
        loss_kmeans_aug = torch.trace(WTW_aug) - torch.trace(FTWTWF_aug)


        return loss_kmeans, loss_kmeans_aug



def _get_triplet_loss(representation, positive_representation):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    num = representation.size(1)
    loss = 0
    n_loss = 0
    nb_sample = 10

    A1 = [representation[a] @ positive_representation[a].T for a in range(representation.size(0))]
    A2 = [representation[a] @ representation[a].T for a in range(representation.size(0))]

    A1 = torch.stack(A1)
    A2 = torch.stack(A2)

    for i in range(num):
        positive = A1[:,i,i]
        p_loss = -torch.mean(torch.nn.functional.logsigmoid(positive))

        n1 = torch.concat((A1[:,i , :i], A1[:, i, i+1:]), 1)
        n2 = torch.concat((A2[:,i , :i], A2[:, i, i+1:]), 1)
        negative = torch.concat((n1, n2), 1)

        negative_random_index = numpy.random.randint(0, negative.shape[1], size=nb_sample)
        for j in range(negative_random_index.size):
            n_loss += 0.1 * -torch.mean(torch.nn.functional.logsigmoid(-negative[:, negative_random_index[j]]))

        loss += (p_loss+n_loss)

    loss = (loss / num)
    return loss

def _get_nce_loss(representation, positive_representation):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    temperature=0.1
    reduction = 'mean'
    # similarity = cosine_similarity(representation[0].detach().cpu().numpy(), positive_representation[0].detach().cpu().numpy())  (150, 150)
    query = representation.reshape(-1, 32)
    positive_key = positive_representation.reshape(-1, 32)

    logits = query @ transpose(positive_key)
    labels = torch.arange(len(query), device=query.device)

    loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)


    return loss



# TCNModel(input_size= 116, output_size=10, num_channels=[20] * 2, kernel_size=3, dropout=0.25)
class ContrastiveTCN(nn.Module):
    def __init__(self, args, input_size, num_channels, kernel_size=2, n_hidden=[96, 64, 32]):
        super(ContrastiveTCN, self).__init__()

        self.tcn = CausalCNNEncoder(args, input_size, num_channels, kernel_size=kernel_size, n_hidden=n_hidden)
        self.kmeans = kmeans(args)

    def forward(self, batch):
        batch_size = batch.size(0)
        length = batch.size(2)

        length_pos_neg = 96
        random_length = 96

        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=length - length_pos_neg + 1, size=batch_size
        )
        end_positive = beginning_samples_pos + length_pos_neg
        end_batches = beginning_batches + random_length

        representation = self.tcn(torch.cat(
            [batch[
             j: j + 1, :,
             beginning_batches[j]: beginning_batches[j] + random_length] for j in range(batch_size)]
        ))  # Anchors representations

        positive_representation = self.tcn(torch.cat(
            [batch[
             j: j + 1, :,
             end_positive[j] - length_pos_neg: end_positive[j]] for j in range(batch_size)]
        ))  # Positive samples representations


        tri_loss = _get_nce_loss(representation, positive_representation)

        ''' Kmeans '''
        representation_emb = representation.reshape(-1, 32)
        positive_representation_emb = positive_representation.reshape(-1, 32)
        # loss_kmeans, loss_kmeans_aug = self.kmeans(representation_emb, positive_representation_emb)




        return representation_emb, positive_representation_emb, tri_loss