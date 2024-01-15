import torch
import torch.nn as nn

# class Chomp1d(torch.nn.Module):
#     """
#     Removes the last elements of a time series.
#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
#     is the number of elements to remove.
#     @param chomp_size Number of elements to remove.
#     """
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :-self.chomp_size]
#
#
# class SqueezeChannels(torch.nn.Module):
#     """
#     Squeezes, in a three-dimensional tensor, the third dimension.
#     """
#     def __init__(self):
#         super(SqueezeChannels, self).__init__()
#
#     def forward(self, x):
#         return x.squeeze(2)
#
#
# class CausalConvolutionBlock(torch.nn.Module):
#     """
#     Causal convolution block, composed sequentially of two causal convolutions
#     (with leaky ReLU activation functions), and a parallel residual connection.
#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
#     @param in_channels Number of input channels.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     @param dilation Dilation parameter of non-residual convolutions.
#     @param final Disables, if True, the last activation function.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, dilation,
#                  final=False):
#         super(CausalConvolutionBlock, self).__init__()
#
#         # Computes left padding so that the applied convolutions are causal
#         padding = (kernel_size - 1) * dilation
#
#         # First causal convolution
#         conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=padding, dilation=dilation
#         ))
#         # The truncation makes the convolution causal
#         chomp1 = Chomp1d(padding)
#         relu1 = torch.nn.LeakyReLU()
#
#         # Second causal convolution
#         conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
#             out_channels, out_channels, kernel_size,
#             padding=padding, dilation=dilation
#         ))
#         chomp2 = Chomp1d(padding)
#         relu2 = torch.nn.LeakyReLU()
#
#         # Causal network
#         self.causal = torch.nn.Sequential(
#             conv1, chomp1, relu1, conv2, chomp2, relu2
#         )
#
#         # Residual connection
#         self.upordownsample = torch.nn.Conv1d(
#             in_channels, out_channels, 1
#         ) if in_channels != out_channels else None
#
#         # Final activation function
#         self.relu = torch.nn.LeakyReLU() if final else None
#
#     def forward(self, x):
#         out_causal = self.causal(x)
#         res = x if self.upordownsample is None else self.upordownsample(x)
#         if self.relu is None:
#             return out_causal + res
#         else:
#             return self.relu(out_causal + res)
#
#
# class CausalCNN(torch.nn.Module):
#     """
#     Causal CNN, composed of a sequence of causal convolution blocks.
#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
#     @param in_channels Number of input channels.
#     @param channels Number of channels processed in the network and of output
#            channels.
#     @param depth Depth of the network.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, out_channels,
#                  kernel_size):
#         super(CausalCNN, self).__init__()
#
#         layers = []  # List of causal convolution blocks
#         dilation_size = 1  # Initial dilation size
#
#         for i in range(depth):
#             in_channels_block = in_channels if i == 0 else channels
#             layers += [CausalConvolutionBlock(
#                 in_channels_block, channels, kernel_size, dilation_size
#             )]
#             dilation_size *= 2  # Doubles the dilation size at each step
#
#         # Last layer
#         layers += [CausalConvolutionBlock(
#             channels, out_channels, kernel_size, dilation_size
#         )]
#
#         self.network = torch.nn.Sequential(*layers)  #causal_cnn
#
#     def forward(self, x):
#         return self.network(x)
#
#
# class CausalCNNEncoder(torch.nn.Module):
#     """
#     Encoder of a time series using a causal CNN: the computed representation is
#     the output of a fully connected layer applied to the output of an adaptive
#     max pooling layer applied on top of the causal CNN, which reduces the
#     length of the time series to a fixed size.
#     Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
#     batch size, `C` is the number of input channels, and `L` is the length of
#     the input. Outputs a three-dimensional tensor (`B`, `C`).
#     @param in_channels Number of input channels.
#     @param channels Number of channels manipulated in the causal CNN.
#     @param depth Depth of the causal CNN.
#     @param reduced_size Fixed length to which the output time series of the
#            causal CNN is reduced.
#     @param out_channels Number of output channels.
#     @param kernel_size Kernel size of the applied non-residual convolutions.
#     """
#     def __init__(self, in_channels, channels, depth, reduced_size,
#                  out_channels, kernel_size):
#         super(CausalCNNEncoder, self).__init__()
#         causal_cnn = CausalCNN(
#             in_channels, channels, depth, reduced_size, kernel_size
#         )
#         # reduce_size = torch.nn.AdaptiveMaxPool1d(1)
#         squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
#         # linear = torch.nn.Linear(reduced_size, out_channels)
#         self.network = torch.nn.Sequential(
#             causal_cnn, squeeze
#         )
#
#     def forward(self, x):
#         return self.network(x)
#
#
# class DRNN(nn.Module):
#
#     def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True):
#         super(DRNN, self).__init__()
#
#         self.dilations = [1, 4, 16]
#         # self.dilations = [1, 16]
#         self.cell_type = cell_type
#         self.batch_first = batch_first
#
#         layers = []
#         cell = nn.GRU
#
#         for i in range(n_layers):
#             if i == 0:
#                 c = cell(n_input, n_hidden[i], dropout=dropout)
#             else:
#                 c = cell(n_hidden[i-1], n_hidden[i], dropout=dropout)
#             layers.append(c)
#         self.cells = nn.Sequential(*layers)
#
#     def forward(self, inputs, hidden=None):
#         if self.batch_first:
#             inputs = inputs.transpose(0, 1)
#         outputs = []
#         for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
#             if hidden is None:
#                 inputs, _ = self.drnn_layer(cell, inputs, dilation)
#             else:
#                 inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
#
#             outputs.append(inputs[-1])
#
#         if self.batch_first:
#             inputs = inputs.transpose(0, 1)  #(N, L, H)
#         return inputs, outputs
#
#     def drnn_layer(self, cell, inputs, rate, hidden=None):
#         n_steps = len(inputs)
#         batch_size = inputs[0].size(0)
#         hidden_size = cell.hidden_size
#
#         inputs, _ = self._pad_inputs(inputs, n_steps, rate)
#         dilated_inputs = self._prepare_inputs(inputs, rate)
#
#         if hidden is None:
#             dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
#         else:
#             hidden = self._prepare_inputs(hidden, rate)
#             dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)
#
#         splitted_outputs = self._split_outputs(dilated_outputs, rate)
#         outputs = self._unpad_outputs(splitted_outputs, n_steps)
#
#         return outputs, hidden
#
#     def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
#         if hidden is None:
#             hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
#
#         dilated_outputs, hidden = cell(dilated_inputs, hidden)
#
#         return dilated_outputs, hidden
#
#     def _unpad_outputs(self, splitted_outputs, n_steps):
#         return splitted_outputs[:n_steps]
#
#     def _split_outputs(self, dilated_outputs, rate):
#         batchsize = dilated_outputs.size(1) // rate
#
#         blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]
#
#         interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
#         interleaved = interleaved.view(dilated_outputs.size(0) * rate,
#                                        batchsize,
#                                        dilated_outputs.size(2))
#         return interleaved
#
#     def _pad_inputs(self, inputs, n_steps, rate):
#         is_even = (n_steps % rate) == 0
#
#         if not is_even:
#             dilated_steps = n_steps // rate + 1
#
#             zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
#                                  inputs.size(1),
#                                  inputs.size(2))
#
#             zeros_ = zeros_.cuda()
#
#             inputs = torch.cat((inputs, zeros_))
#         else:
#             dilated_steps = n_steps // rate
#
#         return inputs, dilated_steps
#
#     def _prepare_inputs(self, inputs, rate):
#         dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
#         return dilated_inputs
#
#     def init_hidden(self, batch_size, hidden_dim):
#         hidden = torch.zeros(batch_size, hidden_dim)
#         hidden = hidden.cuda()
#
#         return hidden
#
#
#
# class drnn(nn.Module):
#     def __init__(self, n_input, n_hidden, in_channels, channels, depth, reduced_size,
#                  out_channels, kernel_size):
#         super(drnn, self).__init__()
#
#
#         self.drnn = DRNN(n_input=n_input, n_hidden=n_hidden, n_layers=3, cell_type='GRU')
#         self.causal_cnn = CausalCNN(
#             in_channels, channels, depth, reduced_size, kernel_size
#         )
#
#
#     def forward(self, inputs):
#         outputs_fw, states_fw = self.drnn(inputs)
#         x = outputs_fw.permute(0, 2, 1)
#         features = self.causal_cnn(x)
#         features = features.permute(0, 2, 1)
#         return features






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
    def __init__(self, args, input_size, num_channels, kernel_size=2):
        super(CausalCNNEncoder, self).__init__()

        causal_cnn = CausalCNN(input_size, num_channels, kernel_size)
        squeeze = SqueezeChannels()

        self.network = torch.nn.Sequential(causal_cnn, squeeze)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)
        output = features.permute(0, 2, 1)
        return output



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



# TCNModel(input_size= 116, output_size=10, num_channels=[20] * 2, kernel_size=3, dropout=0.25)
class ContrastiveTCN(nn.Module):
    def __init__(self, args, input_size, num_channels, kernel_size=2):
        super(ContrastiveTCN, self).__init__()

        self.tcn = CausalCNNEncoder(args, input_size, num_channels, kernel_size=kernel_size)
        # self.kmeans = kmeans(args)

    def forward(self, inputs):
        emb = self.tcn(inputs)
        return emb