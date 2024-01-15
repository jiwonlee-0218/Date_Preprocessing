import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchmetrics.functional import pairwise_euclidean_distance
from sklearn.cluster import AgglomerativeClustering, KMeans
from deep_KM_model import Transformer
from functools import partial
import math
from torch.nn import Parameter

import os
import glob







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









class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*args.timeseries, self.n_clusters, device='cuda'), gain=1)


    def forward(self, h_real):
        W = h_real.transpose(0,1)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        FTWTWF = torch.matmul(torch.matmul(self.F.transpose(0,1), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans









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
        self.drnn = DRNN(n_input=num_channels[-1], n_hidden=n_hidden, n_layers=3, cell_type='GRU')


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


















def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()









def compute_similarity(z, centroids, similarity="EUC"):   # z : (284, 10)
    """
    Function that compute distance between a latent vector z and the clusters centroids.
    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """

    # distance =  torch.sum(((z.unsqueeze(1) - centroids) ** 2), dim=2)
    cos_sim = torch.nn.CosineSimilarity(dim=2)
    distance = cos_sim(z.unsqueeze(1), centroids)
    # distance = pairwise_euclidean_distance(z, centroids)
    return distance







class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, path='.'):
        super(ClusterNet, self).__init__()

        # full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        # full_path = full_path[-1]
        full_path = '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning/TCN_5layer_3Dilated_RNN/TEST_EMOTION_6/Epochs451_BS_16_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_333_loss_0.03183.pt'
        print("I got: " + full_path + " weights")



        self.tcm_drnn = TCM_DRNN(args, input_size=116, num_channels=[64] * 5, kernel_size=15, dropout=0.25, n_hidden=[64, 50, 10])
        checkpoint = torch.load(full_path, map_location=args.device)
        self.tcm_drnn.load_state_dict(checkpoint['model_state_dict'])
        self.tcm_drnn.kmeans.F = checkpoint['cluster_indicator_matrix']


        ## clustering model
        self.alpha_ = 1.0
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.centr_size = 10

        ## centroid
        initial_cluster_centers = torch.zeros(self.n_clusters, self.centr_size, dtype=torch.float)
        nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=True)



    def forward(self, x):

        aa, x_recons, kmeans_loss = self.tcm_drnn(x)
        # aa_np = aa.detach().cpu()

        similarity = compute_similarity(
            aa, self.cluster_centers, similarity=self.similarity
        )

        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)  # (284, 8)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
        Q = Q / sum_columns_Q  # torch.Size([284, 8])



        ## P : ground truth distribution == target distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)  # torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
        P = P / sum_columns_P  # torch.Size([284, 8])


        return aa, x_recons, kmeans_loss, Q, P   # kl loss, all_preds = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])





############### 여기서는 euclidean으로 similarity구할 때, distance = pairwise_euclidean_distance(z, centroids)
class ClusterNet2(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, path='.'):
        super(ClusterNet2, self).__init__()

        # full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        # full_path = full_path[-1]
        full_path = '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning/TCN_5layer_3Dilated_RNN/EMOTION_6cluster_new/Epochs351_BS_8_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_191_loss_0.02950.pt'
        print("I got: " + full_path + " weights")



        self.tcm_drnn = TCM_DRNN(args, input_size=116, num_channels=[64] * 5, kernel_size=5, dropout=0.25, n_hidden=[100, 50, 10])
        checkpoint = torch.load(full_path, map_location=args.device)
        self.tcm_drnn.load_state_dict(checkpoint['model_state_dict'])
        self.tcm_drnn.kmeans.F = checkpoint['cluster_indicator_matrix']


        ## clustering model
        self.alpha_ = 1.0
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.centr_size = 10

        ## centroid
        initial_cluster_centers = torch.zeros(self.n_clusters, self.centr_size, dtype=torch.float)
        nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=True)



    def forward(self, x):

        aa, x_recons, kmeans_loss = self.tcm_drnn(x)


        similarity = compute_similarity(
            aa, self.cluster_centers, similarity=self.similarity
        )

        if torch.isnan(similarity).any():
            print('oops')

        numerator = 1.0 / (1.0 + (similarity / self.alpha_))
        power = float(self.alpha_ + 1) / 2
        numerator = numerator ** power
        Q = numerator / torch.sum(numerator, dim=1, keepdim=True)

        if torch.isnan(Q).any():
            print('oops')


        ## P : ground truth distribution == target distribution
        P = target_distribution(Q)

        if torch.isnan(P).any():
            print('oops')



        return aa, x_recons, Q, P








'''
Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)  # (284, 8)
sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
Q = Q / sum_columns_Q  # torch.Size([284, 8])

P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)  # torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
P = P / sum_columns_P  # torch.Size([284, 8])
'''












############### 여기서는 euclidean으로 similarity구할 때, distance =  torch.sqrt(torch.sum(((z.unsqueeze(1) - centroids) ** 2), dim=2) + 1.e-12)
class ClusterNet3(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self,  args, path='.'):
        super(ClusterNet3, self).__init__()

        # full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        # full_path = full_path[-1]
        full_path = '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning/TCN_5layer_3Dilated_RNN/EMOTION_6cluster_new/Epochs351_BS_8_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_191_loss_0.02950.pt'
        print("I got: " + full_path + " weights")





        self.tcm_drnn = TCM_DRNN(args, input_size=116, num_channels=[64] * 5, kernel_size=5, dropout=0.25,
                                 n_hidden=[100, 50, 10])
        checkpoint = torch.load(full_path, map_location=args.device)
        self.tcm_drnn.load_state_dict(checkpoint['model_state_dict'])
        self.tcm_drnn.kmeans.F = checkpoint['cluster_indicator_matrix']




        ## clustering model
        self.alpha_ = 1.0
        self.n_clusters = args.n_clusters
        self.timeseries = args.timeseries
        self.device = args.device
        self.similarity = args.similarity
        self.centr_size = 10


        self.kld = nn.KLDivLoss(size_average=False, reduction='batchmean')




    def forward(self, x):

        aa, x_recons, kmeans_loss = self.tcm_drnn(x)
        aa = aa.reshape(-1, self.timeseries, self.centr_size)
        aa_np = aa.detach().cpu()

        if torch.isnan(aa).any():
            return aa, x_recons, kmeans_loss
        all_kl_loss = 0
        for i in range(aa_np.shape[0]):  # z.size(0): 16 == batch_size


            ## initial centroid individual
            assignements = KMeans(n_clusters=self.n_clusters).fit_predict(aa_np[i])
            # assignements = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete",
            #                                        affinity="euclidean").fit(z_np[i])

            centroids_ = torch.zeros(
                (self.n_clusters, aa_np[i].shape[1]), device=self.device
            )

            for cluster_ in range(self.n_clusters):
                centroids_[cluster_] = aa_np[i][assignements == cluster_].mean(axis=0)


            cluster_centers = centroids_   # centroids_ : torch.Size([8, 32])


            similarity = compute_similarity(
                aa[i], cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
            )


            ## Q (batch_size , n_clusters)

            Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
            sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
            final_Q = Q / sum_columns_Q

            ## P : ground truth distribution == target distribution
            P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1,-1)
            sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
            final_P = P / sum_columns_P

            kl_loss = self.kld(final_Q.log(), final_P)
            all_kl_loss += kl_loss


        LOSS_KL = ( all_kl_loss / aa_np.shape[0] )

        return aa, x_recons, LOSS_KL










########################### transformer를 위한 kl loss #################################

class ClusterNet4(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self,  args, path='.'):
        super(ClusterNet4, self).__init__()

        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")


        self.trans = Transformer(args)
        checkpoint = torch.load(full_path, map_location=args.device)
        self.trans.load_state_dict(checkpoint['model_state_dict'])




        ## clustering model
        self.alpha_ = 1.0
        self.n_clusters = args.n_clusters
        self.timeseries = args.timeseries
        self.device = args.device
        self.centr_size = 64

        self.loss_function = nn.KLDivLoss(size_average=False)


        ## centroid
        initial_cluster_centers = torch.zeros(
                self.n_clusters, self.centr_size, dtype=torch.float            ## (8, 284, 116)
            )
        nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = Parameter(initial_cluster_centers)


    def init_centroids(self, x):

        X_tensor = x.type(torch.FloatTensor).to(self.device)
        X_tensor = X_tensor.unsqueeze(dim=0).detach()

        features = self.trans.backbone(X_tensor)
        z = features.detach().cpu().squeeze()

        assignements = KMeans(n_clusters=self.n_clusters).fit_predict(z)
        # assignements = AgglomerativeClustering(n_clusters= self.n_clusters, linkage="complete", affinity="euclidean").fit_predict(z)

        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )

        for cluster_ in range(self.n_clusters):
            centroids_[cluster_] = z[assignements == cluster_].mean(axis=0)

        cluster_centers = centroids_

        return cluster_centers


    def forward(self, x):

        z, x_reconstr = self.trans(x)


        kld = 0
        Q_list = []
        P_list = []


        for i in range(z.size(0)):
            similarity = compute_similarity(
                z[i], self.cluster_centers
            )


            numerator = 1.0 / (1.0 + (similarity / self.alpha_))
            power = float(self.alpha_ + 1) / 2
            numerator = numerator ** power
            Q = numerator / torch.sum(numerator, dim=1, keepdim=True)


            ## P : ground truth distribution == target distribution
            P = target_distribution(Q).detach()


            loss = self.loss_function(Q.log(), P) / Q.shape[0]
            kld += loss

            Q_list.append(Q)
            P_list.append(P)

        kld = kld / z.size(0)
        Q = torch.stack(Q_list, 0)
        P = torch.stack(P_list, 0)


        return z, x_reconstr, Q, P, kld


        # similarity = compute_similarity(
        #     aa, self.cluster_centers, similarity=self.similarity
        # )
        #
        # Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)  # (284, 8)
        # sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
        # Q = Q / sum_columns_Q  # torch.Size([284, 8])
        #
        # ## P : ground truth distribution == target distribution
        # P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1,
        #                                                -1)  # torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
        # sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
        # P = P / sum_columns_P  # torch.Size([284, 8])

        # return aa, x_recons, bb