import torch.nn as nn
import torch
from utils import compute_similarity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import gc
import glob
import os


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_2, filter_3, filter_lstm):
        super(TAE_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]


        ## CNN PART
        ### output shape (batch_size, 7 , n_hidden = 284)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=116,
                out_channels=filter_1,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(),
        )


        ## LSTM PART
        ### output shape (batch_size , n_hidden = 284 , 50)
        self.lstm_1 = nn.LSTM(
            input_size= filter_1,
            hidden_size=self.hidden_lstm_1,
            batch_first=True

        )
        self.act_lstm_1 = nn.Tanh()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True

        )
        self.act_lstm_2 = nn.Tanh()


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2)
        x = self.conv_layer_1(x)
        out_cnn = x.permute(0, 2, 1)

        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1_act = self.act_lstm_1(out_lstm1)


        features, _ = self.lstm_2(out_lstm1_act) # (1, 6, 64)
        out_lstm2_act = self.act_lstm_2(features)



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
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.hidden_lstm_2 ,
            out_channels=116,
            kernel_size=1,
            stride=1,
        )

    def forward(self, features):

        ## decoder
        features = features.transpose(1, 2)  ##(batch_size  , n_hidden , pooling)
        out_deconv = self.deconv_layer(features)
        out_deconv = out_deconv.permute(0,2,1)
        return out_deconv




class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_1=64,  filter_lstm=[50, 32]):  # 내 모델에 사용될 구성품을 정의 및 초기화하는 메서드
        super(TAE, self).__init__()


        self.filter_1 = filter_1
        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
        )



        self.tae_decoder = TAE_decoder(
            filter_lstm = self.filter_lstm
        )



    def forward(self, x):        #  init에서 정의된 구성품들을 연결하는 메서드

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features, out_deconv   # features는 clustering을 위해 encoder의 output을 사용














class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, TAE: torch.nn.Module, cluster_centers = None, filter_lstm=[50, 32]):
        super(ClusterNet, self).__init__()



        ######### init with the pretrained autoencoder model
        # self.tae = TAE(args)


        self.tae = TAE ################ modified



        ## clustering model
        self.alpha_ = 1.0
        self.centr_size = filter_lstm[1]  # centroids_의 dimension
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.loss_function = nn.KLDivLoss(reduction='batchmean')

        ## centroid
        # if cluster_centers is None:
        #     initial_cluster_centers = torch.zeros(
        #         self.n_clusters, self.centr_size, dtype=torch.float            ## (8, 284, 116)
        #     )
        #     nn.init.xavier_uniform_(initial_cluster_centers)
        # else:
        #     initial_cluster_centers = cluster_centers
        #
        #
        # self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=True)




    def forward(self, x, all_gt):

        z, x_reconstr = self.tae(x)  # z: (16, 284, 32), x_reconstr: (16, 284, 116)

        ARI = 0
        all_kl_loss = 0
        pred_label = all_gt[0][0]


        z_np = z.detach().cpu()
        for i in range(z_np.size(0)):  # z.size(0): 16 == batch_size


            ## initial centroid individual
            assignements = KMeans(n_clusters=self.n_clusters).fit(z_np[i])
            # assignements = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete",
            #                                        affinity="euclidean").fit(z_np[i])

            centroids_ = torch.zeros(
                (self.n_clusters, z_np[i].shape[1]), device=self.device
            )

            for cluster_ in range(self.n_clusters):
                centroids_[cluster_] = z_np[i][assignements.labels_ == cluster_].mean(axis=0)


            cluster_centers = centroids_   # centroids_ : torch.Size([8, 32])


            similarity = compute_similarity(
                z[i], cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
            )


            ## Q (batch_size , n_clusters)
            Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
            sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
            final_Q = Q / sum_columns_Q

            ## P : ground truth distribution == target distribution
            P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1,-1)
            sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
            final_P = P / sum_columns_P

            kl_loss = self.loss_function(final_Q.log(), final_P.detach())
            all_kl_loss = all_kl_loss + kl_loss

            preds = torch.max(final_Q, dim=1)[1]  # preds: torch.Size([284])
            a = preds.cpu().numpy()
            b = pred_label.cpu().numpy()
            ARI += adjusted_rand_score(a, b)


        LOSS_KL = (all_kl_loss / z_np.size(0))
        all_batch_ARI = (ARI / (x.size(0)))  # batch_size에 대한 ARI

        return z, x_reconstr, LOSS_KL, all_batch_ARI  # kl loss, all_preds = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])
















    '''
    def forward(self, x, all_gt):

        z, x_reconstr = self.tae(x)  # z: (16, 284, 32), x_reconstr: (16, 284, 116)
        z_np = z.detach().cpu()  # z_np: (64, 284, 32)

        ARI = 0
        all_kl_loss = 0
        pred_label = all_gt[0][0]

        fp = []
        fq = []
        z_np = z.detach().cpu()
        for i in range(z_np.size(0)):  # z.size(0): 16 == batch_size

            # z[i] = z[i].detach().cpu().numpy()
            ## initial centroid individual
            assignements = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete",
                                                   affinity="euclidean").fit(z_np[i])

            centroids_ = torch.zeros(
                (self.n_clusters, z_np[i].shape[1]), device=self.device
            )  # centroids_ : torch.Size([8, 32])

            for cluster_ in range(self.n_clusters):
                centroids_[cluster_] = z_np[i][assignements.labels_ == cluster_].mean(axis=0)
            # centroids_ : torch.Size([8, 32])

            cluster_centers = centroids_


            similarity = compute_similarity(
                z[i], cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
            )
            # similarity = compute_similarity(
            #     z[i], self.cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
            # )

            ## Q (batch_size , n_clusters)
            Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)  # (284, 8)
            sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
            final_Q = Q / sum_columns_Q  # torch.Size([284, 8])

            ## P : ground truth distribution == target distribution
            P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1,
                                                           -1)  # torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
            sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
            final_P = P / sum_columns_P  # torch.Size([284, 8])

            fp.append(final_P)
            fq.append(final_Q)

            preds = torch.max(final_Q, dim=1)[1]  # preds: torch.Size([284])
            a = preds.cpu().numpy()
            b = pred_label.cpu().numpy()
            ARI += adjusted_rand_score(a, b)

        fp = torch.stack(fp, 0)  # torch.Size([8, 284, 8])
        fq = torch.stack(fq, 0)  # torch.Size([8, 284, 8])

        all_batch_ARI = (ARI / (x.size(0)))  # batch_size에 대한 ARI

        return z, x_reconstr, fp, fq, all_batch_ARI  # kl loss, all_preds = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])
        # return z, x_reconstr
        '''
























