import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import load_data
import utils
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
import argparse
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture


def writelog(file, line):
    file.write(line + '\n')
    print(line)

def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="EMOTION_1005010_2", help="dataset name")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    # model args
    parser.add_argument("--model_name", default="DTCR_bidirection_dilated_RNN2",help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")

    # parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping: Maximal parameter gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=500, help="Epochs number of the autoencoder training",)
    parser.add_argument("--max_patience", type=int, default=15, help="The maximum patience for pre-training, above which we stop training.",)

    parser.add_argument("--lr_ae", type=float, default=0.01, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss',)
    parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')
    # parser.add_argument("--ae_models", default='full_models/', help='full autoencoder weights')
    # parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--ae_models", default=None, help='full autoencoder weights')
    parser.add_argument("--autoencoder_test", default=None, help='full autoencoder weights')


    return parser



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



class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Encoder, self).__init__()


        self.drnn = DRNN(n_input=n_input, n_hidden=n_hidden, n_layers=3, cell_type='GRU')



    def forward(self, inputs):
        ## encoder
        outputs_fw, states_fw = self.drnn(inputs)

        # inputs_bw = torch.flip(inputs, dims=[1])
        # outputs_bw, states_bw = self.drnn(inputs_bw)

        states_fw = torch.cat(states_fw, 1)
        # states_bw = torch.cat(states_bw, 1)
        # final_states = torch.cat((states_fw, states_bw), 1)
        # return outputs_fw, final_states
        return outputs_fw, states_fw






class kmeans(nn.Module):
    def __init__(self, args):
        super(kmeans, self).__init__()


        self.batch_size = args.batch_size
        self.n_clusters = args.n_clusters
        self.F = torch.nn.init.orthogonal_(torch.randn(self.batch_size*150, self.n_clusters, device='cuda'), gain=1)


    def forward(self, h_real):
        W = h_real.permute(1,0)  # shape: hidden_dim, training_samples_num, m*N
        WTW = torch.matmul(h_real, W)
        # term_F = torch.nn.init.orthogonal_(torch.randn(self.batch_size, self.n_clusters, device='cuda'), gain=1)
        FTWTWF = torch.matmul(torch.matmul(self.F.permute(1,0), WTW), self.F)
        loss_kmeans = torch.trace(WTW) - torch.trace(FTWTWF)  # k-means loss

        return loss_kmeans









class Seq2Seq(nn.Module):
    def __init__(self, args, n_input, n_hidden):
        super(Seq2Seq, self).__init__()



        self.f_enc = Encoder(n_input=n_input, n_hidden=n_hidden)
        self.f_dec = Decoder(n_input=n_hidden[-1], n_hidden = n_input)
        self.kmeans = kmeans(args)

    def forward(self, inputs):


        f_outputs, f_final_hidden_state = self.f_enc(inputs)  #torch.Size([16, 150, 50]), (16, 200)

        aa = f_outputs.reshape(-1, 10)
        kmeans_loss = self.kmeans(aa)

        true_input_recons = self.f_dec(f_outputs)


        return aa, true_input_recons, kmeans_loss


















if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR']
    # samples = samples[:, :, 48:56]  # (1041, 150, 116)



    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1041):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    data_label =  data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_label.npz')
    label = data['label_list_LR']
    label = label[:1041]  #(1041, 150)

    # number of clusters
    args.n_clusters = len(np.unique(label))


    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(sample, label, random_state=42, test_size=0.2)


    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)

    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    train_dl = DataLoader(real_train_ds, batch_size=16)
    real_train_dl = DataLoader(real_train_ds, batch_size=1)

    real_test_ds = TensorDataset(real_test_data, real_test_label)
    test_dl = DataLoader(real_test_ds, batch_size=16)
    real_test_dl = DataLoader(real_test_ds, batch_size=1)


    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))



    if not os.path.exists(os.path.join(directory, 'clusteringtest')):
        os.makedirs(os.path.join(directory, 'clusteringtest'))

    path = os.path.join(directory, args.ae_weights)
    full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
    full_path = full_path[299]
    print("I got: " + full_path + " weights")


    seq2seq = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
    checkpoint = torch.load(full_path, map_location=args.device)
    seq2seq.load_state_dict(checkpoint['model_state_dict'])
    seq2seq = seq2seq.to(args.device)


    real_train_data = real_train_data.type(torch.FloatTensor).to(args.device)
    features, encoder_state_fw = seq2seq.f_enc(real_train_data)
    features = features.detach().cpu().numpy()  #(832, 150, 10)

    labeled_0 = features[0][real_train_label[0] == 1]  # shape
    labeled_1 = features[0][real_train_label[0] == 2]  # face
    plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
    plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
    plt.legend()
    plt.show()
    plt.close()

    gmm = GaussianMixture(n_components=2)
    tm = TimeSeriesKMeans(n_clusters=2, verbose=False)





'''
    # 0 subject 
    kmc = KMeans(n_clusters=2, init="k-means++").fit(features[0])
    pmc = kmc.predict(features[0])
    centroid = kmc.cluster_centers_
    predicted_0 = features[0][pmc == 0] #shape
    predicted_1 = features[0][pmc == 1] #face
    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub0_predicted.png')
    plt.close()

    labeled_0 = features[0][real_train_label[0] == 1] #shape
    labeled_1 = features[0][real_train_label[0] == 2] #face
    plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
    plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub0_label.png')
    plt.close()

    # 100 subject
    kmc = KMeans(n_clusters=2, init="k-means++").fit(features[100])
    pmc = kmc.predict(features[100])
    centroid = kmc.cluster_centers_
    predicted_0 = features[100][pmc == 0]  # shape
    predicted_1 = features[100][pmc == 1]  # face
    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub100_predicted.png')
    plt.close()

    labeled_0 = features[100][real_train_label[100] == 1]  # shape
    labeled_1 = features[100][real_train_label[100] == 2]  # face
    plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
    plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub100_label.png')
    plt.close()




    # 200 subject
    kmc = KMeans(n_clusters=2, init="k-means++").fit(features[200])
    pmc = kmc.predict(features[200])
    centroid = kmc.cluster_centers_
    predicted_0 = features[200][pmc == 0]  # shape
    predicted_1 = features[200][pmc == 1]  # face
    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub200_predicted.png')
    plt.close()

    labeled_0 = features[200][real_train_label[200] == 1]  # shape
    labeled_1 = features[200][real_train_label[200] == 2]  # face
    plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
    plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(directory, 'visualization/') + 'sub200_label.png')
    plt.close()
'''













''' multivariate timeseries kmeans clustering method'''
# def initalize_centroids(X):
#     """
#     Function for the initialization of centroids.
#     """
#
#     tae = model.tae
#     tae = tae.to(args.device)
#     X_tensor = X.type(torch.FloatTensor).to(args.device)
#
#     X_tensor =  X_tensor.detach()
#     z, x_reconstr = tae(X_tensor)
#     print('initialize centroid')
#
#     features = z.detach().cpu()  # z, features: (864, 284, 32)
#
#     km = TimeSeriesKMeans(n_clusters=args.n_clusters, verbose=False, random_state=42)
#     assignements = km.fit_predict(features)
#     # assignements = AgglomerativeClustering(n_clusters= args.n_clusters, linkage="complete", affinity="euclidean").fit(features)
#     # km.inertia_
#     # assignements (864,)
#     # km.cluster_centers_   (8, 284, 32)
#
#     centroids_ = torch.zeros(
#         (args.n_clusters, z.shape[1], z.shape[2]), device=args.device
#     )  # centroids_ : torch.Size([8, 284, 32])
#
#     for cluster_ in range(args.n_clusters):
#         centroids_[cluster_] = features[assignements == cluster_].mean(axis=0)
#     # centroids_ : torch.Size([8, 284, 32])
#
#     cluster_centers = centroids_
#
#     return cluster_centers