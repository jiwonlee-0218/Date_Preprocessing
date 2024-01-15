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
import utils
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
import argparse
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score


def writelog(file, line):
    file.write(line + '\n')
    print(line)

def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="EMOTION_6cluster_ddddddddddd", help="dataset name")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    # model args
    parser.add_argument("--model_name", default="DTCR_bidirection_dilated_RNN2",help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")

    # parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping: Maximal parameter gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=300, help="Epochs number of the autoencoder training",)
    parser.add_argument("--max_patience", type=int, default=15, help="The maximum patience for pre-training, above which we stop training.",)

    parser.add_argument("--lr_ae", type=float, default=0.01, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/autoencoder_kmeans',)
    # parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')
    # parser.add_argument("--ae_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
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
        # self.tae_encoder = TAE_encoder(filter_1=n_input, filter_lstm=n_hidden)


    def forward(self, inputs):
        ## encoder
        outputs_fw, states_fw = self.drnn(inputs)
        states_fw = torch.cat(states_fw, 1)

        return outputs_fw, states_fw



class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_lstm):
        super(TAE_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.hidden_lstm_3 = filter_lstm[2]


        ## CNN PART
        ### output shape (batch_size, 7 , n_hidden = 284)
        # self.conv_layer_1 = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=116,
        #         out_channels=filter_1,
        #         kernel_size=1,
        #         stride=1
        #     ),
        #     nn.LeakyReLU(),
        # )


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

        self.lstm_3 = nn.LSTM(
            input_size=self.hidden_lstm_2,
            hidden_size=self.hidden_lstm_3,
            batch_first=True

        )
        self.act_lstm_3 = nn.Tanh()


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        # x = x.transpose(1,2)
        # x = self.conv_layer_1(x)
        # out_cnn = x.permute(0, 2, 1)

        out_lstm1, _ = self.lstm_1(x)
        out_lstm1_act = self.act_lstm_1(out_lstm1)


        features, _ = self.lstm_2(out_lstm1_act) # (1, 6, 64)
        out_lstm2_act = self.act_lstm_2(features)

        features_2, _ = self.lstm_3(out_lstm2_act)  # (1, 6, 64)
        out_lstm3_act = self.act_lstm_3(features_2)



        return out_lstm3_act, _



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

    def forward(self, inputs):


        f_outputs, f_final_hidden_state = self.f_enc(inputs)  #torch.Size([16, 150, 50]), (16, 200)

        true_input_recons = self.f_dec(f_outputs)


        return f_outputs, true_input_recons


















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
    data_label =  data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label = data['label_list_LR']
    label = label[:1041]  #(1041, 150)

    # number of clusters
    args.n_clusters = len(np.unique(label))

    X_train, X_test, y_train, y_test = train_test_split(samples, label, random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)


    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(sample, label, random_state=42, test_size=0.2)
    real_test_data, real_valid_data, real_test_label, real_valid_label = train_test_split(real_test_data, real_test_label, random_state=42, test_size=0.5)

    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_valid_data, real_valid_label = torch.FloatTensor(real_valid_data), torch.FloatTensor(real_valid_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)

    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    train_dl = DataLoader(real_train_ds, batch_size=16)

    real_valid_ds = TensorDataset(real_valid_data, real_valid_label)
    valid_dl = DataLoader(real_valid_ds, batch_size=16)



    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))

    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(os.path.join(directory, 'models_logs')):
            os.makedirs(os.path.join(directory, 'models_logs'))

        if not os.path.exists(os.path.join(directory, 'models_weights')):
            os.makedirs(os.path.join(directory, 'models_weights'))

        # Text Logging
        f = open(os.path.join(directory, 'setting.log'), 'a')
        writelog(f, '======================')
        writelog(f, 'GPU ID: %s' % (args.gpu_id))
        writelog(f, 'Dataset: %s' % (args.dataset_name))
        writelog(f, '----------------------')
        writelog(f, 'Model Name: %s' % args.model_name)
        writelog(f, 'number of clusters: %d' % args.n_clusters)
        writelog(f, '----------------------')
        writelog(f, 'Epoch: %d' % args.epochs_ae)
        writelog(f, 'Max Patience: %d (10 percent of the epoch size)' % args.max_patience)
        writelog(f, 'Batch Size: %d' % args.batch_size)
        writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
        writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
        writelog(f, '======================')
        f.close()



        print("Pretraining autoencoder... \n")
        writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))

        ## define TAE architecture
        seq2seq = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
        seq2seq = seq2seq.to(args.device)
        print(seq2seq)

        ## MSE loss
        loss_ae = nn.MSELoss()

        ## Optimizer
        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        for epoch in tqdm(range(args.epochs_ae)):

            # training
            seq2seq.train()
            total_loss = 0


            for batch_idx, (inputs, _) in enumerate(train_dl):
                inputs = inputs.type(torch.FloatTensor).to(args.device)  # torch.Size([16batch, 150, 116])


                optimizer.zero_grad()  # 기울기에 대한 정보 초기화
                representation, recon = seq2seq(inputs)
                loss_mse = loss_ae(inputs, recon)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차

                loss_mse.backward()  # 기울기 구함
                optimizer.step()  # 최적화 진행

                total_loss += loss_mse.item()

            train_loss = total_loss / (batch_idx + 1)

            writer.add_scalar("training loss", train_loss, epoch + 1)
            print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

            # validation
            seq2seq.eval()
            with torch.no_grad():
                all_val_loss = 0
                for j, (val_x, val_y) in enumerate(valid_dl):
                    val_x = val_x.type(torch.FloatTensor).to(args.device)
                    v_features, val_reconstr = seq2seq(val_x)
                    val_loss = loss_ae(val_x, val_reconstr)

                    all_val_loss += val_loss.item()

                validation_loss = all_val_loss / (j + 1)

                writer.add_scalar("validation loss", validation_loss, epoch + 1)
                print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

            if epoch == 0:
                min_val_loss = validation_loss

            if validation_loss < min_val_loss:
                torch.save({
                    'model_state_dict': seq2seq.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': validation_loss
                }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                            validation_loss))
                min_val_loss = validation_loss
                print("save weights !!")

        writer.close()
        print("Ending pretraining autoencoder. \n")


    if args.ae_weights is not None and args.epochs_ae > 0:

        path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")

        seq2seq = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
        checkpoint = torch.load(full_path, map_location=args.device)
        seq2seq.load_state_dict(checkpoint['model_state_dict'])
        seq2seq = seq2seq.to(args.device)
        print(seq2seq)



        real_train_data = real_train_data.type(torch.FloatTensor).to(args.device)
        features, recon = seq2seq.f_enc(real_train_data)
        features = features.detach().cpu().numpy()  #(832, 150, 10)
        label = real_train_label.detach().cpu().numpy()

        total_NMI = 0

        f = open(os.path.join(directory, 'KMeans\'s train_NMI_embedding.log'), 'a')

        for i in range(features.shape[0]):
            kmc = KMeans(n_clusters=6).fit(features[i])
            pmc = kmc.predict(features[i])
            # print(assignments.labels_)
            NMI = normalized_mutual_info_score(pmc, label[i])
            writelog(f, 'NMI is: %.4f' % (np.round(NMI, 3)))

            total_NMI += np.round(NMI, 3)

        avg = (total_NMI / features.shape[0])
        writelog(f, '======================')
        writelog(f, 'subject: %d' % (features.shape[0]))
        writelog(f, 'total_NMI is: %.4f' % (total_NMI))
        writelog(f, 'avg_NMI is: %.4f' % (avg))
        f.close()
        print(avg)


        ''' test '''
        real_test_data = real_test_data.type(torch.FloatTensor).to(args.device)
        t_features, t_recon = seq2seq.f_enc(real_test_data)
        t_features = t_features.detach().cpu().numpy()  # (832, 150, 10)
        t_label = real_test_label.detach().cpu().numpy()
        total_test_NMI = 0

        f = open(os.path.join(directory, 'KMeans\'s test_NMI_embedding.log'), 'a')

        for i in range(t_features.shape[0]):
            kmc = KMeans(n_clusters=6).fit(t_features[i])
            pmc = kmc.predict(t_features[i])
            NMI = normalized_mutual_info_score(pmc, t_label[i])
            writelog(f, 'NMI is: %.4f' % (np.round(NMI, 3)))

            total_test_NMI += np.round(NMI, 3)

        avg = (total_test_NMI / t_features.shape[0])
        writelog(f, '======================')
        writelog(f, 'subject: %d' % (t_features.shape[0]))
        writelog(f, 'total_NMI is: %.4f' % (total_test_NMI))
        writelog(f, 'avg_NMI is: %.4f' % (avg))
        f.close()
        print(avg)


        tsne = TSNE(n_components=2)
        features_t = tsne.fit_transform(features[100])
        labeled_0 = features_t[real_train_label[100] == 1]  # shape
        labeled_1 = features_t[real_train_label[100] == 2]  # face
        labeled_2 = features_t[real_train_label[100] == 3]  # shape
        labeled_3 = features_t[real_train_label[100] == 4]  # face
        labeled_4 = features_t[real_train_label[100] == 5]  # shape
        labeled_5 = features_t[real_train_label[100] == 6]  # face
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape1', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face1', color='orange', alpha=0.5)
        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape2', color='yellow', alpha=0.5)
        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face2', color='green', alpha=0.5)
        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape3', color='blue', alpha=0.5)
        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face3', color='purple', alpha=0.5)
        plt.legend()
        plt.savefig( '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/autoencoder_kmeans/DTCR_bidirection_dilated_RNN2/EMOTION_6cluster_ddddddddddd/Epochs300_BS_16_LR_0.01_wdcay_1e-06/visualization/'+ 'sub100_embeddding.png')






        # labeled_0 = features[0][real_train_label[0] == 1]  # shape
        # labeled_1 = features[0][real_train_label[0] == 2]  # face
        # plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
        # plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
        # plt.legend()
        # plt.show()
        # plt.close()
        #
        # gmm = GaussianMixture(n_components=2)
        # tm = TimeSeriesKMeans(n_clusters=2, verbose=False)





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