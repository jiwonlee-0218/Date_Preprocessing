import argparse
import copy
import os
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from encoder_decoder import *
from load_data import *
import encoder_decoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score
from sklearn.decomposition import TruncatedSVD

class RNN_clustering_model(nn.Module):

    def __init__(self, args, fw_n_hidden, bw_n_hidden):
        super(RNN_clustering_model, self).__init__()

        device = args.device
        input_dim = args.input_dim
        decoder_input = fw_n_hidden[-1]
        self.models = encoder_decoder.Encoder_Decoder(n_input=input_dim, fw_n_hidden= fw_n_hidden, bw_n_hidden=bw_n_hidden, device=device)
        self.classifier = encoder_decoder.classifier(in_features=np.sum(fw_n_hidden))
        self.kmeans_optimalize = encoder_decoder.kmeans()

    def forward(self, inputs):


        recons_input, fcn_latent, h = self.models(inputs)
        output_without_softmax = self.classifier(h)

        h_fake, h_real = torch.chunk(h, 2, dim=0)
        loss_kmeans = self.kmeans_optimalize(h_real)

        return recons_input, fcn_latent, loss_kmeans, h_real, output_without_softmax, h
        # h == encoder output, fcn_latent == decoder hidden state

def shuffle_timeseries(data, rate=0.2):

    ordered_index = np.arange(len(data))
    ordered_index.astype(int)

    shuffled_index = np.random.choice(ordered_index, size=int(np.floor(rate * len(data))), replace=False)
    ordered_index[shuffled_index] = -1

    shuffled_index = np.random.permutation(shuffled_index)
    ordered_index[ordered_index == -1] = shuffled_index
    data = data[ordered_index]

    return data


def construct_classification_dataset(dataset):
    real_dataset = copy.deepcopy(dataset)
    fake_dataset = []
    for seq in real_dataset:
        fake_dataset.append(shuffle_timeseries(seq))

    fake_dataset = np.array(fake_dataset)

    label = np.array([1] * fake_dataset.shape[0] + [0] * real_dataset.shape[0])
    dataset = np.concatenate([fake_dataset, real_dataset], axis=0)  # (112, 286, 1)

    print('dataset shape: ', dataset.shape)
    print('label shape:', label.shape)

    return dataset, label



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="UCR_UEA_Kmeans", help="model name")
    parser.add_argument('--dataset', default='SyntheticControl', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")
    parser.add_argument("--visualization", default=True, help='visualization True or None')
    parser.add_argument('--input_dim', default=1, type=int, help="input dimension")

    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/task_clustering/')
    parser.add_argument("--dataset_name", default="UCR_SyntheticControl", help="dataset name")

    args = parser.parse_args()

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)



    ################ Preparing Dataset
    # ''' EMOTION  '''
    # emotion_data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    # emotion_samples = emotion_data['tfMRI_EMOTION_LR']  #(1041, 150, 115)
    # emotion_sample = emotion_samples.reshape(-1, 25, 116) #(6246, 25, 116)
    #
    # e_label = [0, 1, 0, 1, 0, 1] # shape=0, face=1
    # e_label = np.array(e_label)
    # e_label = np.repeat(e_label, 1041)  # (1041, 6)
    #
    # # (6246, 25, 116)=(S, T, D)  #(6246,)=(0 or 1)
    # X_train, X_test, y_train, y_test = train_test_split(emotion_sample, e_label, random_state=42, shuffle=True, test_size=0.2)
    #
    # # number of clusters
    # args.n_clusters = len(np.unique(y_train))
    # args.num_steps = X_train.shape[1]
    # args.embedding_size = X_train.shape[2]
    # print('Label:', np.unique(y_train))


    ################ Preparing UCR Dataset
    train_data, train_label = load_data('Coffee')       #(56, 286, 1)=(S, T, D)  #(56,)=(0 or 1)


    args.n_clusters = len(np.unique(train_label))       #2
    args.num_steps = train_data.shape[1]
    args.embedding_size = train_data.shape[2]


    train_real_fake_data, train_real_fake_label = construct_classification_dataset(train_data)


    cls_data = np.expand_dims(train_real_fake_data, axis=2)
    cls_label_ = np.zeros(shape=(train_real_fake_label.shape[0], len(np.unique(train_real_fake_label))))
    cls_label_[np.arange(cls_label_.shape[0]), train_real_fake_label] = 1

    train_real_fake_data, train_real_fake_labels = torch.FloatTensor(train_real_fake_data), torch.FloatTensor(cls_label_)
    train_real_fake_data = train_real_fake_data.to(args.device)
    train_real_fake_labels = train_real_fake_labels.to(args.device)

    train_ds = TensorDataset(train_real_fake_data, train_real_fake_labels)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)  ###






    # train_data, train_label = torch.FloatTensor(train_data), torch.FloatTensor(train_label)
    # train_ds = TensorDataset(train_data, train_label)
    # train_dl = DataLoader(train_ds, batch_size=4)
    #
    # model = RNN_clustering_model(args).to(device)
    #
    # train_data = train_data.to(args.device)
    # recons_input, fcn_latent = model(train_data)


    ## define Model
    model = RNN_clustering_model(args, fw_n_hidden=[100, 50, 50], bw_n_hidden=[50, 30, 30]).to(device)



    ## Loss
    loss_ae = nn.MSELoss()
    loss_classification = nn.CrossEntropyLoss()
    ## Optimizer
    optimizer_clu = torch.optim.Adam(
        [{"params": model.models.parameters()},{"params": model.classifier.parameters()}],
        lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-6)


    for epoch in range(300):

        # training
        model.train()
        all_loss = 0
        mse = 0
        discrim = 0
        kmea = 0

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)



            optimizer_clu.zero_grad()  # 기울기에 대한 정보 초기화
            recons_input, fcn_latent, loss_kmeans, h_real, output_without_softmax, h = model(inputs)
            loss_mse = loss_ae(inputs, recons_input)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
            loss_discrimination = loss_classification(output_without_softmax, _)
            total = loss_mse + loss_discrimination + loss_kmeans
            total.backward()  # 기울기 구함
            optimizer_clu.step()  # 최적화 진행



            if batch_idx % 10 == 0 and batch_idx != 0:
                U, S, Vh = torch.linalg.svd(h_real.T)
                sorted_indices = torch.argsort(S)
                # topk_evecs = Vh[sorted_indices[:-2 - 1:-1], :]
                topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:2], :]
                F_new = topk_evecs.T
                model.kmeans_optimalize.F.data = F_new

            '''
            # self.F == (batch_idx=0) no update || self.F =! (batch_idx=0) update

            if batch_idx%10 == 0 : # 0 10 20 30 40 50
                if batch_idx != 0: # 10 20 30 40 50
                    U, S, Vh = torch.linalg.svd(h_real)
                    sorted_indices = torch.argsort(S)
                    topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:2], :]
                    F_new = topk_evecs.T
                    encoder_decoder.kmeans().F = F_new
                    previous_F = encoder_decoder.kmeans().F

                previous_F = encoder_decoder.kmeans().F
            else:
                encoder_decoder.kmeans().F = previous_F
            '''


            all_loss += total.item()


        train_loss = all_loss / (batch_idx + 1)
        print("epoch: ",epoch,' ',train_loss)


    # temp = list(zip(train_data, train_label))
    # random.shuffle(temp)
    # aa, bb = zip(*temp)
    # aa, bb = np.array(aa), np.array(bb)
    train_dl = DataLoader(train_data, batch_size=8, shuffle=False)
    model.eval()
    latent_collector = []
    with torch.no_grad():
        for idx, inputs in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            f_outputs, f_final_hidden_state = model.models.f_enc(inputs)
            latent_collector.append(f_final_hidden_state)

    latent_collector = torch.cat(latent_collector).detach().cpu().numpy()
    km = KMeans(2)
    result = km.fit_predict(latent_collector)
    print(result)
    print(train_label)
    print(rand_score(train_label, result))
    print(normalized_mutual_info_score(train_label, result))