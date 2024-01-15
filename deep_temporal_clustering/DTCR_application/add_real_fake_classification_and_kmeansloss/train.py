import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from main_config import get_arguments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import load_data
import utils
from KM_model import Seq2Seq
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE


from tslearn.clustering import TimeSeriesKMeans


def writelog(file, line):
    file.write(line + '\n')
    print(line)



def pretrain_autoencoder(args, real_fake_train_data, real_fake_train_label, kmeans_train, kmeans_test, directory='.'):
    """
    function for the autoencoder pretraining
    """


    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(os.path.join(directory, 'models_logs')):
        os.makedirs(os.path.join(directory, 'models_logs'))

    if not os.path.exists(os.path.join(directory, 'models_weights')):
        os.makedirs(os.path.join(directory, 'models_weights'))

    if not os.path.exists(os.path.join(directory, 'visualization')):
        os.makedirs(os.path.join(directory, 'visualization/EPOCH'))

    # Text Logging
    f = open(os.path.join(directory, 'setting.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'GPU ID: %s' % (args.gpu_id))
    writelog(f, 'Dataset: %s' % (args.dataset_name))
    writelog(f, '----------------------')
    writelog(f, 'Model Name: %s' % args.model_name)
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


    train_ds = TensorDataset(real_fake_train_data, real_fake_train_label)  ###modify
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)



    ## define TAE architecture
    # tae = TAE(args)
    # tae = tae.to(args.device)
    # print(tae)
    seq2seq = Seq2Seq(args, n_input=116, filter_1=64, n_hidden=[64, 32, 20], classification_unit=[64, 2])
    seq2seq = seq2seq.to(args.device)
    print(seq2seq)

    ## MSE loss
    loss_ae = nn.MSELoss()
    loss_cls = nn.CrossEntropyLoss()
    ## Optimizer
    # optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    lamda = 0.1
    best_indicator = 0
    best_epoch = -1
    nmi_best_indicator = 0
    nmi_best_epoch = -1
    for epoch in tqdm(range(args.epochs_ae)):


        # training
        seq2seq.train()
        all_loss = 0
        recon_loss = 0
        classification_loss = 0
        km_loss = 0

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)  #torch.Size([16batch, 150, 116])
            _ = _.type(torch.FloatTensor).to(args.device)

            real_input = inputs[_[:,0] == 1 ]
            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            h_real, real_reconstr, classifier_predict, kmeans_loss = seq2seq(inputs, _)
            # real_reconstr = reconstr[_[:,0] == 1]
            loss_mse = loss_ae(real_input, real_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
            loss_classification = loss_cls(classifier_predict, _)

            total_loss = loss_mse + loss_classification + (0.7 * kmeans_loss)
            total_loss.backward()  # 기울기 구함
            optimizer.step()  # 최적화 진행

            all_loss += total_loss.item()
            recon_loss += loss_mse.item()
            classification_loss += loss_classification.item()
            km_loss += kmeans_loss.item()

        if epoch % 5 == 0 and epoch != 0:
            U, S, Vh = torch.linalg.svd(h_real.T)
            sorted_indices = torch.argsort(S)
            topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:2], :]
            F_new = topk_evecs.T
            seq2seq.kmeans.F.data = F_new

        train_loss = all_loss / (batch_idx + 1)
        recon_loss = recon_loss / (batch_idx + 1)
        classification_loss = classification_loss / (batch_idx + 1)
        km_loss = km_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch+1)
        writer.add_scalar("reconstruction loss", recon_loss, epoch + 1)
        writer.add_scalar("classification loss", classification_loss, epoch + 1)
        writer.add_scalar("kmeans loss", km_loss, epoch + 1)
        print()
        print("total loss for epoch {} is : {}, recon loss: {}, classification loss: {}, kmeans loss: {}".format(epoch + 1, train_loss, recon_loss, classification_loss, km_loss))
        torch.save({
            'model_state_dict': seq2seq.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'loss': train_loss
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1, train_loss))


        # kmeans
        # if (epoch+1) % 5 == 0 or epoch == 0:
        seq2seq.eval()
        with torch.no_grad():
            total_RI = 0
            test_total_RI = 0
            total_NMI = 0
            test_total_NMI = 0

            # kmeans with train data
            for t_idx, (train, train_label) in enumerate(kmeans_train):
                train = train.type(torch.FloatTensor).to(args.device)  # torch.Size([16batch, 150, 116])
                features, encoder_state_fw = seq2seq.f_enc(train)
                features = features.detach().cpu().numpy().squeeze()

                pred_train = utils.cluster_using_kmeans(features, args.n_clusters)
                RI_score_train = utils.ri_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
                NMI_score_train = utils.nmi_score(train_label.detach().cpu().numpy().squeeze(), pred_train)

                total_RI += RI_score_train
                total_NMI += NMI_score_train

                if epoch == 0 or (epoch+1) == 30 or (epoch+1) == 70 or (epoch+1) == 100 or (epoch+1) == 150 or (epoch+1) == 200 or (epoch+1) == 250 or (epoch+1) == 300 or (epoch+1) == 350 or (epoch+1) == 400 or (epoch+1) == 450 or (epoch+1) == 500:
                    if (t_idx+1) == 100:
                        tsne = TSNE(n_components=2)
                        features_t = tsne.fit_transform(features)

                        predicted_0 = features_t[pred_train == 0]  # shape
                        predicted_1 = features_t[pred_train == 1]  # face
                        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
                        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_predicted_epoch_{}.png'.format(epoch+1))
                        plt.close()

                        labeled_0 = features_t[train_label.squeeze() == 1]  # shape
                        labeled_1 = features_t[train_label.squeeze() == 2]  # face
                        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
                        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_labeld_epoch_{}.png'.format(epoch+1))
                        plt.close()


            total_RI = total_RI / (t_idx + 1)
            total_NMI = total_NMI / (t_idx + 1)


            # kmeans with test data
            for tt_idx, (test, test_label) in enumerate(kmeans_test):
                test = test.type(torch.FloatTensor).to(args.device)
                tt_features, tt_encoder_state_fw = seq2seq.f_enc(test)
                tt_features = tt_features.detach().cpu().numpy().squeeze()

                pred_test = utils.cluster_using_kmeans(tt_features, args.n_clusters)
                RI_score_test = utils.ri_score(test_label.detach().cpu().numpy().squeeze(), pred_test)
                NMI_score_test = utils.nmi_score(test_label.detach().cpu().numpy().squeeze(), pred_test)

                test_total_RI += RI_score_test
                test_total_NMI += NMI_score_test
            test_total_RI = test_total_RI / (tt_idx + 1)
            test_total_NMI = test_total_NMI / (tt_idx + 1)


            if test_total_RI > best_indicator:
                best_indicator = test_total_RI
                best_epoch = epoch+1

            if test_total_NMI > nmi_best_indicator:
                nmi_best_indicator = test_total_NMI
                nmi_best_epoch = epoch+1
            print('Epoch {} - {} train:{}\ttest:{}, {} train:{}\ttest:{}'.format(epoch + 1, 'RI', total_RI, test_total_RI, 'NMI', total_NMI, test_total_NMI))


    f = open(os.path.join(directory, 'Rand_Index.log'), 'a')
    writelog(f, 'number of clusters: %d' % args.n_clusters)
    writelog(f, 'best\tRI = {}, epoch = {}\n\n'.format(best_indicator, best_epoch))
    writer.close()

    f = open(os.path.join(directory, 'NMI.log'), 'a')
    writelog(f, 'number of clusters: %d' % args.n_clusters)
    writelog(f, 'best\tNMI = {}, epoch = {}\n\n'.format(nmi_best_indicator, nmi_best_epoch))
    writer.close()
    print("Ending pretraining autoencoder. \n")




















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
    samples = samples[:, :, :]  # (1041, 150, 116)



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
    # construct classification dataset real/fake data and labels
    real_dataset, fake_dataset, true_label, fake_label = load_data.construct_classification_dataset(real_train_data) ################################################
    # make 2d one hot vector for classification task label
    true_cls_label_ = np.zeros(shape=(true_label.shape[0], args.n_clusters))
    true_cls_label_[np.arange(true_cls_label_.shape[0]), true_label] = 1 ##########################################
    fake_cls_label_ = np.zeros(shape=(fake_label.shape[0], args.n_clusters))
    fake_cls_label_[np.arange(fake_cls_label_.shape[0]), fake_label] = 1 ##########################################



    #rf_train_data(1664, 150, 116)
    #rf_train_label(1664,) == (1, 1, 1, ..., 0, 0, 0)
    ''' modify
    temp = list(zip(rf_train_data, rf_train_label))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    cls_train_data, cls_train_label = np.array(res1), np.array(res2)
    #cls_train_data(1664, 150, 116)
    #cls_train_label(1664,) == shuffled(1, 0, 0, ..., 0, 1, 0)
    '''

    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)

    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    real_train_dl = DataLoader(real_train_ds, batch_size=1)

    real_test_ds = TensorDataset(real_test_data, real_test_label)
    real_test_dl = DataLoader(real_test_ds, batch_size=1)







    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))






    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain
        rf_train_data=[]
        rf_train_label=[]
        length = int(real_dataset.shape[0] / (args.batch_size/2))
        for i in range(length):
            rf_train_data.append(real_dataset[i*8:(i*8)+8])
            rf_train_data.append(fake_dataset[i*8:(i*8)+8])

            rf_train_label.append(true_cls_label_[i*8:(i*8)+8])
            rf_train_label.append(fake_cls_label_[i*8:(i*8)+8])


        rf_train_data = np.array(rf_train_data).reshape(-1, 150, 116)
        rf_train_label = np.array(rf_train_label).reshape(-1, 2)

        rf_train_data, rf_train_label = torch.FloatTensor(rf_train_data), torch.FloatTensor(rf_train_label)

        pretrain_autoencoder(args, real_fake_train_data=rf_train_data, real_fake_train_label=rf_train_label, kmeans_train=real_train_dl, kmeans_test=real_test_dl, directory=directory)




    if args.ae_weights is not None and args.epochs_ae > 0:

        if not os.path.exists(os.path.join(directory, 'visualization')):
            os.makedirs(os.path.join(directory, 'visualization'))

        path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[109]
        print("I got: " + full_path + " weights")


        seq2seq = Seq2Seq(args, n_input=8, n_hidden=[8, 6, 2], classification_unit=[64, 2])
        checkpoint = torch.load(full_path, map_location=args.device)
        seq2seq.load_state_dict(checkpoint['model_state_dict'])
        seq2seq = seq2seq.to(args.device)


        real_train_data = real_train_data.type(torch.FloatTensor).to(args.device)
        features, encoder_state_fw = seq2seq.f_enc(real_train_data)
        features = features.detach().cpu().numpy()


        ''' 0 subject '''
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

        ''' 100 subject '''
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




        ''' 200 subject '''
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