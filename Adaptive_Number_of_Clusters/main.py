import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# from model import *
from torch.utils.data import TensorDataset, DataLoader
from nilearn.connectome import ConnectivityMeasure
from torch.nn.utils.rnn import pad_sequence
# from dFC_model import *
from test_model import *
from torch.utils.tensorboard import SummaryWriter
from nilearn.connectome import ConnectivityMeasure
import torch.nn.functional as F
from tqdm import tqdm
import random
import glob
import copy
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.multiprocessing as mp



def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="7_task_masking/masking_R_T", help="dataset name")

    # model args
    parser.add_argument("--model_name", default="GMPool/input_diversity",help="model name")
    # parser.add_argument("--model_name", default="GMPool/input_R_T", help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")



    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=151, help="Epochs number of the autoencoder training",)
    parser.add_argument("--lr_ae", type=float, default=1e-6, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/DataCommon2/jwlee/Adaptive_Number_of_Clusters',)
    parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    # parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')

    return parser


def writelog(file, line):
    file.write(line + '\n')
    print(line)

# def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
#     '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
#     measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
#     mt = measure.fit_transform(input)
#
#     if fisher_t == True:
#         for i in range(len(mt)):
#             mt[i][:] = np.arctanh(mt[i][:])

#     return mt
#
#
# def flatten_upper_triang_values(X):
#     upper_tri = torch.triu(X, diagonal=1) ##torch.Size([8, 116, 116])
#
#     values_list = []
#     for i in range(X.shape[0]):
#         arr = upper_tri[i]
#         values = arr[arr != 0]
#         values_list.append(values)
#         arr = 0
#
#     values_arr = torch.stack(values_list)
#     return values_arr
#
# def make_functional_connectivity(inputs, Group, group_idx):
#
#    group_idx_list = group_idx.tolist()
#    number = Group.tolist()
#    fc_list = []
#
#    tmp = number[0]
#    change = 0
#    index = 0
#
#    for i in range(len(number)):사
#        change += 1
#
#        if tmp != number[i]:
#            x = inputs[index:change - 1, :]
#            sub = np.reshape(x.detach().cpu().numpy(), (1,) + x.shape)
#            fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
#            fc_list.append(fc[0])
#            tmp = number[i]
#            index = i
#
#        if i == len(number) - 1:
#            x = inputs[index:change, :]  ## == aa[index:, :]
#            sub = np.reshape(x.detach().cpu().numpy(), (1,) + x.shape)
#            fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
#            fc_list.append(fc[0])
#
#    return np.array(fc_list)
#
#
# def padding_sequences(sequences, max_len):
#     max_len = max_len #50
#
#     # out = F.pad(a,(0,0,0,max_len-a.shape[0]),'constant',10)  (3, 6) --> (10, 6)
#     sequences[0] = F.pad(sequences[0], (0, 0, 0, max_len - sequences[0].shape[0]), 'constant', 0)
#
#     seqs = pad_sequence(sequences, batch_first=True)
#
#     return seqs


def run(args, train_dl, test_dl, directory='.'):
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
    writelog(f, 'number of labels: %d' % (args.n_labels))
    writelog(f, '----------------------')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()

    print("Training ... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))


    batch = args.batch_size
    time = args.timeseries
    roi = args.roi


    model = GMNetwork(args, batch, time, roi)
    model.to(args.device)



    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_ae)
    # optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)




    for epoch in tqdm(range(args.epochs_ae)):
        total_loss = 0
        total_acc = 0

        model.train()

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            _ = _.type(torch.Tensor).to(args.device).to(torch.int64)

            optimizer.zero_grad()

            out, M, indices = model(inputs)
            loss = loss_function(out, _)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_loss += loss
            a, preds = torch.max(out, dim=1)
            total_acc += (torch.sum(preds==_).item() / args.batch_size) ## 1-batch(8)당 accuracy



        total_loss = total_loss / (batch_idx + 1)
        total_acc = total_acc / (batch_idx + 1)

        writer.add_scalar("training loss", total_loss, epoch)
        writer.add_scalar("training acc", total_acc, epoch)
        print()
        print('loss for epoch {} is : {}'.format(epoch, total_loss))
        print('acc for epoch {} is : {}'.format(epoch, total_acc))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': total_loss,
            # 'scheduler': scheduler
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, total_loss))


        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)

        with torch.no_grad():
            for t_idx, (test, test_label) in enumerate(test_dl):
                test = test.type(torch.FloatTensor).to(args.device)
                test_label = test_label.cpu().numpy()



                out, M, indices = model(test)

                if epoch == 0 or epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40 or epoch == 50 or epoch == 60 or epoch == 70 or epoch == 80 or epoch == 90 or epoch==100\
                        or epoch == 110 or epoch == 120 or epoch == 130 or epoch == 140 or epoch == 150:
                    if (t_idx+1) == 1:
                        # label_list = np.asarray(test_label, dtype=int).tolist()

                        f = open(os.path.join(directory, 'visualization/grouping_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))

                        for i in range(len(test_label)):
                            a = int(test_label[i])
                            writelog(f, 'sub: %d, label: %d' % (i, a))
                            writelog(f, 'Grouping results: %s' % (indices[i].cpu().numpy().astype(int).tolist()))
                            writelog(f, '---------------------------------------------')
                        writelog(f, '========================================================================')
                        writelog(f, '                                            ')
                        f.close()



                        for i in range(len(test_label)):

                            a = int(test_label[i])
                            plt.imshow(M[i].cpu().numpy())
                            plt.colorbar()
                            plt.title('grouping matrix M')
                            plt.savefig(os.path.join(directory, 'visualization/EPOCH/') +
                                        'sub{}label{}_grouping_matrix_M_epoch_{}.png'.format(
                                    i, a, epoch))
                            plt.close()


                        # zero_label_index1 = np.where(test_label == 0)[0][0]
                        # zero_label_index2 = np.where(test_label == 0)[0][1]
                        # one_label_index1 = np.where(test_label == 1)[0][0]
                        # one_label_index2 = np.where(test_label == 1)[0][1]
                        #
                        #
                        # print(test_label)
                        # f = open(os.path.join(directory, 'visualization/grouping_result.log'), 'a')
                        # writelog(f, 'Epochs: %d' % (epoch))
                        # writelog(f, 'sub: %d, label: %d' % (zero_label_index1, np.int(test_label[zero_label_index1])))
                        # writelog(f, 'Grouping results: %s' % (indices[zero_label_index1].cpu().numpy().astype(int).tolist()))
                        # writelog(f, 'sub: %d, label: %d' % (zero_label_index2, np.int(test_label[zero_label_index2])))
                        # writelog(f, 'Grouping results: %s' % (indices[zero_label_index2].cpu().numpy().astype(int).tolist()))
                        # writelog(f, '---------------------------------------------')
                        # writelog(f, 'sub: %d, label: %d' % (one_label_index1, np.int(test_label[one_label_index1])))
                        # writelog(f, 'Grouping results: %s' % (indices[one_label_index1].cpu().numpy().astype(int).tolist()))
                        # writelog(f, 'sub: %d, label: %d' % (one_label_index2, np.int(test_label[one_label_index2])))
                        # writelog(f, 'Grouping results: %s' % (indices[one_label_index2].cpu().numpy().astype(int).tolist()))
                        # writelog(f, '========================================================================')
                        # writelog(f, '                                            ')
                        # f.close()
                        #
                        #
                        # plt.imshow(M[zero_label_index1].cpu().numpy())
                        # plt.colorbar()
                        # plt.title('grouping matrix M')
                        # plt.savefig(
                        #     os.path.join(directory, 'visualization/EPOCH/') + 'sub{}label{}_grouping_matrix_M_epoch_{}.png'.format(
                        #         zero_label_index1, int(test_label[zero_label_index1]), epoch))
                        # plt.close()
                        #
                        # plt.imshow(M[zero_label_index2].cpu().numpy())
                        # plt.colorbar()
                        # plt.title('grouping matrix M')
                        # plt.savefig(
                        #     os.path.join(directory,
                        #                  'visualization/EPOCH/') + 'sub{}label{}_grouping_matrix_M_epoch_{}.png'.format(
                        #         zero_label_index2, int(test_label[zero_label_index2]), epoch))
                        # plt.close()
                        #
                        #
                        #
                        #
                        # plt.imshow(M[one_label_index1].cpu().numpy())
                        # plt.colorbar()
                        # plt.title('grouping matrix M')
                        # plt.savefig(
                        #     os.path.join(directory,
                        #                  'visualization/EPOCH/') + 'sub{}label{}_grouping_matrix_M_epoch_{}.png'.format(
                        #         one_label_index1, int(test_label[one_label_index1]), epoch))
                        # plt.close()
                        #
                        # plt.imshow(M[one_label_index2].cpu().numpy())
                        # plt.colorbar()
                        # plt.title('grouping matrix M')
                        # plt.savefig(
                        #     os.path.join(directory,
                        #                  'visualization/EPOCH/') + 'sub{}label{}_grouping_matrix_M_epoch_{}.png'.format(
                        #         one_label_index2, int(test_label[one_label_index2]), epoch))
                        # plt.close()

                        # plt.imshow(S[0].cpu().numpy())
                        # plt.colorbar()
                        # plt.title('pooling S')
                        # plt.savefig(
                        #     os.path.join(directory,
                        #                  'visualization/EPOCH/') + 'sub0_pooling_S_epoch_{}.png'.format(
                        #         epoch))
                        # plt.close()



                predic = torch.max(out, dim=1)[1].cpu().numpy()
                labels_all = np.append(labels_all, test_label)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)

        writer.add_scalar("test acc", acc, epoch)
        print('test_acc for epoch {} is : {}\n'.format(epoch, acc))









if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()





    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # # 2 label data load
    # data = np.load('/DataCommon/jwlee/EMOTION&MOTOR/hcp_emotion_&_motor.npz')
    # emotion_samples = data['tfMRI_EMOTION_LR'] #(1041, 176, 116)
    # motor_samples = data['tfMRI_MOTOR_LR'] #(1080, 284, 116)
    # motor_samples = motor_samples[:, :176, :] #(1080, 176, 116)



    # 7 label data load
    data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_7tasks.npz')
    emotion_samples = data['tfMRI_EMOTION_LR']  # (1041, 176, 116)
    gambling_samples = data['tfMRI_GAMBLING_LR'] # (1085, 253, 116)
    language_samples = data['tfMRI_LANGUAGE_LR'] # (1044, 316, 116)
    motor_samples = data['tfMRI_MOTOR_LR']  # (1080, 284, 116)
    relational_samples = data['tfMRI_RELATIONAL_LR'] # (1040, 232, 116)
    social_samples = data['tfMRI_SOCIAL_LR'] # (1045, 274, 116)
    wm_samples = data['tfMRI_WM_LR'] # (1082, 405, 116)


    emotion_samples = emotion_samples[:, :176, :]
    gambling_samples = gambling_samples[:, :176, :]
    language_samples = language_samples[:, :176, :]
    motor_samples = motor_samples[:, :176, :]  # (1080, 176, 116)
    relational_samples = relational_samples[:, :176, :]
    social_samples = social_samples[:, :176, :]
    wm_samples = wm_samples[:, :176, :]





########################################################################################################################


    # mean 0, std 1
    emo_results = []
    for ss in range(1041):
        w = emotion_samples[ss]
        emo_results.append(StandardScaler().fit_transform(w))
    sample1 = np.array(emo_results)
    sample1_label = [0 for i in range(1041)]
    sample1_label = np.array(sample1_label)  ## 0 == emotion


    gambling_results = []
    for pp in range(1085):
        k = gambling_samples[pp]
        gambling_results.append(StandardScaler().fit_transform(k))
    sample2 = np.array(gambling_results)
    sample2_label = [1 for i in range(1085)]
    sample2_label = np.array(sample2_label)  ## 1 == gambling


    language_results = []
    for pp in range(1044):
        k = language_samples[pp]
        language_results.append(StandardScaler().fit_transform(k))
    sample3 = np.array(language_results)
    sample3_label = [2 for i in range(1044)]
    sample3_label = np.array(sample3_label)  ## 2 == language


    motor_results = []
    for pp in range(1080):
        k = motor_samples[pp]
        motor_results.append(StandardScaler().fit_transform(k))
    sample4 = np.array(motor_results)
    sample4_label = [3 for i in range(1080)]
    sample4_label = np.array(sample4_label)  ## 3 == motor



    relational_results = []
    for pp in range(1040):
        k = relational_samples[pp]
        relational_results.append(StandardScaler().fit_transform(k))
    sample5 = np.array(relational_results)
    sample5_label = [4 for i in range(1040)]
    sample5_label = np.array(sample5_label)  ## 4 == relational



    social_results = []
    for pp in range(1045):
        k = social_samples[pp]
        social_results.append(StandardScaler().fit_transform(k))
    sample6 = np.array(social_results)
    sample6_label = [5 for i in range(1045)]
    sample6_label = np.array(sample6_label)  ## 5 == social



    wm_results = []
    for pp in range(1082):
        k = wm_samples[pp]
        wm_results.append(StandardScaler().fit_transform(k))
    sample7 = np.array(wm_results)
    sample7_label = [6 for i in range(1082)]
    sample7_label = np.array(sample7_label)  ## 6 == wm








    # minmax
    # mm = MinMaxScaler()
    # emo_results = []
    # for ss in range(1040):
    #     emo_results.append(mm.fit_transform(emotion_samples[ss]))
    # sample1 = np.array(emo_results)
    # sample1_label = [0 for i in range(1040)]
    # sample1_label = np.array(sample1_label) ## 0 == emotion
    #
    # mot_results = []
    # for pp in range(1080):
    #     mot_results.append(mm.fit_transform(motor_samples[pp]))
    # sample2 = np.array(mot_results)
    # sample2_label = [1 for i in range(1080)]
    # sample2_label = np.array(sample2_label) ## 1 == motor



    # samples = np.concatenate((sample1, sample4))  ## samples[1040] == sample2[0]
    # labels = np.concatenate((sample1_label, sample4_label))
    samples = np.concatenate((sample1, sample2, sample3, sample4, sample5, sample6, sample7))   ## samples[1040] == sample2[0]
    labels = np.concatenate((sample1_label, sample2_label, sample3_label, sample4_label, sample5_label, sample6_label, sample7_label))




    # number of clusters
    args.n_labels = len(np.unique(labels))
    args.timeseries = samples.shape[1]
    args.roi = samples.shape[2]




    # train, validation, test
    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(samples, labels, random_state=42, test_size=0.2)



    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)
    # real_train_data, real_train_label = torch.tensor(real_train_data, dtype=torch.float, requires_grad=True), torch.FloatTensor(real_train_label)
    # real_test_data, real_test_label = torch.tensor(real_test_data, dtype=torch.float, requires_grad=True), torch.FloatTensor(real_test_label)

    real_train_ds = TensorDataset(real_train_data, real_train_label)
    real_test_ds = TensorDataset(real_test_data, real_test_label)


    train_dl = DataLoader(real_train_ds, batch_size=args.batch_size)
    test_dl = DataLoader(real_test_ds, batch_size=args.batch_size)





    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))

    if args.ae_weights is None:
        run(args, train_dl=train_dl, test_dl=test_dl, directory=directory)

    if args.ae_weights is not None:

        if not os.path.exists(os.path.join(directory, 'test_visualization')):
            os.makedirs(os.path.join(directory, 'test_visualization/EPOCH'))

        path = os.path.join(directory, args.ae_weights)
        # full_path = sorted(glob.glob(path + '*.pt'), key=os.path.getctime)
        # full_path = full_path[-1]
        full_path = '/DataCommon2/jwlee/Adaptive_Number_of_Clusters/GMPool/input_dFC/7_task_euc/Epochs201_BS_8_LR_1e-05_wdcay_1e-06/models_weights/checkpoint_epoch_118_loss_0.00001.pt'
        print("I got: " + full_path)

        batch = args.batch_size
        time = args.timeseries
        roi = args.roi

        model = GMNetwork(args, batch, time, roi)
        checkpoint = torch.load(full_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(args.device)
        print(model)


        torch.manual_seed(0)
        test_data = DataLoader(real_test_ds, batch_size=batch, shuffle=True)


        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(test_data):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                _ = _.cpu().numpy()

                out, M, indices = model(inputs)

                if (batch_idx + 1) == 1:
                    f = open(os.path.join(directory, 'test_visualization/test_grouping_result.log'), 'a')


                    for i in range(len(_)):
                        a = int(_[i])
                        writelog(f, 'sub: %d, label: %d' % (i, a))
                        writelog(f, 'Grouping results: %s' % (indices[i].cpu().numpy().astype(int).tolist()))
                        writelog(f, '---------------------------------------------')
                    writelog(f, '========================================================================')
                    writelog(f, '                                            ')
                    f.close()

                    for i in range(len(_)):
                        a = int(_[i])
                        plt.imshow(M[i].cpu().numpy())
                        plt.colorbar()
                        plt.title('grouping matrix M')
                        plt.savefig(os.path.join(directory, 'test_visualization/EPOCH/') +
                                    'sub{}label{}_grouping_matrix_M.png'.format(i, a))
                        plt.close()

                predic = torch.max(out, dim=1)[1].cpu().numpy()
                labels_all = np.append(labels_all, _)
                predict_all = np.append(predict_all, predic)

            acc = metrics.accuracy_score(labels_all, predict_all)
            print('test_acc for epoch is : {}\n'.format(acc))


'''
a = torch.ones(25, 300)
b = torch.ones(22, 300)
c = torch.ones(15, 300)
torch.argmax(S[1], dim=1)
'''


