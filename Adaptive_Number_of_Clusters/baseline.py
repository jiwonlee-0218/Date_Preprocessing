import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from model import *
from torch.utils.data import TensorDataset, DataLoader
from nilearn.connectome import ConnectivityMeasure
from torch.nn.utils.rnn import pad_sequence
# from model import *
from dFC_model import *
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
import os
import numpy as np



def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="7_task_dFC_baseline", help="dataset name")

    # model args
    parser.add_argument("--model_name", default="GMPool/input_dFC",help="model name")
    # parser.add_argument("--model_name", default="GMPool/input_R_T", help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")



    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=501, help="Epochs number of the autoencoder training",)
    parser.add_argument("--lr_ae", type=float, default=0.0001, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/DataCommon2/jwlee/Adaptive_Number_of_Clusters',)
    parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    # parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')

    return parser


def writelog(file, line):
    file.write(line + '\n')
    print(line)




def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt

def get_dfc(in_):
    '''
    run SWC for a single subject. Window the BOLD timeseries data then get lambda
    and estimate the FC matrix for each window.
    '''

    func = in_[0]
    window_size = in_[1]
    window_shape = in_[2]
    n_nodes = in_[3]
    step = in_[4]

    n_nodes = len(func[:, 0])
    if window_shape == 'rectangle':
        window = np.ones(window_size)
    elif window_shape == 'hamming':
        window = np.hamming(window_size)
    elif window_shape == 'hanning':
        window = np.hanning(window_size)
    else:
        raise Exception('%s window shape not recognised. Choose rectangle, hamming or hanning.' % window_shape)

    inds = range(0, len(func[0]), step)
    nwindows = len(inds)
    dfc = np.zeros([nwindows, int((n_nodes*n_nodes-n_nodes) / 2)])
    windowed_func = np.zeros([nwindows, n_nodes, window_size])

    for i in range(nwindows):
        this_sec = func[:, inds[i]:inds[i] + window_size]
        windowed_func[i] = this_sec * window



    for i in range(nwindows):
        fc = connectivity(np.expand_dims(windowed_func[i].T, 0), type='correlation', vectorization=True, fisher_t=False)
        dfc[i, :] = fc

    return dfc



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



    model = Transformer2(n_classes = args.n_labels, vocab_size = 6670, d_model = 512, word_pad_len = 44, hidden_size = 1024, n_heads = 4, n_encoders = 3, dropout =0.0)
    model.to(args.device)



    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)






    for epoch in tqdm(range(args.epochs_ae)):
        total_loss = 0
        total_acc = 0

        model.train()

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            _ = _.type(torch.Tensor).to(args.device).to(torch.int64)

            optimizer.zero_grad()

            out = model(inputs)
            loss = loss_function(out, _)

            loss.backward()
            optimizer.step()


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
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, total_loss))


        model.eval()
        test_total_loss = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)

        with torch.no_grad():
            for t_idx, (test, test_label) in enumerate(test_dl):
                test = test.type(torch.FloatTensor).to(args.device)
                test_label = test_label.cpu().numpy()

                out = model(test)
                test_loss = loss_function(out, torch.from_numpy(test_label).to(args.device).to(torch.int64))

                test_total_loss += test_loss
                predic = torch.max(out, dim=1)[1].cpu().numpy()
                labels_all = np.append(labels_all, test_label)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        test_total_loss = test_total_loss / (t_idx + 1)


        writer.add_scalar("test loss", test_total_loss, epoch)
        writer.add_scalar("test acc", acc, epoch)
        print('loss for epoch {} is : {}'.format(epoch, test_total_loss))
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





    # 7 label data load
    data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_7tasks.npz')
    emotion_samples = data['tfMRI_EMOTION_LR']  # (1041, 176, 116)
    gambling_samples = data['tfMRI_GAMBLING_LR'] # (1085, 253, 116)
    language_samples = data['tfMRI_LANGUAGE_LR'] # (1044, 316, 116)
    motor_samples = data['tfMRI_MOTOR_LR']  # (1080, 284, 116)
    relational_samples = data['tfMRI_RELATIONAL_LR'] # (1040, 232, 116)
    social_samples = data['tfMRI_SOCIAL_LR'] # (1045, 274, 116)
    wm_samples = data['tfMRI_WM_LR'] # (1082, 405, 116)


    emotion_samples = emotion_samples[:200, :176, :]
    gambling_samples = gambling_samples[:200, :176, :]
    language_samples = language_samples[:200, :176, :]
    motor_samples = motor_samples[:200, :176, :]  # (1080, 176, 116)
    relational_samples = relational_samples[:200, :176, :]
    social_samples = social_samples[:200, :176, :]
    wm_samples = wm_samples[:200, :176, :]



    # mean 0, std 1
    emo_results = []
    for ss in range(200):
        w = emotion_samples[ss]
        emo_results.append(StandardScaler().fit_transform(w))
    emotion_sample1 = np.array(emo_results)
    sample1_label = [0 for i in range(200)]
    sample1_label = np.array(sample1_label)  ## 0 == emotion


    gambling_results = []
    for pp in range(200):
        k = gambling_samples[pp]
        gambling_results.append(StandardScaler().fit_transform(k))
    gambling_sample2 = np.array(gambling_results)
    sample2_label = [1 for i in range(200)]
    sample2_label = np.array(sample2_label)  ## 1 == gambling


    language_results = []
    for pp in range(200):
        k = language_samples[pp]
        language_results.append(StandardScaler().fit_transform(k))
    language_sample3 = np.array(language_results)
    sample3_label = [2 for i in range(200)]
    sample3_label = np.array(sample3_label)  ## 2 == language


    motor_results = []
    for pp in range(200):
        k = motor_samples[pp]
        motor_results.append(StandardScaler().fit_transform(k))
    motor_sample4 = np.array(motor_results)
    sample4_label = [3 for i in range(200)]
    sample4_label = np.array(sample4_label)  ## 3 == motor



    relational_results = []
    for pp in range(200):
        k = relational_samples[pp]
        relational_results.append(StandardScaler().fit_transform(k))
    relational_sample5 = np.array(relational_results)
    sample5_label = [4 for i in range(200)]
    sample5_label = np.array(sample5_label)  ## 4 == relational



    social_results = []
    for pp in range(200):
        k = social_samples[pp]
        social_results.append(StandardScaler().fit_transform(k))
    social_sample6 = np.array(social_results)
    sample6_label = [5 for i in range(200)]
    sample6_label = np.array(sample6_label)  ## 5 == social



    wm_results = []
    for pp in range(200):
        k = wm_samples[pp]
        wm_results.append(StandardScaler().fit_transform(k))
    wm_sample7 = np.array(wm_results)
    sample7_label = [6 for i in range(200)]
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


########################################################################################################################
########################################################################################################################


    task_list = [emotion_sample1, gambling_sample2, language_sample3, motor_sample4, relational_sample5, social_sample6, wm_sample7]
    fc_task_list = []
    for i in task_list:
        dfc_ = [None] * i.shape[0]
        for s in range(i.shape[0]):
            dfc_[s] = get_dfc((i[s].T, 4, 'rectangle', 116, 4))

        fc_task_array = np.array(dfc_)
        fc_task_list.append(fc_task_array)




########################################################################################################################

    samples = np.concatenate((fc_task_list[0], fc_task_list[1], fc_task_list[2], fc_task_list[3], fc_task_list[4], fc_task_list[5], fc_task_list[6]))   ## samples[1040] == sample2[0]
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
        path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(path + '*.pt'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path)

        batch = 16
        time = args.timeseries
        roi = args.roi

        model = GMNetwork(args, batch, time, roi)
        checkpoint = torch.load(full_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(args.device)
        print(model)

        test_data = DataLoader(real_test_ds, batch_size=batch)


        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(test_data):

                inputs = inputs.type(torch.FloatTensor).to(args.device)
                _ = _.type(torch.Tensor).to(args.device).to(torch.int64)
                out, M, S = model(inputs)

                _ = _.cpu().numpy()
                predic = torch.max(out, dim=1)[1].cpu().numpy()
                labels_all = np.append(labels_all, _)
                predict_all = np.append(predict_all, predic)




'''
a = torch.ones(25, 300)
b = torch.ones(22, 300)
c = torch.ones(15, 300)
torch.argmax(S[1], dim=1)
'''


