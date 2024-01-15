import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.st_gcn import Model
from tqdm import tqdm
import random
from scipy import stats
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import glob


def writelog(file, line):
    file.write(line + '\n')
    print(line)





def training_function():
    ###### **model parameters**
    W = 50  # window size


    ###### **training parameters**
    LR = 0.001  # learning rate
    batch_size = 16


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

    # state_dict = torch.load('checkpoint.pth')
    # net.load_state_dict(state_dict)

    # train_data = np.load('data/train_data_1200_1.npy')
    # train_label = np.load('data/train_label_1200_1.npy')
    # test_data = np.load('data/test_data_1200_1.npy')
    # test_label = np.load('data/test_label_1200_1.npy')
    #
    # print(train_data.shape)

    ###### start training model



    W = 50
    final_testing_accuracy = 0
    testing_acc_curr_fold = []
    #print('-'*80)
    #print("Window Size {}".format(W))
    #print('-'*80)
    for fold in range(1, 6):
        os.makedirs(os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model', str(fold)), exist_ok=True)
        os.makedirs(os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model_weights', str(fold)), exist_ok=True)
        print('-'*80)
        print("Window Size {}, Fold {}".format(W, fold))
        print('-'*80)
        best_test_acc_curr_fold = 0
        best_test_epoch_curr_fold = 0
        best_edge_imp_curr_fold = []
        train_data = np.load('/home/jwlee/HMM/ST_GCN/data/task_fMRI/train_data_'+str(fold)+'.npy')
        train_label = np.load('/home/jwlee/HMM/ST_GCN/data/task_fMRI/train_label_'+str(fold)+'.npy')
        test_data = np.load('/home/jwlee/HMM/ST_GCN/data/task_fMRI/test_data_'+str(fold)+'.npy')
        test_label = np.load('/home/jwlee/HMM/ST_GCN/data/task_fMRI/test_label_'+str(fold)+'.npy')

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, random_state=42, test_size=0.2)
        X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        X_test, y_test = torch.FloatTensor(test_data), torch.FloatTensor(test_label)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=16)  ###
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=16)
        test_ds = TensorDataset(X_val, y_val)
        test_dl = DataLoader(test_ds, batch_size=1)



        net = Model(1, 7, None, True)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

        best_score = 0.0
        for epoch in tqdm(range(30)):  # number of mini-batches

            net.train()
            loss_accumulate = 0.0
            acc_accumulate = 0.0

            for batch_idx, (inputs, _) in enumerate(train_dl):


                # construct a mini-batch by sampling a window W for each subject
                train_data_batch = np.zeros((inputs.shape[0], 1, W, 116, 1))
                train_label_batch = _

                for i in range(inputs.shape[0]):
                    r1 = random.randint(0, train_data.shape[2] - W)
                    train_data_batch[i] = inputs[i, :, r1:r1 + W, :, :]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = train_label_batch.float().to(device)
                #train_data_batch_dev = train_data_batch_dev.squeeze()
                # forward + backward + optimize
                optimizer.zero_grad()
                #net.hidden = net.init_hidden(batch_size)
                outputs = net(train_data_batch_dev)
                loss = criterion(outputs, train_label_batch_dev.long())
                loss.backward()
                optimizer.step()

                # print training statistics
                loss_accumulate += loss.item()
                pred = outputs.argmax(1)
                acc_accumulate += (torch.sum(pred.cpu() == train_label_batch_dev.cpu()).item() / batch_size)

            total_loss = loss_accumulate / (batch_idx + 1)
            total_acc = acc_accumulate / (batch_idx + 1)
            print('loss for epoch {} is : {}'.format(epoch, total_loss))
            print('acc for epoch {} is : {}'.format(epoch, total_acc))


            net.eval()
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            with torch.no_grad():
                for j, (val_x, val_y) in enumerate(valid_dl):
                    val_data_batch = np.zeros((val_x.shape[0], 1, W, 116, 1))
                    val_label_batch = val_y

                    for i in range(val_x.shape[0]):
                        r1 = random.randint(0, test_data.shape[2] - W)
                        val_data_batch[i] = val_x[i, :, r1:r1 + W, :, :]

                    val_data_batch_dev = torch.from_numpy(val_data_batch).float().to(device)
                    outputs = net(val_data_batch_dev)
                    pred = outputs.argmax(1).cpu().numpy()
                    predict_all = np.append(predict_all, pred)
                    labels_all = np.append(labels_all, val_label_batch.cpu().detach().numpy())

            val_acc = metrics.accuracy_score(labels_all, predict_all)
            print('val_acc for epoch {} is : {}'.format(epoch, val_acc))


            if best_score < val_acc:
                best_score = val_acc
                best_epoch = epoch

                torch.save({
                    'fold': fold,
                    'loss': total_loss,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model_weights', str(fold), 'checkpoint_epoch_{}.pth'.format(epoch)))

                f = open(os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model', str(fold), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')



        test_predict_all = np.array([], dtype=int)
        test_labels_all = np.array([], dtype=int)
        net = Model(1, 7, None, True)
        # load model
        path = os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model_weights', str(fold))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        net.load_state_dict(checkpoint['model'])
        net.to(device)

        net.eval()
        with torch.no_grad():
            for test_batch_idx, (test_inputs, test_labelsss) in tqdm(enumerate(test_dl)):
                test_data_batch = np.zeros((test_inputs.shape[0], 1, W, 116, 1))
                test_label_batch = test_labelsss

                for i in range(test_inputs.shape[0]):
                    r1 = random.randint(0, test_data.shape[2] - W)
                    test_data_batch[i] = test_inputs[i, :, r1:r1 + W, :, :]

                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                outputs = net(test_data_batch_dev)
                pred = outputs.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, pred)
                labels_all = np.append(labels_all, test_label_batch.cpu().detach().numpy())

            test_acc = metrics.accuracy_score(test_labels_all, test_predict_all)
            f = open(os.path.join('/home/jwlee/HMM/ST_GCN/result2', 'model', str(fold), 'test_acc.log'), 'a')
            writelog(f, 'test_acc: %.4f' % (test_acc))
            f.close()
            print('test_acc is : {}'.format(test_acc))




if __name__ == '__main__':
    # GPU Configuration
    gpu_id = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    training_function()