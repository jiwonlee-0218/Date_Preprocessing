import os
import random

import matplotlib.pyplot as plt
import torch
import os
import argparse
import numpy as np
from dataset import *
from net.st_gcn import Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import glob




def writelog(file, line):
    file.write(line + '\n')
    print(line)




def step(model, criterion, inputs, label, device='cpu', optimizer=None):
    if optimizer is None: model.eval()
    else: model.train()



    # run model
    logit = model(inputs.to(device))
    loss = criterion(logit, label.to(device))


    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, logit


def training_function(args):
    # make directories
    os.makedirs(os.path.join(args.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, 'summary'), exist_ok=True)



    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # torch.cuda.manual_seed_all(args.seed)



    # define dataset
    dataset = DatasetHCPTask(args.sourcedir, roi=args.roi, crop_length=args.crop_length, k_fold=args.k_fold) #전체데이터 5944
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) # 전체데이터 batch8로 나눈

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(args.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(args.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'loss' : 0,
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'scheduler': None}


    # start experiment
    for k in range(checkpoint['fold'], args.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(args.targetdir, 'model', str(k)), exist_ok=True) #'./result/stagin_experiment/model/0' ('./result/stagin_experiment' 'model' '0')
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)  #5 fold로 나누고 지금 fold 즉 k가 몇인지를 보고 그에 맞게끔 shuffle

        # define model
        batch = args.batch_size
        time = dataset.crop_length
        roi = dataset.num_nodes
        n_labels= dataset.num_classes
        spa_hidden_dim = 64
        hidden_dim = 128
        W = 50
        net = Model(1, n_labels, None, True)
        net.to(device)



        if checkpoint['model_state_dict'] is not None: net.load_state_dict(checkpoint['model_state_dict'])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_ae, weight_decay=0.001)
        if checkpoint['optimizer_state_dict'] is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




        best_score = 0.0
        # best_epoch = 0

        # start training
        for epoch in range(checkpoint['epoch'], args.epochs_ae):


            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            acc_accumulate = 0.0


            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                inputs = x['timeseries'] #torch.Size([8, 176, 116])
                label = x['label']


                # train_data_batch = np.zeros((inputs.shape[0], 1, W, 116, 1))
                # train_label_batch = label

                # for i in range(inputs.shape[0]):
                #     r1 = random.randint(0, time - W)
                #     train_data_batch[i] = inputs[i, :, r1:r1 + W, :, :]

                # train_data_batch = torch.from_numpy(train_data_batch).float()

                loss, logit = step(
                    model = net,
                    criterion = criterion,
                    inputs = inputs,
                    label = label,
                    device=device,
                    optimizer=optimizer
                )



                loss_accumulate += loss.detach().cpu().numpy()
                pred = logit.argmax(1)  # logit: torch.Size([2, 7]), device='cuda:0')
                acc_accumulate += ( torch.sum(pred.cpu() == label).item()  / batch )





            total_loss = loss_accumulate / len(dataloader)
            total_acc = acc_accumulate / len(dataloader)



            print()
            print('loss for epoch {} is : {}'.format(epoch, total_loss))
            print('acc for epoch {} is : {}'.format(epoch, total_acc))



            # eval
            dataset.set_fold(k, train=False)
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            for i, x in enumerate(dataloader):
                with torch.no_grad():
                    # process input data
                    inputs = x['timeseries']
                    label = x['label']


                    # val_data_batch = np.zeros((inputs.shape[0], 1, W, 116, 1))
                    # val_label_batch = label

                    # for i in range(inputs.shape[0]):
                    #     r1 = random.randint(0, time - W)
                    #     val_data_batch[i] = inputs[i, :, r1:r1 + W, :, :]

                    # val_data_batch = torch.from_numpy(val_data_batch).float()


                    val_loss, val_logit = step(
                        model=net,
                        criterion=criterion,
                        inputs = inputs,
                        label=label,
                        device=device,
                        optimizer=None,
                    )

                    pred = val_logit.argmax(1).cpu().numpy()
                    predict_all = np.append(predict_all, pred)
                    labels_all = np.append(labels_all, label)



            val_acc = metrics.accuracy_score(labels_all, predict_all)
            print('val_acc for epoch {} is : {}'.format(epoch, val_acc))


            if best_score < val_acc:
                best_score = val_acc
                best_epoch = epoch

                torch.save({
                    'fold': k,
                    'loss': total_loss,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(epoch)))


                f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')







def tt_function(args):

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # define dataset
    dataset = DatasetHCPTask_test(args.sourcedir, roi=args.roi, crop_length=args.crop_length, k_fold=args.k_fold)  # 전체데이터 1484
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    f1score_list = []
    for k in range(args.k_fold):
        os.makedirs(os.path.join(args.targetdir, 'attention', str(k)), exist_ok=True)


        # define model
        batch = args.batch_size
        time = dataset.crop_length
        roi = dataset.num_nodes
        n_labels = dataset.num_classes
        spa_hidden_dim = 64
        hidden_dim = 128
        W = 50
        net = Model(1, n_labels, None, True)

        # load model
        path = os.path.join(args.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        net.load_state_dict(checkpoint['model'])
        net.to(device)


        criterion = torch.nn.CrossEntropyLoss()



        # eval
        probabilities = []
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)



        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():

                inputs = x['timeseries']
                label = x['label']

                # test_data_batch = np.zeros((inputs.shape[0], 1, W, 116, 1))
                # test_label_batch = label

                # for i in range(inputs.shape[0]):
                #     r1 = random.randint(0, time - W)
                #     test_data_batch[i] = inputs[i, :, r1:r1 + W, :, :]

                # test_data_batch = torch.from_numpy(test_data_batch).float()

                test_loss, test_logit = step(
                    model=net,
                    criterion=criterion,
                    inputs=inputs,
                    label=label,
                    device=device,
                    optimizer=None,
                )


                probabilities.append(torch.nn.functional.softmax(test_logit.squeeze(), -1))
                pred = test_logit.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, pred)
                labels_all = np.append(labels_all, label)


        y_probabilities = torch.stack(probabilities).cpu().detach().numpy()
        auc_score = metrics.roc_auc_score(labels_all, y_probabilities, average='macro', multi_class='ovr')
        test_acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average='macro')
        # f = open(os.path.join(args.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        # writelog(f, 'test_acc: %.4f' % (test_acc))
        # f.close()
        print('test_acc is : {}'.format(test_acc))
        print('f1_score is : {}'.format(f1_score))
        print('auc_score is : {}'.format(auc_score))
        print('---------------------------')
        f1score_list.append(f1_score)

    print('total_f1score_mean: ', np.mean(f1score_list))
    print('total_f1score_std: ', np.std(f1score_list))






if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='MY-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU id") #############################
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_1') #################################
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='/home/jwlee/HMM/ST_GCN/data/task_fMRI')
        parser.add_argument('-dt', '--targetdir', type=str, default='/home/jwlee/HMM/ST_GCN/result')



        # model args
        parser.add_argument("--model_name", default="GMPool", help="model name")
        parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
        parser.add_argument('--crop_length', type=int, default=176)

        # training args

        parser.add_argument("--batch_size", default=16, type=int, help="batch size")
        parser.add_argument("--epochs_ae", type=int, default=30, help="Epochs number of the autoencoder training", )
        parser.add_argument("--lr_ae", type=float, default= 0.001, help="Learning rate of the autoencoder training", )




        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=True, help='pre-trained autoencoder weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)
    print(args.exp_name)

    # training_function(args)

    if args.weights is not None:
        tt_function(args)
