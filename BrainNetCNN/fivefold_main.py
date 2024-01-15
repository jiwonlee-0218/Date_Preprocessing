import os
import random

import matplotlib.pyplot as plt
import torch
import os
import argparse
import numpy as np
from dataset import *
import module_brainnet as md
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import glob
from main_model import *
import utils as ut


def writelog(file, line):
    file.write(line + '\n')
    print(line)


def step(args, model, criterion, inputs, label, device='cpu', optimizer=None):
    if optimizer is None:
        model.eval_models()
    else:
        model.train_models()

    # run model
    logit = model(inputs.to(device))
    loss = criterion(logit, label.to(device))
    l1_loss = ut.l1_regularization(args.wspars, model.CLSBlock, device)


    loss_all = loss + l1_loss

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss_all.backward()
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
    args.device = device
    print(args.device)
    # torch.cuda.manual_seed_all(args.seed)

    # define dataset
    dataset = DatasetHCPTask(args.sourcedir, roi=args.atlas, crop_length=args.crop_length, k_fold=args.k_fold)  # 전체데이터 5944
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)  # 전체데이터 batch8로 나눈

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(args.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(args.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'loss': 0,
            'epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None}


    # start experiment
    for k in range(checkpoint['fold'], args.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(args.targetdir, 'model', str(k)),
                    exist_ok=True)  # './result/stagin_experiment/model/0' ('./result/stagin_experiment' 'model' '0')
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)  # 5 fold로 나누고 지금 fold 즉 k가 몇인지를 보고 그에 맞게끔 shuffle

        # define model
        model = BrainNetCNN(args)
        model.to(device)



        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_ae, momentum=0.9, weight_decay=args.weight_decay)



        # define logging objects
        summary_writer = SummaryWriter(os.path.join(args.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(args.targetdir, 'summary', str(k), 'val'), )


        best_score = 0.0
        # best_epoch = 0



        # start training
        for epoch in range(checkpoint['epoch'], args.epochs_ae):

            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            acc_accumulate = 0.0

            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                inputs = x['timeseries']  # torch.Size([4, 150, 116])
                label = x['label']

                loss, logit = step(
                    args = args,
                    model=model,
                    criterion=criterion,
                    inputs=inputs,
                    label=label,
                    device=device,
                    optimizer=optimizer
                )

                loss_accumulate += loss.detach().cpu().numpy()
                pred = logit.argmax(1)  # logit: torch.Size([2, 7]), device='cuda:0')
                acc_accumulate += (torch.sum(pred.cpu() == label).item() / args.batch_size)


            total_loss = loss_accumulate / len(dataloader)
            total_acc = acc_accumulate / len(dataloader)

            # summarize results
            summary_writer.add_scalar('training loss', total_loss, epoch)
            summary_writer.add_scalar("training acc", total_acc, epoch)
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

                    val_loss, val_logit = step(
                        args=args,
                        model=model,
                        criterion=criterion,
                        inputs=inputs,
                        label=label,
                        device=device,
                        optimizer=None,
                    )

                    pred = val_logit.argmax(1).cpu().numpy()
                    predict_all = np.append(predict_all, pred)
                    labels_all = np.append(labels_all, label)

            val_acc = metrics.accuracy_score(labels_all, predict_all)
            print('val_acc for epoch {} is : {}'.format(epoch, val_acc))
            summary_writer_val.add_scalar('val acc', val_acc, epoch)

            if best_score < val_acc:
                best_score = val_acc
                best_epoch = epoch

                torch.save({
                    'fold': k,
                    'loss': total_loss,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(epoch)))

                f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')

        # finalize fold
        torch.save(model.state_dict(), os.path.join(args.targetdir, 'model', str(k), 'model.pth'))  # 모든 fold마다
        checkpoint.update({'loss': 0, 'epoch': 0, 'model': None, 'optimizer': None})

    summary_writer.close()
    summary_writer_val.close()



def tt_function(args):
    os.makedirs(os.path.join(args.targetdir, 'attention'), exist_ok=True)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(device)

    # define dataset
    dataset = DatasetHCPTask_test(args.sourcedir, roi=args.atlas, crop_length=args.crop_length, k_fold=args.k_fold)  # 전체데이터 1484
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)


    f1score_list = []
    for k in range(args.k_fold):


        # define model
        model = BrainNetCNN(args)

        # load model
        path = os.path.join(args.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        # eval
        probabilities = []
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)


        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                inputs = x['timeseries']
                label = x['label']

                test_loss, test_logit = step(
                    args=args,
                    model=model,
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
        test_acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average='macro')
        auc_score = metrics.roc_auc_score(labels_all, y_probabilities, average='macro', multi_class='ovr')



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








if __name__ == '__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='MY-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_4')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='/home/jwlee/HMM/BrainNetCNN/data')
        parser.add_argument('-dt', '--targetdir', type=str, default='/home/jwlee/HMM/BrainNetCNN/result')

        # model args
        parser.add_argument("--model_name", default="GMPool", help="model name")
        parser.add_argument("--prec_glob", type=str, default='filt_noglobal', choices=['filt_global', 'filt_noglobal'])
        parser.add_argument("--prec_type", type=str, default='dparsf', choices=['cpac', 'dparsf'])
        parser.add_argument('--atlas', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
        parser.add_argument('--crop_length', type=int, default=176)

        # training args

        parser.add_argument("--batch_size", default=14, type=int, help="batch size")
        parser.add_argument("--epochs_ae", type=int, default=100, help="Epochs number of the autoencoder training", )
        parser.add_argument("--lr_ae", type=float, default=0.01, help="Learning rate of the autoencoder training", )
        parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for Adam optimizer", )


        parser.add_argument("--nroi", type=int, default=116)
        parser.add_argument("--input_size", type=int, default=200)
        parser.add_argument("--E2Efilters", type=int, default=32)
        parser.add_argument("--E2Nfilters", type=int, default=64)
        parser.add_argument("--N2Gfilters", type=int, default=30)
        parser.add_argument("--clshidden", type=int, default=128)
        parser.add_argument("--dp", type=float, default=0.5)  # dropout
        parser.add_argument("--wspars", type=float, default=0) # dropout
        parser.add_argument("--rp", type=int, default=2)




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
