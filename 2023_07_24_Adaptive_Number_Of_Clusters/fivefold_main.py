import os
import random
import torch
import os
import util
import argparse
import numpy as np
from dataset import *
from main_model import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import glob




def writelog(file, line):
    file.write(line + '\n')
    print(line)




def step(model, criterion, inputs, fc, label, device='cpu', optimizer=None):
    if optimizer is None: model.eval()
    else: model.train()



    # run model
    logit = model(inputs.to(device), fc.to(device))
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

    # set seed and device
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # torch.cuda.manual_seed_all(args.seed)



    # define dataset
    dataset = DatasetHCPTask(args.sourcedir, roi=args.roi, crop_length=args.crop_length, k_fold=args.k_fold) #전체데이터 5944
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) # 7428을 batch8로 나눈 743

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
        hidden_dim = 128
        model = GMNetwork(roi, hidden_dim, n_labels)
        model.to(device)



        if checkpoint['model_state_dict'] is not None: model.load_state_dict(checkpoint['model_state_dict'])
        criterion = torch.nn.CrossEntropyLoss()


        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs_ae, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=args.max_lr/args.lr_ae, final_div_factor=1000)
        if checkpoint['optimizer_state_dict'] is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])


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
                fc = util.bold.get_minibatch_fc( x['timeseries'])
                inputs = x['timeseries'] #torch.Size([4, 150, 116])
                label = x['label']


                loss, logit = step(
                    model = model,
                    criterion = criterion,
                    inputs = inputs,
                    fc = fc,
                    label = label,
                    device=device,
                    optimizer=optimizer
                )



                loss_accumulate += loss.detach().cpu().numpy()
                pred = logit.argmax(1)  # logit: torch.Size([2, 7]), device='cuda:0')
                acc_accumulate += ( torch.sum(pred.cpu() == label).item()  / batch )


            if scheduler is not None:
                scheduler.step()


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
                    fc = util.bold.get_minibatch_fc(x['timeseries'])
                    inputs = x['timeseries']
                    label = x['label']


                    val_loss, val_logit = step(
                        model=model,
                        criterion=criterion,
                        inputs = inputs,
                        fc = fc,
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
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(epoch)))


                f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')

        # finalize fold
        torch.save(model.state_dict(), os.path.join(args.targetdir, 'model', str(k), 'model.pth')) #모든 fold마다
        checkpoint.update({'loss' : 0, 'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})


    summary_writer.close()
    summary_writer_val.close()


def tt_function(args):
    os.makedirs(os.path.join(args.targetdir, 'attention'), exist_ok=True)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # define dataset
    dataset = DatasetHCPTask_test(args.sourcedir, roi=args.roi, crop_length=args.crop_length, k_fold=args.k_fold)  # 전체데이터 1484
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)


    for k in range(args.k_fold):
        os.makedirs(os.path.join(args.targetdir, 'attention', str(k)), exist_ok=True)

        # define model
        batch = args.batch_size
        time = dataset.crop_length
        roi = dataset.num_nodes
        n_labels = dataset.num_classes
        hidden_dim = 128
        model = GMNetwork(roi, hidden_dim, n_labels)

        # load model
        path = os.path.join(args.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)


        criterion = torch.nn.CrossEntropyLoss()



        # eval
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        fold_fc_attention = {'fc': [], 'time-attention': []}


        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():

                inputs = x['timeseries']
                label = x['label']

                test_loss, test_logit, fc_attention, test_new_X = step(
                    model=model,
                    criterion=criterion,
                    inputs=inputs,
                    label=label,
                    device=device,
                    optimizer=None,
                )

                pred = test_logit.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, pred)
                labels_all = np.append(labels_all, label)

                fold_fc_attention['fc'].append(fc_attention['fc'].detach().cpu().numpy())
                fold_fc_attention['time-attention'].append(fc_attention['time-attention'].detach().cpu().numpy())

        test_acc = metrics.accuracy_score(labels_all, predict_all)
        print('test_acc is : {}'.format(test_acc))

        for key, value in fold_fc_attention.items():
            os.makedirs(os.path.join(args.targetdir, 'attention', str(k), key), exist_ok=True)
            for idx, task in enumerate(dataset.task_list):
                np.save(os.path.join(args.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, labels_all) if l == idx]))




if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='MY-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="3", help="GPU id")
        parser.add_argument('-s', '--seed', type=int, default=0)
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_2')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='/DataCommon2/jwlee/2023_07_24_Adaptive_Number_Of_Clusters/data')
        parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/2023_07_24_Adaptive_Number_Of_Clusters/attention_FC_10/result')



        # model args
        parser.add_argument("--model_name", default="GMPool", help="model name")
        parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
        parser.add_argument('--crop_length', type=int, default=176)

        # training args

        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--epochs_ae", type=int, default=121, help="Epochs number of the autoencoder training", )
        parser.add_argument("--lr_ae", type=float, default=1e-5, help="Learning rate of the autoencoder training", )
        parser.add_argument('--max_lr', type=float, default=3e-5)
        parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for Adam optimizer", )



        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)
    print(args.exp_name)

    training_function(args)

    if args.weights is not None:
        tt_function(args)