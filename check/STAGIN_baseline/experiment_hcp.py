import os
import random
import torch
import numpy as np
from model import *
from dataload_craddock import dataloader, dataloader_hcp
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import glob
import util
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch.nn as nn


def writelog(file, line):
    file.write(line + '\n')
    print(line)


def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()


    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints) #logit torch.Size([16])
    # logit = torch.sigmoid(logit)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho


    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss, attention, latent, reg_ortho









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


    for k in range(args.k_fold):
        os.makedirs(os.path.join(args.targetdir, 'model', str(k)), exist_ok=True)
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)

        # define dataset
        # tr_loader, vl_loader, ts_loader = dataloader(args, k)
        tr_loader, vl_loader = dataloader_hcp(args, k)



        model = ModelSTAGIN(
            input_dim=tr_loader[0].shape[2],
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            cls_token=args.cls_token,
            readout=args.readout
        )
        model.to(device)
        # criterion = torch.nn.BCELoss()
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=(  int((tr_loader[0].shape[0]) / args.minibatch_size)), pct_start=0.2, div_factor=args.max_lr / args.lr, final_div_factor=1000)



        # define logging objects
        summary_writer = SummaryWriter(os.path.join(args.targetdir, 'summary',  str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(args.targetdir, 'summary',  str(k), 'val'), )



        best_score = 0.0
        for ep in range(args.num_epochs):
            # num_trn = tr_loader[0][:20].shape[0]
            # num_val = vl_loader[0][:10].shape[0]
            # data_shf, labl_shf = tr_loader[0][:20].to(device), tr_loader[2][:20].to(device)
            # val_data_shf, val_labl_shf = vl_loader[0][:10].to(device), vl_loader[2][:10].to(device)


            num_trn = tr_loader[0].shape[0]
            num_val = vl_loader[0].shape[0]
            # trn_shf = torch.randperm(num_trn) #shuffle
            data_shf, labl_shf = tr_loader[0].to(device), tr_loader[1].to(device)
            # val_shf = torch.randperm(num_val) #shuffle
            val_data_shf, val_labl_shf = vl_loader[0].to(device), vl_loader[1].to(device)





            loss_accumulate = 0.0
            acc_accumulate = 0.0
            tr_predict_all = np.array([], dtype=int)
            tr_prob_all = np.array([], dtype=int)
            tr_labels_all = np.array([], dtype=int)

            for i in tqdm(range(int(math.ceil(num_trn / args.minibatch_size)))):
                batchx, batchy = data_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)], labl_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)] #torch.Size([16, 115, 200]), torch.Size([16])
                dyn_a, sampling_points = util.bold.process_dynamic_fc(batchx, args.window_size, args.window_stride, args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
                sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22


                if i==0: dyn_v = repeat(torch.eye(tr_loader[0].shape[2]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=args.minibatch_size)
                if len(dyn_a) < args.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = batchx.permute(1, 0, 2)  # torch.Size([115, 16, 200])
                label = batchy




                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=args.reg_lambda,
                    clip_grad=args.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler
                )

                # @ main task
                loss_accumulate += loss.detach().cpu().numpy()
                # probability = logit.detach().cpu().numpy().reshape(-1)
                prediction = logit.argmax(1).cpu().numpy()

                print(prediction)
                tr_predict_all = np.append(tr_predict_all, prediction)
                # tr_prob_all = np.append(tr_prob_all, probability)
                tr_labels_all = np.append(tr_labels_all, label.cpu())




            total_loss = loss_accumulate / int(math.ceil(num_trn / args.minibatch_size))
            total_acc = metrics.accuracy_score(tr_labels_all, tr_predict_all)
            # tn, fp, fn, tp = confusion_matrix(tr_labels_all, tr_predict_all).ravel()
            # sensitivity = tp / (tp + fn)
            # specificity = tn / (tn + fp)
            # auc = roc_auc_score(tr_labels_all, tr_prob_all)


            # @ main task
            summary_writer.add_scalar('training loss', total_loss, ep)
            summary_writer.add_scalar("training acc", total_acc, ep)
            print(dict(epoch=ep, train_loss=total_loss, train_acc=total_acc))








            predict_all = np.array([], dtype=int)
            prob_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            for i in tqdm(range(int(math.ceil(num_val / args.minibatch_size)))):
                with torch.no_grad():
                    val_batchx, val_batchy = val_data_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)], val_labl_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)]  # torch.Size([16, 115, 200]), torch.Size([16])

                    dyn_a, sampling_points = util.bold.process_dynamic_fc(val_batchx, args.window_size,
                                                                         args.window_stride,
                                                                         args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
                    sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22
                    if i==0: dyn_v = repeat(torch.eye(tr_loader[0].shape[2]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=args.minibatch_size)
                    if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(tr_loader[0].shape[2]), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=args.minibatch_size)
                    if len(dyn_a) < args.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                    t = val_batchx.permute(1, 0, 2)  # torch.Size([115, 16, 200])
                    label = val_batchy

                    logit, loss, attention, latent, reg_ortho = step(
                        model=model,
                        criterion=criterion,
                        dyn_v=dyn_v,
                        dyn_a=dyn_a,
                        sampling_endpoints=sampling_endpoints,
                        t=t,
                        label=label,
                        reg_lambda=args.reg_lambda,
                        clip_grad=args.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None)

                    loss_accumulate += loss.detach().cpu().numpy()
                    # probability = logit.detach().cpu().numpy().reshape(-1)
                    prediction = logit.argmax(1).cpu().numpy()
                    print(prediction)

                    predict_all = np.append(predict_all, prediction)
                    # prob_all = np.append(prob_all, probability)
                    labels_all = np.append(labels_all, label.cpu())



            val_loss = loss_accumulate / int(math.ceil(num_val / args.minibatch_size))
            val_acc = metrics.accuracy_score(labels_all, predict_all)
            # tn, fp, fn, tp = confusion_matrix(labels_all, predict_all).ravel()
            # val_sensitivity = tp / (tp + fn)
            # val_specificity = tn / (tn + fp)
            # val_auc = roc_auc_score(labels_all, prob_all)





            summary_writer_val.add_scalar('val acc', val_acc, ep)
            print(dict(epoch=ep, val_acc=val_acc))
            print('-----------------------------------------------------------------')


            if best_score < val_acc:
                best_score = val_acc
                best_epoch = ep

                # save checkpoint
                torch.save({
                    'fold': k,
                    'loss': total_loss,
                    'epoch': ep,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(ep)))

                f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.5f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print('best score !!!')
                print()


    summary_writer.close()
    summary_writer_val.close()






def tt_function(argv):
    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    logger = util.logger.LoggerBrainNetFormer(argv.k_fold, args.num_classes)

    for k in range(args.k_fold):

        # define dataset
        tr_loader, vl_loader, ts_loader = dataloader_v4(args, (k+1))

        model = BrainNetFormer(
            n_region=tr_loader[0].shape[2],
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            window_size=args.window_size)

        path = os.path.join(argv.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)


        criterion = torch.nn.BCEWithLogitsLoss()

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        loss_accumulate = 0.0


        test_data, test_label = ts_loader[0], ts_loader[2]
        for i in range(test_data.shape[0]):
            with torch.no_grad():
                batchx_t, batchy_t = test_data[i], test_label[i]


                dyn_t, sampling_points = util.bold.process_dynamic_t(batchx_t, args.window_size,
                                                                     args.window_stride,
                                                                     args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
                dyn_a, sampling_points = util.bold.process_dynamic_fc(batchx_t, args.window_size,
                                                                      args.window_stride,
                                                                      args.dynamic_length)  # torch.Size([16, 22, 200, 200])

                sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22

                a = util.bold.process_static_fc(batchx_t)  # torch.Size([16, 200, 200])
                t = batchx_t.permute(1, 0, 2)  # torch.Size([115, 16, 200])
                label = batchy_t

                logit, loss, attention = step(
                    model=model,
                    criterion=criterion,
                    dyn_t=dyn_t,
                    dyn_a=dyn_a,
                    sampling_points=sampling_points,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    a=a,
                    label=label,
                    train=False,
                    clip_grad=args.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)

                # @ main task
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()

        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss', loss_accumulate / int(math.ceil(test_data.shape[0] / args.minibatch_size)))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']

        summary_writer.flush()
        print(metrics)
        f = open(os.path.join(argv.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        writelog(f, 'test_acc: %.4f' % (metrics['accuracy']))
        f.close()
        print('test_acc is : {}'.format(metrics['accuracy']))



if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_1_sero_140_v2')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-b', '--minibatch_size', type=int, default=16)
        parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/check/STAGIN/result')


        # dataset args
        parser.add_argument("--prec_glob", type=str, default=False, choices=[True, False])
        parser.add_argument("--prec_type", type=str, default='dparsf', choices=['cpac', 'dparsf'])
        parser.add_argument("--atlas", type=str, default='cc200', choices=['aal', 'ho', 'cc200'])
        parser.add_argument('--input_size', type=int, default=200)

        # training args
        parser.add_argument('--window_size', type=int, default=50)
        parser.add_argument('--window_stride', type=int, default=3)
        parser.add_argument('--dynamic_length', type=int, default=140)

        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--max_lr', type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for Adam optimizer", )
        parser.add_argument('--reg_lambda', type=float, default=0.00001)
        parser.add_argument('--clip_grad', type=float, default=0.0)
        parser.add_argument('--num_epochs', type=int, default=30)
        parser.add_argument('--num_heads', type=int, default=1)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--sparsity', type=int, default=30)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean'])
        parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])
        parser.add_argument("--rp", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=7)


        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=None, help='pre-trained ESTA weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)
    print(args.exp_name)

    training_function(args)

    if args.weights is not None:
        tt_function(args)  #test function



