import os
import util
import random
import torch
import numpy as np
# from model import *
from BrainNetFormer import *
from dataset import *
from tqdm import tqdm
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import glob
from sklearn import metrics


def writelog(file, line):
    file.write(line + '\n')
    print(line)


def step(model, criterion, dyn_t, dyn_a, sampling_points, sampling_endpoints, t, a, label, subtask_label, reg_lambda,
         reg_subtask, train=True, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    # @ main task prediction
    logit, attention = model(dyn_t.to(device), dyn_a.to(device), t.to(device), a.to(device), sampling_endpoints)
    loss = criterion(logit, label.to(device))

    # @ subtask prediction
    # loss_subtask = criterion(logit_subtask, subtask_label.to(device))
    # loss += loss_subtask*reg_subtask

    # @ optimize model
    if (train):
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    return logit, loss, attention


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)

    # GPU Configuration
    gpu_id = argv.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    argv.device = device
    print(argv.device)
    torch.cuda.manual_seed_all(argv.seed)

    # define dataset
    if argv.dataset == 'rest':
        dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    elif argv.dataset == 'task':
        dataset = DatasetHCPTask(argv.sourcedir, subtask_type=argv.subtask_type, roi=argv.roi,
                                 dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    else:
        raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    num_timepoints = len(list(range(0, argv.dynamic_length - argv.window_size, argv.window_stride)))
    subtask_nums = 2 if argv.subtask_type == 'simple' else 25
    print('subtask_nums: ', subtask_nums)

    # @ ************************************ start experiment **************************************** @#
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)
        os.makedirs(os.path.join(argv.targetdir, 'model_weights', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = BrainNetFormer(
            n_region=dataset.num_nodes,  # n_region是顶点的个数N
            hidden_dim=argv.hidden_dim,  # hidden_dim是特征的维度D
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            subtask_nums=subtask_nums,
            window_size=argv.window_size)
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs,
                                                        steps_per_epoch=len(dataloader), pct_start=0.2,
                                                        div_factor=argv.max_lr / argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerBrainNetFormer(argv.k_fold, dataset.num_classes)
        logger_subtask = util.logger.LoggerBrainNetFormer(argv.k_fold, subtask_nums)

        best_score = 0.0
        ''' training step '''
        # @ ******************************** start training ***********************************@#
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            # @ ********************************* train ***************************************@#
            logger.initialize(k)
            logger_subtask.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            loss_subtask_accumulate = 0.0

            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):  #
                # process input data
                # @ 预处理数据
                dyn_t, sampling_points = util.bold.process_dynamic_t(x['timeseries'], argv.window_size,
                                                                     argv.window_stride,
                                                                     argv.dynamic_length)  # torch.Size([2, 34, 116, 50]) len(sampling_points)=34
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                      argv.window_stride,
                                                                      argv.dynamic_length)  # torch.Size([2, 34, 116, 116])
                sampling_endpoints = [p + argv.window_size for p in sampling_points]  # len(sampling_endpoints)=34

                a = util.bold.process_static_fc(x['timeseries'])  # torch.Size([2, 116, 116])
                t = x['timeseries'].permute(1, 0, 2)  # torch.Size([150, 2, 116])
                label = x['label']
                # subtask_label = x['subtask_label'].permute(1,0)[sampling_points].permute(1,0)
                subtask_label = x['subtask_label']  # whole time중 짤린 150구간의 subtask_label torch.Size([2, 150])
                subtask_label = rearrange(subtask_label, 'b n -> (b n)')  # torch.Size([300])
                # @ 向前传播+反向传播
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
                    subtask_label=subtask_label,
                    reg_lambda=argv.reg_lambda,
                    reg_subtask=argv.reg_subtask,
                    train=True,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler)

                # @ main task
                pred = logit.argmax(1)  # @预测值
                prob = logit.softmax(1)  # @预测概率
                loss_accumulate += loss.detach().cpu().numpy()  # @累计loss
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(),
                           prob=prob.detach().cpu().numpy())

                # @ subtask
                # pred_subtask = logit_subtask.argmax(1)
                # prob_subtask = logit_subtask.softmax(1)
                # loss_subtask_accumulate += loss_subtask.detach().cpu().numpy()
                # logger_subtask.add(k=k, pred=pred_subtask.detach().cpu().numpy(), true=subtask_label.detach().cpu().numpy(), prob=prob_subtask.detach().cpu().numpy())

                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i + epoch * len(dataloader))

            # @ main task
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate / len(dataloader), epoch)
            summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key == 'fold']
            print("train:", metrics)

            # @ subtask
            # samples_subtask = logger_subtask.get(k)
            # metrics_subtask = logger_subtask.evaluate(k)
            # summary_writer.add_scalar('loss_subtask', loss_subtask_accumulate/len(dataloader), epoch)
            # summary_writer.add_pr_curve('precision-recall_subtask', samples_subtask['true'], samples_subtask['prob'][:,1], epoch)
            # [summary_writer.add_scalar(key+'_subtask', value, epoch) for key, value in metrics_subtask.items() if not key=='fold']
            # summary_writer.flush()

            logger.initialize(k)
            dataset.set_fold(k, train=False)
            for i, x in enumerate(dataloader):
                with torch.no_grad():
                    dyn_t, sampling_points = util.bold.process_dynamic_t(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)  # torch.Size([2, 34, 116, 50]) len(sampling_points)=34
                    dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)  # torch.Size([2, 34, 116, 116])
                    sampling_endpoints = [p + argv.window_size for p in sampling_points]  # len(sampling_endpoints)=34

                    a = util.bold.process_static_fc(x['timeseries'])  # torch.Size([2, 116, 116])
                    t = x['timeseries'].permute(1, 0, 2)  # torch.Size([150, 2, 116])
                    label = x['label']
                    # subtask_label = x['subtask_label'].permute(1,0)[sampling_points].permute(1,0)
                    subtask_label = x['subtask_label']  # whole time중 짤린 150구간의 subtask_label torch.Size([2, 150])
                    subtask_label = rearrange(subtask_label, 'b n -> (b n)')  # torch.Size([300])
                    # @ 向前传播+反向传播
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
                        subtask_label=subtask_label,
                        reg_lambda=argv.reg_lambda,
                        reg_subtask=argv.reg_subtask,
                        train=False,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None)

                    # @ main task
                    pred = logit.argmax(1)  # @预测值
                    prob = logit.softmax(1)  # @预测概率
                    loss_accumulate += loss.detach().cpu().numpy()  # @累计loss
                    logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(),
                               prob=prob.detach().cpu().numpy())

                    # @ subtask
                    # pred_subtask = logit_subtask.argmax(1)
                    # prob_subtask = logit_subtask.softmax(1)
                    # loss_subtask_accumulate += loss_subtask.detach().cpu().numpy()
                    # logger_subtask.add(k=k, pred=pred_subtask.detach().cpu().numpy(), true=subtask_label.detach().cpu().numpy(), prob=prob_subtask.detach().cpu().numpy())

            metrics = logger.evaluate(k)
            print("val:", metrics)
            if best_score < metrics['accuracy']:
                best_score = metrics['accuracy']
                best_epoch = epoch

                # save checkpoint
                torch.save({
                    'fold': k,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(argv.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(epoch)))

                f = open(os.path.join(argv.targetdir, 'model', str(k), 'best_acc.log'), 'a')
                writelog(f, 'best_acc: %.4f for epoch: %d' % (best_score, best_epoch))
                f.close()
                print()
                print('-----------------------------------------------------------------')

        # finalize fold
        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    summary_writer.close()
    summary_writer_val.close()


def tt_function(argv):
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)
    # GPU Configuration
    gpu_id = argv.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # define dataset
    if argv.dataset == 'rest':
        dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    elif argv.dataset == 'task':
        dataset = DatasetHCPTask_test(argv.sourcedir, subtask_type=argv.subtask_type, roi=argv.roi,
                                      dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    else:
        raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    num_timepoints = len(list(range(0, argv.dynamic_length - argv.window_size, argv.window_stride)))
    subtask_nums = 25
    logger = util.logger.LoggerBrainNetFormer(argv.k_fold, dataset.num_classes)
    logger_subtask = util.logger.LoggerBrainNetFormer(argv.k_fold, subtask_nums)

    for k in range(argv.k_fold):
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        model = BrainNetFormer(
            n_region=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            subtask_nums=subtask_nums,
            window_size=argv.window_size)

        path = os.path.join(argv.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        # define logging objects
        probabilities = []
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        logger_subtask.initialize(k)
        loss_accumulate = 0.0
        loss_subtask_accumulate = 0.0

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                if i == 392:
                    continue
                # process input data
                dyn_t, sampling_points = util.bold.process_dynamic_t(x['timeseries'], argv.window_size,
                                                                     argv.window_stride)
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                      argv.window_stride)
                sampling_endpoints = [p + argv.window_size for p in sampling_points]
                # print(dyn_t.shape, dyn_a.shape)
                a = util.bold.process_static_fc(x['timeseries'])
                t = x['timeseries'].permute(1, 0, 2)
                # a =
                label = x['label']
                # subtask_label = x['subtask_label'].permute(1,0)[sampling_points].permute(1,0)
                subtask_label = x['subtask_label']
                subtask_label = rearrange(subtask_label, 'b n -> (b n)')
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
                    subtask_label=subtask_label,
                    reg_lambda=argv.reg_lambda,
                    reg_subtask=argv.reg_subtask,
                    train=False,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)

                # @ main task
                # pred = logit.argmax(1)
                prob = logit.softmax(1)
                # logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                # loss_accumulate += loss.detach().cpu().numpy()

                probabilities.append(prob.squeeze().cpu().numpy())
                y_pred = logit.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, y_pred)
                labels_all = np.append(labels_all, label)




                # @ subtask
                # pred_subtask = logit_subtask.argmax(1)
                # prob_subtask = logit_subtask.softmax(1)
                # loss_subtask_accumulate += loss_subtask.detach().cpu().numpy()
                # logger_subtask.add(k=k, pred=pred_subtask.detach().cpu().numpy(), true=subtask_label.detach().cpu().numpy(), prob=prob_subtask.detach().cpu().numpy())






        # summarize results
        # samples = logger.get(k)
        # metrics = logger.evaluate(k)
        # summary_writer.add_scalar('loss', loss_accumulate / len(dataloader))
        # summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
        # [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']
        #
        # summary_writer.flush()
        # print(metrics)
        # f = open(os.path.join(argv.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        # writelog(f, 'test_acc: %.4f' % (metrics['accuracy']))
        # f.close()
        # print('test_acc is : {}'.format(metrics['accuracy']))


        # samples_subtask = logger_subtask.get(k)
        # metrics_subtask = logger_subtask.evaluate(k)
        # summary_writer.add_scalar('loss_subtask', loss_subtask_accumulate/len(dataloader))
        # summary_writer.add_pr_curve('precision-recall_subtask', samples_subtask['true'], samples_subtask['prob'][:,1])
        # [summary_writer.add_scalar(key+'_subtask', value) for key, value in metrics_subtask.items() if not key=='fold']

        # summary_writer.flush()
        # print("subtask:", metrics_subtask)
        #
        # # finalize fold
        # logger.to_csv(argv.targetdir, k)
        # logger_subtask.to_csv(argv.targetdir, k)

        y_probabilities = np.array(probabilities)
        auc_score = metrics.roc_auc_score(labels_all, y_probabilities, average='macro', multi_class='ovr')
        test_acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average='macro')

        print('test_acc is : {}'.format(test_acc))
        print('f1_score is : {}'.format(f1_score))
        print('auc_score is : {}'.format(auc_score))
        print('---------------------------')



