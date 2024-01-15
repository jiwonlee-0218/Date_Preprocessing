import os
import util
import random
import torch
import numpy as np
from model import *
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import glob
from sklearn import metrics


def writelog(file, line):
    file.write(line + '\n')
    print(line)



def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)
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
    if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
    elif argv.dataset=='hcp-task': dataset = DatasetHCPTask(argv.sourcedir, roi=argv.roi, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    elif argv.dataset=='ukb-rest': dataset = DatasetUKBRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
    else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=argv.num_workers, pin_memory=True)

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

    # start experiment
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True) #'./result/stagin_experiment/model/0' ('./result/stagin_experiment' 'model' '0')
        os.makedirs(os.path.join(argv.targetdir, 'model_weights', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)  #5 fold로 나누고 지금 fold 즉 k가 몇인지를 보고 그에 맞게끔 shuffle

        # define model
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        )
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=argv.lr, weight_decay=argv.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)


        best_score = 0.0
        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2) #torch.Size([150, 2, 116])
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                pred = logit.argmax(1) if dataset.num_classes > 1 else logit  #logit: torch.Size([2, 7]), pred: tensor([5, 5], device='cuda:0')
                print(pred)
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
            summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
            if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush()
            print(metrics)




            logger.initialize(k)
            dataset.set_fold(k, train=False)
            for i, x in enumerate(dataloader):
                with torch.no_grad():
                    # process input data
                    # if torch.all(x['label'] != x['label'][0]):

                    dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                    sampling_endpoints = [p+argv.window_size for p in sampling_points]
                    if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                    if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                    if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                    t = x['timeseries'].permute(1,0,2)
                    label = x['label']

                    logit, loss, attention, latent, reg_ortho = step(
                        model=model,
                        criterion=criterion,
                        dyn_v=dyn_v,
                        dyn_a=dyn_a,
                        sampling_endpoints=sampling_endpoints,
                        t=t,
                        label=label,
                        reg_lambda=argv.reg_lambda,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None,
                    )
                    pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                    print(pred)
                    prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                    logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())


            metrics = logger.evaluate(k)
            print(metrics)
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
    if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
    elif argv.dataset=='hcp-task': dataset = DatasetHCPTask_test(argv.sourcedir, roi=argv.roi, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold)
    elif argv.dataset=='ukb-rest': dataset = DatasetUKBRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
    elif argv.dataset=='ucla-rest': dataset = DatasetFMRIPREP(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples, prefix='ucla')
    else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    logger = util.logger.LoggerSTAGIN(argv.k_fold, dataset.num_classes)



    f1score_list = []
    for k in range(argv.k_fold):
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        )

        path = os.path.join(argv.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define logging objects
        probabilities = []
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        fold_attention = {'node_attention': [], 'time_attention': []}
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )


                probabilities.append(torch.nn.functional.softmax(logit.squeeze(), -1))
                pred = logit.argmax(1).cpu().numpy()
                predict_all = np.append(predict_all, pred)
                labels_all = np.append(labels_all, label)


                # pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                # prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                # logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy()) ####################################
                # loss_accumulate += loss.detach().cpu().numpy()
                # reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                #
                # fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                # fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                # latent_accumulate.append(latent.detach().cpu().numpy())

        # summarize results
        # samples = logger.get(k)
        # metrics = logger.evaluate(k)
        # summary_writer.add_scalar('loss', loss_accumulate/len(dataloader))
        # summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader))
        # summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
        # [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
        # [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for key, value in attention.items()]
        # summary_writer.flush()
        # print(metrics)
        # f = open(os.path.join(argv.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        # writelog(f, 'test_acc: %.4f' % (metrics['accuracy']))
        # f.close()
        # print('test_acc is : {}'.format(metrics['accuracy']))

        y_probabilities = torch.stack(probabilities).cpu().detach().numpy()
        auc_score = metrics.roc_auc_score(labels_all, y_probabilities, average='macro', multi_class='ovr')
        test_acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average='macro')


        print('test_acc is : {}'.format(test_acc))
        print('f1_score is : {}'.format(f1_score))
        print('auc_score is : {}'.format(auc_score))
        print('---------------------------')
        f1score_list.append(f1_score)

    print('total_f1score_mean: ', np.mean(f1score_list))
    print('total_f1score_std: ', np.std(f1score_list))


