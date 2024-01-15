import os
import util
import random
import torch
import numpy as np
from dataset import *
from main_model import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid









def step(model, criterion, dyn_v, t, fc, label, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention = model(dyn_v.to(device), t.to(device), fc.to(device))
    loss = criterion(logit, label.to(device))


    # optimize model
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
    print(device)
    torch.cuda.manual_seed_all(argv.seed)



    # define dataset
    if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
    elif argv.dataset=='hcp-task': dataset = DatasetHCPTask(argv.sourcedir, roi=argv.roi, crop_length=argv.crop_length, k_fold=argv.k_fold)
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

        # set dataloader
        dataset.set_fold(k, train=True)  #5 fold로 나누고 지금 fold 즉 k가 몇인지를 보고 그에 맞게끔 shuffle

        # define model
        model = EigenvectorAttentionModel(
            time_length = dataset.crop_length,
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token
        )
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerEigenvectorAttentionModel(argv.k_fold, dataset.num_classes)

        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0


            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                # dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                fc = util.bold.get_minibatch_fc(x['timeseries'])

                if i==0: dyn_v = repeat(torch.eye(argv.crop_length), 'n1 n2 -> b n1 n2', b=argv.minibatch_size)
                t = x['timeseries'] #torch.Size([4, 150, 116])
                label = x['label']


                logit, loss, attention = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    t=t,
                    fc=fc,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit  #logit: torch.Size([2, 7]), pred: tensor([5, 5], device='cuda:0')
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                loss_accumulate += loss.detach().cpu().numpy()


                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)

            if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush() #생성된 데이터 파일을 텐서보드를 통해서 Web에서 볼 수 있음
            print(metrics)

            # save checkpoint음
            torch.save({
                'fold': k,
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))


            logger.initialize(k)
            dataset.set_fold(k, train=False)
            for i, x in enumerate(dataloader):
                with torch.no_grad():
                    # process input data
                    # dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                    fc = util.bold.get_minibatch_fc(x['timeseries'])

                    if i==0: dyn_v = repeat(torch.eye(argv.crop_length), 'n1 n2 -> b n1 n2', b=argv.minibatch_size)
                    t = x['timeseries']
                    label = x['label']

                    logit, loss, attention  = step(
                        model=model,
                        criterion=criterion,
                        dyn_v=dyn_v,
                        t=t,
                        fc=fc,
                        label=label,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None,
                    )
                    pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                    prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                    logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
            metrics = logger.evaluate(k)
            print(metrics)

        # finalize fold
        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth')) #모든 fold마다
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    summary_writer.close()
    summary_writer_val.close()
    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))





# def test(argv):
#     os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
#     # define dataset
#     if argv.dataset=='hcp-rest': dataset = DatasetHCPRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
#     elif argv.dataset=='hcp-task': dataset = DatasetHCPTask(argv.sourcedir, roi=argv.roi, crop_length=argv.crop_length, k_fold=argv.k_fold)
#     elif argv.dataset=='ukb-rest': dataset = DatasetUKBRest(argv.sourcedir, roi=argv.roi, k_fold=argv.k_fold, target_feature=argv.target_feature, smoothing_fwhm=argv.fwhm, regression=argv.regression, num_samples=argv.num_samples)
#     else: raise
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=argv.num_workers, pin_memory=True)
#     logger = util.logger.LoggerEigenvectorAttentionModel(argv.k_fold, dataset.num_classes)
#
#     for k in range(argv.k_fold):
#         os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)
#
#         model = EigenvectorAttentionModel(
#             time_length=dataset.crop_length,
#             input_dim=dataset.num_nodes,
#             hidden_dim=argv.hidden_dim,
#             num_classes=dataset.num_classes,
#             num_heads=argv.num_heads,
#             num_layers=argv.num_layers,
#             sparsity=argv.sparsity,
#             dropout=argv.dropout,
#             cls_token=argv.cls_token
#         )
#         model.to(device)
#         model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
#         criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()
#
#         # define logging objects
#         fold_attention = {'time-attention_1': [], 'time-attention_2': []}
#         summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
#
#         logger.initialize(k)
#         dataset.set_fold(k, train=False)
#         loss_accumulate = 0.0
#
#
#         for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
#             with torch.no_grad():
#                 # process input data
#                 fc = util.bold.get_minibatch_fc(x['timeseries'])
#
#                 if i == 0: dyn_v = repeat(torch.eye(argv.crop_length), 'n1 n2 -> b n1 n2', b=argv.minibatch_size)
#                 t = x['timeseries']  # torch.Size([4, 150, 116])
#                 label = x['label']
#
#                 logit, loss, attention = step(
#                     model=model,
#                     criterion=criterion,
#                     dyn_v=dyn_v,
#                     t=t,
#                     fc=fc,
#                     label=label,
#                     clip_grad=argv.clip_grad,
#                     device=device,
#                     optimizer=None,
#                     scheduler=None,
#                 )
#                 pred = logit.argmax(1) if dataset.num_classes > 1 else logit
#                 prob = logit.softmax(1) if dataset.num_classes > 1 else logit
#                 logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy()) ####################################
#                 loss_accumulate += loss.detach().cpu().numpy()
#
#
#                 fold_attention['time-attention_1'].append(attention['time-attention_1'].detach().cpu().numpy())
#                 fold_attention['time-attention_2'].append(attention['time-attention_2'].detach().cpu().numpy())
#
#
#         # summarize results
#         samples = logger.get(k)
#         metrics = logger.evaluate(k)
#         summary_writer.add_scalar('loss', loss_accumulate/len(dataloader))
#         summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
#         [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
#         [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for key, value in attention.items()]
#         summary_writer.flush()
#         print(metrics)
#
#         # finalize fold
#         logger.to_csv(argv.targetdir, k)
#
#         for key, value in fold_attention.items():  #len(fold_attention)=67 첫번째 key는 node_attention, 두번째 key는
#             os.makedirs(os.path.join(argv.targetdir, 'attention', str(k), key), exist_ok=True)  #os.path.join(argv.targetdir, 'attention', str(k), key) == './result/stagin_experiment/attention/0/node_attention'
#             for idx, task in enumerate(dataset.task_list):
#                 np.save(os.path.join(argv.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, samples['true']) if l==idx]))
#
#
#         del fold_attention
#
#     # finalize experiment
#     logger.to_csv(argv.targetdir)
#     final_metrics = logger.evaluate()
#     print(final_metrics)
#     summary_writer.close()
#     torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))

if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()

    # run and analyze experiment
    if not any([argv.train, argv.test]): argv.train = argv.test = True
    if argv.train: train(argv)
    # if argv.test: test(argv)
