import os
import random
import torch
import argparse
import numpy as np
from dataset import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics




def writelog(file, line):
    file.write(line + '\n')
    print(line)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.utils import weight_norm
import math
from nilearn.connectome import ConnectivityMeasure
import copy
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=1, in_chans=116, embed_dim=90):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)



    def forward(self, x):
        B, H, W = x.shape # (64, 284, 116)
        x = x.transpose(1,2)  # (16, 116, 284)
        x = self.proj(x).transpose(1, 2)  # (64, 80, 284) -> (64, 284, 80)
        return x  # (64, 284, 80) patch sequence = 284 patch 1개당 80 dimension


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Parameters
    ----------
    d_model : int
        Size of word embeddings
    word_pad_len : int
        Length of the padded sentence
    dropout : float
        Dropout
    """
    def __init__(self, d_model: int, word_pad_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()

        self.pe = torch.tensor([
            [pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)]
            for pos in range(word_pad_len)
        ])  # (batch_size, word_pad_len, emb_size)

        # PE(pos, 2i) = sin(pos / 10000^{2i / d_model})
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        # PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word embeddings
        Returns
        -------
        position encoded embeddings : torch.Tensor (batch_size, word_pad_len, emb_size)
            Word Embeddings + Positional Encoding
        """
        # word embeddings + positional encoding
        embeddings = embeddings + nn.Parameter(self.pe, requires_grad=True).to('cuda')
        embeddings = self.dropout(embeddings)
        return embeddings

''' transformer '''
class Transformer_model(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self,  ntoken, d_model, n_head, word_pad_len, dropout=0.0): # 116 -> 64 -> 128
        super(Transformer_model, self).__init__()
        self.src_mask = None
        self.embedding = PatchEmbed(in_chans=ntoken, embed_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, word_pad_len, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead= n_head, dim_feedforward=128, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.fc = nn.Linear(word_pad_len * d_model, 8)


    def forward(self, inputs):
        src = self.embedding(inputs)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        encoder_out = output.view(output.size(0), -1)
        scores = self.fc(encoder_out)

        return scores





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


def train(args):
    # make directories
    os.makedirs(os.path.join(args.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, 'summary'), exist_ok=True)

    # set seed and device 절대 풀지말것
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device
    print(device)
    # torch.cuda.manual_seed_all(args.seed)



    # define dataset
    dataset = DatasetHCPTask(args.sourcedir, roi=args.roi, crop_length=args.crop_length, k_fold=args.k_fold) #전체데이터 7428개 나옴
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) # 7428개 batch8로 나눈 929

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

        model = Transformer_model(ntoken=116, d_model=64, n_head=4, word_pad_len=time, dropout=0.0)
        model.to(device)



        if checkpoint['model_state_dict'] is not None: model.load_state_dict(checkpoint['model_state_dict'])
        criterion = torch.nn.CrossEntropyLoss()


        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
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
                inputs = x['timeseries'] #torch.Size([4, 150, 116])
                label = x['label']


                loss, logit = step(
                    model = model,
                    criterion = criterion,
                    inputs = inputs,
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
                    inputs = x['timeseries']
                    label = x['label']


                    loss, logit  = step(
                        model=model,
                        criterion=criterion,
                        inputs = inputs,
                        label=label,
                        device=device,
                        optimizer=None,
                    )

                    pred = logit.argmax(1).cpu().numpy()
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






if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='MY-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
        parser.add_argument('-s', '--seed', type=int, default=0)
        parser.add_argument('-n', '--exp_name', type=str, default='basic_experiment')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='/DataCommon2/jwlee/2023_07_17_Adaptive_Number_of_Clusters/data')
        parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/2023_07_17_Adaptive_Number_of_Clusters/masking_R_T/result')



        # model args
        parser.add_argument("--model_name", default="GMPool", help="model name")
        parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'destrieux', 'harvard_oxford'])
        parser.add_argument('--crop_length', type=int, default=176)

        # training args

        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--epochs_ae", type=int, default=101, help="Epochs number of the autoencoder training", )
        parser.add_argument("--lr_ae", type=float, default=1e-6, help="Learning rate of the autoencoder training", )
        parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for Adam optimizer", )




        parser.add_argument('--train', action='store_true')  # 옵션이 지정되면 True 를 대입하고 지정하지 않으면 False 를 대입
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--num_workers', type=int, default=4)

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)

    # run and analyze experiment
    if not any([args.train, args.test]): args.train = args.test = True
    if args.train: train(args)

