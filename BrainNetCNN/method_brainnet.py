import os
import time
import csv
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import module_brainnet as md
import utils as ut
# from dataset.dataload_wonsik_mdd import dataloader
# from dataset.data_loader_abide import dataloader_abide
# from dataset.dataload_wonsik import dataloader
from dataload_abide_202308 import dataloader

# Define experiment class
class experiment():

    def __init__(self, args):

        """ Build network """
        self.E2EBlock = md.E2EBlock(args.nroi, args.E2Efilters, 0.5).to(args.device)
        self.E2EBlock2 = md.E2EBlock2(args.nroi, args.E2Efilters, 0.5).to(args.device)
        self.E2NBlock = md.E2NBlock(args.nroi, args.E2Efilters, args.E2Nfilters, 0.5).to(args.device)
        self.N2GBlock = md.N2GBlock(args.nroi, args.E2Nfilters, args.N2Gfilters, 0.5).to(args.device)
        self.CLSBlock = md.CLSBlock(args.N2Gfilters, args.clshidden, 0.5).to(args.device)
        self.cls_criterion = nn.CrossEntropyLoss().to(args.device)

        self.train_params = list(self.E2EBlock.parameters()) + list(self.E2EBlock2.parameters()) + \
                            list(self.E2NBlock.parameters()) + \
                            list(self.N2GBlock.parameters()) + \
                            list(self.CLSBlock.parameters())

        self.st = args

    def train_models(self):
        self.E2EBlock.train(), self.E2EBlock2.train()
        self.E2NBlock.train(), self.N2GBlock.train()
        self.CLSBlock.train()

    def eval_models(self):
        self.E2EBlock.eval(), self.E2EBlock2.eval(),
        self.E2NBlock.eval(), self.N2GBlock.eval(),
        self.CLSBlock.eval()

    def optimizer(self):
        return optimizer.Adam(self.train_params, lr=self.st.lr, weight_decay=self.st.wdecay)

    def training(self, save_path, ckpt_path, save=False):
        """ Load dataset """
        # tr_loader, vl_loader, ts_loader = dataloader(self.st, self.st.fold, self.st.bs)
        tr_loader, vl_loader, ts_loader = dataloader(self.st, self.st.fold)
        # tr_loader, vl_loader, ts_loader = dataloader_abide(self.st)

        opt = self.optimizer()
        writer = SummaryWriter(save_path + "/logs/")

        best_loss_epoch = 0
        best_loss_ep = np.inf
        best_acc = 0
        al_auc, al_acc, al_sen, al_spc = [], [], [], []
        al_auc_v, al_acc_v, al_sen_v, al_spc_v = [], [], [], []
        al_auc_t, al_acc_t, al_sen_t, al_spc_t = [], [], [], []
        for ep in range(self.st.epoch):
            # Train
            self.train_models()
            pred_all = np.empty(shape=(0), dtype=np.float32)
            real_all = np.empty(shape=(0), dtype=np.float32)
            prob_all = np.empty(shape=(0), dtype=np.float32)
            num_trn = tr_loader[0].shape[0]
            trn_shf = torch.randperm(num_trn)
            data_shf, labl_shf = tr_loader[0][trn_shf].to(self.st.device), tr_loader[1][trn_shf].to(self.st.device)
            for i in range(int(num_trn / self.st.bs)):
            # for bidx, batch in enumerate(zip(tr_loader)):
                opt.zero_grad()
                batchx, batchy = data_shf[i * self.st.bs:(i + 1) * (self.st.bs)], labl_shf[i * self.st.bs:(i + 1) * (self.st.bs)]
                # batchx, batchy = batch[0]  # [batch, 1, time, roi]
                # batchx, batchy, _, _ = batch[0]

                out1 = self.E2EBlock(batchx.unsqueeze(1).cuda())
                out1 = self.E2EBlock2(out1)
                out2 = self.E2NBlock(out1)
                out3 = self.N2GBlock(out2)
                logit = self.CLSBlock(out3)

                loss = self.cls_criterion(logit, batchy.cuda())
                l1_loss = ut.l1_regularization(self.st.wspars, self.CLSBlock, self.st.device)

                pred, prob = ut.prediction(logit)
                real_all = np.concatenate([real_all, batchy.detach().cpu().numpy()])
                pred_all = np.concatenate([pred_all, pred.detach().cpu().numpy()])
                prob_all = np.concatenate([prob_all, prob.detach().cpu().numpy()])

                """ Objective function """
                loss_all = loss + l1_loss
                loss_all.backward()
                opt.step()
                opt.zero_grad()
            #
            # if save:
            #     results = ut.cal_all_metric(real_all, pred_all, prob_all)
            #     al_auc.append(results[0]), al_acc.append(results[1]), al_sen.append(results[2]), al_spc.append(results[3])
            #     self.save_model(ckpt_path, ep)
            #     writer.add_scalars("Train", {"class": loss_all, 'ACC': results[1]}, ep)
            #     print(ep, results[1])

            with torch.no_grad():
                self.eval_models()
                for i in range(1):
                # for bidx_v, batch_v in enumerate(zip(vl_loader)):
                    batchx_v, batchy_v = vl_loader[0], vl_loader[1]  # [batch, 1, time, roi]
                    # batchx_v, batchy_v = batch_v[0]  # [batch, 1, time, roi]
                    # batchx_v, batchy_v, _, _ = batch_v[0]

                    out1_v = self.E2EBlock(batchx_v.unsqueeze(1).cuda())
                    out1_v = self.E2EBlock2(out1_v)
                    out2_v = self.E2NBlock(out1_v)
                    out3_v = self.N2GBlock(out2_v)
                    logit_v = self.CLSBlock(out3_v)
                    loss_v = self.cls_criterion(logit_v, batchy_v.cuda())

                pred_v, prob_v = ut.prediction(logit_v)
                results_v = ut.cal_all_metric(batchy_v.detach().cpu().numpy(), pred_v.detach().cpu().numpy(), prob_v.detach().cpu().numpy())
                if loss_v < best_loss_ep:
                    best_loss_ep = ep
                writer.add_scalars("Valid", {"class": loss_v, 'ACC': results_v[1]}, ep)
                print(ep, results_v[1])

                if np.abs(results_v[2] - results_v[3]) < 0.1:
                    al_auc_v.append(results_v[0]), al_acc_v.append(results_v[1])
                    al_sen_v.append(results_v[2]), al_spc_v.append(results_v[3])
                    for i in range(1):
                    # for bidx_t, batch_t in enumerate(zip(ts_loader)):
                    #     batchx_t, batchy_t = batch_t[0]
                        batchx_t, batchy_t = ts_loader[0], ts_loader[1]  # [batch, 1, time, roi]
                        # batchx_t, batchy_t, _, _ = batch_t[0]  # [batch, 1, time, roi]
                        out1_t = self.E2EBlock(batchx_t.unsqueeze(1).cuda())
                        out1_t = self.E2EBlock2(out1_t)
                        out2_t = self.E2NBlock(out1_t)
                        out3_t = self.N2GBlock(out2_t)
                        logit_t = self.CLSBlock(out3_t)
                        loss_t = self.cls_criterion(logit_t, batchy_t.cuda())
                    pred_t, prob_t = ut.prediction(logit_t)
                    results_t = ut.cal_all_metric(batchy_t.detach().cpu().numpy(), pred_t.detach().cpu().numpy(),
                                                  prob_t.detach().cpu().numpy())
                    al_auc_t.append(results_t[0]), al_acc_t.append(results_t[1]), al_sen_t.append(results_t[2]), al_spc_t.append(results_t[3])
                    writer.add_scalars("Test", {"class": loss_t, 'ACC': results_t[1]}, ep)
                    print(ep, results_t[1])
                    print("=============")

        best_auc_ep = ut.bestepoch_acc(al_auc_v)
        best_acc_ep = ut.bestepoch_acc(al_acc_v)

        final = [al_auc_t[best_auc_ep], al_acc_t[best_auc_ep], al_sen_t[best_auc_ep], al_spc_t[best_auc_ep],
                 al_auc_t[best_acc_ep], al_acc_t[best_acc_ep], al_sen_t[best_acc_ep], al_spc_t[best_acc_ep],
                 al_auc_t[best_loss_ep], al_acc_t[best_loss_ep], al_sen_t[best_loss_ep], al_spc_t[best_loss_ep]]

        return [best_auc_ep, best_acc_ep, best_loss_ep], final

    # def save_model(self, ckpt_path, ep):
    #     torch.save({"E2EBlock": self.E2EBlock.state_dict(),
    #                 "E2EBlock2": self.E2EBlock2.state_dict(),
    #                 "E2NBlock": self.E2NBlock.state_dict(),
    #                 "N2GBlock": self.N2GBlock.state_dict(),
    #                 "CLSBlock": self.CLSBlock.state_dict()}, ckpt_path + "/E%d.pt" % (ep))

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """ Fix the seed """
    seed1 = 5930
    np.random.seed(seed1)
    os.environ["PYTHONHASHSEED"] = str(seed1)
    torch.cuda.manual_seed(seed1)
    torch.cuda.manual_seed_all(seed1)  # if you are using multi-GPU
    torch.manual_seed(seed1)
    random.seed(seed1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

    """ Path """
    main_path = "/DataCommon2/eskang/Transformer/Journal/202308_ABIDE/BrainNetCNN/"
    # main_path = "/DataCommon/eskang/abide_test/"
    case_path = main_path + "E2E{}_E2N{}_N2G{}_H{}_l1{}_l2{}_lr{}/rp{}/".format(args.E2Efilters, args.E2Nfilters,
                                                                          args.N2Gfilters, args.clshidden,
                                                                          args.wspars, args.wdecay, args.lr,
                                                                                         args.rp)

    save_path = ut.create_dir(case_path + "cv{}/".format(args.fold))
    ckpt_path = ut.create_dir(case_path + "cv{}/ckpt/".format(args.fold))
    ut.save_args(case_path, args)


    exp = experiment(args)
    bep, results_f = exp.training(save_path, ckpt_path, save=False)

    if args.fold == 1:
        arr_auc = np.zeros((5, 4))
        arr_acc = np.zeros((5, 4))
        arr_los = np.zeros((5, 4))
    else:
        arr_auc = np.load(case_path + 'results_fold.npz')['VAL_AUC']
        arr_acc = np.load(case_path + 'results_fold.npz')['VAL_ACC']
        arr_los = np.load(case_path + 'results_fold.npz')['VAL_LOS']

    arr_auc[args.fold - 1, 0], arr_acc[args.fold - 1, 0], arr_los[args.fold - 1, 0] = results_f[0], results_f[4], results_f[8]
    arr_auc[args.fold - 1, 1], arr_acc[args.fold - 1, 1], arr_los[args.fold - 1, 1] = results_f[1], results_f[5], results_f[9]
    arr_auc[args.fold - 1, 2], arr_acc[args.fold - 1, 2], arr_los[args.fold - 1, 2] = results_f[2], results_f[6], results_f[10]
    arr_auc[args.fold - 1, 3], arr_acc[args.fold - 1, 3], arr_los[args.fold - 1, 3] = results_f[3], results_f[7], results_f[11]

    if args.fold == 5:
        summary = open(case_path + 'summary.txt', 'w', encoding='utf-8')
        summary.write("Criterion\tAUC\tACC\tSEN\tSPC\n")
        auc_fmean = np.array(arr_auc).mean(0)
        acc_fmean = np.array(arr_acc).mean(0)
        los_fmean = np.array(arr_los).mean(0)

        summary.write('VAL_AUC\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(auc_fmean[0], auc_fmean[1], auc_fmean[2],
                                                                       auc_fmean[3]) + '\n')
        summary.write('VAL_ACC\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(acc_fmean[0], acc_fmean[1], acc_fmean[2],
                                                                       acc_fmean[3]) + '\n')
        summary.write('VAL_LOS\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(los_fmean[0], los_fmean[1], los_fmean[2],
                                                                       los_fmean[3]) + '\n')

    np.savez(case_path + 'results_fold.npz', VAL_AUC=arr_auc, VAL_ACC=arr_acc, VAL_LOS=arr_los)

    # ut.remove_file_train(ckpt_path, bep, args.epoch)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="BrainNetCNN")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    # parser.add_argument("--window", type=int, default=30)
    # parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--combat", type=str2bool, default='True')
    parser.add_argument("--prec_glob", type=str, default='filt_noglobal', choices=['filt_global', 'filt_noglobal'])
    parser.add_argument("--prec_type", type=str, default='dparsf', choices=['cpac', 'dparsf'])
    parser.add_argument("--atlas", type=str, default='cc200', choices=['aal', 'ho', 'cc200'])

    parser.add_argument("--seed_rp", type=int, default=1210)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--rp", type=int, default=2)

    parser.add_argument("--nroi", type=int, default=200)
    parser.add_argument("--input_size", type=int, default=200)
    parser.add_argument("--E2Efilters", type=int, default=32)
    parser.add_argument("--E2Nfilters", type=int, default=64)
    parser.add_argument("--N2Gfilters", type=int, default=30)
    parser.add_argument("--clshidden", type=int, default=128)
    parser.add_argument("--dp", type=float, default=0.5)  # dropout
    parser.add_argument("--wspars", type=float, default=0)  # dropout
    parser.add_argument("--wdecay", type=float, default=5e-3)  # dropout
    ARGS = parser.parse_args()
    main(ARGS)