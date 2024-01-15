import os
import numpy as np
import random
import sys


import torch
import torch.nn as nn
from settings import setting as setting
import dataset.dataload as ld

from models.model_adaptive_mask import AdaptiveMask
from models.model_prototype import ProtoClassifier
from models.model_relation import FCTransformer
from models.model_decoder import CLS_Decoder
import utilss.load as lo
import utilss.make as mk
from experiments.exp import train_fc, train_pfc, evaluate



def main(args, paths_all, f):

    gpu_id = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)

    model_df = AdaptiveMask(args).to(device)
    model_tf = FCTransformer(args).to(device)
    model_cl = ProtoClassifier(args).to(device)
    model_dc = CLS_Decoder(args).to(device)

    train_data, valid_data, test_data = ld.dataloader(args, f)
    _, mpath = paths_all

    model_list = [model_df, model_tf, model_cl, model_dc]
    model_list, optimizer, optimizer2 = lo.load_optimizer(args, model_list)

    criterion_cls = nn.CrossEntropyLoss(reduction="mean").to(device)
    criterion_rec = nn.L1Loss(reduction="mean").to(device)
    loss_list = [criterion_cls, criterion_rec]

    best = float("inf")
    val_result = [[0]]
    for epoch in range(1, args.epoch + 1):

        if val_result[0][0] > 0.6 and epoch % 2 == 0: # After training prototypes to some extent
            # Step 2
            train_pfc(args, device, model_list, optimizer2, train_data, loss_list)
            val_loss, val_result = evaluate(args, device, epoch, model_list, valid_data, loss_list, 'val')

        else:
            # Step 1
            train_fc(args, device, model_list, optimizer, train_data, loss_list)
            val_loss, val_result = evaluate(args, device, epoch, model_list, valid_data, loss_list, 'val')

        if val_loss[0] < best:
            best = val_loss[0]
            best_ep = epoch
            tst_loss, tst_result = evaluate(args, device, epoch, model_list, valid_data, loss_list, 'tst')
            best_perf = tst_result[0]

            # SAVE
            # torch.save({'model_df': model_df.state_dict(),
            #             'model_tf': model_tf.state_dict(),
            #             'model_cl': model_cl.state_dict(),
            #             'model_dc': model_dc.state_dict()}, mpath + "best_model.pt")

    # AUC, ACC, SEN, SPC
    final = [best_ep, best_perf[1], best_perf[0], best_perf[2], best_perf[3]]

    return final


if __name__ == "__main__":

    args = setting.get_args()
    pre_path = './Set_the_Path/'
    paths = mk.mk_paths(args, args.fold, pre_path)

    final = main(args, paths, args.fold)
    print("Test Results", final)
