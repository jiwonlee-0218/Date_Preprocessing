import os

import warnings

warnings.filterwarnings("ignore")

import sys
import math
from pathlib import Path

import torch
import torchvision

import datasets_utils

import utils
import matplotlib.pyplot as plt


def train_one_epoch(SiT_model, data_loader, optimizer, epoch,  args, all_loss):
    # save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    # Path(save_recon).mkdir(parents=True, exist_ok=True)
    # bz = args.batch_size
    # plot_ = True if args.rec_head == 1 else False

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (im, _) in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule
        # it = len(data_loader) * epoch + it  # global training iteration
        # for i, param_group in enumerate(optimizer.param_groups):
        #     param_group["lr"] = lr_schedule[it]
        #     if i == 0:  # only the first group is regularized
        #         param_group["weight_decay"] = wd_schedule[it]

        # if args.drop_replace > 0:
        #     im_corr, im_mask = datasets_utils.GMML_replace_list(im, im_corr, im_mask, drop_type=args.drop_type,
        #                                                         max_replace=args.drop_replace, align=args.drop_align)

        # move to gpu
        # im = [im.cuda(non_blocking=True) for im in im]
        im = im.type(torch.FloatTensor).to(args.device)
        # rot = [r.type(torch.LongTensor).cuda(non_blocking=True) for r in rot]
        # im_corr = [c.cuda(non_blocking=True) for c in im_corr]
        # im_mask = [m.cuda(non_blocking=True) for m in im_mask]

        recons_l, recons_l_w, rec_imgs, orig_imgs = SiT_model(im)



        # -------------------------------------------------
        # if plot_ == True and utils.is_main_process():  # and args.saveckp_freq and epoch % args.saveckp_freq == 0:
        #     plot_ = False
        #     # validating: check the reconstructed images
        #     print_out = save_recon + '/epoch_' + str(epoch).zfill(5) + '.jpg'
        #     imagesToPrint = torch.cat([im[0][0: min(15, bz)].cpu(), im_corr[0][0: min(15, bz)].cpu(),
        #                                rec_imgs[0: min(15, bz)].cpu()], dim=0)
        #     torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, range=(-1, 1))

        loss = recons_l_w

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # model update
        optimizer.zero_grad()
        param_norms = None
        loss.backward()
        if args.clip_grad:
            param_norms = utils.clip_gradients(SiT_model, args.clip_grad)
        optimizer.step()

        all_loss += loss.item()


        # logging
        torch.cuda.synchronize()

        metric_logger.update(recons_l=recons_l.item() if hasattr(recons_l, 'item') else 0.)
        metric_logger.update(recons_l_w=recons_l_w.item() if hasattr(recons_l, 'item') else 0.)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])


        if it == 781:
            plt.imshow(rec_imgs[0].permute(1, 2, 0).cpu().detach())
            plt.savefig('/home/jwlee/HMM/deep_temporal_clustering/SiT/result/recon{}.png'.format(epoch))
            plt.close()
            plt.imshow(orig_imgs[0].permute(1, 2, 0).cpu().detach())
            plt.savefig('/home/jwlee/HMM/deep_temporal_clustering/SiT/result/orig_imgs{}.png'.format(epoch))
            plt.close()

    train_loss = all_loss / (it + 1)
    print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}