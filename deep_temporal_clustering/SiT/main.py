import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import datetime
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from losses import SimCLR
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets_utils
import prepare_datasets
from engine import train_one_epoch

import utils
import vision_transformer_SiT as vits
from vision_transformer_SiT import RECHead, ContrastiveHead


def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help="Name of architecture to train.")
    parser.add_argument('--img_size', default=32, type=int, help="Input size to the Transformer.")
    parser.add_argument('--patch_size', default=8, type=int, help="patch size to the Transformer.")

    ##################### Pre-text tasks
    # Reconstruction parameters TASK 1
    parser.add_argument('--rec_head', default=1, type=float, help="use recons head or not")
    parser.add_argument('--drop_perc', type=float, default=0.7, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.35, help='Replace X percentage of the input image')

    parser.add_argument('--drop_align', type=int, default=0, help='Set to patch size to align corruption with patch size')
    parser.add_argument('--drop_type', type=str, default='noise', help='Type of alien concept')
    parser.add_argument('--drop_only', type=int, default=1, help='consider only the loss from corrupted patches')

    # SimCLR parameters TASK 3
    parser.add_argument('--simCLR_head', default=0, type=float, help="use simclr head or not")
    parser.add_argument('--simCLR_tempr', default=0.5, type=float, help="Simclr tempreture")
    parser.add_argument('--simCLR_outdim', default=256, type=int, help="Dimensionality of the head output.")

    # Rotation parameters TASK 2
    parser.add_argument('--rot_head', default=0, type=float, help="use rotation head or not")

    # Usage of uncertainty
    parser.add_argument('--use_uncert', default=1, type=float, help="Using uncertainty for multi-task learning")
    #####################################

    # Dataset
    parser.add_argument('--data_set', default='CIFAR10', type=str,
                        choices=['STL10', 'MNIST', 'CIFAR10', 'CIFAR100', 'Flowers', 'Aircraft', 'Cars', 'ImageNet',
                                 'TinyImageNet', 'Pets'],
                        help='Name of the dataset.')
    parser.add_argument('--data_location', default='.', type=str, help='Dataset location.')

    # Hyper-parameters
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=500, type=int, help="Number of epochs of training.")

    parser.add_argument('--weight_decay', type=float, default=0.04, help="weight decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final value of the weight decay.")

    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Target LR at the end of optimization.")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="Whether or not to use half precision for training.")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm.")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")

    # Misc
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
    return parser


def train_SiTv2(args):
    # utils.init_distributed_mode(args)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    args.epochs += 1

    ################ Preparing Dataset
    # transform = datasets_utils.DataAugmentationSiT(args)
    dataset, _ = prepare_datasets.build_dataset(args, True, trnsfrm=transforms.ToTensor(), training_mode='SSL')
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    img, label = next(iter(data_loader))
    print(img.shape, label.shape)

    print(f"==> {args.data_set} training set is loaded.")
    print(f"-------> The dataset consists of {len(dataset)} images.")
    print(f"-------> The dataset size: {img[0,0,:,:].shape}")





    ################ Create Transformer
    SiT_model = vits.__dict__[args.model](img_size=[args.img_size])  ### ENCODER ###
    n_params = sum(p.numel() for p in SiT_model.parameters() if p.requires_grad)

    SiT_model = FullpiplineSiT(args, SiT_model)
    SiT_model = SiT_model.cuda()

    # SiT_model = nn.parallel.DistributedDataParallel(SiT_model, device_ids=[args.gpu_id])
    print(f"==> {args.model} model is created.")
    print(f"-------> The model has {n_params} parameters.")

    ################ optimization ...
    # Create Optimizer
    params_groups = utils.get_params_groups(SiT_model)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs



    # Initialize schedulers
    # lr_schedule = utils.cosine_scheduler(args.lr * (args.batch_size * utils.get_world_size()) / 256.,
    #                                      args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    # wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader))

    ################ Resume Training if exist
    to_restore = {"epoch": 0} # dict
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth"),
    #     run_variables=to_restore, SiT_model=SiT_model,
    #     optimizer=optimizer, fp16_scaler=fp16_scaler)
    start_epoch = to_restore["epoch"]

    ################ Training
    start_time = time.time()
    print(f"==> Start training from epoch {start_epoch}")



    for epoch in range(start_epoch, args.epochs):
        all_loss = 0
        # data_loader.sampler.set_epoch(epoch)

        # Train an epoch
        train_stats = train_one_epoch(SiT_model, data_loader, optimizer, epoch, args, all_loss)

        save_dict = {'SiT_model': SiT_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args}
        #
        #
        # utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        # if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        #     utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        # if utils.is_main_process():
        #     with (Path(args.output_dir) / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


class FullpiplineSiT(nn.Module):

    def __init__(self, args, backbone):
        super(FullpiplineSiT, self).__init__()

        embed_dim = backbone.embed_dim

        self.rec = args.rec_head
        self.drop_only = args.drop_only

        self.rot = args.rot_head
        self.simCLR = args.simCLR_head

        # create full model
        self.backbone = backbone  #### ENCODER ####
        self.rec_head = RECHead(embed_dim, patch_size=args.patch_size) if (args.rec_head == 1) else nn.Identity()  #### DECODER ####
        # self.contr_head = ContrastiveHead(embed_dim, args.simCLR_outdim) if (args.simCLR_head == 1) else nn.Identity()
        # self.rot_head = ContrastiveHead(embed_dim, 4) if (args.rot_head == 1) else nn.Identity()

        # create learnable parameters for the MTL task Multi-Task Learning
        self.use_uncert = args.use_uncert
        self.rec_w = nn.Parameter(torch.tensor([1.0])) if (args.rec_head == 1 and args.use_uncert == 1) else 0
        # self.contr_w = nn.Parameter(torch.tensor([1.0])) if (args.simCLR_head == 1 and args.use_uncert == 1) else 0
        # self.rot_w = nn.Parameter(torch.tensor([1.0])) if (args.rot_head == 1 and args.use_uncert == 1) else 0

        # self.simCLR_loss = SimCLR(args.simCLR_tempr)
        # self.rot_loss = torch.nn.CrossEntropyLoss()

    def uncertaintyLoss(self, loss_, scalar_):
        loss_w = (0.5 / (scalar_ ** 2) * loss_ + torch.log(1 + scalar_ ** 2)) if (self.use_uncert == 1) else loss_
        return loss_w

    def forward(self, im):

        encoded_out = self.backbone(im)  # torch.Size([64, 3, 32, 32]) -> torch.Size([64, 18, 768])

        # calculate rotation loss
        # if self.rot == 1:
        #     loss_rot = self.rot_loss(self.rot_head(x[:, 0]), torch.cat(rot[:2]))
        #     loss_rot_w = self.uncertaintyLoss(loss_rot, self.rot_w)
        # else:
        #     loss_rot, loss_rot_w = 0, 0
        #
        # # calculate contrastive loss
        # if self.simCLR == 1:
        #     loss_contr = self.simCLR_loss(self.contr_head(x[:, 1]))
        #     loss_contr_w = self.uncertaintyLoss(loss_contr, self.contr_w)
        # else:
        #     loss_contr, loss_contr_w = 0, 0



        # calculate reconstruction loss
        if self.rec == 1:
            recons_imgs = self.rec_head(encoded_out[:, 2:])  # torch.Size([64, 16, 768])가 rec_head로 들어간다 -> torch.Size([64, 3, 32, 32])
            recloss = F.l1_loss(recons_imgs, im, reduction='none')
            # loss_rec = recloss[torch.cat(im_mask[0:]) == 1].mean() if (self.drop_only == 1) else recloss.mean()
            loss_rec = recloss.mean()

            loss_rec_w = self.uncertaintyLoss(loss_rec, self.rec_w)

        else:
            loss_rec, loss_rec_w = 0, 0
            recons_imgs = None

        return loss_rec, loss_rec_w, recons_imgs, im


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiTv2', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiTv2(args)
