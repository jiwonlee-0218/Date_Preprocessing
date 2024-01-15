from fMRI_train import *
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import argparse
import torch.nn as nn

import os


from pathlib import Path

import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F





import fMRI_vision_transformer_SiT as vits
from fMRI_vision_transformer_SiT import RECHead


def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="MOTOR", help="dataset name")
    parser.add_argument('--model', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help="Name of architecture to train.")
    parser.add_argument('--img_size', default=284, type=int, help="Input size to the Transformer.")
    parser.add_argument('--patch_size', default=1, type=int, help="patch size to the Transformer.")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    ##################### Pre-text tasks
    # Reconstruction parameters TASK 1
    parser.add_argument('--rec_head', default=1, type=float, help="use recons head or not")
    parser.add_argument('--drop_perc', type=float, default=0.7, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.35, help='Replace X percentage of the input image')

    parser.add_argument('--drop_align', type=int, default=0, help='Set to patch size to align corruption with patch size')
    parser.add_argument('--drop_type', type=str, default='noise', help='Type of alien concept')
    parser.add_argument('--drop_only', type=int, default=1, help='consider only the loss from corrupted patches')


    # Usage of uncertainty
    parser.add_argument('--use_uncert', default=0, type=float, help="Using uncertainty for multi-task learning")
    #####################################



    # Hyper-parameters
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=400, type=int, help="Number of epochs of training.")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum epochs numer of the full model training", )

    parser.add_argument('--weight_decay', type=float, default=0.04, help="weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--gamma", default=0.2, type=float, help="loss_weights")


    # Training/Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Gradient clipping: Maximal parameter gradient norm.")

    # Misc
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')

    # GPU
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")


    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/SiT/', )
    parser.add_argument("--model_name", default="LAB_TEST", help="model name")
    parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')
    # parser.add_argument("--weights", default='models_weights/', help='pre-trained autoencoder weights')
    parser.add_argument("--SiT_models", default=None, help='full autoencoder weights')
    # parser.add_argument("--SiT_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--SiT_test", default=None, help='full autoencoder weights')

    return parser









def writelog(file, line):
    file.write(line + '\n')
    print(line)






def test_function(args, X_test_data):


    print("Test pretraining model ...")
    if not os.path.exists(os.path.join(directory, 'SiT_models_pictures')):
        os.makedirs(os.path.join(directory, 'SiT_models_pictures'))

    path = os.path.join(directory, args.weights)
    full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
    full_path = full_path[-1]
    print("I got: " + full_path + " weights")

    test_SiT_model = vits.__dict__[args.model](img_size=[args.img_size])  ### ENCODER ###

    test_SiT_model = FullpiplineSiT(args, test_SiT_model)
    test_SiT_model = test_SiT_model.cuda()



    checkpoint = torch.load(full_path, map_location=args.device)
    test_SiT_model.load_state_dict(checkpoint['model_state_dict'])

    test_SiT_model.eval()
    with torch.no_grad():
        X_test_data = X_test_data.to(args.device)
        test_recons_l, test_rec_imgs, test_orig_imgs, test_represent = test_SiT_model(X_test_data)

        test_loss = test_recons_l
        print("Test SiT loss is : %.4f" % test_loss)

    f = open(os.path.join(os.path.join(directory, 'SiT_models_pictures/'), 'SiT\'s test loss.log'),'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'Test SiT loss loss is : %.4f' % test_loss)
    f.close()

    ########### TEST data's first subject ##############
    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(test_orig_imgs[1].cpu().T)
    plt.title("Original_X")
    plt.colorbar()
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'orig.png')

    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(test_rec_imgs[1].cpu().detach().T)
    plt.title("Reconstruction")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'recon.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(test_represent[1].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(test_represent[1][:150, :].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented_shorts.png')




def DTC_test_function(args, X_test_data, y_test_data):
    print("Test for DTC Model...")
    if not os.path.exists(os.path.join(directory, 'full_models_pictures')):
        os.makedirs(os.path.join(directory, 'full_models_pictures'))

    E_path = os.path.join(directory, args.SiT_models)
    E_full_path = sorted(glob.glob(E_path + '*.pt'), key=os.path.getctime)
    E_full_path = E_full_path[-1]
    print("I got: " + E_full_path + " embedding")

    dtc_model = torch.load(E_full_path)
    dtc_model = dtc_model.to(args.device)

    all_gt = []
    dtc_model.eval()
    with torch.no_grad():
        X_test_data = X_test_data.to(args.device)
        all_gt.append(y_test_data.cpu().detach())

        represent, x_reconstr, loss_KL, ARI, SiT_loss = dtc_model(X_test_data, all_gt)

        total_loss = args.gamma * SiT_loss + (1-args.gamma) * loss_KL

        print("Total Loss is : %.4f" % (total_loss), "MSE Loss is : %.4f" % (args.gamma * SiT_loss), "KL Loss is : %.4f" % ((1-args.gamma) * loss_KL),
              "ARI : %.4f" % (ARI))

    f = open(os.path.join(os.path.join(directory, 'full_models_pictures/'), 'DTC Model\'s test loss and ARI.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, 'loss weight for clustering: %s' % str((1-args.gamma)))
    writelog(f, '======================')
    writelog(f, 'DTC Model\'s test total loss is : %.4f,  test MSE loss is : %.4f,  test KL loss is : %.4f' % (
    total_loss, (args.gamma * SiT_loss), ((1-args.gamma) * loss_KL)))
    writelog(f, 'DTC Model\'s ARI is : %.4f' % ARI)
    f.close()


    ########### TEST data's first subject ##############
    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(X_test_data[1].cpu().T)
    plt.title("Original_X")
    plt.colorbar()
    plt.savefig(os.path.join(directory, 'full_models_pictures/') + 'orig.png')

    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(x_reconstr[1].cpu().detach().T)
    plt.title("Reconstruction")
    plt.savefig(os.path.join(directory, 'full_models_pictures/') + 'recon.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(represent[1].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'full_models_pictures/') + 'represented.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(represent[1][:150, :].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'full_models_pictures/') + 'represented_shorts.png')





class FullpiplineSiT(nn.Module):

    def __init__(self, args, backbone):
        super(FullpiplineSiT, self).__init__()

        embed_dim = backbone.embed_dim
        in_chans = backbone.in_chans

        self.rec = args.rec_head
        self.drop_only = args.drop_only


        # create full model
        self.backbone = backbone  #### ENCODER ####
        self.rec_head = RECHead(embed_dim, in_chans) if (args.rec_head == 1) else nn.Identity()  #### DECODER ####


        # create learnable parameters for the MTL task Multi-Task Learning
        self.use_uncert = args.use_uncert
        self.rec_w = nn.Parameter(torch.tensor([1.0])) if (args.rec_head == 1 and args.use_uncert == 1) else 0




    def uncertaintyLoss(self, loss_, scalar_):   # uncertainty를 사용하여 각 task loss의 weight를 주려한다.
        loss_w = (0.5 / (scalar_ ** 2) * loss_ + torch.log(1 + scalar_ ** 2)) if (self.use_uncert == 1) else loss_
        return loss_w

    def forward(self, x):

        encoded_out = self.backbone(x) # torch.Size([8, 284, 116]) -> torch.Size([8, 286, 80])



        # calculate reconstruction loss
        if self.rec == 1:
            recon = self.rec_head(encoded_out)  # torch.Size([64, 16, 768])가 rec_head로 들어간다 -> torch.Size([64, 3, 32, 32])
            recloss = F.mse_loss(recon, x) ############# loss

            # loss_rec = recloss[torch.cat(im_mask[0:]) == 1].mean() if (self.drop_only == 1) else recloss.mean()  masking 아직 안했으므로
            loss_rec_w = self.uncertaintyLoss(recloss, self.rec_w)


        else:
            loss_rec, loss_rec_w = 0, 0
            recons_imgs = None

        return recloss, recon, x, encoded_out













if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiTv2', parents=[get_args_parser()])
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)

    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, str(args.gamma),
                             'Epochs' + str(args.epochs) + '_BS_' + str(args.batch_size) + '_LR_' + str(
                                 args.lr) + '_wdcay_' + str(args.weight_decay))

    ################ Preparing Dataset
    data = np.load('/DataCommon/jwlee/MOTOR_LR/hcp_motor.npz')
    samples = data['tfMRI_MOTOR_LR']
    samples = samples[:1080]  # (1041, 284, 116)

    label = data['label_MOTOR_LR']
    label = label[:1080]  # (1041, 284)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1080):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(sample, label, random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)  ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)  ###

    # number of clusters
    args.n_clusters = len(np.unique(y_train))

    sub, label = next(iter(train_dl))
    print(sub.shape, label.shape)

    print(f"-------> The dataset consists of {len(train_dl) * args.batch_size} subjects.")  ## 864 subject
    print(f"-------> The dataset size: {sub[0, :, :].shape}")






    if args.weights is None and args.epochs > 0:
        train_SiTv2(args, train_dl, valid_dl, directory)

    if args.weights is not None and args.SiT_models is None and args.SiT_test is not None:
        test_function(args, X_test_data= X_test)


    if args.weights is not None and args.SiT_models is None and args.SiT_test is None:  #### weight ok, but model is none
        training_function(args, directory, train_dl, valid_dl)


    if args.weights is not None and args.SiT_models is not None:  ### weight ok, model ok
        DTC_test_function(args, X_test_data= X_test, y_test_data = y_test)


