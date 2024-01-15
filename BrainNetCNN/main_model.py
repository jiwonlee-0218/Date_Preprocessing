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
from einops import rearrange
import module_brainnet as md
from dataload_abide_202308 import dataloader





''' transformer '''
class BrainNetCNN(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args):
        super(BrainNetCNN, self).__init__()


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


    def forward(self, inputs):

        inputs = torch.unsqueeze(inputs, 1)  # torch.Size([8, 1, 176, 116])

        out1 = self.E2EBlock(inputs)
        out1 = self.E2EBlock2(out1)
        out2 = self.E2NBlock(out1)
        out3 = self.N2GBlock(out2)
        logit = self.CLSBlock(out3)



        return logit


    def flatten_upper_triang_values(self, X):  ##torch.Size([116, 116])


        mask = torch.triu(torch.ones_like(X), diagonal=1)
        A_triu_1d = X[mask == 1]

        return A_triu_1d


    def corrcoef(self, x):
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        return c


