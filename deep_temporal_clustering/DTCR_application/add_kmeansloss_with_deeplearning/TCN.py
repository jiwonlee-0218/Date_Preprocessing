import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision import datasets, transforms
import matplotlib.pyplot as plt



import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="EMOTION_6cluster", help="dataset name")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    # model args
    parser.add_argument("--model_name", default="TCN_practice",help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")

    # parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping: Maximal parameter gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=351, help="Epochs number of the autoencoder training",)
    parser.add_argument("--max_patience", type=int, default=15, help="The maximum patience for pre-training, above which we stop training.",)

    parser.add_argument("--lr_ae", type=float, default=0.01, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning',)
    # parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')
    # parser.add_argument("--ae_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--ae_models", default=None, help='full autoencoder weights')
    parser.add_argument("--autoencoder_test", default=None, help='full autoencoder weights')




    return parser



def data_generator(root, batch_size):
    train_set = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader




class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
                                nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                stride=stride, padding=0, dilation=dilation)
                                )

        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
                                nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                stride=stride, padding=0, dilation=dilation)
                                )
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.unsqueeze(2)
        out = self.net(x).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



# TCNModel(input_size= 116, output_size=10, num_channels=[20] * 2, kernel_size=3, dropout=0.25)
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)


    def forward(self, inputs):
        emb = self.tcn(inputs)
        y = self.decoder(emb)
        return emb, y


if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR'] #(1041, 150, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1041):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)


    # train, validation, test
    data_label = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label = data_label['label_list_LR']  #(1041, 150)
    # label = label[:1041]

    # number of clusters
    args.n_clusters = len(np.unique(label))
    args.timeseries = label.shape[1]


    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(sample, label, random_state=42, test_size=0.2)


    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)

    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    train_dl = DataLoader(real_train_ds, batch_size=16)
    real_train_dl = DataLoader(real_train_ds, batch_size=1)

    real_test_ds = TensorDataset(real_test_data, real_test_label)
    test_dl = DataLoader(real_test_ds, batch_size=16)
    real_test_dl = DataLoader(real_test_ds, batch_size=1)

    root = './data/mnist'
    batch_size = args.batch_size
    train_loader, test_loader = data_generator(root, batch_size)
    permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()



    input_channels = 116
    seq_length = args.timeseries

    model = TCNModel(input_size= input_channels, output_size= input_channels, num_channels=[64] * 2, kernel_size=3, dropout=0.25)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
    loss_ae = nn.MSELoss()



    for epoch in tqdm(range(args.epochs_ae)):

        # training
        model.train()
        all_loss = 0
        recon_loss = 0
        km_loss = 0

        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data = data.type(torch.FloatTensor).to(args.device)  #torch.Size([16batch, 150, 116])
        #     data = data.view(-1, input_channels, seq_length)
        #
        #     optimizer.zero_grad()  # 기울기에 대한 정보 초기화
        #     h, recon = model(data)
        #     loss_mse = loss_ae(data, recon)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차


        for batch_idx, (data, target) in enumerate(train_dl):
            data = data.type(torch.FloatTensor).to(args.device)  #torch.Size([16batch, 150, 116])
            data = data.permute(0, 2, 1)

            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            h, recon = model(data)
            loss_mse = loss_ae(data, recon)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차

