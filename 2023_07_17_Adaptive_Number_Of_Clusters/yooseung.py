import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import os
from torchvision.transforms import Compose, ToTensor
# from transform import source_transform, adapt_transform
from tqdm import tqdm
import model
from utils import loop_iterable, set_requires_grad
from utils import GradientReversal, loop_iterable
from fourier import fft_mixup, fft_amp_mix, mixup, fft_amp_replace
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from dataset import minmax_scaler
import dataset
import GPUtil
import random
import fft_model

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="4")
    # gpu id number

    parser.add_argument('--model_name', type=str, default='trained_inter')
    # model save name

    parser.add_argument('--batch_size', type=int, default=4)
    # batch size
    parser.add_argument('--init_lr', type=float, default=1e-4)
    # learning rate
    parser.add_argument('--epochs', type=int, default=100)
    # number of epochs
    parser.add_argument('--source', type=str, default='adni1')
    # source domain
    parser.add_argument('--target', type=str, default='adni2')

    args, _ = parser.parse_known_args()
    return args


args = parse_args()

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_FILE = 'models/adcn/source_intra_fft_{}_acc.pt'.format(args.source)

net = fft_model.Net(dropout=0.5)
net.to(device)
net.load_state_dict(torch.load(MODEL_FILE))
feature_extractor = net.feature_extractor
clf = net.classifier

discriminator = nn.Sequential(
    GradientReversal(lambda_=0.1),
    nn.Linear(128*2*3*2, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

optimizer = torch.optim.Adam(list(discriminator.parameters()) + list(net.parameters()), lr=args.init_lr)
half_batch = args.batch_size // 2

source_path = "/DataRead/ysshin/task/{}_data_adcn.npy".format(args.source)
src_label_path ="/DataRead/ysshin/task/{}_label_adcn.npy".format(args.source)
target_path = "/DataRead/ysshin/task/data_8_2/{}_data_adcn_82.npy".format(args.target)
trg_label_path ="/DataRead/ysshin/task/data_8_2/{}_label_adcn_82.npy".format(args.target)

def do_epoch(model, source_loader, target_loader, optim=None):
    batches = zip(source_loader, target_loader)
    n_batches = min(len(source_loader), len(target_loader))
    total_loss = 0
    total_acc = 0
    total_size = 0
    y_true = []
    y_pred = []

    for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
        target_f = fft_mixup(source_x, target_x)
        source_x = minmax_scaler(source_x)
        target_f = minmax_scaler(target_f)

        x = torch.cat([source_x, target_f])
        x = x.to(device)

        domain_y = torch.cat([torch.ones(source_x.shape[0]), torch.zeros(target_f.shape[0])])
        domain_y = domain_y.to(device)
        label_y = source_labels.to(device)
        features = feature_extractor(x).view(x.shape[0], -1)
        domain_preds = discriminator(features).squeeze()
        label_preds = clf(features[:source_x.shape[0]])
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
        label_loss = F.cross_entropy(label_preds, label_y)
        attention = net.SpatialGate.attention.squeeze()
        s = attention[:2, :, :, :]
        t = attention[2:, :, :, :]
        delta = s - t
        att_loss = 0.5 * (delta[0].pow(2).sum() + delta[1].pow(2).sum()) / (2 * 193 * 229 * 193)

        loss = label_loss + domain_loss + att_loss

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()

        _, pred = torch.max(label_preds, 1)
        total_size += label_y.size(0)
        total_acc += (pred == label_y).sum().item()

        y_pred.append(pred.cpu().detach().tolist())
        y_true.append(label_y.cpu().detach().tolist())

    auc_score = roc_auc_score(sum(y_true, []), sum(y_pred, []), average="macro")
    acc_score = total_acc / total_size
    mean_loss = total_loss / n_batches

    return mean_loss, auc_score, acc_score

source_dataset = dataset.MyDataset(data_path=source_path, label_path=src_label_path, transform=None)
target_dataset = dataset.MyDataset(data_path=target_path, label_path=trg_label_path, transform=None)

#skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(source_dataset)):
    print('Fold {}'.format(fold))
    print('-----------------------------')

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=half_batch, drop_last=True,
                                                sampler=SubsetRandomSampler(train_idx))

    val_loader = torch.utils.data.DataLoader(source_dataset, batch_size=half_batch, drop_last=True,
                                             sampler=SubsetRandomSampler(val_idx))


    target_loader = torch.utils.data.DataLoader(target_dataset, half_batch, shuffle=True)

    best_auc = 0

    for epoch in range(0, args.epochs):
        net.train()
        train_loss, train_auc, train_acc = do_epoch(net, source_loader, target_loader, optim=optimizer)

        net.eval()
        with torch.no_grad():
            val_loss, val_auc, val_acc = do_epoch(net, val_loader, target_loader, optim=None)

        tqdm.write(
            f'Epoch {epoch:03d}: train_loss = {train_loss:.4f} , train_acc = {train_acc:.4f} , train_auc = {train_auc:.4f} ---- '
            f'val_loss = {val_loss: .4f} , val_acc = {val_acc:.4f} , val_auc = {val_auc:.4f}')

        if val_auc > best_auc:
            tqdm.write(f'Fold {fold} Saving model... Selection: val_auc')
            best_auc = val_auc
            torch.save(net.state_dict(), "models/adcn/folds/{}_{}/{}_auc_fold{}.pt".format(args.source, args.target, args.model_name, fold))

        # if val_loss < best_loss:
        #     tqdm.write(f'Saving model... Selection: val_loss')
        #     best_loss = val_loss
        #     torch.save(net.state_dict(), "models/{}_loss.pt".format(args.model_name))