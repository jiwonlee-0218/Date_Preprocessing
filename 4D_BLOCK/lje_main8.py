import os
import GPUtil
from datetime import datetime, timedelta
from glob import glob
import random
import pandas as pd
import numpy as np
import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# import torchvision.models as models
from torchvision.models import convnext_large as m
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
# from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
# from transformers import AdamW
import warnings
warnings.filterwarnings(action='ignore')

"""
main8은 main7 기반으로 augmentation들을 대폭 추가했어
HorizontalFlip, RandomContrast, RandomBrightness, ColorJitter 적용함
scheduler도 짜잘하게 바꿈
나머진 동일
"""


##### GPU Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="3")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##### Hyperparameter Setting
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':2e-4, # from 3e-4 to 2e-4
    'BATCH_SIZE':128,
    # 'BATCH_SIZE':64,
    'SEED':41,
    'NUM_WORKERS':5,
    'PATIENCE':8,
    'NUM_FOLDS':5
}
path = '/DataRead/mjy/Dacon/open'
save_path = '/home/jwlee/HMM/4D_BLOCK/model'
os.makedirs(save_path, exist_ok=True)

##### Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


##### CustomDataset
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_name = img_path.split('/')[-1]

        if self.label_list is not None:
            # image = cv2.imread(img_path)
            image = cv2.imread(f'{path}/train_synthesized/{img_name}')
        else:
            image = cv2.imread(f'{path}/test/{img_name}')

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        item = {'image':image}
        if self.label_list is not None:
            item['label'] = torch.FloatTensor(self.label_list[index])
            # label = torch.FloatTensor(self.label_list[index])
            # return image, label
        # else:
        #     return image
        return item

    def __len__(self):
        return len(self.img_path_list)


##### Model Define
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.backbone = models.efficientnet_b0(pretrained=True)
        # self.backbone = models.convnext_base(pretrained=True)
        # self.backbone = models.convnext_large(pretrained=True)
        # self.backbone = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224-22k-1k")
        self.backbone = m(pretrained=True)
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        # x = self.backbone(x).logits
        x = F.sigmoid(self.classifier(x))
        return x


##### Train
def train(df_train, df_val, model, optimizer, scheduler, device, memo):
    datetime_train = (datetime.today() + timedelta(hours=9)).strftime('%Y%m%d_%H%M%S')
    model_path = f'{save_path}/{datetime_train}{memo}.pth'

    train_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.RandomContrast(p=0.5),
        A.RandomBrightness(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])

    train_dataset = CustomDataset(df_train['img_path'].values, df_train.iloc[:,2:12].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'])

    val_dataset = CustomDataset(df_val['img_path'].values, df_val.iloc[:,2:12].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

    model.to(device)
    # criterion = nn.BCELoss().to(device)
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    best_val_acc = 0
    best_val_loss = np.inf
    patience = 0

    # for epoch in range(1, CFG['EPOCHS'] + 1):
    for epoch in tqdm(range(1, CFG['EPOCHS'] + 1), desc='Epoch'):
        model.train()
        train_loss = []
        # for imgs, labels in tqdm(iter(train_loader)):
        for batch in tqdm(train_loader, total=len(train_loader), desc='Batch'):
            # imgs = imgs.float().to(device)
            # labels = labels.to(device)
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_loss)

        # if best_val_acc < _val_acc:
        #     best_val_acc = _val_acc
        #     print("Best model changed")

        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            # torch.save(model.state_dict(), '/DataRead/mjy/Dacon/open/submission/ConvNext_large.pth')
            torch.save(model.state_dict(), model_path)   #torch.save(model.state_dict(), PATH)
            print("!!!!!!!! Best model changed")
            patience = 0
        else:
            patience += 1
            print(f'''PATIENCE: {patience} of {CFG['PATIENCE']}''')
        if patience == CFG['PATIENCE']:
            break
    return model_path


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc='Valid'):
            # imgs = imgs.float().to(device)
            # labels = labels.to(device)
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            probs = model(imgs)

            loss = criterion(probs, labels)

            probs = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds = probs > 0.5
            batch_acc = (labels == preds).mean()

            val_acc.append(batch_acc)
            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)

    return _val_loss, _val_acc


def implementationTrain(df_train=None, df_val=None, memo=''):
    if df_train is None or df_val is None:
        assert df_train is None and df_val is None, 'ERROR'
        df = pd.read_csv(f'{path}/train.csv')
        df = df.sample(frac=1)
        train_len = int(len(df) * 0.8)
        df_train = df[:train_len]
        df_val = df[train_len:]

    model = Model()
    # model.eval()
    # optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    optimizer = AdamW(model.parameters(), lr= CFG["LEARNING_RATE"])
    # optimizer = AdamW(model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,threshold_mode='abs',min_lr=1e-8, verbose=True)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=514, num_training_steps=6425)
    # num_warmup_steps=(total_samples // bs) * 2 = (32994//128)*2 = 514
    # num_total_steps = (total_samples // bs) * n_epochs = (32994//128)*25 = 6425
    model_path = train(df_train, df_val, model, optimizer, scheduler, device, memo)
    return model_path


##### Test
def inference(model_path, device):
    df_test = pd.read_csv(f'{path}/test.csv')
    test_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])
    test_dataset = CustomDataset(df_test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = Model()
    model.to(device)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)

    model.eval()
    probabilities = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc='Test'):
            # imgs = imgs.float().to(device)
            # labels = labels.to(device)
            imgs = batch['image'].to(device)

            probs = model(imgs)

            probs = probs.cpu().detach().numpy()
            probabilities += probs.tolist()
            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return probabilities, predictions


##### K-fold cross validation
def cross_validation_train():
    df = pd.read_csv(f'{path}/train.csv')
    from sklearn.model_selection import StratifiedShuffleSplit
    # df['label_sum'] = df.iloc[:,2:].sum(axis=1)
    # minor_idx = df[(df['label_sum']==8)|(df['label_sum']==9)].index.values
    df['label'] = df.agg(lambda x: f"{x['A']}{x['B']}{x['C']}{x['D']}{x['E']}{x['F']}{x['G']}{x['H']}{x['I']}{x['J']}",
                         axis=1)
    minor_idx = df[(df['label'] == '1111111011') | (df['label'] == '1110011111') | (df['label'] == '1100000110') | (
                df['label'] == '0101000011')].index.values
    df.iloc[minor_idx, -1] = '999'
    split = StratifiedShuffleSplit(n_splits=CFG['NUM_FOLDS'], test_size=0.2)
    train_fold_idx = {}
    val_fold_idx = {}
    for fold, (train_idx, val_idx) in enumerate(split.split(df, df['label'])):
        train_fold_idx[fold+1] = train_idx
        val_fold_idx[fold+1] = val_idx
    # df = df.drop(columns=['label_sum', 'label'])
    df = df.drop(columns=['label'])

    model_path_list = []
    # for fold in range(1, CFG['NUM_FOLDS']+1):
    for fold in train_fold_idx.keys():
        print(f'--------------------FOLD {fold}')
        print(train_fold_idx[fold])
        df_train = df.iloc[train_fold_idx[fold]]
        df_val = df.iloc[val_fold_idx[fold]]
        model_path = implementationTrain(df_train, df_val, memo=f'(Fold{fold})')
        model_path_list.append(model_path)
    return model_path_list


def cross_validation_inference(model_path_list):
    submit = pd.read_csv(f'{path}/submission/sample_submission.csv')
    probs_all = np.zeros((len(submit), 10))
    cv_name = ''
    for model_path in model_path_list:
        model_name = model_path.split('/')[-1].split('.')[0]
        cv_name += f'{model_name}_'

        probs, preds = inference(model_path, device)
        probs_all += np.array(probs)

    probs_all /= len(model_path_list)
    preds_all = probs_all > 0.5
    # preds_all = preds_all.astype(int).tolist()
    preds_all = preds_all.astype(int)
    submit.iloc[:, 1:] = preds_all
    submit.head()
    submit.to_csv(f'{save_path}/{cv_name[:-1]}.csv', index=False)



# model_path = implementationTrain()
# model_path = '/DataCommon3/jelee/practice/model/20230118_151206(Fold2).pth'
# model_name = model_path.split('/')[-1].split('.')[0]
# probs, preds = inference(model_path, device)
#
# submit = pd.read_csv(f'{path}/submission/sample_submission.csv')
# submit.iloc[:,1:] = preds
# submit.head()
# submit.to_csv(f'{save_path}/{model_name}.csv', index=False)


model_path_list = cross_validation_train()
cross_validation_inference(model_path_list)
