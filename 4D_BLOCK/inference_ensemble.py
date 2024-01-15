import random
import pandas as pd
import numpy as np
import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="0")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test = pd.read_csv('/DataRead/mjy/Dacon/open/test.csv')

"""
Hyperparameter Setting
"""
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':128,
    'SEED':41
}



path_1 = "/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115/fold_0/Epoch_4loss_0.03428647781793888.pth"
path_2 = "/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115/fold_1/Epoch_5loss_0.037788069466702066.pth"
path_3 = "/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115/fold_2/Epoch_9loss_0.040516047729537465.pth"
path_4 = "/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115/fold_3/Epoch_5loss_0.042403484927490354.pth"
path_5 = "/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115/fold_4/Epoch_2loss_0.043155661855752654.pth"








"""
CustomDataset
"""
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = img_path[1:]

        # image = cv2.imread(img_path)
        image = cv2.imread('/DataRead/mjy/Dacon/open' + img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = torch.FloatTensor(self.label_list[index])
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)


train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])




test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)





"""
Model Define
"""

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        # self.backbone = models.efficientnet_b0(pretrained=True)
        # self.backbone = models.convnext_base(pretrained=True)
        self.backbone = models.convnext_large(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.sigmoid(self.classifier(x))
        return x




# import models
# model = BaseModel()

model1 = BaseModel()
model1.load_state_dict(torch.load(path_1, map_location='cpu'))
model1.to(device)
model1.eval()

model2 = BaseModel()
model2.load_state_dict(torch.load(path_2, map_location='cpu'))
model2.to(device)
model2.eval()

model3 = BaseModel()
model3.load_state_dict(torch.load(path_3, map_location='cpu'))
model3.to(device)
model3.eval()

model4 = BaseModel()
model4.load_state_dict(torch.load(path_4, map_location='cpu'))
model4.to(device)
model4.eval()

model5 = BaseModel()
model5.load_state_dict(torch.load(path_5, map_location='cpu'))
model5.to(device)
model5.eval()











def inference(test_loader, device):

    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            probs1 = model1(imgs)
            probs2 = model2(imgs)
            probs3 = model3(imgs)
            probs4 = model4(imgs)
            probs5 = model5(imgs)

            probs1 = probs1.cpu().detach().numpy()
            probs2 = probs2.cpu().detach().numpy()
            probs3 = probs3.cpu().detach().numpy()
            probs4 = probs4.cpu().detach().numpy()
            probs5 = probs5.cpu().detach().numpy()

            probs = probs1 + probs2 + probs3 + probs4 + probs5
            probs_total = probs / 5

            preds = probs_total > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return predictions

preds = inference(test_loader, device)


submit = pd.read_csv('/DataRead/mjy/Dacon/open/submission/sample_submission.csv')
submit.iloc[:,1:] = preds
submit.head()
submit.to_csv('/DataRead/mjy/Dacon/open/submission/ConvNext_kfold_0115_submit.csv', index=False)

