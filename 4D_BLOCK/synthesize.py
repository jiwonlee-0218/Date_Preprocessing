import os
import random
import cv2
import numpy as np
from glob import glob
import torch
import GPUtil
from torchvision.datasets import LSUN
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.utils.data


##### GPU Configuration
gpu_id = 2
if gpu_id == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = devices
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH = '/DataRead/mjy/Dacon/open'
BACKGROUND = "/DataRead/mjy/Dacon/open/JW/model/BACKGROUND"
save_path = '/DataRead/mjy/Dacon/open/test_synthesized'
FOREGROUND = '/DataRead/mjy/Dacon/open/JW/model/FOREGROUND'
os.makedirs(save_path, exist_ok=True)

def onChange(pos):
    pass


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train'):

    # Compose transforms

    # Get dataset
    lsun_dset = LSUN(root=path_to_data, classes=[dataset])
    bg_idx_list = list(range(len(lsun_dset)))
    num = np.random.choice(bg_idx_list)
    a, b = lsun_dset[num]
    a = a.resize((400, 400))
    a.save(os.path.join(BACKGROUND, f'background_{num}.jpg'))
    aa = cv2.imread(os.path.join(BACKGROUND, f'background_{num}.jpg'))
    # Create dataloader
    return aa



img_list = sorted(glob(os.path.join(DATA_PATH, 'test/*.jpg')))

# cv2.namedWindow("test")
# cv2.createTrackbar("threshold", "test", 0, 255, onChange)
# cv2.setTrackbarPos("threshold", "test", 127)

for img_l in img_list[:]:
    file_name = img_l.split('/')[-1]
    print(file_name)
    img = cv2.imread(img_l)
    # cv2.imwrite(os.path.join(save_path, f'{file_name}'), img)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (60, 40, 300, 320)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    # b, g, r = cv2.split(img)
    # rgba = [b, g, r, alpha]
    # dst = cv2.merge(rgba, 4)
    #
    # cv2.imwrite(os.path.join(save_path, f'new_{file_name}'), dst)  ## dst == foreground

    ## background

    bb = get_lsun_dataloader(path_to_data='/DataCommon3/jelee/practice/lsun-master/', dataset='classroom_train')



    # bb = cv2.imread(BACKGROUND)
    background_img = cv2.resize(bb, dsize=(400, 400))
    # cv2.imwrite('/DataRead/mjy/Dacon/open/JW/model/background.jpg', background_img)

    alphaalpha = alpha / 255.
    alphaalpha = np.repeat(np.expand_dims(alphaalpha, axis=2), 3, axis=2) #alphaalpha.shape (400, 400, 3)
    foreground = cv2.multiply(alphaalpha, img.astype(float))  #(400, 400, 3)
    background = cv2.multiply(1. - alphaalpha, background_img.astype(float))  #(400, 400, 3)

    cv2.imwrite(os.path.join(FOREGROUND, f'foreground_{file_name}'), foreground)
    result = cv2.add(foreground, background).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path, f'result_{file_name}'), result)