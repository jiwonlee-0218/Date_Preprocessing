import os
import torch


timeseries_list, label_list, id = torch.load('/DataCommon2/jwlee/2023_07_24_Adaptive_Number_Of_Clusters/data/hcp-task_roi-aal.pth')


train_timeseries_list = []
train_label_list = []

test_timeseries_list = []
test_label_list = []



'EMOTION'
emotion_timeseries_list = timeseries_list[:1043]
emotion_label_list = label_list[:1043]

'GAMBLING'
gambling_timeseries_list = timeseries_list[1043:2128]
del gambling_timeseries_list[1053]
gambling_label_list = label_list[1043:2128]
del gambling_label_list[1053]

'LANGUAGE'
language_timeseries_list = timeseries_list[2128:3175]
language_label_list = label_list[2128:3175]

'MOTOR'
motor_timeseries_list = timeseries_list[3175:4255]
motor_label_list = label_list[3175:4255]

'RELATIONAL'
relational_timeseries_list = timeseries_list[4255:5295]
relational_label_list = label_list[4255:5295]

'SOCIAL'
social_timeseries_list = timeseries_list[5295:6344]
social_label_list = label_list[5295:6344]

'WM'
wm_timeseries_list = timeseries_list[6344:]
wm_label_list = label_list[6344:]




task_timeseries_list = [emotion_timeseries_list, gambling_timeseries_list, language_timeseries_list, motor_timeseries_list, relational_timeseries_list, social_timeseries_list, wm_timeseries_list]
for j in task_timeseries_list:
    for k in j[:-212]:
        train_timeseries_list.append(k)

for j in task_timeseries_list:
    for k in j[-212:]:
        test_timeseries_list.append(k)



task_label_list = [emotion_label_list, gambling_label_list, language_label_list, motor_label_list, relational_label_list, social_label_list, wm_label_list]
for j in task_label_list:
    for k in j[:-212]:
        train_label_list.append(k)


for j in task_label_list:
    for k in j[-212:]:
        test_label_list.append(k)







torch.save((train_timeseries_list, train_label_list), '/home/jwlee/HMM/ST_GCN/data/train_hcp-task_roi-aal.pth')
torch.save((test_timeseries_list, test_label_list), '/home/jwlee/HMM/ST_GCN/data/test_hcp-task_roi-aal.pth')

