import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
import glob

targetdir = './result'
os.makedirs(os.path.join(targetdir, 'MIMO_model_weights'), exist_ok=True)


# Get Boston Housing dataset
delimiter = ','
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
BostonTrain = pd.read_csv("./train.csv", sep=delimiter)
BostonVal = pd.read_csv("./validation.csv", sep=delimiter)

# Extract input and output data and convert to torch tensors
train_X = BostonTrain.drop(['MEDV'], axis=1)
train_X = torch.tensor(train_X.values, dtype=torch.float32)
train_Y = BostonTrain[['MEDV']]
train_Y = torch.tensor(train_Y.values, dtype=torch.float32)
Val_X = BostonVal.drop(['MEDV'], axis=1)
Val_X = torch.tensor(Val_X.values, dtype=torch.float32) #torch.Size([137, 13])
Val_Y = BostonVal[['MEDV']]
Val_Y = torch.tensor(Val_Y.values, dtype=torch.float32)

# sample input data for MIMO Regression model: independent for each heads
def sample_data(X, Y, num_heads, batch_size):
    sampled_X_data = []
    sampled_Y_data = []
    for _ in range(num_heads):
        indices = torch.randperm(len(X))[:batch_size]
        sampled_X_batch = X[indices]
        sampled_X_data.append(sampled_X_batch)
        sampled_Y_batch = Y[indices]
        sampled_Y_data.append(sampled_Y_batch)

    stacked_X_data = torch.stack(sampled_X_data, dim=1)
    stacked_Y_data = torch.stack(sampled_Y_data, dim=1)

    return stacked_X_data, stacked_Y_data

X_tensor, Y_tensor = sample_data(train_X, train_Y, num_heads=3, batch_size=48)



class Encoder(nn.Module):
    def __init__(self, encode_dim):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, encode_dim)
        )

    def forward(self, x):
        encoded_x = self.encoder_layer(x).squeeze()
        return encoded_x

class MultiheadCritic(nn.Module):
    def __init__(self, encode_dim, num_heads):
        super(MultiheadCritic, self).__init__()
        self.multihead_layer = nn.Sequential(
            nn.Linear(encode_dim*num_heads, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_heads)
        )

    def forward(self, x):
        multihead_output = self.multihead_layer(x)
        return multihead_output

class MIMORegressionModel(nn.Module):
    num_heads: int = 3
    encode_dim: int = 16

    def __init__(self):
        super(MIMORegressionModel, self).__init__()
        self.encoders = nn.ModuleList([Encoder(self.encode_dim) for _ in range(self.num_heads)])
        self.multihead_model = MultiheadCritic(self.encode_dim, self.num_heads)

    def forward(self, x):
        encoded_x = [encoder(x[:, i, :].unsqueeze(1)) for i, encoder in enumerate(self.encoders)]
     
        stacked_encoded_x = torch.stack(encoded_x, dim=1)
        reshape_input = stacked_encoded_x.view(stacked_encoded_x.shape[0], -1)
        mimo_output = self.multihead_model(reshape_input).unsqueeze(2)
        return mimo_output

model = MIMORegressionModel()

# Define loss function and optimizer
def mimo_loss(x, y):
    loss = ((x - y) ** 2).mean(0)
    loss = loss.sum(0)
    return loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
# num_epochs = 2000
# for epoch in tqdm(range(num_epochs)):
#     optimizer.zero_grad()
#
#     X_tensor, Y_tensor = sample_data(train_X, train_Y, num_heads=3, batch_size=48)
#
#     # Forward pass
#     outputs = model(X_tensor)
#     # Compute loss
#     loss = mimo_loss(outputs, Y_tensor)
#
#     # Backpropagation
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         # calculate the accuracy(MSE error) of model for validation data
#         repeated_val_X = Val_X.unsqueeze(1).repeat(1,3,1) #torch.Size([137, 3, 13])
#         predictions = model(repeated_val_X)
#
#         # choose the prediction value: mean or median (in here, I used mean)
#         pred_val = torch.mean(predictions, axis=1)
#         val_loss = criterion(pred_val, Val_Y)
#         print(f"Epoch [{epoch}/{num_epochs}], Acc(MSE): {val_loss.item()}")
#
#         if epoch == 0:
#             lowest_loss = val_loss
#
#         if val_loss < lowest_loss:
#             lowest_loss = val_loss
#             lowest_epoth = epoch
#
#             torch.save({
#                 'loss': lowest_loss,
#                 'epoch': epoch,
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict()},
#                 os.path.join('./result/MIMO_model_weights', 'checkpoint_epoch_{}.pth'.format(epoch)))
#
#             print("save !!")

# =================================================================
# make a code for testing
# 1. predict the values for given test set (shape will be [batch_size, num_heads, 1(value)])
# 2-1. choose the prediction value for free (it can be mean or median for each heads -> shape will be [batch_size, 1])
# 2-2. calculate the uncertainty(variance) for each heads (shape will be [batch_size, 1])
# 3. submit the results of 2-1 and 2-2 for 1 csv file
# there will be 2 columns. first column will be preiction value and second column will be uncertainty(variance)
# ==================================================================



# Extract values from DataFrame and convert to torch tensors
BostonTest = pd.read_csv("./test.csv", sep=delimiter)
test_X = torch.tensor(BostonTest.values, dtype=torch.float32) #torch.Size([51, 13])
print(test_X.shape)


model = MIMORegressionModel()

# load model
path = os.path.join(targetdir, 'MIMO_model_weights')
full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
print(full_path)
checkpoint = torch.load(full_path)
model.load_state_dict(checkpoint['model'])


def evaluation(test_dataset):
    with torch.no_grad():

        repeated_test_X = test_X.unsqueeze(1).repeat(1,3,1)
        predictions = model(repeated_test_X) #torch.Size([51, 3, 1])

        pred_test = torch.mean(predictions, axis=1) #torch.Size([51, 1])
        uncertainty = torch.var(predictions, axis=1) #torch.Size([51, 1])

    return pred_test, uncertainty

pred_test, uncertainty = evaluation(test_X)


combined_tensor = torch.cat((pred_test, uncertainty), dim=1)
numpy_array = combined_tensor.numpy() #(51, 2)
column_indices = ['prediction value', 'uncertainty']


# df = pd.DataFrame(numpy_array, columns=column_indices)
# df.to_csv('regression_MIMO_test.csv', index=False)