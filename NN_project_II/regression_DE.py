import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import pandas as pd
import seaborn as split_rngs
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import glob




targetdir = './result'
os.makedirs(os.path.join(targetdir, 'DE_model_weights'), exist_ok=True)

# Get Boston Housing dataset
delimiter = ','
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
BostonTrain = pd.read_csv("./train.csv", sep=delimiter)
BostonVal = pd.read_csv("./validation.csv", sep=delimiter)

# Extract values from DataFrame and convert to torch tensors
train_X = BostonTrain.drop(['MEDV'], axis=1)
train_X = torch.tensor(train_X.values, dtype=torch.float32) #torch.Size([318, 13])
train_Y = BostonTrain[['MEDV']]
train_Y = torch.tensor(train_Y.values, dtype=torch.float32) #torch.Size([318, 1])
Val_X = BostonVal.drop(['MEDV'], axis=1)
Val_X = torch.tensor(Val_X.values, dtype=torch.float32) #torch.Size([137, 13])
Val_Y = BostonVal[['MEDV']]
Val_Y = torch.tensor(Val_Y.values, dtype=torch.float32) #torch.Size([137, 1])
print(Val_X.shape)




# Define a simple linear regression model2
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear(x)

# Create an ensemble of models
num_ensembles = 3
ensemble = [LinearRegressionModel() for _ in range(num_ensembles)]

# # Define loss function and optimizer for each model
# criterion = nn.MSELoss()  # Mean Squared Error loss
# optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in ensemble]
#
# # Training loop for each model
# num_epochs = 2000
# for epoch in tqdm(range(num_epochs)):
#     for model, optimizer in zip(ensemble, optimizers):
#         # Forward pass
#         outputs = model(train_X)
#         # print(outputs)
#         loss = criterion(outputs, train_Y)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if epoch % 100 == 0:
#         # calculate the accuracy(MSE error) of model for validation data
#         predictions = torch.stack([model(Val_X) for model in ensemble]) #torch.Size([3, 137, 1])
#
#         # choose the prediction value: mean or median (in here, I used mean)
#         pred_val = torch.mean(predictions, axis=0) #torch.Size([137, 1])
#         val_loss = criterion(pred_val, Val_Y)
#         print(f"Epoch [{epoch}/{num_epochs}], Acc(MSE): {val_loss.item()}")
#
#         if epoch == 0:
#             lowest_loss = val_loss
#
#         if val_loss < lowest_loss:
#             lowest_loss = val_loss
#             lowest_epoch = epoch
#
#             torch.save({
#                 'loss': lowest_loss,
#                 'epoch': epoch,
#                 'model1': ensemble[0].state_dict(),
#                 'model2': ensemble[1].state_dict(),
#                 'model3': ensemble[2].state_dict(),
#                 'optimizer': optimizer.state_dict()},
#                 os.path.join('./result/DE_model_weights', 'checkpoint_epoch_{}.pth'.format(epoch)))
#
#             print("save !!")

# =================================================================
# make a code for testing
# 1. predict the values for given test set (shape will be [num_ensemble, batch_size, 1(value)])
# 2-1. choose the prediction value for free (it can be mean or median for each model -> shape will be [batch_size, 1])
# 2-2. calculate the uncertainty(variance) for each model (shape will be [batch_size, 1])
# 3. submit the results of 2-1 and 2-2 for 1 csv file
# there will be 2 columns. first column will be preiction value and second column will be uncertainty(variance)
# ==================================================================

# Extract values from DataFrame and convert to torch tensors
BostonTest = pd.read_csv("./test.csv", sep=delimiter)
test_X = torch.tensor(BostonTest.values, dtype=torch.float32) #torch.Size([51, 13])
print(test_X.shape)


num_ensembles = 3
ensemble = [LinearRegressionModel() for _ in range(num_ensembles)]



# load model
path = os.path.join(targetdir, 'DE_model_weights')
full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
print(full_path)
checkpoint = torch.load(full_path)
ensemble[0].load_state_dict(checkpoint['model1'])
ensemble[1].load_state_dict(checkpoint['model2'])
ensemble[2].load_state_dict(checkpoint['model3'])





def evaluation(test_dataset):
    with torch.no_grad():

        test_predictions_1 = ensemble[0](test_dataset)
        test_predictions_2 = ensemble[1](test_dataset)
        test_predictions_3 = ensemble[2](test_dataset)
        test_predictions = torch.stack([test_predictions_1, test_predictions_2, test_predictions_3]) # torch.Size([3, 51, 1])


        pred_test = torch.mean(test_predictions, axis=0)  # torch.Size([51, 1])
        uncertainty = torch.var(test_predictions, axis=0) # torch.Size([51, 1])

    return pred_test, uncertainty

pred_test, uncertainty = evaluation(test_X)


combined_tensor = torch.cat((pred_test, uncertainty), dim=1)
numpy_array = combined_tensor.numpy() #(51, 2)
column_indices = ['prediction value', 'uncertainty']


df = pd.DataFrame(numpy_array, columns=column_indices)
df.to_csv('regression_DE_test.csv', index=False)

