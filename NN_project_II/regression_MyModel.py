import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import pandas as pd
import seaborn as split_rngs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os
import glob




targetdir = './result'
os.makedirs(os.path.join(targetdir, 'MyModel_weights'), exist_ok=True)

# Get Boston Housing dataset
delimiter = ','
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
BostonTrain = pd.read_csv("./train.csv", sep=delimiter)
BostonVal = pd.read_csv("./validation.csv", sep=delimiter)
BostonTest = pd.read_csv("./test.csv", sep=delimiter)

# Extract values from DataFrame and convert to torch tensors
train_X = BostonTrain.drop(['MEDV'], axis=1)
train_X = torch.tensor(train_X.values, dtype=torch.float32) #torch.Size([318, 13])
train_Y = BostonTrain[['MEDV']]
train_Y = torch.tensor(train_Y.values, dtype=torch.float32) #torch.Size([318, 1])
Val_X = BostonVal.drop(['MEDV'], axis=1)
Val_X = torch.tensor(Val_X.values, dtype=torch.float32) #torch.Size([137, 13])
Val_Y = BostonVal[['MEDV']]
Val_Y = torch.tensor(Val_Y.values, dtype=torch.float32) #torch.Size([137, 1])
test_X = torch.tensor(BostonTest.values, dtype=torch.float32) #torch.Size([51, 13])





# Define a simple linear regression model
# GBR=GradientBoostingRegressor(n_estimators=1000, learning_rate=.05, loss='huber', max_depth=4,max_features=.3, min_samples_leaf=3, random_state=24)
# GBR.fit(train_X, train_Y)
# print("=====GradientBoostingRegressor=====")
# y_train_pred = GBR.predict(train_X)
# print('MAE train: %.3f' % (mean_absolute_error(train_Y.numpy(), y_train_pred) ))
# plt.scatter(y_train_pred, y_train_pred-train_Y.squeeze().numpy(), c='blue', marker='o', label='Training data')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
# plt.xlim([-10,50])
# plt.show()
# plt.close()
#
# tree_predictions =[]
# for tree in GBR.estimators_:
#     tree_predictions.append(tree[0].predict(test_X))
# # NumPy 배열로 변환
# tree_predictions = np.array(tree_predictions)
# # 예측값의 분산 계산
# variance_predictions = np.var(tree_predictions, axis=0) #(51,)
# # 예측값 계산
# y_test_pred = GBR.predict(test_X) #(51,)
#
# combined_array = np.column_stack((y_test_pred, variance_predictions))
# column_indices = ['prediction value', 'uncertainty']
# df = pd.DataFrame(combined_array, columns=column_indices)
# df.to_csv('regression_MyModel[GBR]_test.csv', index=False)





##########################################################################################################################



# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64, bias=True),
            nn.ELU(),
            nn.Linear(64, 64, bias=True),
            nn.ELU(),
            nn.Linear(64, 64, bias=True),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


num_ensembles = 5
ensemble = [LinearRegressionModel() for _ in range(num_ensembles)]


# Define loss function and optimizer for each model
criterion = nn.MSELoss()  # Mean Squared Error loss
opt = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in ensemble]
#
# Training loop for each model
# num_epochs = 3000
# for epoch in tqdm(range(num_epochs)):
#     for model, optimizer in zip(ensemble, opt):
#
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
#         Val_predictions = torch.stack([model(Val_X) for model in ensemble]) #torch.Size([3, 137, 1])
#
#
#         # choose the prediction value: mean or median (in here, I used mean)
#         val_loss = criterion(Val_predictions, Val_Y)
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
#                 'model4': ensemble[3].state_dict(),
#                 'model5': ensemble[4].state_dict(),
#                 'optimizer': optimizer.state_dict()},
#                 os.path.join('./result/MyModel_weights', 'checkpoint_epoch_{}.pth'.format(epoch)))
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





# load model
path = os.path.join(targetdir, 'MyModel_weights')
full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
print(full_path)
checkpoint = torch.load(full_path)
ensemble[0].load_state_dict(checkpoint['model1'])
ensemble[1].load_state_dict(checkpoint['model2'])
ensemble[2].load_state_dict(checkpoint['model3'])
ensemble[1].load_state_dict(checkpoint['model2'])
ensemble[2].load_state_dict(checkpoint['model3'])






def evaluation(val_dataset, test_dataset):
    with torch.no_grad():



        val_predictions_1 = ensemble[0](val_dataset)
        val_predictions_2 = ensemble[1](val_dataset)
        val_predictions_3 = ensemble[2](val_dataset)
        val_predictions_4 = ensemble[3](val_dataset)
        val_predictions_5 = ensemble[4](val_dataset)
        val_predictions = torch.stack([val_predictions_1, val_predictions_2, val_predictions_3, val_predictions_4, val_predictions_5])  # torch.Size([5, 51, 1])

        val_pred = torch.mean(val_predictions, axis=0)  # torch.Size([51, 1])
        val_uncertainty = torch.var(val_predictions, axis=0)  # torch.Size([51, 1])




        test_predictions_1 = ensemble[0](test_dataset)
        test_predictions_2 = ensemble[1](test_dataset)
        test_predictions_3 = ensemble[2](test_dataset)
        test_predictions_4 = ensemble[3](test_dataset)
        test_predictions_5 = ensemble[4](test_dataset)
        test_predictions = torch.stack([test_predictions_1, test_predictions_2, test_predictions_3, test_predictions_4, test_predictions_5])  # torch.Size([5, 51, 1])

        test_pred = torch.mean(test_predictions, axis=0)  # torch.Size([51, 1])
        test_uncertainty = torch.var(test_predictions, axis=0)  # torch.Size([51, 1])



    return val_pred.numpy(), val_uncertainty.numpy(), test_pred.numpy(), test_uncertainty.numpy()

val_predictions, val_uncertainty, test_predictions, test_uncertainty = evaluation(Val_X, test_X) #(137, 1), (137, 1), (51, 1), (51, 1)

val_predictions_flatten = val_predictions.flatten() #(137,)
val_y = Val_Y.flatten().numpy() #(137,)

plt.scatter(val_predictions_flatten, val_y)
plt.plot([0, 50], [0, 50], '--k')
plt.show()
plt.close()



numpy_array = np.concatenate((test_predictions, test_uncertainty), 1) #(51, 2)
column_indices = ['prediction value', 'uncertainty']


df = pd.DataFrame(numpy_array, columns=column_indices)
# df.to_csv('regression_MyModel[linearregression]_test.csv', index=False)








