import torch.nn as nn

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()


        self.fc1 = nn.Linear(in_features=6670, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)
        # self.fc4 = nn.Linear(in_features=64, out_features=2)



    def forward(self, x):  # x : (1, 284, 116)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)

        return x