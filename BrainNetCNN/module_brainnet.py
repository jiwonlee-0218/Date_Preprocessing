import torch
import torch.nn as nn

class E2EBlock(nn.Module):

    def __init__(self, nroi, num_filter, drop_rate):
        super(E2EBlock, self).__init__()

        self.kernel_size = nroi
        # Spatial Conv
        self.conv1 = nn.Sequential(nn.Conv2d(1, num_filter, kernel_size=[nroi, 1]),
                                   nn.BatchNorm2d(num_filter),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate))
        self.conv2 = nn.Sequential(nn.Conv2d(1, num_filter, kernel_size=[1, nroi]),
                                   nn.BatchNorm2d(num_filter),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate))

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return torch.cat([a] * self.kernel_size, 2) + torch.cat([b] * self.kernel_size, 3)

class E2EBlock2(nn.Module):

    def __init__(self, nroi, num_filter, drop_rate):
        super(E2EBlock2, self).__init__()

        self.kernel_size = nroi
        # Spatial Conv
        self.conv1 = nn.Sequential(nn.Conv2d(num_filter, num_filter, kernel_size=[nroi, 1]),
                                   nn.BatchNorm2d(num_filter),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate))
        self.conv2 = nn.Sequential(nn.Conv2d(num_filter, num_filter, kernel_size=[1, nroi]),
                                   nn.BatchNorm2d(num_filter),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate))

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return torch.cat([a] * self.kernel_size, 2) + torch.cat([b] * self.kernel_size, 3)


class E2NBlock(nn.Module):
    def __init__(self, nroi, num_filter_before, num_filter, drop_rate):
        super(E2NBlock, self).__init__()

        self.kernel_size = nroi
        self.conv = nn.Sequential(nn.Conv2d(num_filter_before, num_filter, kernel_size=[nroi, 1]),
                                  nn.BatchNorm2d(num_filter),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=drop_rate))

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        a = self.conv(x)
        return a


class N2GBlock(nn.Module):
    def __init__(self, nroi, num_filter_before, num_filter, drop_rate):
        super(N2GBlock, self).__init__()

        self.kernel_size = nroi
        self.conv = nn.Sequential(nn.Conv2d(num_filter_before, num_filter, kernel_size=[1, nroi]),
                                  nn.BatchNorm2d(num_filter),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p=drop_rate))
        self.conv = nn.Sequential(nn.Conv2d(num_filter_before, num_filter, kernel_size=[1, nroi]),
                                  nn.BatchNorm2d(num_filter),
                                  nn.ReLU(),
                                  nn.Dropout(p=drop_rate))


        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        a = self.conv(x)
        return a



class CLSBlock(nn.Module):
    def __init__(self, num_filter_a, num_hidden, drop_rate):
        super(CLSBlock, self).__init__()
        self.dense = nn.Sequential(nn.Linear(num_filter_a, num_hidden),
                                   # nn.BatchNorm1d(num_hidden),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate),
                                   nn.LeakyReLU(),
                                   nn.Linear(num_hidden, 32),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate),
                                   nn.Linear(32, 7))

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        a = self.dense(x.squeeze(-1).squeeze(-1))
        return a




class CLSBlock_proto(nn.Module):
    def __init__(self, num_filter_a, num_hidden, drop_rate):
        super(CLSBlock_proto, self).__init__()
        self.dense = nn.Sequential(nn.Linear(num_filter_a, num_hidden),
                                   # nn.BatchNorm1d(num_hidden),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate),
                                   nn.LeakyReLU(),
                                   nn.Linear(num_hidden, 32),
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=drop_rate),
                                   nn.Linear(32, 2))

        self.ProtoCLS = nn.Parameter(torch.randn(2, 2), requires_grad=True)

        """ initialize """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logit = self.dense(x.squeeze(-1).squeeze(-1))
        proto = self.ProtoCLS

        features_square = torch.sum(torch.pow(logit, 2),1, keepdim=True)
        centers_square = torch.sum(torch.pow(proto, 2), 0, keepdim=True)
        features_into_centers = 2*torch.matmul(logit, proto)
        dist_sim = -(features_square+centers_square-features_into_centers)

        return logit, dist_sim

def proto_regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]
    return distance