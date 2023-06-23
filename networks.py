import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # Calculate the output size of the convnet based on input size
        conv_out_size = self._get_conv_out_size((1, 360, 360))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def _get_conv_out_size(self, shape):
        o = self.convnet(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()

        # Adjust the input size of fc1 based on the output size of embedding_net
        embedding_out_size = self._get_embedding_out_size((1, 360, 360))
        self.fc1 = nn.Linear(embedding_out_size, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

    def _get_embedding_out_size(self, shape):
        o = self.embedding_net(torch.zeros(1, *shape))
        return o.size(1)

#
# class EmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()
#         self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2))
#
#         self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 2)
#                                 )
#
#     def forward(self, x):
#         output = self.convnet(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output
#
#     def get_embedding(self, x):
#         return self.forward(x)
#
#
# class EmbeddingNetL2(EmbeddingNet):
#     def __init__(self):
#         super(EmbeddingNetL2, self).__init__()
#
#     def forward(self, x):
#         output = super(EmbeddingNetL2, self).forward(x)
#         output /= output.pow(2).sum(1, keepdim=True).sqrt()
#         return output
#
#     def get_embedding(self, x):
#         return self.forward(x)
#
#
# class ClassificationNet(nn.Module):
#     def __init__(self, embedding_net, n_classes):
#         super(ClassificationNet, self).__init__()
#         self.embedding_net = embedding_net
#         self.n_classes = n_classes
#         self.nonlinear = nn.PReLU()
#         self.fc1 = nn.Linear(2, n_classes)
#
#     def forward(self, x):
#         output = self.embedding_net(x)
#         output = self.nonlinear(output)
#         scores = F.log_softmax(self.fc1(output), dim=-1)
#         return scores
#
#     def get_embedding(self, x):
#         return self.nonlinear(self.embedding_net(x))
#

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)