import torch


def euclidean_distance(vectors):
    # unpack the vectors into separate tensors
    featsA, featsB = vectors
    # compute the sum of squared distances between the tensors
    sumSquared = torch.sum(torch.square(featsA - featsB), dim=1, keepdim=True)
    # return the euclidean distance between the tensors
    # return torch.sqrt(torch.max(sumSquared, torch.tensor(torch.finfo(sumSquared.dtype).eps)))
    return sumSquared


import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


        # label = 0 --> not similiar

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                              2))
        return loss_contrastive


import torch


def find_label_with_th(output1, output2):
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    threshold = 0.5 * torch.max(euclidean_distance)  # Half of the maximum distance

    label = torch.where(euclidean_distance > threshold, torch.tensor(1), torch.tensor(0))

    return label
