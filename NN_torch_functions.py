import torch


# 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

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
import numpy as np

import torch

def euclidean_distance(tensor1, tensor2):
    # Ensure the tensors have the same shape
    assert tensor1.size() == tensor2.size(), "Tensors must have the same shape"

    # Compute the element-wise squared difference
    squared_diff = torch.pow(tensor1 - tensor2, 2)

    # Sum the squared differences along appropriate axes
    summed_diff = torch.sum(squared_diff)

    # Take the square root of the summed differences
    distance = torch.sqrt(summed_diff)

    return distance


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        # label = 0 --> not similiar

    def forward(self, output1, output2, label):
        # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        euclid_distance = euclidean_distance(output1,output2)
        print("output1",output1.shape)

        # loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
        #                               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
        #                                                       2))
        loss_contrastive = (label) * torch.pow(euclid_distance, 2) + (1 - label) * torch.pow(euclid_distance, 2)
        print("output1",output1.shape)
        return loss_contrastive


import torch


def find_label_with_th(output1, output2):
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    threshold = 0.5 * torch.max(euclidean_distance)  # Half of the maximum distance

    label = torch.where(euclidean_distance > threshold, torch.tensor(1), torch.tensor(0))

    return label
