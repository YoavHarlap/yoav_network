import torch

def euclidean_distance(vectors):
    # unpack the vectors into separate tensors
    featsA, featsB = vectors
    # compute the sum of squared distances between the tensors
    sumSquared = torch.sum(torch.square(featsA - featsB), dim=1, keepdim=True)
    # return the euclidean distance between the tensors
    # return torch.sqrt(torch.max(sumSquared, torch.tensor(torch.finfo(sumSquared.dtype).eps)))
    return sumSquared
