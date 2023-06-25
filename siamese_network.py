from torchvision.datasets import MNIST
from torchvision import transforms
# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from dataset import make_cryo_imgs_arr, makeDataset, makeSiameseDataset
import matplotlib.pyplot as plt

from NN_torch_functions import euclidean_distance, ContrastiveLoss, find_label_with_th
import IPython

from torchvision.datasets import MNIST
from torchvision import transforms

# Get data
len_images_arr = 600
len_train = 300
images, labels = make_cryo_imgs_arr(len_images_arr, make_all_images=False)

siamese_train_dataset = makeSiameseDataset(images[:len_train], labels[:len_train], transform=None)
siamese_test_dataset = makeSiameseDataset(images[-len_train:], labels[-len_train:], transform=None)

n_classes = 2

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit

import numpy as np

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

mnist_classes = ['0', '1']
colors = ['#1f77b4', '#ff0000']  # Blue and red colors

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

batch_size = 20
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss

margin = 1.
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 0
log_interval = 100
fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


def plot_embeddings(embeddings, targets, xlim=None, ylim=None, n_classes=n_classes):
    plt.figure(figsize=(10, 10))
    for i in range(n_classes):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)




#
# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if cuda:
#                 # images = images.cuda()
#                 images = [img.cuda() for img in images]
#             print(images[0].shape)
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labels
#
# import matplotlib.pyplot as plt
#
# def extract_embeddings(data_loader, model):
#     with torch.no_grad():
#         model.eval()
#         for batch_idx, (data, target) in enumerate(data_loader):
#             target = target if len(target) > 0 else None
#             if not type(data) in (tuple, list):
#                 data = (data,)
#             if cuda:
#                 data = tuple(d.cuda() for d in data)
#                 if target is not None:
#                     target = target.cuda()
#
#             outputs = model(*data)
#
#             if type(outputs) not in (tuple, list):
#                 outputs = (outputs,)
#             loss_inputs = outputs
#             if target is not None:
#                 target = (target,)
#                 loss_inputs += target
#
#     return embeddings, labels


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            
            if cuda:
                images = [img.cuda() for img in images]
            # length = len(images)
            mod = model(*images)
            length = vec_to_points(mod, embeddings, k)
            # embeddings[k:k+length] = model(*images).numpy()
            labels[k:k+length] = target.numpy()
            k += length
    return embeddings, labels

def vec_to_points(vector,embeddings,index):
    vector0 = vector[0]
    vector1 = vector[1]
    length = len(vector0)

    for i in range(length):
        embeddings[index+i] = [vector0[i],vector1[i]]
    return length

train_embeddings_cl, train_labels_cl = extract_embeddings(siamese_train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl)
val_embeddings_cl, val_labels_cl = extract_embeddings(siamese_test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl)
plt.show()





