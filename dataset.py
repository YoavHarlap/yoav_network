from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import glob, os
import random
from scipy.stats import bernoulli
from scipy import ndimage


def load_mrcs_from_path_to_arr(path):
    images = []
    images_filenames = []
    os.chdir(path)
    for file in glob.glob("*.mrc"):
        # print(file)
        mrc = mrcfile.open(file, 'r')
        mrc_imgs = np.array(mrc.data)
        mrc.close()
        # print(mrc_imgs.shape)
        # print(mrc_imgs.ndim)

        if (mrc_imgs.ndim == 2):  # was just 1 pic in mrc file
            mrc_imgs = [mrc_imgs]

        for i in range(len(mrc_imgs)):
            images.append(mrc_imgs[i])
            images_filenames.append(file)
    return images

import random



def make_imgs_arr_from_labels(labels, good_imgs, outliers_imgs,make_all_images):
    imgs = []
    new_labels = []
    k = 0
    p = 0
    if(make_all_images):
        labels = [1]*len(good_imgs) + [0]*len(outliers_imgs)
        random.shuffle(labels)

    for i in range(len(labels)):
        if (labels[i] != 1):
            new_labels.append([1,0])
            # labels[i] = -1
            img = outliers_imgs[k]
            k = k + 1
        else:
            img =good_imgs[p]
            p = p + 1
            new_labels.append([0,1])

        img = torch.tensor(img).unsqueeze(0)
        print(img.shape)
        imgs.append(img)
    #imgs = np.array(imgs)
    new_labels = torch.tensor(new_labels)
    return imgs, labels


def make_cryo_imgs_arr(len_arr,make_all_images=False):
    good_imgs_path = "/data/yoavharlap/eman_particles/good"
    outliers_imgs_path = "/data/yoavharlap/eman_particles/outliers"
    good_imgs = load_mrcs_from_path_to_arr(good_imgs_path)
    outliers_imgs = load_mrcs_from_path_to_arr(outliers_imgs_path)

    random.shuffle(good_imgs)
    random.shuffle(outliers_imgs)

    print(len(good_imgs))
    print(len(outliers_imgs))
    p = 2 / 3
    random_labels = bernoulli.rvs(p, size=len_arr)
    imgs, true_labels = make_imgs_arr_from_labels(random_labels, good_imgs, outliers_imgs,make_all_images)
    return imgs, true_labels


class makeDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
