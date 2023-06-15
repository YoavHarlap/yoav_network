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
    # new_labels = torch.tensor(new_labels)
    # new_imgs = torch.tensor(new_labels)
    return imgs, labels

    # return imgs, labels


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





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:32:38 2023

@author: yoavharlap
"""

# import the necessary packages

import numpy as np
import torch


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    pair_indexex = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    labels = np.array(labels)
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
        pair_indexex.append([idxA,idxB,1])
        pair_indexex.append([idxA,negIdx,1])
    # return a 2-tuple of our image pairs and labels

    return pairImages,np.array(pairLabels)

class makeSiameseDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.pair_images, self.labels = make_pairs(images, labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair_images = self.pair_images[idx]
        imgA  = pair_images[0]
        imgB = pair_images[1]
        label = self.labels[idx]
        # if self.transform:
        #     image = self.transform(pair_images)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return imgA, imgB, label
