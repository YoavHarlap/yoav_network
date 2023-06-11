# Import dependencies
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import make_cryo_imgs_arr, makeDataset


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (360 - 6) * (360 - 6), 2)
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')

with open('/home/yoavharlap/PycharmProjects/yoav_network/model_state5.pt', 'rb') as f:
    print("pppppp")
    clf.load_state_dict(load(f))

len_images_arr = 600
len_train = 400
test_images, labels = make_cryo_imgs_arr(len_images_arr)

import numpy as np
sum = 0
error_indexes = []
for index in range(len(labels)):
    label = labels[index]
    img = test_images[index][0]
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = np.array(img)
    print("hi", img.shape)

    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    # img_tensor = img.to('cuda')
    c = clf(img_tensor)
    # print(c)
    # print("---")
    # print(torch.argmax(c))
    # print("label: ", label)
    if(torch.argmax(c)!=label):
        sum = sum +1
        error_indexes.append(index)
print("sum eror:",sum,"from", index+1)
print(labels[error_indexes])
