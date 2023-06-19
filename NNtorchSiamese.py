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
# Get data
# train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# dataset = DataLoader(train, 32)
from livelossplot import PlotLosses
from NN_torch_functions import euclidean_distance, ContrastiveLoss,find_label_with_th
import IPython

# Create an instance of the PlotLosses class
liveplot = PlotLosses()

len_images_arr = 600
len_train = 500

images, labels = make_cryo_imgs_arr(len_images_arr, make_all_images=True)
training_data = makeSiameseDataset(images[:len_train], labels[:len_train])
test_data = makeSiameseDataset(images[:len_train], labels[:len_train])

train_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)


# examples = next(iter(train_dataloader))
#
# for label, img  in enumerate(examples):
#    plt.imshow(img.permute(1,2,0))
#    plt.show()
#    print(f"Label: {label}")

# 1,360,360 - 2 classes

# Image Classifier Neural Network
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

    def forward(self, x1, x2):
        # output1 = self.model(x[0])
        # # output2 = self.model(x[1])
        # output2 = (output1+3)*7
        output1 = self.model(x1)
        output2 = self.model(x2)
        return output1, output2


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
# Define your custom weights for each class
class_weights = torch.tensor([1.0, 1.0]).to('cuda')  # Higher weight for label 0

# Create the CrossEntropyLoss criterion with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

loss_fn = nn.CrossEntropyLoss()
contrastive_loss = ContrastiveLoss()
# Training flow
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("hi00")
    loss_values = []
    accuracy = []
    num_epochs = 15
    for epoch in range(num_epochs):
        logs = {}
        total_correct = 0
        total_loss = 0
        total_images = 0
        total_val_loss = 0

        # model.train()
        for i, (imagesA, imagesB, target) in enumerate(train_dataloader):
            # print("11")
            # print("22")

            # imagesA,imagesB = data[0].to('cuda:0'),data[1].to('cuda:0')
            # images = [imagesA,imagesB]
            imgsA = imagesA.to('cuda:0')
            imgsB = imagesB.to('cuda:0')

            labels = target.to('cuda:0')

            # Forward propagation
            outputs = clf(imgsA, imgsB)

            # Calculating loss with softmax to obtain cross entropy loss

            # loss = criterion(outputs, labels)
            # outputs = euclidean_distance(outputs).to('cuda:0')
            labels = labels.squeeze()

            # labels_one_hot = nn.functional.one_hot(labels, num_classes=2)

            # loss = loss_fn(outputs, labels_one_hot.float())  # ....>

            # loss = loss_fn(outputs, labels)  # ....>
            loss = contrastive_loss(outputs[0], outputs[1], labels)
            opt.zero_grad()
            # Backward prop
            loss.backward()

            # Updating gradients
            opt.step()

            # Total number of labels
            total_images += labels.size(0)

            # Obtaining predictions from max value
            # _, predicted = torch.max(outputs.detach(), 1)


            predicted = find_label_with_th(outputs[0],outputs[1])
            # Calculate the number of correct answers



            correct = (predicted == labels).sum().item()

            total_correct += correct
            total_loss += loss.item()
        logs['log loss'] = total_loss / total_images
        logs['Accuracy'] = ((total_correct / total_images) * 100)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, i + 1, len(test_dataloader), (total_loss / total_images),
                      (total_correct / total_images) * 100))

        # Testing the model
        clf.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            total_losss = 0

            for imagesA, imagesB, target in test_dataloader:
                imgsA = imagesA.to('cuda:0')
                imgsB = imagesB.to('cuda:0')
                labels = target.to('cuda:0')
                outputs = clf(imgsA, imgsB)
                labels = labels.squeeze()
                predicted = find_label_with_th(outputs[0], outputs[1])

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_losss += loss.item()

                accuracy = correct / total

                # Check if predicted label and correct label are equal
                equal_labels = predicted == labels

                # Iterate over the batch
                # for i in range(len(equal_labels)):
                #     if equal_labels[i].item() == 1:
                #         # Retrieve the pair of images
                #         imgA = imagesA[i][0]
                #         imgB = imagesB[i][0]
                #
                #         outputs = clf(imgsA, imgsB)
                #
                #         euclidean_distance = torch.nn.functional.pairwise_distance(outputs[0], outputs[1])
                #
                #         plt.subplot(121)
                #         plt.imshow(imgA.cpu().numpy())
                #         plt.title("Image A")
                #
                #         plt.subplot(122)
                #         plt.imshow(imgB.cpu().numpy())
                #         plt.title("Image B")
                #
                #         plt.show()

            print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        liveplot.update(logs)
        # liveplot.draw()

    with open('/home/yoavharlap/PycharmProjects/yoav_network/model_siamese.pt', 'wb') as f:
        save(clf.state_dict(), f)
