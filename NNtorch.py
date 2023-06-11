# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from dataset import make_cryo_imgs_arr, makeDataset
import matplotlib.pyplot as plt
# Get data
# train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# dataset = DataLoader(train, 32)
from livelossplot import PlotLosses
import IPython
# Create an instance of the PlotLosses class
liveplot = PlotLosses()

len_images_arr = 600
len_train = 1500

images, labels = make_cryo_imgs_arr(len_images_arr, make_all_images=True)
training_data = makeDataset(images[:len_train], labels[:len_train])
test_data = makeDataset(images[:len_train], labels[:len_train])

train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=50, shuffle=False)


# 1,28,28 - classes 0-9

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

    def forward(self, x):
        return self.model(x)



# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
# Define your custom weights for each class
class_weights = torch.tensor([1.0, 1.0]).to('cuda') # Higher weight for label 0

# Create the CrossEntropyLoss criterion with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("hi00")
    loss_values = []
    accuracy = []
    # for epoch in range(15):  # train for 10 epochs
    #     for batch in train_dataloader:
    #         X, y = batch
    #         X, y = X.to('cuda'), y.to('cuda')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)
    #
    #         # Apply backprop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #
    #
    #     print(f"Epoch:{epoch} loss is {loss.item()}")
    #     loss_values.append(loss.item())
    #
    #     total_loss = 0.0
    #     total_correct = 0
    #     total_samples = 0
    #     #
    #     # # Iterate over the test data
    #     for batch_data, batch_labels in test_dataloader:
    #         # Move batch data and labels to the appropriate device
    #         batch_data = batch_data.to(device)
    #         batch_labels = batch_labels.to(device)
    #
    #         # Forward pass: compute predictions
    #         batch_predictions = clf(batch_data)
    #
    #         # Compute accuracy
    #         _, predicted_labels = torch.max(batch_predictions, 1)
    #         total_correct += (predicted_labels == batch_labels).sum().item()
    #
    #         # Update the total number of samples
    #         total_samples += batch_labels.size(0)
    #
    #     # Calculate average loss and accuracy
    #     accuracy.append(total_correct / total_samples)
    #
    #
    # # Plot the test loss
    # plt.plot(loss_values)
    # # plt.plot(accuracy)
    # plt.xlabel('Iterations')
    # plt.ylabel('Test Loss')
    # plt.title('Test Loss Curve')
    # plt.show()
    num_epochs  =15
    for epoch in range(num_epochs):
        logs = {}
        total_correct = 0
        total_loss = 0
        total_images = 0
        total_val_loss = 0

        # model.train()
        for i, (data, target) in enumerate(train_dataloader):
            images = data.to('cuda:0')
            labels = target.to('cuda:0')

            # Forward propagation
            outputs = clf(images)

            # Calculating loss with softmax to obtain cross entropy loss

            # loss = criterion(outputs, labels)

            loss = loss_fn(outputs, labels)  # ....>

            opt.zero_grad()
            # Backward prop
            loss.backward()

            # Updating gradients
            opt.step()

            # Total number of labels
            total_images += labels.size(0)

            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.detach(), 1)

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

            for data, target in test_dataloader:
                images = data.to('cuda:0')
                labels = target.to('cuda:0')
                outputs = clf(images)

                _, predicted = torch.max(outputs.detach(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_losss += loss.item()

                accuracy = correct / total

            print('Test Accuracy of the model: {} %'.format(100 * correct / total))
            logs['val_' + 'log loss'] = total_losss / total
            logs['val_' + 'Accuracy'] = ((correct / total) * 100)
        # logs['accuracy'] = accuracy
        liveplot.update(logs)
        # liveplot.draw()

    with open('/home/yoavharlap/PycharmProjects/yoav_network/model_state5.pt', 'wb') as f:
        save(clf.state_dict(), f)

