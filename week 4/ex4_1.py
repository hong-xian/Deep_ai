import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# hyper parameter
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

train_dataset = datasets.MNIST(root="./MNIST", train=True,
                               transform=transforms.ToTensor(),
                               download=False)
test_dataset = datasets.MNIST(root="./MNIST", train=False,
                              transform=transforms.ToTensor(),
                              download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2)))
        self.classifier = nn.Sequential(nn.Linear(500, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), 500)
        x = self.classifier(x)
        return x


cnn = NN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


loss_list = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        output = cnn(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            loss_list.append(loss)
            print("Epoch: [{} / {}], Step: [{} / {}], loss:{:.8f}".
                  format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        output = cnn(images)
        _, predicted = torch.max(output, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct / total
    print("Accuracy on 10000 images is: {} %".format(100 * accuracy))

plt.plot(loss_list)
plt.show()
