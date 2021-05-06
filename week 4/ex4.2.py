import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)
batch_size = 32
num_classes = 2
learning_rate = 1e-4
num_epochs = 200
t0 = time()


# change data into Dataset and use DataLoader
class DataFromH5File(torch.utils.data.Dataset):
    def __init__(self, filepath):
        h5file = h5py.File(filepath, 'r')
        key_list = []
        for key in h5file.keys():
            key_list.append(key)
        self.x_data = torch.from_numpy(np.array(h5file[key_list[1]]).reshape((-1, 3, 64, 64))).float()
        self.y_data = torch.from_numpy(np.array(h5file[key_list[2]]))
        self.list_classes = h5file["list_classes"]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]


if torch.cuda.is_available():
    train_set = DataFromH5File("/home/liushuang/PycharmProjects/lab/mydata/ex4_2/train_happy.h5")
    test_set = DataFromH5File("/home/liushuang/PycharmProjects/lab/mydata/ex4_2/test_happy.h5")
else:
    train_set = DataFromH5File("/Users/xiujing/PycharmProjects/lab/mydata/ex4_2/train_happy.h5")
    test_set = DataFromH5File("/Users/xiujing/PycharmProjects/lab/mydata/ex4_2/test_happy.h5")
# torch.Size([600, 3, 64, 64])
# torch.Size([150, 3, 64, 64])
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=True)


# build a model
class NN(nn.Module):
    def __init__(self, num_of_classes):
        super(NN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2)))
        self.classifier = nn.Sequential(nn.Linear(1152, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 20),
                                        nn.ReLU(),
                                        nn.Linear(20, num_of_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), 1152)
        x = self.classifier(x)
        return x


model = NN(num_of_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        loss_list.append(loss.item())
        print("Epoch: [{} / {}], loss:{:.8f}".
              format(epoch + 1, num_epochs, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct / total
    print("Accuracy on 150 images is: {} %".format(100 * accuracy))
    print("running time is:{:.4f}".format(time() - t0))
    plt.plot(loss_list)
    plt.title("on my cpu running time is:{:.4f}".format(time() - t0))
    plt.show()
