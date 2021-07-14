import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


batch_size = 128


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_classes):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        output = h_n[-1, :, :]
        output = self.classifier(output)
        return output


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
train_dataset = torchvision.datasets.MNIST(root="/Users/xiujing/PycharmProjects/lab/MNIST", train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root="/Users/xiujing/PycharmProjects/lab/MNIST", train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)
# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
net = Rnn(28, 10, 2, 20)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        outputs = net(torch.squeeze(images, 1))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print('train loss: %.3f | Acc: %.3f' % (train_loss / total, 100. * correct / total))
    return train_loss / total


def test(epoch):
    print("\nEpoch: %d" % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = net(torch.squeeze(images, 1))
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print('test loss: %.3f | Acc: %.3f' % (test_loss / total, 100. * correct / total))
    return test_loss / total


loss_list = []
for j in range(20):
    loss_list.append(train(j))
    test(j)
plt.plot(loss_list)
plt.show()


