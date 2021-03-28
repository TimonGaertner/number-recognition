from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import classes
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import os
import graph


showimages=False
showgraph=True

epochs = 20


kwargs = {'num_workers':0, 'pin_memory': True}

trainloader = torch.utils.data.DataLoader(datasets.MNIST('/data', train=True, download=True,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                                  ])), batch_size=80, shuffle = True, **kwargs)

testloader = torch.utils.data.DataLoader(datasets.MNIST('/data', train=False,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                                  ])), batch_size=80, shuffle = True, **kwargs)





# imshow and the if statement under to show an example image of the dataset

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if showimages == True:
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=4)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2420, 784)
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


net = Net()
net.cuda()

if os.path.isfile('savednet.pt'):
    net = torch.load('savednet.pt')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train
def train(epoch):
        net.train()
        for i, (data, target) in enumerate(trainloader, 0):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(trainloader.dataset),
                       100. * i / len(trainloader), loss.item()))





def test():
    net.eval()
    testloss = 0
    correct = 0

    for data, target in testloader:
        data= data.cuda()
        target= target.cuda()
        outputs=net(data)
        testloss += F.nll_loss(outputs, target, reduction='sum').item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).cpu().sum().item()
    loss= testloss/len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    graph.append2y(100. * correct / len(testloader.dataset))
    if  graph.showgraph==True:
        graph.main()
    elif showgraph==True:
        graph.showgraph=True






for epoch in range(1, epochs):
    train(epoch)
    test()
    torch.save(net, 'savednet.pt')