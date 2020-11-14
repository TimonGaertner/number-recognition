import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable




kwargs = {'num_workers':1, 'pin_memory': True}
traindata = torch.utils.data.DataLoader(datasets.MNIST('data', train = True, download = True, transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,),(0.3081,))])), batch_size = 48, shuffle = True, **kwargs)
testdata = torch.utils.data.DataLoader(datasets.MNIST('data', train = False, transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,),(0.3081,))])), batch_size = 48, shuffle = True, **kwargs)

optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.7)

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(traindata):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        out = model(data)
        criterion = func.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id*len(data), len(train.dataset), 100. * batch_id / len(traindata), loss.data[0]
        ))
        