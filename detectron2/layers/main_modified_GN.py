from __future__ import print_function

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from group_norm  import *

from torch.utils.data import DataLoader, Dataset, TensorDataset

from gamma_correction import *

from Clustering import *

import random

import os

import numpy as np


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

'''class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)

        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4 * 4 * 50, 500)

        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4 * 4 * 50)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

'''

class Net(nn.Module):
    '''def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(50, 120, kernel_size=5)
        self.conv2_bn = GroupNorm2d(3, 120)
        self.dense1 = nn.Linear(in_features=3000, out_features=128)
        self.dense1_bn = nn.GroupNorm(128, 128)
        self.dense2 = nn.Linear(128, 10)'''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(10)
        self.dense1 = nn.Linear(in_features=250, out_features=60)
        self.dense1_bn = nn.BatchNorm1d(60)
        self.dense2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(-1, 250) #reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    g = 1.786
    image, label = [x for x in iter(train_loader).next()]
    image_vector = []
    for img in image:
        img = img.to('cpu').numpy()
        image_vector.append(img.reshape((img.shape[0] * img.shape[1] * img.shape[2], 1)))
    image_vector = np.asarray(image_vector)
    Data = data_preparation(n_cluster=2, data=image_vector[:, :, 0])
    Group_unified_1 = []
    Group_unified_2 = []
    labels_1 = []
    labels_2 = []
    labels = torch.zeros(label.shape)
    inxx_l1 = 0
    inxx_l2 = 0
    label_l1 = []
    label_l2 = []
    label = label.to('cpu').numpy()
    for inx, val in enumerate(Data.labels_):
        if val == 1:
            tmp_img = np.transpose(image[inx, :, :, :].to('cpu').numpy(), (1, 2, 0))
            tmp_img = adjust_gamma(tmp_img, gamma=g)
            Group_unified_1.append(tmp_img)
            labels_1.append(label[inx])
            #labels[inx] = label[inx]
            label_l1.append(inx)
            #inxx_l1 += 1
        else:
            Group_unified_2.append(np.transpose(image[inx, :, :, :].to('cpu').numpy(), (1, 2, 0)))
            labels_2.append(label[inx])
            #labels[inx] = label[inx]
            label_l2.append(inx)
            #inxx_l2 += 1
    inxxxx = 0
    with open(('cifar10_label_1' + '.txt'), 'w') as fpn:
        for inxxxx in range(len(label_l1)):
            fpn.write(str(label_l1[inxxxx]) + '\n')
        #inxxxx += 1
    inxxxx = 0
    with open(('cifar10_label_2' + '.txt'), 'w') as fpn:
        for inxxxx in range(len(label_l2)):
            fpn.write(str(label_l2[inxxxx]) + '\n')
    #### shaffel the data
    Group_unified_1 = np.transpose(np.asarray(Group_unified_1), (0, 3, 1, 2))
    Group_unified_2 = np.transpose(np.asarray(Group_unified_2), (0, 3, 1, 2))
    #Group_1 = np.asarray(Group_1)
    #Group_2 = np.asarray(Group_2)
    a_1 = [k for k in range(Group_unified_1.shape[0]//args.batch_size_modified)]
    a_2 = [k for k in range(Group_unified_2.shape[0] // args.batch_size_modified)]
    len1 = len(a_1)
    len2 = len(a_2)
    random.shuffle(a_1)
    #labels_1_new = torch.zeros(labels_1.shape)
    #labels_2_new = torch.zeros(labels_2.shape)
    '''for i in range(a_1):
        labels_1_new[i] = labels_1[a_1[i]]'''

    random.shuffle(a_2)
    '''for i in range(a_2):
        labels_2_new[i] = labels_2[a_2[i]]'''


    Group_unified_1 = torch.from_numpy(Group_unified_1)
    Group_unified_2 = torch.from_numpy(Group_unified_2)
    labels_1 = torch.from_numpy(np.array(labels_1))
    labels_2 = torch.from_numpy(np.array(labels_2))
    labels_1 = labels_1.type(torch.LongTensor)
    labels_2 = labels_2.type(torch.LongTensor)
    #for batch_idx, (data, target) in enumerate(train_loader):
    inx = 0
    idx_1 = 0
    idx_2 = 0
    for idx in range(len(a_1) + len(a_2)):
        rnd = random.randint(0, 1)
        '''for xb, yb in data:
            xb_np = xb.numpy()
            xb_np = np.transpose(xb_np, (1, 2, 0))
            xb_GC = adjust_gamma(xb_np, 3)'''
        #Group_unified.narrow(0, inx*args.batch_size_modified, args.batch_size_modified)
        if rnd == 0:
            if len1 > 0:
                data = Group_unified_1.narrow(0, a_1[idx_1]*args.batch_size_modified,
                                              args.batch_size_modified).requires_grad_()
                target = labels_1.narrow(0, a_1[idx_1]*args.batch_size_modified, args.batch_size_modified)
                idx_1 += 1
                len1 -= 1
            else:
                pass
        else:
            if len2 > 0:
                data = Group_unified_2.narrow(0, a_2[idx_2] * args.batch_size_modified,
                                              args.batch_size_modified).requires_grad_()
                target = labels_2.narrow(0, a_2[idx_2] * args.batch_size_modified, args.batch_size_modified)
                idx_2 += 1
                len2 -= 1
            else:
                pass

        data = data.type(torch.FloatTensor)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        Output = model(data)
        output = torch.zeros(Output.shape[0], 2)
        for i in range(Output.shape[0]):
            output[i, 0] = Output[i, 0]
            output[i, 1] = 10
        output = output.to(device)
        '''print(max(target.to('cpu')))
        print(min(Output))
        print(max(Output))'''
        loss = F.nll_loss(Output, target)

        loss.backward()

        optimizer.step()

        if inx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, inx , len(image),

                       100. * inx / len(image), loss.item()))

        inx += args.batch_size_modified
def test(args, model, device, test_loader):
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=60000, metavar='N',

                        help='input batch size for training (default: 6400)')

    parser.add_argument('--batch-size-modified', type=int, default=60, metavar='N',

                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',

                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',

                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',

                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',

                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,

                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',

                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',

                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,

                        help='For Saving the current Model')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(

        datasets.CIFAR10('../data', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                       ])),

        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(

        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize((0.1307,), (0.3081,))

        ])),

        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)

        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()