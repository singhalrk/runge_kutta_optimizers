'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from logger import KeepProgress
import random

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18

from torch.autograd import Variable

from modified_rk4 import RK4
# from adam_rk4 import AdamRK4

from rk2_heun import RK2_heun
from rk2_ralston import RK2
import json


#import tensorboard_logger


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default = 200, type=int, help='number of epochs')

parser.add_argument('--optimizer', type=str, default="SGD", metavar='N',
                    help='which optimizer to use')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')

parser.add_argument('--epoch_step', default='[-10,-50]', type=str, help='lr steps')
parser.add_argument('--lr_decay', default=0.1, type=float, help='lr decay')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../cifar_data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../cifar_data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
def create_optimizer(args, lr):
    if args.optimizer == "RK4":
       optimizer = RK4(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "RK2_momentum":
       optimizer = RK2(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.wd)

    elif args.optimizer == "RK2":
       optimizer = RK2(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "RK2_heun":
       optimizer = RK2_heun(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)

    elif args.optimizer == "SGD_nesterov":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, nesterov=True, weight_decay=args.wd)

    elif args.optimizer == "SGD_momentum" :
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.wd)

    elif args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)

    return optimizer

def lr_adjust(optimizer, epoch, args, epoch_step) :
    lr = args.lr * (args.lr_decay ** (epoch_step.index(epoch) + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr updated to {0:.3f}'.format(lr))

optimizer = create_optimizer(args, args.lr)
print(args.optimizer)
print(args.lr)

basefile = str(random.randint(1,100000)) + '_' + str(random.randint(1,100000))
progress = KeepProgress(net,args,basefile)

epoch_step = json.loads(args.epoch_step)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print(basefile)

    optimizer = create_optimizer(args, args.lr)
    if epoch in epoch_step:
        lr_adjust(optimizer, epoch, args, epoch_step)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

        def closure():
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        outputs = net(inputs)
        #loss = closure()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx%100 == 0 :
            progress.train_progress({'train_loss': loss.data[0], 'train_accuracy':100.*correct/total})
            # progress.train_progress([epoch, batch_idx * len(inputs), len(trainloader.dataset),
            #     100. * batch_idx / len(trainloader), loss.data[0]])


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    test_loss /= len(testloader.dataset)
    # progress.test_progress([test_loss, correct, len(testloader.dataset),
    #     100. * correct / len(testloader.dataset)])

    vals = {'test_loss':loss.data[0], 'test_accuracy':(100.*correct/total)}
    progress.test_progress(vals)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)
