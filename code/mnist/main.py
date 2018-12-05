import torch
import torch.nn as nn
from torch.autograd import Variable


import torchvision.datasets as dsets
import torchvision.transforms as transforms

import random
import json
import argparse
import torch.backends.cudnn as cudnn
import time
from tqdm import tqdm

from model import CNN

import torch.optim as optim

from modified_rk4 import RK4
from rk2_heun import RK2_heun

from rk2_ralston_f import RK2


from logger import KeepProgress

parser = argparse.ArgumentParser(description='MNIST')

parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')

parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default='[30,50]', type=str, help='lr steps')
parser.add_argument('--lr_decay', default=0.1, type=float, help='lr decay')
parser.add_argument('--batch_size', default=128, type=int)

def data_set(args):
    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def get_optimizer(net, args) :
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.wd)

    if args.optimizer == "RK4":
       optimizer = RK4(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "RK2_momentum":
       optimizer = RK2(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.wd)

    elif args.optimizer == "RK2":
       optimizer = RK2(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "RK2_heun":
       optimizer = RK2_heun(net.parameters(), lr=args.lr, momentum=0.0, weight_decay = args.wd)

    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)

    elif args.optimizer == "SGD_nesterov":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, nesterov=args.nesterov, weight_decay=args.wd)

    elif args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
        #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer

def lr_adjust(optimizer, epoch, args, epoch_step) :
    lr = args.lr * (args.lr_decay ** (epoch_step.index(epoch) + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr updated to {}'.format(lr))

criterion = nn.CrossEntropyLoss()

args = parser.parse_args()

base_ = str(random.randint(1,100000)) + '_' + str(random.randint(1,100000))
print(base_)

net = CNN()
optimizer = get_optimizer(net, args)

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def main():
    epoch_step = json.loads(args.epoch_step)

    print(args)
    progress = KeepProgress(net, args, base=base_)



    train_loader, test_loader = data_set(args)

    def train(epoch) :
        if epoch in epoch_step:
            lr_adjust(optimizer, epoch, args, epoch_step)

        net.train()
        train_loss = 0; correct = 0; total = 0;

        for t, (images, labels) in tqdm(enumerate(train_loader)) :
            images = Variable(images)
            labels = Variable(labels)

            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
                # print(use_cuda)

            def closure() :
                net.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            outputs = net(images)

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            if t%10 == 0 :
                print('\n')
                print('{0:.2f} %'.format(100 * t /len(train_loader.dataset)))
                progress.train_progress({ 'train_accuracy':100*correct/total,'train_loss':loss.data[0], 'epoch': epoch})

    def test(epoch):
        net.eval()
        correct = 0; test_loss = 0

        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output = net(data)
            test_loss += criterion(output, target).data[0]
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / len(test_loader.dataset)))

        progress.test_progress({'test_loss':test_loss/len(test_loader.dataset), 'test_accuracy':100 * correct/len(test_loader.dataset),'epoch': epoch})


    for t in range(args.epochs):
        print(t)

        start = time.clock()
        train(t)
        print('Time Taken', time.clock() - start)

        test(t)

if __name__ == '__main__':
    main()
