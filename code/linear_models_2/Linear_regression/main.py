import torchvision.datasets as dsets
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim
# from tqdm import tqdm
import argparse
import random
import json

import torch.backends.cudnn as cudnn


from models import LinearRegression

from rk2_heun import RK2_heun
from rk2_ralston import RK2
from modified_rk4 import RK4
from logger import KeepProgress


parser = argparse.ArgumentParser(description='Linear Models')
parser.add_argument('--epochs', default = 50, type=int, help='number of epochs')

parser.add_argument('--optimizer', type=str, default="SGD", metavar='N',
                    help='which optimizer to use')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')

parser.add_argument('--epoch_step', default='[-10,-50]', type=str, help='lr steps')
parser.add_argument('--lr_decay', default=0.1, type=float, help='lr decay')
parser.add_argument('--model', default='MSE', type=str)




parser.add_argument('--dataset', default='mnist', type=str)
### experiment with batch size
### and adam
parser.add_argument('--batch_size', default=128, type=int)

def get_optimizer(net, args, lr) :
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
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, nesterov=args.nesterov, weight_decay=args.wd)

    elif args.optimizer == "SGD_momentum" :
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.wd)

    elif args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
        #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return optimizer

def lr_adjust(optimizer, epoch, args, epoch_step) :
    lr = args.lr * (args.lr_decay ** (epoch_step.index(epoch) + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr updated to {}'.format(lr))


def loss_fn(opt) :
    return nn.MSELoss()

def data_set(opt):

    # normalize dataset
    # train_set = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    # test_set = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_set = dsets.MNIST(root='../mnist_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    test_set = dsets.MNIST(root='../mnist_data', train=False,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))


    train_load = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False)

    return train_load, test_load


def main() :
    use_cuda = torch.cuda.is_available()
    opt = parser.parse_args()

    print(opt)

    input_size = 784
    #num_classes = 10
    output_size = 10

    net = LinearRegression(input_size, output_size)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    epoch_step = json.loads(opt.epoch_step)

    criterion = loss_fn(opt)
    train_loader, test_loader = data_set(opt)
    # optimizer = get_optimizer(net, opt)

    base_ = str(random.randint(1,100000)) + '_' + str(random.randint(1,100000))
    progress = KeepProgress(net, opt, base=base_)
    optimizer = get_optimizer(net, opt, opt.lr)
    def train(epoch) :

        if epoch in epoch_step:
            lr_adjust(optimizer, epoch, opt, epoch_step)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for t, (images, labels) in enumerate(train_loader) :
            inputs, targets = Variable(images.view(-1, 28*28)), Variable(labels)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            def closure() :
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

            if t%100 == 0 :
                print(t)
                progress.train_progress({ 'train_accuracy':100*correct/total,'train_loss':loss.data[0]})

    def test(epoch) :
        correct = 0
        total = 0
        test_loss = 0
        for inputs, targets in test_loader:

            inputs = Variable(inputs.view(-1, 28*28), volatile=True)
            targets = Variable(targets)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            print(use_cuda)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data[0]

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

        progress.test_progress({'test_loss':test_loss/len(test_loader.dataset), 'test_accuracy':100*correct/total})

    for epoch in range(opt.epochs):
        train(epoch)
        test(epoch)


if __name__ == '__main__' :
    main()
