import os

# import sys
# import time
# import math
# import argparse

import numpy as np
import torch

import pickle

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std



class KeepProgress():

    def __init__(self,net,args,grad_norm = None) :
        self.net = net
        self.args = args

        self.model_data = {}
        self.grad_norm = grad_norm
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        self.epochs = 0
        self.train_plotss = 0


    def prepare_data(self) :
        return None

    def get_grad_norm(self) :
        if self.grad_norm is not None:
            for i,p in enumerate(self.net.parameters()):
                self.grad_norm['param %s'%i].append(torch.norm(p.data))
        else:
            self.grad_norm = {}
            for i,p in enumerate(self.net.parameters()):
                self.grad_norm['param %s'%i] = [torch.norm(p.data)]


    def train_progress(self,vals) :
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(*vals))
        self.train_loss.append(vals[-1])
        self.train_plotss += 1

    def test_progress(self,vals) :
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(*vals))
        self.epochs += 1

        self.test_loss.append(vals[0])
        self.test_accuracy.append(vals[-1])
        self.get_grad_norm()

        if self.epochs%2 == 0: self.pickle_data()
    """
    keep track of lr update, weight decay, other variables

    Another Idea, create file during initialization

    """

    def pickle_data(self) :

        # make data directory
        directory = "optimizer_plots"

        if not os.path.exists(directory):
            os.makedirs(directory)

        print(".... saving data ....")

        base = directory + "/Optimizer_" + str(self.args.optimizer) + "_lr=" +str(self.args.lr) + "_"

        """save test loss and accuracy """
        Test_Loss = base + "test_loss.p"
        pickle.dump(self.test_loss, open(Test_Loss, 'wb'))

        Test_Accuracy = base + "test_accuracy.p"
        pickle.dump(self.test_accuracy, open(Test_Accuracy, "wb"))

        """ save train loss and accuracy """
        Train_Loss = base + "train_loss.p"
        pickle.dump(self.train_loss, open(Train_Loss, 'wb'))

        # Train_Accuracy = base + "train_accuracy.p"
        # pickle.dump(self.test_accuracy, open(Train_Accuracy, "wb"))

        """ save gradient norms  """
        grad_norm_file = base + "grad_norm.p"
        pickle.dump(self.grad_norm, open(grad_norm_file, 'wb'))



#   """Optimizer_RK2_lr=0.01_grad_norm.p"""
