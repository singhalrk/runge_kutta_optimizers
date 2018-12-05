import os
import torch
import pickle


class KeepProgress():
    exp_number = 0

    def __init__(self, net, args, base, grad_norm=None):
        self.net = net
        self.args = args

        self.model_data = {}
        self.grad_norm = grad_norm
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        self.epochs = 0
        self.train_plots = 0
        self.train_best = {'loss': [1e+9, 0], 'accuracy': [0, 0]}
        self.test_best = {'loss': [1e+9, 0], 'accuracy': [0, 0]}

        self.model_data['lr'] = self.args.lr
        self.directory = "optimizer_plots/"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # New_optimizer_plots/RK2_randomNumber_randomNumber
        self.base = str(self.args.optimizer) + '_' + str(base)
        if not os.path.exists(self.directory + self.base):
            os.makedirs(self.directory + self.base)

        print(self.base)
        self.__class__.exp_number += 1
        self.model_data['exp_number'] = self.__class__.exp_number

    def get_grad_norm(self):
        if self.grad_norm is not None:
            for i, p in enumerate(self.net.parameters()):
                self.grad_norm['param %s' % i].append(torch.norm(p.data))
        else:
            self.grad_norm = {}
            for i, p in enumerate(self.net.parameters()):
                self.grad_norm['param %s' % i] = [torch.norm(p.data)]

    def train_progress(self, vals):
        self.train_loss.append(vals['train_loss'])
        self.train_accuracy.append(vals['train_accuracy'])
        self.train_plots += 1

        dum = self.train_best['loss'][0]
        dum2 = self.train_best['accuracy'][0]

        if vals['train_loss'] <= dum:
            self.train_best['loss'] = [vals['train_loss'], self.train_plots]

        if vals['train_accuracy'] > dum2:
           self.train_best['accuracy'] = [vals['train_accuracy'],
                                          self.train_plots]

        print("train loss = {}, train accuracy = {}".format(vals['train_loss'],
                                                            vals['train_accuracy']))

        self.pickle_data()

    def test_progress(self,vals) :
        self.epochs += 1
        self.test_loss.append(vals['test_loss'])
        self.test_accuracy.append(vals['test_accuracy'])
        # self.get_grad_norm()

        dum = self.test_best['loss'][0]
        dum2 = self.test_best['accuracy'][0]
        if vals['test_loss'] <= dum:
            self.test_best['loss'] = [vals['test_loss'], self.train_plots]

        if vals['test_accuracy'] >= dum2:
            self.test_best['accuracy'] = [vals['test_accuracy'],
                                          self.train_plots]

        print("test loss = {}, test accuracy = {}".format(vals['test_loss'],
                                                          vals['test_accuracy']))

        if self.epochs % 1 == 0:
            self.pickle_data()

    def pickle_data(self):
        if self.train_plots % 5 == 0:
            print(".... saving data ....")

        self.model_data['args'] = self.args

        self.model_data['train best perf'] = self.train_best
        self.model_data['test best perf'] = self.test_best

        self.model_data['test loss'] = self.test_loss
        self.model_data['test accuracy'] = self.test_accuracy

        self.model_data['train accuracy'] = self.train_accuracy
        self.model_data['train loss'] = self.train_loss

        # self.model_data['grad norm'] = self.grad_norm

        model_file = self.directory + self.base
        model_file += "/lr_" + str(1. / self.args.lr) + '_model.p'

        pickle.dump(self.model_data, open(model_file, 'wb'))
