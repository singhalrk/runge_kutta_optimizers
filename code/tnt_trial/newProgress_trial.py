import os
import torch
import pickle

class KeepProgress():

    def __init__(self,net,args) :
        self.net = net
        self.args = args

        self.model_data = {}
        # self.grad_norm = grad_norm
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        self.epochs = 0
        self.train_plots = 0

        self.model_data['lr'] = self.args['lr']
        self.model_data['lr change'] = 0


    def get_grad_norm(self) :
        if self.grad_norm is not None:
            for i,p in enumerate(self.net.parameters()):
                self.grad_norm['param %s'%i].append(torch.norm(p.data))
        else:
            self.grad_norm = {}
            for i,p in enumerate(self.net.parameters()):
                self.grad_norm['param %s'%i] = [torch.norm(p.data)]


    def train_progress(self,vals) :
        self.train_loss.append(vals['train_loss'])
        self.train_accuracy.append(vals['train_accuracy'])
        self.train_plots += 1

        self.pickle_data()


    def test_progress(self,vals) :
        self.epochs += 1
        self.test_loss.append(vals['test_loss'])
        self.test_accuracy.append(vals['test_accuracy'])
        # self.get_grad_norm()

        if self.epochs%2 == 0: self.pickle_data()
    """
    keep track of lr update, weight decay, other variables

    What to do when args.lr changes
    """

    def pickle_data(self) :

        # make data directory
        directory = "New_optimizer_plots/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        # New_optimizer_plots/RK2
        base = str(self.args['optimizer'])

        if not os.path.exists(directory + base):
            os.makedirs(directory + base)

        print(".... saving data ....")

        """ save model data dictionary  """

        # lr update check
        # + "_lr=" +str(self.args.lr) + "_"
        # save lr update for the first time along with epoch

        self.model_data['args'] = self.args

        if self.model_data['args'] != self.args and self.model_data['lr change'] == 0:
            self.model_data['updated args'] = self.args
            self.model_data['lr change'] = 1
            self.model_data['lr epoch change'] = self.epoch

        self.model_data['test loss'] = self.test_loss
        self.model_data['test accuracy'] = self.test_accuracy


        self.model_data['train accuracy'] = self.train_accuracy
        self.model_data['train loss'] = self.train_loss

        # self.model_data['grad norm'] = self.grad_norm

        # example =  New_optimizer_plots/RK2/lr_10_model.p
        model_file = directory +  base + "/lr_" + str(1. / self.args['lr']) + '_model.p'
        pickle.dump(self.model_data, open( model_file , 'wb'))
