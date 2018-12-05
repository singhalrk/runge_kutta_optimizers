# import torch
import pickle
import os


""""
/home/rs4070/optimizer_models/VAE/New_optimizer_plots/RK2

pickle.load(open(base[i] + "_train_loss.p","rb"))
"""

def get_model_data(optimizer, model_file) :
    filename_models = model_file + "New_optimizer_plots/" + optimizer

    pickle_files = os.listdir(filename_models)
    model_str = [(filename) for filename in pickle_files if ".p" in filename]

    print(model_str)

    nModels = len(model_str)
    model_lr = [(1./float(model_str[i].split('_')[1])) for i in range(nModels)]

    print('number of ' + optimizer + ' models = %s'%nModels)
    print(model_lr)

    models = []
    for i in range(nModels):
        buf = {}
        buf['lr'] = model_lr[i]
        buf['model data'] = pickle.load(open(filename_models + '/' + model_str[i],"rb"))
        models.append(buf)

    return models


# q = get_model_data('RK2', '/home/rs4070/optimizer_models/VAE/')
# print(q[-1]['model data'].keys())
# print(q[-1]['model data']['args'])
