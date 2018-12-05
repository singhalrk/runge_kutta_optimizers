""""
/home/rs4070/optimizer_models/wideresnet/New_optimizer_plots/RK2_randomNumber

pickle.load(open(base+ ".p","rb"))
"""
import os
import pickle

def get_model_data2(optimizer, model_file) :
    filename_models = model_file

    pickle_files = os.listdir(filename_models)
    optimizer_models = [(p) for p in pickle_files if p.split("_")[0] == optimizer]

    optimizer_models = []
    for p in pickle_files:
        if len(optimizer.split("_"))>1:
            if p.split("_")[0] == optimizer.split("_")[0] and p.split("_")[1]==optimizer.split("_")[1]:
                optimizer_models.append(p)

        else:
            if p.split("_")[0] == optimizer and p.split("_")[1].isdigit() :
                optimizer_models.append(p)


    optimizer_files = [(os.listdir(filename_models + f)) for f in optimizer_models]

    model_str = []
    for filename in optimizer_files:
        model_str.append([])
        for file_name in filename:
            if '.p' in file_name:
                model_str[-1].append(file_name)

    print(optimizer_models)

    nModels = len(model_str)
    model_lr = [([(1./float(model_str[i][j].split('_')[1])) for j in range(len(model_str[i]))]) for i in range(nModels)]


    models = []
    for i in range(nModels):
#         models.append([])
        for j in range(len(model_str[i])):
            buf = {}
            buf['lr'] = model_lr[i][j]
            buf['model number'] = '_%s'%i
            buf['model data'] = pickle.load(open(filename_models + '/' + optimizer_models[i] + '/'  + model_str[i][j],"rb"))
            models.append(buf)

    print(optimizer + " learning rates = %s"%model_lr)

    return models



# a = get_model_data2('RK2', '/home/rs4070/optimizer_models/resnet_cifar/optimizer_plots/')
# print((a))
