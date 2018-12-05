import pandas as pd
import pickle
import os
import numpy as np

import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.figure_factory as ff

from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.palettes import Paired12

# import cufflinks as cf

""""
/home/rs4070/optimizer_models/wideresnet/New_optimizer_plots/RK2_randomNumber

pickle.load(open(base+ ".p","rb"))

/Users/rsinghal/Desktop/DeepLearning/Scripts/rk4/rk4-new/mnist/optimizer_plots
"""

# key_ = ['optimizer', 'lr', 'momentum', 'wd', 'epochs', 'epoch_step', 'lr_decay']
key_ = ['optimizer', 'lr', 'epochs']


def get_data(filename):
    _ = os.listdir(filename)
    model_files = [(ff) for ff in _ if ff.split('_')[-1].isnumeric()]
    model_dict = {}

    for m_name in model_files:
        # print(m_name, os.listdir(filename + '/' + m_name))
        arr = os.listdir(filename + '/' + m_name)
        model_dict[m_name] = pickle.load(open(filename + '/' + m_name + '/' + arr[0], "rb"))
    return model_dict

def get_details(data, key=key_) :
    names = list(data.keys())
    interest = ['best train loss', 'best train acc', 'best test loss', 'best test acc']

    arr = []
    for nm in names:
        dummy = []

        dummy.append(data[nm]['train best perf']['loss'][0])
        dummy.append(data[nm]['train best perf']['accuracy'][0])
        dummy.append(data[nm]['test best perf']['loss'][0])
        dummy.append(data[nm]['test best perf']['accuracy'][0])

        for val in key:
            dummy.append(vars(data[nm]['args'])[val])
        arr.append(dummy)

    df = pd.DataFrame(arr, columns=interest + key, index=names)
    # plot_details(df.T)
    return df

def process_data(filename, optim_list=['RK2','sgd'], lr_list=[0.1]):
    # show model details in a table - done in function get_details
    raw_data = get_data(filename)
    model_names = list(raw_data.keys())
    details = get_details(raw_data)

    # dict_keys(['lr', 'exp_number', 'args', 'train best perf', 'test best perf', 'test loss', 'test accuracy', 'train accuracy', 'train loss'])

    # ['best train loss', 'best train acc', 'best test loss', 'best test acc','optimizer', 'lr', 'epochs']

    options = dict(plot_width=400, plot_height=300,tools="pan,wheel_zoom,box_zoom,box_select,lasso_select")

    p = figure(width=800, height=600, title='MNIST')

    train_loss = figure(title='Train Loss', **options)
    test_loss = figure(title='Test Loss', **options)

    train_accuracy = figure(title='Train Accuracy', **options)
    test_accuracy = figure(title='Test Accuracy', **options)

    nm_color = Paired12[2:]
    for nm in model_names:
        if details.loc[nm]['optimizer'] not in optim_list:
            continue
        elif details.loc[nm]['lr'] not in lr_list:
            continue
        dummy = raw_data[nm]
        # tr_loss = pd.Series(dummy['train loss'])
        # test_loss = pd.Series(dummy['test loss'])

        tr_loss_log = pd.Series(np.log(dummy['train loss']))
        test_loss_log = pd.Series(np.log(dummy['test loss']))

        test_acc = pd.Series(dummy['test accuracy'])
        tr_acc = pd.Series(dummy['train accuracy'])

        nm_plot = '{}, lr={}'.format(details.loc[nm]['optimizer'], details.loc[nm]['lr'])

        train_loss.line(tr_loss_log.index, tr_loss_log, legend=nm_plot, line_color=nm_color[model_names.index(nm)], alpha=5.5)

        test_loss.line(test_loss_log.index, test_loss_log, legend=nm_plot, line_color=nm_color[model_names.index(nm)], alpha=5.5)

        train_accuracy.line(tr_acc.index, tr_acc, legend=nm_plot, line_color=nm_color[model_names.index(nm)], alpha=5.5)

        test_accuracy.line(test_acc.index, test_acc, legend=nm_plot, line_color=nm_color[model_names.index(nm)], alpha=5.5)

    train_loss.legend.location = "top_right"
    test_loss.legend.location = "top_right"
    train_accuracy.legend.location = "bottom_right"
    test_accuracy.legend.location = "bottom_right"

    p = gridplot([[train_loss, test_loss], [train_accuracy, test_accuracy]], toolbar_location='right')
    show(p)

def plots(df):
    data = [go.Scatter(x=df.index, y=df[col], name=col) for col in df.columns]
    py.iplot(data)

def plot_details(df_models):
    table = ff.create_table(df_models)
    py.iplot(table)

