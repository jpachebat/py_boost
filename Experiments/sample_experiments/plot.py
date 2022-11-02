import pickle

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pandas as pd


def plot(params, set='test', x_axis='iter'):

    if set == 'test':
        t = 1
    
    if set == 'train':
        t = 0

    with open(f'results/{params["ds_name"]}/2/metrics.json', 'rb') as f:
        metric2 = pickle.load(f)

    with open(f'results/{params["ds_name"]}/2/time.json', 'rb') as f:
        time2 = pickle.load(f)

    with open(f'results/{params["ds_name"]}/3/metrics.json', 'rb') as f:
        metric3 = pickle.load(f)

    with open(f'results/{params["ds_name"]}/3/time.json', 'rb') as f:
        time3 = pickle.load(f)

    fig_acc = go.Figure(layout_title_text=f"accuracy on {set}")
    fig_bce = go.Figure(layout_title_text=f"bce on {set}")

    for subsample in metric2:
        

        acc2 = metric2[subsample][:, t, 0] ## acc on: t=1 test, t=0 train

        bce2 = metric2[subsample][:, t, 1] ## bce on: t=1 test, t=0 train
        
        if x_axis == 'iter':
            x2 = [i for i in range(time2[subsample].shape[0])]
            
        if x_axis == 'time':
            x2 = time2[subsample]

        fig_acc.add_trace(go.Scatter(x=x2, y=acc2,
                    mode='lines',
                    name=f'2nd, subsample = {subsample}'))

        fig_bce.add_trace(go.Scatter(x=x2, y=bce2,
                            mode='lines',
                            name=f'2nd, subsample = {subsample}'))

    for subsample in metric3:

        acc3 = metric3[subsample][:, t, 0] ## acc on: t=1 test, t=0 train

        bce3 = metric3[subsample][:, t, 1] ## bce on: t=1 test, t=0 train
        
        if x_axis == 'iter':
            x3 = [i for i in range(time3[subsample].shape[0])]
            
        if x_axis == 'time':
            x3 = time3[subsample]

        fig_acc.add_trace(go.Scatter(x=x3, y=acc3,
                    mode='lines',
                    name=f'3rd, subsample = {subsample}'))

        fig_bce.add_trace(go.Scatter(x=x3, y=bce3,
                            mode='lines',
                            name=f'3rd, subsample = {subsample}'))

    fig_acc.show()
    fig_bce.show()