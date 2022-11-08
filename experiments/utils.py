
import pickle

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pandas as pd

def plot(params, x='iter'):
    print('Training with following parameters:')
    print(pd.DataFrame(params))
    with open(f'results/{params["ds_name"]}/callbacks/{params["order"]}.json', 'rb') as f:
        cb = pickle.load(f)
    if x == 'iter':
        xs = [i for i in range(params['ntrees'])]
    fig1 = go.Figure(layout_title_text=f"accuracy")
    fig2 = go.Figure(layout_title_text=f"bce")

    for lambda_l2 in params['lambdas']:
        
        metrics = cb[lambda_l2]['metrics']
        if x == 'time':
            xs = cb[lambda_l2]['time']
        y1 = metrics[:, 0, 0] ## Accuracy on train
        y2 = metrics[:, 2, 0] ## acc on test
        y3 = metrics[:, 0, 1] ## bce on test
        y4 = metrics[:, 2, 1] ## bce on test

        fig1.add_trace(go.Scatter(x=xs, y=y1,
                            mode='lines',
                            name=f'acc-train lambda={lambda_l2}'))
        fig1.add_trace(go.Scatter(x=xs, y=y2,
                            mode='lines',
                            name=f'acc-valid lambda={lambda_l2}'))
        fig2.add_trace(go.Scatter(x=xs, y=y3,
                            mode='lines',
                            name=f'bce-train lambda={lambda_l2}'))
        fig2.add_trace(go.Scatter(x=xs, y=y4,
                            mode='lines',
                            name=f'bce-valid lambda={lambda_l2}'))

    fig1.show()
    fig2.show()
