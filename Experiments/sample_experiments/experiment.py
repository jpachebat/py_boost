import sys
sys.path.append("../..")
import X_py_boost

import cupy as cp
import numpy as np
import time 

## Callbacks
from X_py_boost.callbacks.callback import Callback, MultipleMetricsHistory, TimeHistory

from X_py_boost import GradientBoosting
from X_py_boost.gpu.losses import AccuracyMetric, BCEMetric

import os
import pickle

def experiment(params):
    data = np.load(f'data/' + params['ds_name'] + '.npz', allow_pickle=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    eval_sets = [{'X': X_test, 'y': y_test},]

    time = {}
    metrics = {}

    for subsample in params['subsample']:

        metrics_history = MultipleMetricsHistory( [AccuracyMetric(), BCEMetric()] ) # pass the metric list here
        time_history = TimeHistory()
        model = GradientBoosting(loss='bce', order=params['order'], colsample=params['colsample'], subsample=subsample, ntrees=params['ntrees'], max_depth=params['max_depth'], lambda_l2=params['lambda_l2'], es=params['es'], callbacks=[metrics_history, time_history], verbose=params['verbose'])
        
        model.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test},])

        time[subsample] = time_history.history - time_history.history[0]
        metrics[subsample] = metrics_history.history

    os.makedirs(f"results/{params['ds_name']}/{params['order']}", exist_ok=True)

    with open(f'results/{params["ds_name"]}/{params["order"]}/metrics.json', 'wb') as f:
        pickle.dump(metrics, f)

    with open(f'results/{params["ds_name"]}/{params["order"]}/time.json', 'wb') as f:
        pickle.dump(time, f)


