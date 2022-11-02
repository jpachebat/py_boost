import sys
sys.path.append("../..")
import X_py_boost

from X_py_boost import GradientBoosting
from X_py_boost.gpu.losses import AccuracyMetric, BCEMetric

import numpy as np
import pickle
import time
import os

from X_py_boost.callbacks.callback import Callback

class MultipleMetricsHistory(Callback):
    
    def __init__(self, metric_list):
        
        # assume list of callable(y_true, y_pred)
        self.metric_list = metric_list
        self.history = None
        
    def before_train(self, build_info):
        
        self.history = []
        # postprocess fn defined by loss function
        self.postprocess_fn = build_info['model'].loss.postprocess_output
        
    def after_iteration(self, build_info):
        
        iter_info = []
        # single train dataset
        train = build_info['data']['train']
        # lists of val datasets
        valid = build_info['data']['valid']
        # calc metric on train data
        y_pred = self.postprocess_fn(train['ensemble'])
        
        metrics = []
        for fn in self.metric_list:
            metrics.append(float(fn(train['target'], y_pred)))
        iter_info.append(metrics)
        
        # calc metrics on validation sets
        for y_true, y_pred in zip(valid['target'], valid['ensemble']):
            y_pred = self.postprocess_fn(y_pred)
            metrics = []
            for fn in self.metric_list:
                metrics.append(float(fn(y_true, y_pred)))
            iter_info.append(metrics)

        self.history.append(iter_info)
        
        return False
        
    def after_train(self, build_info):
        
        self.history = np.array(self.history)

class TimeHistory(Callback):

    def __init__(self):
        
        # assume list of callable(y_true, y_pred)
        self.start_time = None
        self.history = None
    
    def before_train(self, build_info):
        
        self.history = []
        self.start_time = time.time()

    def after_iteration(self, build_info):
        
        self.history.append(time.time() - self.start_time)
        
        return False

    def after_train(self, build_info):
        
        self.history = np.array(self.history)

        
os.makedirs("results/X_py_boost/3", exist_ok=True)
os.makedirs("results/X_py_boost/4", exist_ok=True)


lambdas = {'Higgs': [100000],
          'epsilon': [10000],
          'covertype': [1, 10], 
          'electricity': [1000],
          'MiniBooNE': [10000]}

ds_names = ['Higgs', 'epsilon', 
    'covertype', 'MiniBooNE', 'electricity']

"""

## Order 3
metrics = {}
times = {}


for ds_name in ds_names:


    print(f'Training on {ds_name}, order 3:')
    metrics[ds_name] = {}
    times[ds_name] = {}
    npzfile = np.load('data/' + ds_name + '.npz', allow_pickle=True)
    X_train, X_test, y_train, y_test = npzfile['X_train'], npzfile['X_test'], npzfile['y_train'], npzfile['y_test']
    eval_sets = [{'X': X_test, 'y': y_test},]
    for lambda_l2 in lambdas[ds_name]:

        metrics_history = MultipleMetricsHistory( [AccuracyMetric(), BCEMetric()] )
        time_history = TimeHistory()
        model = GradientBoosting(loss='bce', order=3, ntrees=10000, lambda_l2=lambda_l2, es=1000, 
                                     callbacks=[metrics_history, time_history], verbose=1000)
        model.fit(X_train, y_train, eval_sets=eval_sets)
        metrics[ds_name][lambda_l2] = metrics_history.history
        times[ds_name][lambda_l2] = time_history.history

    with open(f'results/X_py_boost/3/metrics.json', 'wb') as f:
                pickle.dump(metrics, f)

    with open(f'results/X_py_boost/3/times.json', 'wb') as f:
                pickle.dump(times, f)

"""
## Order 4
metrics = {}
times = {}


for ds_name in ds_names:
    
    print(f'Training on {ds_name}, order 4:')
    metrics[ds_name] = {}
    times[ds_name] = {}
    npzfile = np.load('data/' + ds_name + '.npz', allow_pickle=True)
    X_train, X_test, y_train, y_test = npzfile['X_train'], npzfile['X_test'], npzfile['y_train'], npzfile['y_test']
    eval_sets = [{'X': X_test, 'y': y_test},]
    for lambda_l2 in lambdas[ds_name]:
        
        metrics_history = MultipleMetricsHistory( [AccuracyMetric(), BCEMetric()] )
        time_history = TimeHistory()
        model = GradientBoosting(loss='bce', order=4, ntrees=10000, lambda_l2=lambda_l2, es=1000, 
                                     callbacks=[metrics_history, time_history], verbose=1000)
        model.fit(X_train, y_train, eval_sets=eval_sets)
        metrics[ds_name][lambda_l2] = metrics_history.history
        times[ds_name][lambda_l2] = time_history.history

with open(f'results/X_py_boost/4/metrics.json', 'wb') as f:
            pickle.dump(metrics, f)
        
with open(f'results/X_py_boost/4/times.json', 'wb') as f:
            pickle.dump(times, f)