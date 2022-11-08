import sys
sys.path.append("../..")
import X_py_boost

import cupy as cp
import numpy as np
import time 


## Callbacks

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


## Training functions
from X_py_boost import GradientBoosting
from X_py_boost.gpu.losses import AccuracyMetric, BCEMetric



class experiment():

    def __init__(self, params):
        self.params = params
        
    def train_model(self, data, order=2, ntrees=2000, max_depth=8, lambda_l2=1):

        (X_train, X_test, X_valid, y_train, y_test, y_valid) = data
        eval_sets = [{'X': X_train, 'y': y_train}, {'X': X_valid, 'y': y_valid}]

        metrics_history = MultipleMetricsHistory( [AccuracyMetric(), BCEMetric()] ) # pass the metric list here
        time_history = TimeHistory()
        model = GradientBoosting(loss='bce', order=order, ntrees=ntrees, max_depth=max_depth, lambda_l2=lambda_l2, es=3000, callbacks=[metrics_history, time_history], verbose=1000)

        model.fit(X_train, y_train, eval_sets=eval_sets)


        # should contain np.ndarray of size (n_iter, n_datasets, n_metrics) 
        # n_datasets = n_val_datasets + 1, fist one is train
        

        return(model, metrics_history.history, time_history.history)

    def get_results(self):
        
        import pickle 
        import os

        ## load data
        npzfile = np.load('data/' + ds_name + '.npz', allow_pickle=True)
        X_train, X_test, X_valid, y_train, y_test, y_valid = npzfile['X_train'], npzfile['X_test'], npzfile['X_valid'], npzfile['y_train'], npzfile['y_test'], npzfile['y_valid']
        data = (X_train, X_test, X_valid, y_train, y_test, y_valid)


        ## get results.
        callbacks = {}
        models = {}

        for lambda_l2 in self.params['lambdas']:
            model, metric_history, time_history = self.train_model(data, order=self.params['order'], ntrees=self.params['ntrees'], max_depth=self.params['max_depth'], lambda_l2=lambda_l2)
            callbacks[lambda_l2] = {'metrics': metric_history, 'time': time_history}
            models[lambda_l2] = model

        os.makedirs(f"results/{ self.params['ds_name'] }/callbacks", exist_ok=True)
        os.makedirs(f"results/{ self.params['ds_name'] }/models", exist_ok=True)

        with open(f'results/{self.params["ds_name"]}/callbacks/{self.params["order"]}.json', 'wb') as f:
            pickle.dump(callbacks, f)


        with open(f'results/{self.params["ds_name"]}/models/{self.params["order"]}.json', 'wb') as f:
            pickle.dump(models, f)
