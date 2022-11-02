import numpy as np

## Get results for a model

def get_results(model):
    return list(np.array(model.history).reshape(len(model.history)))


## Generate synthetic data
def generate_synthetic_data():

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000000, n_features=100, n_informative=10, n_clusters_per_class=4, random_state=0)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return(X_train, X_test, y_train, y_test)

## Results for py_boost
def accuracy_per_time_py_boost(X_train, X_test, y_train, y_test):

    import numpy as np

    from py_boost import GradientBoosting
    import time


    start = time.time()

    model = GradientBoosting(loss='bce', ntrees=1000, es=500, max_depth=10, lambda_l2=0.001, verbose=10000)
    model.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test},])

    end = time.time()


    time = end - start #computation time 

    y = get_results(model)

    time_per_iter = time / len(y)
    x = [i * time_per_iter  for i in range(len(y))]

    return(x, y)

def accuracy_per_time_X_py_boost(X_train, X_test, y_train, y_test):

    import numpy as np

    import sys
    sys.path.append("../")
    from X_py_boost import GradientBoosting
    import time


    start = time.time()

    model = GradientBoosting(loss='bce4', ntrees=1000, es=500, max_depth=10, lambda_l2=0.001, verbose=10000)
    model.fit(X_train, y_train, eval_sets=[{'X': X_test, 'y': y_test},])

    end = time.time()


    time = end - start #computation time 

    y = get_results(model)

    time_per_iter = time / len(y)
    x = [i * time_per_iter  for i in range(len(y))]

    return(x, y)

## plot results
def plot_res(x2, y2, x4, y4):

    fig = go.Figure(layout_title_text="BCE, synthetic data")
    fig.add_trace(go.Scatter(x=x2, y=y2,
                        mode='lines',
                        name='2nd order'))
    fig.add_trace(go.Scatter(x=x4, y=y4,
                        mode='lines',
                        name='4th order'))

    fig.show()




