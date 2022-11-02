from sklearn.datasets import fetch_openml

def get_data(data_id=None):

    dataset = fetch_openml(data_id=data_id)

    data = dataset['data']
    label = dataset['target']

    X = data.values.astype('float32')

    if (data_id in [44120]):
        label_01 = (label == 'UP')
        y = label_01.values.astype('int32')

    if (data_id in [44121, 44127, 44129, 44130, 44131, 44089]):
        y = label.values.astype('int32')

    if (data_id in [44122, 44123, 44124]):
        label_01 = (label == 'P')
        y = label_01.values.astype('int32')


    if (data_id in [44125]):
        label_01 = (label == 'h')
        y = label_01.values.astype('int32')

    if (data_id in [44126]):
        label_01 = (label == '2')
        y = label_01.values.astype('int32')

    if (data_id in [44128, 44090, 44091]):
        label_01 = (label == 'True')
        y = label_01.values.astype('int32')

    return(X, y)