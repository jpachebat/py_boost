import numpy as np

datasets_NC = {
    'electricity': 44120,
    'covertype': 44121,
    'pol': 44122,
    'house_16H': 44123,
    'kdd_ipums_la_97-small': 44124,
    'MagicTelescope': 44125,
    'bank-marketing': 44126,
    'phoneme': 44127,
    'MiniBooNE': 44128,
    'Higgs': 44129,
    'eye_movements': 44130,
    'jannis': 44131,
    'credit': 44089,
    'california': 44090,
    'wine': 44091
}

from sklearn.datasets import fetch_openml

## Get X, y values given the dataset's name

def import_dataset(ds_name=None):

    print(f"loading {ds_name}")

    dataset = fetch_openml(data_id=datasets_NC[ds_name])
    data = dataset['data']
    label = dataset['target']
    X = data.values.astype('float32')

    if ds_name == 'electricity':
        y = (label == 'UP').values.astype('int32')

    if ds_name in ['pol', 'house_16H', 'kdd_ipums_la_97-small']:
        y = (label == 'P').values.astype('int32')

    if ds_name in ['MagicTelescope']:
        y = (label == 'h').values.astype('int32')
    
    if ds_name in ['MiniBooNE', 'california', 'wine']:
        y = (label == 'True').values.astype('int32')

    if ds_name in ['bank-marketing', 'phoneme']:
        y = (label == '2').values.astype('int32')
    
    if ds_name in ['covertype', 'Higgs', 'eye_movements', 'jannis', 'credit']:
        y = label.values.astype('int32')

    return(X, y)


from sklearn.model_selection import train_test_split

for ds_name in datasets_NC:

    X, y = import_dataset(ds_name)

    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=0.30, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_t, y_t, test_size=0.50, random_state=42) 

    np.savez('data/' + ds_name, X_train=X_train, X_test=X_test, X_valid=X_valid, y_train=y_train, y_test=y_test, y_valid=y_valid)