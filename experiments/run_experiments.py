from experiments import *

datasets_name = [
    'electricity',
    'covertype',
    'pol',
    'house_16H',
    'kdd_ipums_la_97-small',
    'MagicTelescope',
    'bank-marketing',
    'phoneme',
    'MiniBooNE',
    'Higgs',
    'eye_movements',
    'jannis',
    'credit',
    'california',
    'wine'
]

for ds_name in datasets_name:
    for order in [2, 3, 4]:

        params = {
        'ds_name': ds_name,
        'order': order,
        'lambdas': [0.1, 1, 10, 100, 1000, 10000, 100000],
        'max_depth': 6,
        'ntrees': 2500
            }

        exp = experiment(params)
        exp.get_results()
