### Higher Order Optimization for Gradient Boosting Decision Trees

Here is the code and data for NeurIPS' 2022 Higher Order Optimization workshop paper: "Higher Order Optimization for Gradient Boosting Decision Trees".

It is based on [Py_Boost](https://github.com/sb-ai-lab/Py-Boost). 

## Installation

To run the experiments, you have to download the repo and install requirements.

We use the same requirements as Py_Boost. To install the requirements, install Py_Boost and uninstall it. More on installation on their [page](https://github.com/sb-ai-lab/Py-Boost). 


## Data

Data in the data folder were obtained by running the `create_data.py` file. Each dataset is retrieved from OpenML and split in 80% train, 20% test.

If you wish to recreate data, run:
```bash
cd experiments
python create_data.py
```

## Experiments

To run the experiments, you must meet the requirements of [Py_Boost](https://github.com/sb-ai-lab/Py-Boost).


