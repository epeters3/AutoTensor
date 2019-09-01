# Automatic Neural Net Hyperparameter Tuning

This package can be used to automatically fit a neural network (one hidden layer) to a given dataset and conduct a hyperparameter search in order to improve the accuracy of the model. We plan to add support for conducting neural architecture search as well, where the package explores a variety of neural network layer types, numbers of layers and their composition, and the layers' hyperparameters as well. 

## Basic Usage

### Installation

To get up and running, clone this repository then install the dependencies:

```shell
git clone https://github.com/epeters3/AutoTensor.git
cd AutoTensor
pip install -r requirements.txt
```

Note: This repository requires Python 3.

### Usig Via the CLI

```shell
python -m AutoTensor.main --path path_to_dataset [--target-index index_of_target_column] [--val-ratio validation_set_size] [--test-ratio test_set_size]
```

This will load the dataset found at `path_to_dataset` and attempt to conduct hyperparameter seearch on a neural network with one hidden layer to find a good architecture for this model that maximizes test set accuracy. It will attempt to predict the class label of the column identified by `--target-index`. Note that if `--target-index` is not supplied, the package will assume the last column in the dataset is the target column.

### Using Via Python

```python
from AutoTensor import find_optimal_model
import numpy as np

# Use your dataset. This is an example.
X = np.random.rand(8,2)
y = np.random.choice(['a', 'b'], 8)

find_optimal_model(X, y, val_ratio = 0.15, test_ratio = 0.15)
```

## Credits

While there are many AutoML systems and Neural Architecture Search methods (NAS), this project was inspired by these two bodies of work:

1. [The Google AutoML Project](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)
2. [Work done by Baker, Gupta, Naik, & Rasker](https://arxiv.org/abs/1611.02167)

One of the inspirations behind the package's approach:

> "To find an appropriate model size, it's best to start with relatively few layers and parameters, then begin increasing the size of the layers or adding new layers until you see diminishing returns on the validation loss." - Tensorflow Documentation