from __future__ import print_function
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from AutoTensor.plot import plot_tf_history
from AutoTensor.tensorflow.model_builder import model_builder
from AutoTensor.utils import print_no_nl


def evaluate_model(config, data, num_classes, verbose):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=config["patience"])

    print_no_nl("compiling...")
    model = model_builder(config, num_classes)

    print_no_nl("fitting...")
    history = model.fit(
        data.train_data,
        data.train_labels,
        validation_data=(data.val_data, data.val_labels),
        callbacks=[early_stop],
        epochs=config["max_epochs"],
        verbose=verbose)

    # print("history.history.val_acc:\n{}".format(
    #     str(history.history["val_acc"])))
    if verbose > 0:
        plot_tf_history(history, "AutoTensor/reports/model-history.png")
    print_no_nl("testing...")
    test_loss, test_acc = model.evaluate(
        data.test_data, data.test_labels, verbose=verbose)
    print_no_nl("done")
    return test_acc