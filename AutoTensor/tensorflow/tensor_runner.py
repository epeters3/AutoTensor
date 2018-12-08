import json

import tensorflow as tf
from tensorflow import keras

from AutoTensor.plot import plot_tf_history
from AutoTensor.tensorflow.model_builder import model_builder


def evaluate_model(config, data, verbose):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=config["patience"])

    model = model_builder(config)

    history = model.fit(
        data.train_data,
        data.train_labels,
        validation_data=(data.val_data, data.val_labels),
        callbacks=[early_stop],
        epochs=config["max_epochs"],
        verbose=verbose)

    if verbose > 0:
        print("history.history.val_acc:\n{}".format(
            str(history.history["val_acc"])))
        plot_tf_history(history, "AutoTensor/reports/model-history.png")

    test_loss, test_acc = model.evaluate(data.test_data, data.test_labels)
    return test_acc