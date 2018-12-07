import json

import tensorflow as tf
from tensorflow import keras


def evaluate_model(config, train_data, train_labels, val_data, val_labels,
                   test_data, test_labels, max_epochs, patience):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience)
    json_config = json.dumps(config)
    model = keras.models.model_from_json(json_config)
    model.fit(
        train_data, train_labels, callbacks=[early_stop], epochs=max_epochs)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return test_acc