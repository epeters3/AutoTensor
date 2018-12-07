# Using Q-Learning To Optimize Neural Net Hyperparameters

Author: Evan Peterson

## Credits

This project was inspired by these two bodies of work:

1. [The Google AutoML Project](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)
2. [Work done by Baker, Gupta, Naik, & Rasker](https://arxiv.org/abs/1611.02167)

## Brainstorming - DELETE ME

> "To find an appropriate model size, it's best to start with relatively few layers and parameters, then begin increasing the size of the layers or adding new layers until you see diminishing returns on the validation loss." - Tensorflow Documentation

**Possible states (TF Keras NN Hyperparameters)**:

-   In Layer Setup:
    -   Number of layers (`int`) (maybe don't try at first, adds complexity that I still need to reason through)
    -   Layer type
        -   Dropout rate (`double`)- "The fraction of the layer's features that are being zeroed-out; it is usually set between 0.2 and 0.5"
        -   Number of nodes in each layer (`int`)
        -   Activation type of the layer (`func` or `string`)
            -   `tf.nn.relu`
            -   `tf.nn.sigmoid`
            -   `"softmax"` - may only be valid for the output layer of the model
        -   The kernel regularizer
            -   Lx regularization coefficient (`double`)
-   In Compile Setup:
    -   Loss function (`string`)
        -   `"binary_crossentropy"`
        -   `"mean_squared_error"`
        -   `"categorical_crossentropy"`
    -   Optimizer (`func result` or `string`)
        -   `tf.train.AdamOptimizer()`
        -   `tf.train.RMSPropOptimizer()`
        -   `tf.train.GradientDescentOptimizer()`
        -   `"rmsprop"`
    -   Metrics
        -   `"categorical_accuracy"`
        -   `"mae"` - Mean absolute error
        -   `"accuracy"`
-   When training/fitting:

    -   epochs (`int`)
    -   Batch size (`int`)
    -   Patience of the early stopping callback (`int`)
    -   steps_per_epoch (`ing`) - "the number of training steps the model runs before it moves to the next epoch."

    ```python
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop])
    ```

**Data to record**:

-   A history of all `(action, state)` pairs tried

**Things to Try**:

-   Epsilon-greedy exploration
-   Remember to normalize the data beforehand
-   Include `None` as an option on all valid hyperparameters
-   Allow the Q-Learning model to add or subtract to numeric hyperparameters, or multiply/divide if it makes sense for that single parameter.
-   Make sure its hooked up to GPU: [https://www.tensorflow.org/guide/using_gpu](https://www.tensorflow.org/guide/using_gpu)
-   Store and edit the model configurations as dictionaries, then convert to JSON string:
    ```python
    # Serialize a model to JSON format
    json_config = model.to_json()
    # Convert it to an editable python dictionary
    config = json.loads(json_config)
    # ... make edits
    # Serialize back to json
    new_json_config = json.dumps(config)
    # Train the model on that new config
    fresh_model = tf.keras.models.model_from_json(new_json_config)
    ```
